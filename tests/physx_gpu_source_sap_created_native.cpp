// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_created_native.h"
#include "physx_gpu/source_sap_dma_native.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
namespace {
constexpr std::uint32_t sentinel=0xcdcdcdcdu;
ShaderBinary read_shader(const char* path) {
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("Missing BP_UPDATE_CREATEDPAIRS shader");
    const std::vector<char> data{
        std::istreambuf_iterator<char>{in},std::istreambuf_iterator<char>{}};
    ShaderBinary out;
#if defined(FYZZXYZZY_TEST_D3D12)
    out.format=ShaderFormat::Dxil;
#else
    out.format=ShaderFormat::Spirv;
#endif
    out.bytes.assign(data.begin(),data.end());
    return out;
}
void require(bool b,const char* message) {
    if(!b)throw std::runtime_error(message);
}
void run(Context& context,SourceSapCreatedNativeStage& stage,
         bool shared,bool odd) {
    constexpr std::uint32_t capacity=6u,slots=10u,createdHandle=4u;
    const std::uint32_t previous=odd?2u:1u;
    const std::uint32_t currentIndex=previous;
    // created, new bounds, old bounds, projection[3], handles[3].
    std::array<std::vector<std::uint32_t>,9> input{};
    input[0]={createdHandle};
    input[1].resize(capacity*6u);
    input[2].assign(capacity*6u,sentinel);
    for(std::uint32_t handle=0;handle<capacity;++handle)
        for(std::uint32_t i=0;i<6u;++i)
            input[1][handle*6u+i]=handle*100u+i;
    for(std::size_t i=3;i<9u;++i)input[i].assign(slots,sentinel);
    std::array<BufferId,9> buffer{};
    std::array<std::size_t,9> offset{};
    std::array<CudaDeviceAddressMap::Address,9> pointer{};
    CudaDeviceAddressMap map;
    BufferId root=0u;
    if(shared) {
        std::size_t allocation=0u;
        for(std::size_t i=0;i<input.size();++i) {
            allocation=(allocation+255u)&~std::size_t{255u};
            offset[i]=allocation;
            allocation+=input[i].size()*4u;
        }
        root=context.create_buffer(allocation,MemoryPlacement::DeviceLocal);
        for(std::size_t i=0;i<input.size();++i) {
            buffer[i]=root;
            pointer[i]=map.register_slice(root,offset[i],input[i].size()*4u);
            context.upload(root,offset[i],input[i].data(),input[i].size()*4u);
        }
    } else {
        for(std::size_t i=0;i<input.size();++i) {
            const auto bytes=input[i].size()*4u;
            buffer[i]=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
            pointer[i]=map.register_allocation(buffer[i],bytes);
            context.upload(buffer[i],0u,input[i].data(),bytes);
        }
    }
    const auto ptr=[]<class T>(CudaDeviceAddressMap::Address address) {
        return reinterpret_cast<T>(static_cast<std::uintptr_t>(address));
    };
    physx::PxgBroadPhaseDesc source{};
    source.numPreviousHandles=previous;
    source.numCreatedHandles=1u;
    source.updateData_createdHandles=ptr.operator()<decltype(source.updateData_createdHandles)>(pointer[0]);
    source.newIntegerBounds=ptr.operator()<decltype(source.newIntegerBounds)>(pointer[1]);
    source.oldIntegerBounds=ptr.operator()<decltype(source.oldIntegerBounds)>(pointer[2]);
    for(std::uint32_t axis=0;axis<3u;++axis) {
        using Projection=std::remove_reference_t<decltype(source.boxProjections[axis])>;
        using Handle=std::remove_reference_t<decltype(source.boxHandles[0][axis])>;
        source.boxProjections[axis]=ptr.operator()<Projection>(pointer[3u+axis]);
        source.boxHandles[0][axis]=ptr.operator()<Handle>(pointer[6u+axis]);
    }
    SourceSapDmaNative stream(context,map);
    auto& commands=stream.commands();
    auto invalid=source;
    invalid.boxHandles[0][2]=ptr.operator()<
        std::remove_reference_t<decltype(source.boxHandles[0][2])>>(
            static_cast<CudaDeviceAddressMap::Address>(0x1234u));
    bool rejected=false;
    try{stage.append(commands,invalid,map,{capacity,slots});}
    catch(const std::exception&){rejected=true;}
    require(rejected && commands.empty(),
            "Bad source created-pair endpoint pointer mutated stream");
    rejected=false;
    try{stage.append(commands,source,map,{capacity,2u});}
    catch(const std::exception&){rejected=true;}
    require(rejected && commands.empty(),
            "Insufficient source created-pair padding did not reject");
    stage.append(commands,source,map,{capacity,slots});
    require(commands.size()==1u,"Source created-pair kernel missing");
    const auto& cmd=std::get<DispatchCommand>(commands.front());
    require(cmd.groups.x==32u && cmd.buffers.size()==9u,
            "Original source created-pair dispatch shape changed");
    stream.wait_at_fetch();
    auto read=[&](std::size_t slot) {
        std::vector<std::uint32_t> observed(input[slot].size());
        context.download(buffer[slot],shared?offset[slot]:0u,
                         observed.data(),observed.size()*4u);
        return observed;
    };
    auto expectedOld=input[2];
    for(std::uint32_t a=0;a<3u;++a) {
        expectedOld[createdHandle*6u+a]=0xff7fffffu;
        expectedOld[createdHandle*6u+3u+a]=0x00800000u;
    }
    require(read(2u)==expectedOld,"Original created volume old bounds not emptied");
    require(read(1u)==input[1],"Original created volume new integer bounds changed");
    for(std::uint32_t axis=0;axis<3u;++axis) {
        auto projections=input[3u+axis];
        auto handles=input[6u+axis];
        projections[2u*currentIndex]=input[1][createdHandle*6u+axis];
        projections[2u*currentIndex+1u]=input[1][createdHandle*6u+3u+axis];
        handles[2u*currentIndex]=(createdHandle<<3u)|3u;
        handles[2u*currentIndex+1u]=(createdHandle<<3u)|2u;
        if(odd) {
            projections[2u*(previous+1u)]=0xffffffffu;
            projections[2u*(previous+1u)+1u]=0xffffffffu;
            handles[2u*(previous+1u)]=0x3fffffffu;
            handles[2u*(previous+1u)+1u]=0x3fffffffu;
        }
        require(read(3u+axis)==projections,
                "Source created-pair projections or odd radix padding wrong");
        require(read(6u+axis)==handles,
                "Source created-pair handles or odd radix padding wrong");
    }
    source.numCreatedHandles=0u;
    stage.append(commands,source,map,{});
    require(commands.empty(),"Quiet source created-pair frame dispatched");
    if(shared)context.destroy_buffer(root);
    else for(auto id:buffer)context.destroy_buffer(id);
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("Expected created-pairs shader");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        SourceSapCreatedNativeStage stage(context,read_shader(argv[1]));
        for(bool shared:{false,true})
            for(bool odd:{false,true})
                run(context,stage,shared,odd);
        std::cout<<"Source GPU SAP created-pair kernel PASS\n";
    }catch(const std::exception& ex) {
        std::cerr<<"Source GPU SAP created-pair kernel FAILED: "<<ex.what()<<'\n';
        return 1;
    }
}
