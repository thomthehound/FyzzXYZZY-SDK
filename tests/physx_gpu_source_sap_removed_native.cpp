// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_removed_native.h"
#include "physx_gpu/source_sap_dma_native.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
namespace {
constexpr std::uint32_t sentinel=0xcdcdcdcdu;
ShaderBinary read_shader(const char* path) {
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("Missing source SAP removal shader");
    const std::vector<char> contents{
        std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
    ShaderBinary result;
#if defined(FYZZXYZZY_TEST_D3D12)
    result.format=ShaderFormat::Dxil;
#else
    result.format=ShaderFormat::Spirv;
#endif
    result.bytes.assign(contents.begin(),contents.end());
    return result;
}
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}

void run(Context& context,SourceSapRemovedNativeStage& stage,
         bool shared) {
    constexpr std::uint32_t capacity=6,slots=12,removedHandle=2;
    // removed / sap[3] / previous handles[3] / old bounds /
    // current projections[3]. Original PhysX descriptor pointer topology.
    std::array<std::vector<std::uint32_t>,11> input{};
    input[0]={removedHandle};
    for(std::uint32_t axis=0;axis<3u;++axis) {
        input[1u+axis].assign(capacity*2u,sentinel);
        input[4u+axis].resize(slots);
        input[8u+axis].resize(slots);
        for(std::uint32_t i=0;i<slots;++i) {
            input[4u+axis][i]=((i+axis)*8u)|((i&1u)?0u:1u);
            input[8u+axis][i]=1000u+axis*100u+i;
        }
        input[1u+axis][removedHandle*2u]=2u+axis;
        input[1u+axis][removedHandle*2u+1u]=7u+axis;
    }
    input[7].assign(capacity*6u,sentinel);
    std::array<BufferId,11> buffers{};
    std::array<CudaDeviceAddressMap::Address,11> addresses{};
    std::array<std::size_t,11> offsets{};
    CudaDeviceAddressMap map;
    BufferId root=0u;
    if(shared) {
        std::size_t allocation=0u;
        for(std::size_t i=0;i<buffers.size();++i) {
            allocation=(allocation+255u)&~std::size_t{255u};
            offsets[i]=allocation;
            allocation+=input[i].size()*4u;
        }
        root=context.create_buffer(allocation,MemoryPlacement::DeviceLocal);
        for(std::size_t i=0;i<buffers.size();++i) {
            buffers[i]=root;
            addresses[i]=map.register_slice(root,offsets[i],input[i].size()*4u);
            context.upload(root,offsets[i],input[i].data(),input[i].size()*4u);
        }
    } else {
        for(std::size_t i=0;i<buffers.size();++i) {
            const auto size=input[i].size()*4u;
            buffers[i]=context.create_buffer(size,MemoryPlacement::DeviceLocal);
            addresses[i]=map.register_allocation(buffers[i],size);
            context.upload(buffers[i],0u,input[i].data(),size);
        }
    }
    physx::PxgBroadPhaseDesc desc{};
    const auto ptr=[]<typename T>(CudaDeviceAddressMap::Address value) {
        return reinterpret_cast<T>(static_cast<std::uintptr_t>(value));
    };
    desc.numPreviousHandles=3u;
    desc.numCreatedHandles=1u;
    desc.numRemovedHandles=1u;
    desc.updateData_removedHandles=ptr.operator()<decltype(desc.updateData_removedHandles)>(addresses[0]);
    for(std::size_t a=0;a<3u;++a) {
        desc.boxSapBox1D[a]=ptr.operator()<decltype(desc.boxSapBox1D[a])>(addresses[1u+a]);
        desc.boxHandles[1][a]=ptr.operator()<decltype(desc.boxHandles[1][a])>(addresses[4u+a]);
        desc.boxProjections[a]=ptr.operator()<decltype(desc.boxProjections[a])>(addresses[8u+a]);
    }
    desc.oldIntegerBounds=ptr.operator()<decltype(desc.oldIntegerBounds)>(addresses[7]);
    SourceSapDmaNative stream(context,map);
    auto& commands=stream.commands();
    const auto reject=[&](auto run_bad) {
        const auto before=commands.size();
        bool failed=false;
        try{run_bad();}catch(const std::exception&){failed=true;}
        require(failed && commands.size()==before,
                "Invalid source SAP removal partially submitted GPU commands");
    };
    auto bad=desc;
    bad.boxHandles[1][2]=ptr.operator()<decltype(desc.boxHandles[1][2])>(
        static_cast<CudaDeviceAddressMap::Address>(0x1234u));
    reject([&]{stage.append_mark_removed(commands,bad,map,{capacity,slots});});
    reject([&]{stage.append_mark_removed(commands,desc,map,{capacity,3u});});
    reject([&]{stage.append_remove_projections(commands,desc,map,{capacity,7u});});
    stage.append_mark_removed(commands,desc,map,{capacity,slots});
    require(commands.size()==1u,"Source removed-pair stage missing dispatch");
    require(std::get<DispatchCommand>(commands[0]).groups.x==32u,
            "Source removed-pair grid is not original 32x256");
    stream.wait_at_fetch();

    auto read=[&](std::size_t index){
        std::vector<std::uint32_t> data(input[index].size());
        context.download(buffers[index],shared?offsets[index]:0u,
                         data.data(),data.size()*4u);
        return data;
    };
    for(std::uint32_t axis=0;axis<3u;++axis) {
        auto want=input[4u+axis];
        want[2u+axis]|=4u;
        want[7u+axis]|=4u;
        require(read(4u+axis)==want,
                "Original previous-frame SAP handle was not marked deleted");
        require(read(1u+axis)==input[1u+axis],
                "First removal kernel incorrectly invalidated SAP indices");
        require(read(8u+axis)==input[8u+axis],
                "First removal kernel modified current projections");
    }
    auto expectedBounds=input[7];
    for(std::uint32_t axis=0;axis<3u;++axis) {
        expectedBounds[removedHandle*6u+axis]=0xff7fffffu;
        expectedBounds[removedHandle*6u+3u+axis]=0x00800000u;
    }
    require(read(7u)==expectedBounds,
            "Original removed bounds did not receive PxgIntegerAABB::setEmpty");

    bad=desc;
    bad.boxProjections[2]=ptr.operator()<decltype(desc.boxProjections[2])>(
        static_cast<CudaDeviceAddressMap::Address>(0x1234u));
    reject([&]{stage.append_remove_projections(commands,bad,map,{capacity,slots});});
    stage.append_remove_projections(commands,desc,map,{capacity,slots});
    require(commands.size()==1u,"Source deleted-projection stage missing dispatch");
    stream.wait_at_fetch();
    for(std::uint32_t axis=0;axis<3u;++axis) {
        auto sap=input[1u+axis];
        sap[removedHandle*2u]=0xffffffffu;
        sap[removedHandle*2u+1u]=0xffffffffu;
        require(read(1u+axis)==sap,"Source deleted-projection SAP indices mismatch");
        auto projected=input[8u+axis];
        projected[2u+axis]=0xffffffffu;
        projected[7u+axis]=0xffffffffu;
        require(read(8u+axis)==projected,
                "Source removed projections were not invalidated before WOR sort");
    }
    auto quiet=desc;
    quiet.numRemovedHandles=0u;
    stage.append_mark_removed(commands,quiet,map,{});
    stage.append_remove_projections(commands,quiet,map,{});
    require(commands.empty(),"Quiet source SAP removal appended GPU dispatch");
    if(shared)context.destroy_buffer(root);
    else for(const auto b:buffers)context.destroy_buffer(b);
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::invalid_argument("Expected mark and projection shader paths");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        SourceSapRemovedNativeStage stage(
            context,read_shader(argv[1]),read_shader(argv[2]));
        run(context,stage,false);
        run(context,stage,true);
        std::cout<<"Source SAP deletion GPU stages PASS\n";
    }catch(const std::exception& e){
        std::cerr<<"Source SAP deletion GPU stages FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
