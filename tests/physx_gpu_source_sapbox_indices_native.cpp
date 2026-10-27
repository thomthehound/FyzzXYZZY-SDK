// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sapbox_indices_native.h"
#include "physx_gpu/source_sap_dma_native.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
namespace {
ShaderBinary load(const char* file) {
    std::ifstream in(file,std::ios::binary);
    if(!in)throw std::runtime_error("Missing BP_INITIALIZE_SAPBOX shader");
    const std::vector<char> bytes{std::istreambuf_iterator<char>{in},
                                  std::istreambuf_iterator<char>{}};
    ShaderBinary value;
#if defined(FYZZXYZZY_TEST_D3D12)
    value.format=ShaderFormat::Dxil;
#else
    value.format=ShaderFormat::Spirv;
#endif
    value.bytes.assign(bytes.begin(),bytes.end());
    return value;
}
void check(bool cond,const char* message) {
    if(!cond)throw std::runtime_error(message);
}
void run(Context& context,SourceSapBoxIndicesNativeStage& stage,
         bool shared,bool is_new) {
    CudaDeviceAddressMap map;
    constexpr std::uint32_t handles=4u,capacity=6u,slots=12u;
    constexpr std::uint32_t untouched=0xcdcdcdcdu;
    std::array<std::vector<std::uint32_t>,9> initial{},expected{};
    for(std::uint32_t axis=0;axis<3u;++axis) {
        auto& ordered=initial[axis];
        ordered.resize(slots,untouched);
        const std::array<std::uint32_t,4> order{
            axis%4u,(axis+3u)%4u,(axis+1u)%4u,(axis+2u)%4u};
        for(std::uint32_t i=0;i<4u;++i) {
            ordered[i]=(order[i]<<3u)|1u|4u;
            ordered[4u+i]=(order[3u-i]<<3u)|4u;
        }
    }
    for(std::uint32_t i=3u;i<9u;++i)
        initial[i].resize(capacity*2u,untouched);
    expected=initial;
    for(std::uint32_t axis=0;axis<3u;++axis) {
        const auto& ordered=initial[axis];
        auto& sap=expected[(is_new?6u:3u)+axis];
        for(std::uint32_t i=0;i<handles*2u;++i) {
            const auto tagged=ordered[i];
            sap[(tagged>>3u)*2u+((tagged&1u)?0u:1u)]=i;
        }
    }
    std::array<BufferId,9> buffers{};
    std::array<CudaDeviceAddressMap::Address,9> pointers{};
    std::array<std::size_t,9> offsets{};
    BufferId backing=0u;
    if(shared) {
        std::size_t bytes=0u;
        for(std::size_t i=0;i<initial.size();++i){
            bytes=(bytes+255u)&~std::size_t{255u};
            offsets[i]=bytes;
            bytes+=initial[i].size()*4u;
        }
        backing=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
        for(std::size_t i=0;i<initial.size();++i){
            buffers[i]=backing;
            pointers[i]=map.register_slice(backing,offsets[i],initial[i].size()*4u);
            context.upload(backing,offsets[i],initial[i].data(),initial[i].size()*4u);
        }
    }else{
        for(std::size_t i=0;i<initial.size();++i){
            const auto bytes=initial[i].size()*4u;
            buffers[i]=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
            pointers[i]=map.register_allocation(buffers[i],bytes);
            context.upload(buffers[i],0u,initial[i].data(),bytes);
        }
    }
    physx::PxgBroadPhaseDesc source{};
    for(std::uint32_t axis=0;axis<3u;++axis){
        using Handles=std::remove_reference_t<decltype(source.boxHandles[0][0])>;
        using Sap=std::remove_reference_t<decltype(source.boxSapBox1D[0])>;
        source.boxHandles[0][axis]=reinterpret_cast<Handles>(
            static_cast<std::uintptr_t>(pointers[axis]));
        source.boxSapBox1D[axis]=reinterpret_cast<Sap>(
            static_cast<std::uintptr_t>(pointers[3u+axis]));
        source.boxNewSapBox1D[axis]=reinterpret_cast<Sap>(
            static_cast<std::uintptr_t>(pointers[6u+axis]));
    }
    SourceSapDmaNative stream(context,map);
    auto& commands=stream.commands();
    auto reject=[&](auto fn){
        bool threw=false;
        try{fn();}catch(const std::exception&){threw=true;}
        check(threw&&commands.empty(),"Invalid source SAP indices mutated stream");
    };
    auto bad=source;
    using Sap=std::remove_reference_t<decltype(source.boxSapBox1D[0])>;
    if(is_new)bad.boxNewSapBox1D[2]=reinterpret_cast<Sap>(std::uintptr_t{0x1234u});
    else bad.boxSapBox1D[2]=reinterpret_cast<Sap>(std::uintptr_t{0x1234u});
    reject([&]{stage.append(commands,bad,map,{capacity,slots},handles,is_new);});
    reject([&]{stage.append(commands,source,map,{handles-1u,slots},handles,is_new);});
    reject([&]{stage.append(commands,source,map,{capacity,2u*handles-1u},handles,is_new);});
    stage.append(commands,source,map,{capacity,slots},handles,is_new);
    check(commands.size()==1u,"Original source box indices must emit one kernel");
    const auto& dispatch=std::get<DispatchCommand>(commands[0]);
    check(dispatch.groups.x==1u&&dispatch.buffers.size()==6u,
          "Source BP_INITIALIZE_SAPBOX dispatch shape changed");
    stream.submit();
    stream.wait_at_fetch();
    for(std::size_t i=0;i<initial.size();++i){
        std::vector<std::uint32_t> observed(initial[i].size());
        context.download(buffers[i],shared?offsets[i]:0u,
                         observed.data(),observed.size()*4u);
        check(observed==expected[i],
              "Source SAP indexing changed other source buffers or produced wrong indices");
    }
    commands.clear();
    stage.append(commands,physx::PxgBroadPhaseDesc{},map,{},0u,is_new);
    check(commands.empty(),"Zero-handle source SAP indexing appended work");
    if(shared)context.destroy_buffer(backing);
    else for(const auto id:buffers)context.destroy_buffer(id);
}
}
int main(int argc,char** argv){
    try{
        if(argc!=2)throw std::invalid_argument("Require SAP indices shader");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        SourceSapBoxIndicesNativeStage stage(context,load(argv[1]));
        for(const bool is_new:{false,true})
            for(const bool shared:{false,true})
                run(context,stage,shared,is_new);
        std::cout<<"Source SAP box indices source allocations PASS\n";
    }catch(const std::exception& ex){
        std::cerr<<"Source SAP box indices failed: "<<ex.what()<<'\n';
        return 1;
    }
    return 0;
}
