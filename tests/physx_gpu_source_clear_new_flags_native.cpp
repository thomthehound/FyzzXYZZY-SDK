// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_clear_new_flags_native.h"
#include <algorithm>
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
ShaderBinary load(const char* name) {
    std::ifstream input(name,std::ios::binary);
    if(!input)throw std::runtime_error("Missing source clear-new shader");
    const std::vector<char> bytes{std::istreambuf_iterator<char>{input},
                                  std::istreambuf_iterator<char>{}};
    ShaderBinary shader;
#if defined(FYZZXYZZY_TEST_D3D12)
    shader.format=ShaderFormat::Dxil;
#else
    shader.format=ShaderFormat::Spirv;
#endif
    shader.bytes.assign(bytes.begin(),bytes.end());
    return shader;
}
void check(bool condition,const char* error) {
    if(!condition)throw std::runtime_error(error);
}
void run(Context& context,SourceClearNewFlagsNativeStage& stage,bool shared) {
    constexpr std::uint32_t boxes=5u;
    CudaDeviceAddressMap map;
    std::array<BufferId,7> buffers{};
    std::array<CudaDeviceAddressMap::Address,7> addresses{};
    std::array<std::vector<std::uint32_t>,7> values{};
    values[0]={1u,4u};
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        values[axis+1u].resize(boxes*2u);
        values[axis+4u].resize(boxes*2u);
        for(std::uint32_t h=0u;h<boxes;++h) {
            values[axis+1u][h*2u]=h*2u;
            values[axis+1u][h*2u+1u]=h*2u+1u;
            values[axis+4u][h*2u]=(h<<3u)|5u;
            values[axis+4u][h*2u+1u]=(h<<3u)|4u;
        }
    }
    auto expected=values;
    for(std::uint32_t axis=0u;axis<3u;++axis)
        for(const auto h:{1u,4u}) {
            expected[axis+4u][h*2u]=(h<<3u)|1u;
            expected[axis+4u][h*2u+1u]=h<<3u;
        }
    BufferId backing=0;
    std::array<std::size_t,7> offsets{};
    if(shared) {
        std::size_t bytes=0u;
        for(std::size_t i=0;i<values.size();++i) {
            bytes=(bytes+255u)&~std::size_t{255u};
            offsets[i]=bytes;
            bytes+=values[i].size()*sizeof(std::uint32_t);
        }
        backing=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
        for(std::size_t i=0;i<values.size();++i) {
            buffers[i]=backing;
            addresses[i]=map.register_slice(
                backing,offsets[i],values[i].size()*sizeof(std::uint32_t));
            context.upload(backing,offsets[i],values[i].data(),
                           values[i].size()*sizeof(std::uint32_t));
        }
    } else {
        for(std::size_t i=0;i<values.size();++i) {
            const auto bytes=values[i].size()*sizeof(std::uint32_t);
            buffers[i]=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
            addresses[i]=map.register_allocation(buffers[i],bytes);
            context.upload(buffers[i],0u,values[i].data(),bytes);
        }
    }
    physx::PxgBroadPhaseDesc source{};
    source.numPreviousHandles=3u;
    source.numCreatedHandles=2u;
    source.updateData_createdHandles=
        reinterpret_cast<decltype(source.updateData_createdHandles)>(
            static_cast<std::uintptr_t>(addresses[0]));
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        using Sap=std::remove_reference_t<decltype(source.boxSapBox1D[0])>;
        using Handles=std::remove_reference_t<decltype(source.boxHandles[0][0])>;
        source.boxSapBox1D[axis]=reinterpret_cast<Sap>(
            static_cast<std::uintptr_t>(addresses[axis+1u]));
        source.boxHandles[0][axis]=reinterpret_cast<Handles>(
            static_cast<std::uintptr_t>(addresses[axis+4u]));
    }
    std::vector<BatchCommand> operations;
    auto reject=[&](auto fn){
        bool failed=false;
        try{fn();}catch(const std::exception&){failed=true;}
        check(failed && operations.empty(),
              "Invalid source SAP descriptor appended partial work");
    };
    auto bad=source;
    using Handles=std::remove_reference_t<decltype(source.boxHandles[0][0])>;
    bad.boxHandles[0][2]=reinterpret_cast<Handles>(std::uintptr_t{0x1234u});
    reject([&]{stage.append(operations,bad,map,{boxes,2u*boxes});});
    reject([&]{stage.append(operations,source,map,{1u,2u*boxes});});
    stage.append(operations,source,map,{boxes,2u*boxes});
    check(operations.size()==1u,"Source clear-new must append exactly one dispatch");
    const auto& command=std::get<DispatchCommand>(operations.front());
    check(command.groups.x==32u && command.buffers.size()==7u,
          "Source clear-new grid or binding count changed");
    for(const auto& view:command.buffers)
        check(view.buffer!=0u,"Source clear-new lost a device allocation");
    context.wait_submission(context.submit_batch(operations));
    for(std::size_t i=0;i<values.size();++i) {
        std::vector<std::uint32_t> actual(values[i].size());
        context.download(buffers[i],shared?offsets[i]:0u,actual.data(),
                         actual.size()*sizeof(std::uint32_t));
        check(actual==expected[i],
              "Source clear-new changed wrong endpoints or retained new flags");
    }
    operations.clear();
    source.numCreatedHandles=0u;
    source.updateData_createdHandles=nullptr;
    stage.append(operations,source,map,{});
    check(operations.empty(),"Source no-created frame dispatched BP_CLEAR_NEWFLAG");
    if(shared)context.destroy_buffer(backing);
    else for(const auto id:buffers)context.destroy_buffer(id);
}
} // namespace
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("Require clear-new shader");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        SourceClearNewFlagsNativeStage stage(context,load(argv[1]));
        run(context,stage,false);
        run(context,stage,true);
        std::cout<<"Source SAP clear-new original allocations PASS\n";
    } catch(const std::exception& ex) {
        std::cerr<<"Source SAP clear-new FAILED: "<<ex.what()<<'\n';
        return 1;
    }
    return 0;
}
