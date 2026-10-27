// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_updated_native.h"
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
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("Missing source updated-pairs shader");
    const std::vector<char> bytes{
        std::istreambuf_iterator<char>{f},std::istreambuf_iterator<char>{}};
    ShaderBinary result;
#if defined(FYZZXYZZY_TEST_D3D12)
    result.format=ShaderFormat::Dxil;
#else
    result.format=ShaderFormat::Spirv;
#endif
    result.bytes.assign(bytes.begin(),bytes.end());
    return result;
}
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
void run(Context& context,SourceSapUpdatedNativeStage& stage,bool shared) {
    constexpr std::uint32_t capacity=6u, slots=12u;
    // changed, added, removed bitmap words; sap[3], projections[3],
    // encoded bounds. Every pointer is a true source descriptor member.
    std::array<std::vector<std::uint32_t>,10> input{};
    input[0]={0b011111u}; // handles 0,1,2,3,4 changed
    input[1]={0b000010u}; // handle 1 added: exclude
    input[2]={0b000100u}; // handle 2 removed: exclude
    for(std::uint32_t axis=0;axis<3u;++axis) {
        auto& sap=input[3u+axis];
        sap.resize(capacity*2u,sentinel);
        auto& proj=input[6u+axis];
        proj.resize(slots,sentinel);
        for(std::uint32_t handle=0;handle<capacity;++handle) {
            // Reverse the endpoint order: test actual SAP indirection
            // rather than assuming array slot equals handle number.
            sap[handle*2u]=2u*(capacity-1u-handle);
            sap[handle*2u+1u]=2u*(capacity-1u-handle)+1u;
        }
    }
    input[9].resize(capacity*6u);
    for(std::uint32_t handle=0;handle<capacity;++handle)
        for(std::uint32_t component=0;component<6u;++component)
            input[9][handle*6u+component]=100u*handle+component;

    std::array<BufferId,10> buffer{};
    std::array<CudaDeviceAddressMap::Address,10> pointer{};
    std::array<std::size_t,10> offset{};
    CudaDeviceAddressMap map;
    BufferId root=0;
    if(shared) {
        std::size_t bytes=0;
        for(std::size_t i=0;i<input.size();++i) {
            bytes=(bytes+255u)&~std::size_t{255u};
            offset[i]=bytes;
            bytes+=input[i].size()*4u;
        }
        root=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
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
    physx::PxgBroadPhaseDesc desc{};
    desc.aabbMngr_changedHandleBitMapWordCounts=1u;
    desc.aabbMngr_changedHandleMap=ptr.operator()<decltype(desc.aabbMngr_changedHandleMap)>(pointer[0]);
    desc.aabbMngr_addedHandleMap=ptr.operator()<decltype(desc.aabbMngr_addedHandleMap)>(pointer[1]);
    desc.aabbMngr_removedHandleMap=ptr.operator()<decltype(desc.aabbMngr_removedHandleMap)>(pointer[2]);
    for(std::uint32_t axis=0;axis<3u;++axis) {
        using Sap=std::remove_reference_t<decltype(desc.boxSapBox1D[0])>;
        using Projection=std::remove_reference_t<decltype(desc.boxProjections[0])>;
        desc.boxSapBox1D[axis]=ptr.operator()<Sap>(pointer[3u+axis]);
        desc.boxProjections[axis]=ptr.operator()<Projection>(pointer[6u+axis]);
    }
    desc.newIntegerBounds=ptr.operator()<decltype(desc.newIntegerBounds)>(pointer[9]);
    SourceSapDmaNative stream(context,map);
    auto& commands=stream.commands();
    const auto reject=[&](auto fn) {
        auto count=commands.size();
        bool threw=false;
        try{fn();}catch(const std::exception&){threw=true;}
        require(threw&&count==commands.size(),
                "Invalid source changed AABB bitmap partially appended GPU work");
    };
    auto bad=desc;
    bad.boxProjections[2]=ptr.operator()<std::remove_reference_t<decltype(desc.boxProjections[2])>>(
        static_cast<CudaDeviceAddressMap::Address>(0x1234u));
    reject([&]{stage.append(commands,bad,map,{capacity,slots});});
    reject([&]{stage.append(commands,desc,map,{capacity,0u});});
    reject([&]{stage.append(commands,desc,map,{33u,slots});});
    stage.append(commands,desc,map,{capacity,slots});
    require(commands.size()==1u,"Source updated-pairs stage did not append one dispatch");
    const auto& dispatch=std::get<DispatchCommand>(commands.front());
    require(dispatch.groups.x==32u && dispatch.buffers.size()==10u,
            "Source BP_UPDATE_UPDATEDPAIRS grid or source bindings changed");
    stream.wait_at_fetch();
    for(std::uint32_t axis=0;axis<3u;++axis) {
        auto expected=input[6u+axis];
        for(const std::uint32_t handle : {0u,3u,4u}) {
            const std::uint32_t minIndex=input[3u+axis][handle*2u];
            const std::uint32_t maxIndex=input[3u+axis][handle*2u+1u];
            expected[minIndex]=input[9][handle*6u+axis];
            expected[maxIndex]=input[9][handle*6u+axis+3u];
        }
        std::vector<std::uint32_t> observed(slots);
        context.download(buffer[6u+axis],shared?offset[6u+axis]:0u,
                         observed.data(),observed.size()*4u);
        require(observed==expected,
                "Source changed AABB updates did not preserve bitmap exclusion and SAP ranks");
    }
    auto quiet=desc;
    quiet.aabbMngr_changedHandleBitMapWordCounts=0u;
    stage.append(commands,quiet,map,{});
    require(commands.empty(),"Empty source changed AABB bitmap appended GPU commands");
    if(shared)context.destroy_buffer(root);
    else for(const auto id:buffer)context.destroy_buffer(id);
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("Expected updated-pairs shader");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        SourceSapUpdatedNativeStage stage(context,read_shader(argv[1]));
        run(context,stage,false);
        run(context,stage,true);
        std::cout<<"Source AABB manager bitmap projection GPU updates PASS\n";
        return 0;
    }catch(const std::exception& ex) {
        std::cerr<<"Source changed AABB bitmap GPU stage FAILED: "<<ex.what()<<'\n';
        return 1;
    }
}
