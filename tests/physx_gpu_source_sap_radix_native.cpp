// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_sap_radix_native.h"
#include "physx_gpu/source_sap_update_head_native.h"
#include "physx_gpu/source_sap_update_middle_native.h"
#include "physx_gpu/source_sap_update_native.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
namespace {
ShaderBinary shader(const char* path) {
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("Missing source SAP radix shader");
    const std::vector<char> bytes{
        std::istreambuf_iterator<char>{file},std::istreambuf_iterator<char>{}};
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
void run(Context& context,const SourceSapRadixNativeStage& stage,
         bool wr,bool shared) {
    // WR: 3 previous boxes -> 6 live endpoints, padded to 8.
    // WOR: 2 previous + 1 created -> 6 live endpoints, padded to 8.
    constexpr std::uint32_t boxes=3u,keys=8u;
    constexpr std::uint32_t perAxis=7u;
    constexpr std::uint32_t hist=512u;
    // [projection, rank, tempProjection, tempRank, radixCounts,
    //  previousHandles, currentHandles] for each of the three axes.
    std::array<std::vector<std::uint32_t>,21> initial{};
    std::array<std::vector<std::pair<std::uint32_t,std::uint32_t>>,3> expected{};
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        const auto base=axis*perAxis;
        for(std::uint32_t slot=0;slot<perAxis;++slot)
            initial[base+slot].assign(slot==4u?hist:keys,0xcdcdcdcdu);
        auto& projection=initial[base+0u];
        auto& rank=initial[base+1u];
        auto& old=initial[base+5u];
        auto& current=initial[base+6u];
        for(std::uint32_t i=0u;i<6u;++i) {
            // Deliberate equal projections to verify stable ordering.
            projection[i]=100u+16u*((7u*i+3u*axis)%4u);
            rank[i]=0xf0000000u;
            old[i]=0x1000u+axis*256u+i;
            current[i]=0x2000u+axis*256u+i;
            expected[axis].push_back({
                projection[i],wr?i:current[i]});
        }
        projection[6u]=projection[7u]=0xffffffffu;
        expected[axis].push_back({0xffffffffu,wr?6u:0x3fffffffu});
        expected[axis].push_back({0xffffffffu,wr?7u:0x3fffffffu});
        if(!wr)current[6u]=current[7u]=0x3fffffffu;
        std::stable_sort(expected[axis].begin(),expected[axis].end(),
            [](const auto& a,const auto& b){return a.first<b.first;});
    }
    std::array<BufferId,21> buffers{};
    std::array<CudaDeviceAddressMap::Address,21> pointers{};
    std::array<std::size_t,21> offsets{};
    CudaDeviceAddressMap map;
    BufferId root=0u;
    if(shared) {
        std::size_t total=0u;
        for(std::size_t i=0;i<initial.size();++i) {
            total=(total+255u)&~std::size_t{255u};
            offsets[i]=total;
            total+=initial[i].size()*sizeof(std::uint32_t);
        }
        root=context.create_buffer(total,MemoryPlacement::DeviceLocal);
        for(std::size_t i=0;i<initial.size();++i) {
            buffers[i]=root;
            const auto bytes=initial[i].size()*4u;
            pointers[i]=map.register_slice(root,offsets[i],bytes);
            context.upload(root,offsets[i],initial[i].data(),bytes);
        }
    } else {
        for(std::size_t i=0;i<initial.size();++i) {
            const auto bytes=initial[i].size()*4u;
            buffers[i]=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
            pointers[i]=map.register_allocation(buffers[i],bytes);
            context.upload(buffers[i],0u,initial[i].data(),bytes);
        }
    }
    const auto ptr=[]<class T>(CudaDeviceAddressMap::Address p) {
        return reinterpret_cast<T>(static_cast<std::uintptr_t>(p));
    };
    physx::PxgBroadPhaseDesc source{};
    std::array<physx::PxgRadixSortDesc,6> radix{};
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        const auto start=axis*perAxis;
        using Rank=std::remove_reference_t<decltype(source.boxProjectionRanks[axis])>;
        using Handles=std::remove_reference_t<decltype(source.boxHandles[0][axis])>;
        source.boxProjectionRanks[axis]=ptr.operator()<Rank>(pointers[start+1u]);
        source.boxHandles[1][axis]=ptr.operator()<Handles>(pointers[start+5u]);
        source.boxHandles[0][axis]=ptr.operator()<Handles>(pointers[start+6u]);
        auto& forward=radix[axis];
        auto& reverse=radix[axis+3u];
        using Word=std::uint32_t*;
        forward.inputKeys=ptr.operator()<Word>(pointers[start+0u]);
        forward.inputRanks=ptr.operator()<Word>(
            pointers[start+(wr?1u:6u)]);
        forward.outputKeys=ptr.operator()<Word>(pointers[start+2u]);
        forward.outputRanks=ptr.operator()<Word>(pointers[start+3u]);
        forward.radixBlockCounts=ptr.operator()<Word>(pointers[start+4u]);
        reverse.inputKeys=forward.outputKeys;
        reverse.inputRanks=forward.outputRanks;
        reverse.outputKeys=forward.inputKeys;
        reverse.outputRanks=forward.inputRanks;
        reverse.radixBlockCounts=forward.radixBlockCounts;
    }
    SourceSapDmaNative stream(context,map);
    auto& commands=stream.commands();
    auto bad=radix;
    bad[5u].outputRanks=reinterpret_cast<std::uint32_t*>(
        static_cast<std::uintptr_t>(0x1234u));
    bool rejected=false;
    try {
        if(wr)stage.append_wr(commands,source,bad,map,boxes);
        else stage.append_wor(commands,bad,map,2u,1u);
    }catch(const std::exception&){rejected=true;}
    require(rejected && commands.empty(),
            "Invalid late original radix descriptor partially appended GPU work");
    if(wr)stage.append_wr(commands,source,radix,map,boxes);
    else stage.append_wor(commands,radix,map,2u,1u);
    require(commands.size()==(wr?18u:16u),
            "Source sort did not enqueue all eight count/scatter passes");
    const auto& sorted=std::get<DispatchCommand>(commands[wr?1u:0u]);
    require(sorted.groups.x==32u && sorted.groups.y==3u &&
            sorted.buffers.size()==6u,
            "Original source radix count dispatch shape changed");
    // No artificial per-pass synchronization: the existing GPU source
    // stream submits and reaps once at the actual fetch boundary.
    stream.wait_at_fetch();
    auto read=[&](std::size_t index) {
        std::vector<std::uint32_t> values(initial[index].size());
        context.download(buffers[index],shared?offsets[index]:0u,
                         values.data(),values.size()*4u);
        return values;
    };
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        const auto base=axis*perAxis;
        const auto projections=read(base);
        const auto ranks=read(base+(wr?1u:6u));
        for(std::uint32_t i=0u;i<keys;++i) {
            require(projections[i]==expected[axis][i].first,
                    "Source radix stable projection order differs from PhysX");
            require(ranks[i]==expected[axis][i].second,
                    "Source radix rank/handle payload permutation is not stable");
        }
        require(read(base+5u)==initial[base+5u],
                "Source WR radix mutated previous-frame handle history");
        if(wr) {
            const auto curr=read(base+6u);
            for(std::uint32_t i=0u;i<6u;++i) {
                const auto rank=expected[axis][i].second;
                require(curr[i]==initial[base+5u][rank],
                        "Source WR handle gather used wrong previous rank");
            }
        } else {
            require(read(base+1u)==initial[base+1u],
                    "Source WOR radix overwrote previous WR rank history");
        }
    }
    const auto count=commands.size();
    if(wr)stage.append_wr(commands,source,radix,map,0u);
    else stage.append_wor(commands,radix,map,0u,0u);
    require(commands.size()==count,"Idle source radix frame appended work");
    if(shared)context.destroy_buffer(root);
    else for(auto buffer:buffers)context.destroy_buffer(buffer);
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=7)throw std::invalid_argument("Require source radix shaders and wave count/scatter");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        SourceSapRadixNativeStage stage(
            context,shader(argv[1]),shader(argv[2]),
            shader(argv[3]),shader(argv[4]),shader(argv[5]),false,shader(argv[6]));
        SourceSapRadixNativeStage portable(
            context,shader(argv[1]),shader(argv[2]),
            shader(argv[3]),shader(argv[4]),shader(argv[5]),true,shader(argv[6]));
        run(context,stage,true,false);
        run(context,portable,false,true);
        std::cout<<"Source WR/WOR stable radix, ranks and original handles PASS\n";
    }catch(const std::exception& e) {
        std::cerr<<"Source WR/WOR radix GPU FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
