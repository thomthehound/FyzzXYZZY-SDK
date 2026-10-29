// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_aggregate_pair_admission_native.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

namespace {
void require(bool good,const char* message) {
    if(!good)throw std::runtime_error(message);
}
ShaderBinary load_shader(const char* path) {
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("Source aggregate pair slot shader unavailable");
    ShaderBinary shader;
#if defined(FYZZXYZZY_TEST_D3D12)
    shader.format=ShaderFormat::Dxil;
#else
    shader.format=ShaderFormat::Spirv;
#endif
    shader.bytes.assign(std::istreambuf_iterator<char>{in},
                        std::istreambuf_iterator<char>{});
    require(!shader.bytes.empty(),"Empty source aggregate pair admission shader");
    return shader;
}
void verify(Context& ctx,ShaderBinary admit,ShaderBinary publish) {
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(ctx,addresses);
    SourceAggregatePairAdmissionNativeStage stage(
        ctx,std::move(admit),std::move(publish));
    struct Allocation {
        BufferId gpu;
        CudaDeviceAddressMap::Address device;
    };
    std::vector<Allocation> roots;
    const auto create=[&](std::size_t bytes) {
        const auto gpu=ctx.create_buffer(bytes);
        const auto device=addresses.register_slice(gpu,0,bytes);
        roots.push_back({gpu,device});
        return device;
    };
    const auto up=[&](CudaDeviceAddressMap::Address address,
                       const void* data,std::size_t bytes) {
        const auto view=addresses.resolve(address,bytes);
        ctx.upload(view.buffer,view.offset,data,bytes);
    };
    const auto down=[&](CudaDeviceAddressMap::Address address,
                       void* data,std::size_t bytes) {
        const auto view=addresses.resolve(address,bytes);
        ctx.download(view.buffer,view.offset,data,bytes);
    };
    constexpr std::uint32_t rootCapacity=4u,pairCapacity=6u;
    const auto descriptor=create(sizeof(physx::PxgBroadPhaseDesc));
    const auto found=create(rootCapacity*8u);
    const auto pairs=create(pairCapacity*12u);
    const auto indices=create(pairCapacity*4u);
    const auto freeList=create(8u);
    const auto pairCount=create(4u);
    const auto overflow=create(12u);
    SourceAggregatePairAdmissionNativeBuffers buffers{
        descriptor,found,pairs,indices,freeList,pairCount,overflow,
        rootCapacity,pairCapacity};
    const auto invalid=[&](auto changes) {
        auto candidate=buffers;
        changes(candidate);
        const auto before=stream.commands().size();
        bool rejected=false;
        try {stage.append(stream,addresses,candidate);}
        catch(const std::invalid_argument&) {rejected=true;}
        require(rejected && stream.commands().size()==before,
                "Invalid aggregate pair source views modified the GPU stream");
    };
    invalid([](auto& x){x.aggregate_pair_capacity=0u;});
    invalid([](auto& x){x.found_aggregate_pairs=x.aggregate_pairs;});

    physx::PxgBroadPhaseDesc source{};
    source.sharedFoundAggPairIndex=3u;
    up(descriptor,&source,sizeof(source));
    const std::array<std::uint32_t,8u> firstFound{{
        1u,10u,2u,11u,3u,12u,0u,0u}};
    up(found,firstFound.data(),sizeof(firstFound));
    const std::array<std::uint32_t,6u> emptyIndices{{
        1u,3u,0u,0u,0u,0u}};
    up(indices,emptyIndices.data(),sizeof(emptyIndices));
    const std::array<std::uint32_t,2u> initialFree{{2u,4u}};
    up(freeList,initialFree.data(),sizeof(initialFree));
    stage.append(stream,addresses,buffers);
    require(stream.commands().size()==2u,
            "Source aggregate GPU admission omitted its two original stages");
    stream.wait_at_fetch();
    std::array<std::uint32_t,2u> afterFree{};
    std::array<std::uint32_t,3u> afterOverflow{};
    std::uint32_t afterCount=0;
    std::array<std::uint32_t,pairCapacity*3u> pairWords{};
    down(freeList,afterFree.data(),sizeof(afterFree));
    down(overflow,afterOverflow.data(),sizeof(afterOverflow));
    down(pairCount,&afterCount,sizeof(afterCount));
    down(pairs,pairWords.data(),sizeof(pairWords));
    require(afterFree==std::array<std::uint32_t,2u>{{0u,5u}} &&
            afterCount==5u &&
            afterOverflow==std::array<std::uint32_t,3u>{{5u,0u,0u}},
            "Original GPU aggregate free-index and pair slot accounting differs");
    for(const auto [slot,expectedA,expectedB]:
        {std::array<std::uint32_t,3u>{3u,1u,10u},
         std::array<std::uint32_t,3u>{1u,2u,11u},
         std::array<std::uint32_t,3u>{4u,3u,12u}}) {
        require(pairWords[slot*3u]==expectedA &&
                pairWords[slot*3u+1u]==expectedB &&
                pairWords[slot*3u+2u]==1u,
                "Original aggregate pair slot reused wrong source root or flags");
    }
    // Exhaust the original source pair pool in a second frame. The
    // attempted additional slot must be counted, not written out of
    // bounds or silently hidden from Sc's future overflow check.
    source.sharedFoundAggPairIndex=2u;
    up(descriptor,&source,sizeof(source));
    const std::array<std::uint32_t,8u> secondFound{{
        20u,21u,22u,23u,0u,0u,0u,0u}};
    up(found,secondFound.data(),sizeof(secondFound));
    stage.append(stream,addresses,buffers);
    stream.wait_at_fetch();
    down(freeList,afterFree.data(),sizeof(afterFree));
    down(overflow,afterOverflow.data(),sizeof(afterOverflow));
    down(pairCount,&afterCount,sizeof(afterCount));
    down(pairs,pairWords.data(),sizeof(pairWords));
    require(afterFree==std::array<std::uint32_t,2u>{{0u,6u}} &&
            afterCount==6u &&
            afterOverflow==std::array<std::uint32_t,3u>{{7u,1u,0u}} &&
            pairWords[5u*3u]==20u && pairWords[5u*3u+1u]==21u &&
            pairWords[5u*3u+2u]==1u,
            "Original aggregate GPU slot capacity overflow did not preserve valid prefix");

    source.sharedFoundAggPairIndex=rootCapacity+2u;
    up(descriptor,&source,sizeof(source));
    stage.append(stream,addresses,buffers);
    stream.wait_at_fetch();
    down(overflow,afterOverflow.data(),sizeof(afterOverflow));
    require(afterOverflow[2u]==1u,
            "Source aggregate found-root report overflow was not retained");
    require(stream.submission_stats().command_batches==3u,
            "Source aggregate pair admission introduced extra GPU submissions");

    stream.wait_at_fetch();
    for(auto it=roots.rbegin();it!=roots.rend();++it) {
        addresses.unregister_slice(it->device);
        ctx.destroy_buffer(it->gpu);
    }
    require(addresses.live_allocations()==0u,
            "Source GPU aggregate pair slot views leaked");
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::invalid_argument(
            "Expected source aggregate admission and publication kernels");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,load_shader(argv[1]),load_shader(argv[2]));
        std::cout<<"Original PhysX aggregate GPU pair-slot admission PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Original PhysX aggregate GPU pair-slot admission FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
