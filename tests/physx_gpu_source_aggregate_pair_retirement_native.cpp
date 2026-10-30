// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_aggregate_pair_admission_native.h"
#include "physx_gpu/source_aggregate_pair_retirement_native.h"
#include "physx_port/gpubroadphase/include/PxgAggregate.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

namespace {
static_assert(sizeof(physx::PxgAggregatePair)==12u);
static_assert(offsetof(physx::PxgAggregatePair,isNew)==8u);
static_assert(offsetof(physx::PxgAggregatePair,isDead)==9u);
void require(bool good,const char* message) {
    if(!good)throw std::runtime_error(message);
}
ShaderBinary load(const char* path) {
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error(
        "Original aggregate pair retirement GPU shader is missing");
    ShaderBinary shader;
#if defined(FYZZXYZZY_TEST_D3D12)
    shader.format=ShaderFormat::Dxil;
#else
    shader.format=ShaderFormat::Spirv;
#endif
    shader.bytes.assign(std::istreambuf_iterator<char>(in),
                        std::istreambuf_iterator<char>());
    require(!shader.bytes.empty(),"Empty source GPU pair retirement shader");
    return shader;
}
void verify(Context& context,ShaderBinary count,ShaderBinary mark,
            ShaderBinary publish,ShaderBinary admit,ShaderBinary admit_publish) {
    constexpr std::uint32_t capacity=520u;
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourceAggregatePairRetirementNativeStage retirement(
        context,std::move(count),std::move(mark),std::move(publish));
    SourceAggregatePairAdmissionNativeStage admission(
        context,std::move(admit),std::move(admit_publish));
    struct Allocation {BufferId gpu;CudaDeviceAddressMap::Address addr;std::size_t bytes;};
    std::vector<Allocation> allocated;
    const auto alloc=[&](std::size_t bytes) {
        const auto gpu=context.create_buffer(bytes+4u);
        auto address=addresses.register_slice(gpu,0u,bytes+4u);
        allocated.push_back({gpu,address,bytes});
        constexpr std::uint32_t canary=0xfedcba98u;
        context.upload(gpu,bytes,&canary,4u);
        return address;
    };
    const auto up=[&](CudaDeviceAddressMap::Address addr,
                      const void* src,std::size_t bytes) {
        const auto view=addresses.resolve(addr,bytes);
        context.upload(view.buffer,view.offset,src,bytes);
    };
    const auto down=[&](CudaDeviceAddressMap::Address addr,
                        void* dst,std::size_t bytes) {
        const auto view=addresses.resolve(addr,bytes);
        context.download(view.buffer,view.offset,dst,bytes);
    };
    const auto pairs=alloc(capacity*12u);
    const auto bitmap=alloc(capacity*4u);
    const auto histogram=alloc(3u*4u);
    const auto freeIndices=alloc(capacity*4u);
    const auto freeList=alloc(8u);
    const auto pairCount=alloc(4u);
    const auto removed=alloc(4u);
    const auto descriptor=alloc(sizeof(physx::PxgBroadPhaseDesc));
    const auto foundRoots=alloc(4u*8u);
    const auto overflow=alloc(12u);

    SourceAggregatePairRetirementNativeBuffers buffers{
        pairs,bitmap,histogram,freeIndices,freeList,pairCount,removed,capacity};
    auto invalid=buffers;
    invalid.free_indices=invalid.aggregate_pairs;
    bool rejected=false;
    try {retirement.append(stream,addresses,invalid);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected && stream.commands().empty(),
        "Original pair retirement admitted aliased source GPU records");
    invalid=buffers;
    invalid.aggregate_pair_capacity=0u;
    rejected=false;
    try {retirement.append(stream,addresses,invalid);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected && stream.commands().empty(),
        "Original pair retirement accepted zero pair capacity");

    std::array<std::uint32_t,capacity*3u> words{};
    for(std::uint32_t i=0u;i<capacity;++i) {
        words[i*3u]=10u+i;
        words[i*3u+1u]=1000u+i;
        // Slot 77 was retired in an earlier frame and is already
        // present in the source free-index stack.
        words[i*3u+2u]=i==77u?0x00000101u:1u;
    }
    up(pairs,words.data(),sizeof(words));
    std::array<std::uint32_t,capacity> deleted{};
    constexpr std::array<std::uint32_t,5u> retired{{1u,255u,256u,257u,519u}};
    for(const auto slot:retired)deleted[slot]=1u;
    up(bitmap,deleted.data(),sizeof(deleted));
    std::array<std::uint32_t,capacity> freeStack{};
    freeStack[0]=77u;
    up(freeIndices,freeStack.data(),sizeof(freeStack));
    const std::array<std::uint32_t,2u> oldFree{{1u,capacity}};
    up(freeList,oldFree.data(),sizeof(oldFree));
    up(pairCount,&oldFree[1],4u);
    const std::uint32_t zero=0u;
    up(removed,&zero,4u);
    retirement.append(stream,addresses,buffers);
    require(stream.commands().size()==3u,
        "Source aggregate pair retirement altered its original three-stage order");
    stream.wait_at_fetch();
    std::array<std::uint32_t,2u> afterFree{};
    std::uint32_t afterRemoved=0u;
    down(freeList,afterFree.data(),sizeof(afterFree));
    down(removed,&afterRemoved,4u);
    down(freeIndices,freeStack.data(),sizeof(freeStack));
    down(pairs,words.data(),sizeof(words));
    require(afterFree==std::array<std::uint32_t,2u>{{6u,capacity}} &&
            afterRemoved==retired.size() && freeStack[0]==77u,
        "Original aggregate GPU retirement lost the free-list high-watermark");
    const auto source_pair=[&](std::uint32_t slot) {
        physx::PxgAggregatePair pair{};
        std::memcpy(&pair,&words[slot*3u],sizeof(pair));
        return pair;
    };
    for(std::size_t i=0u;i<retired.size();++i) {
        const auto slot=retired[i];
        const auto pair=source_pair(slot);
        require(freeStack[1u+i]==slot &&
                pair.actorHandle0==slot+10u &&
                pair.actorHandle1==slot+1000u &&
                pair.isNew && pair.isDead && pair.pad==0u,
            "Original aggregate GPU retirement corrupted source pair layout or free ordering");
    }
    for(const auto slot:{0u,254u,258u,518u})
        require(source_pair(slot).isNew && !source_pair(slot).isDead,
            "Original GPU retirement marked a surviving pair dead");

    // The next source GPU admission must reuse the newly appended free
    // slots LIFO rather than increasing original maxIndex or touching
    // the unrelated first reserved free index.
    SourceAggregatePairAdmissionNativeBuffers admit_buffers{
        descriptor,foundRoots,pairs,freeIndices,freeList,pairCount,overflow,
        4u,capacity};
    physx::PxgBroadPhaseDesc bp{};
    bp.sharedFoundAggPairIndex=3u;
    up(descriptor,&bp,sizeof(bp));
    const std::array<std::uint32_t,8u> added{{
        200u,201u,202u,203u,204u,205u,0u,0u}};
    up(foundRoots,added.data(),sizeof(added));
    admission.append(stream,addresses,admit_buffers);
    require(stream.commands().size()==2u,
            "Original aggregate free slot reuse bypassed two source admission stages");
    stream.wait_at_fetch();
    down(freeList,afterFree.data(),sizeof(afterFree));
    down(pairs,words.data(),sizeof(words));
    require(afterFree==std::array<std::uint32_t,2u>{{3u,capacity}},
            "Source aggregate GPU admission did not consume recycled free slots");
    for(std::uint32_t i=0u;i<3u;++i) {
        const auto slot=retired[retired.size()-1u-i];
        const auto pair=source_pair(slot);
        require(pair.actorHandle0==200u+2u*i &&
                pair.actorHandle1==201u+2u*i &&
                pair.isNew && !pair.isDead && pair.pad==0u,
            "Original GPU free-list reuse failed to replace dead source pair records");
    }
    // A subsequent frame with no deleted persistent pairs must leave
    // the source free stack and pair records untouched. The current
    // remove bitmap is entirely GPU-owned by the collision stage.
    deleted.fill(0u);
    up(bitmap,deleted.data(),sizeof(deleted));
    retirement.append(stream,addresses,buffers);
    stream.wait_at_fetch();
    std::array<std::uint32_t,2u> afterEmpty{};
    std::uint32_t removedEmpty=~0u;
    down(freeList,afterEmpty.data(),sizeof(afterEmpty));
    down(removed,&removedEmpty,sizeof(removedEmpty));
    require(afterEmpty==afterFree && removedEmpty==0u,
            "Original aggregate empty-frame retirement changed persistent free IDs");

    require(stream.submission_stats().command_batches==3u,
            "Original aggregate retirement/reuse introduced extra GPU fence batches");
    for(auto it=allocated.rbegin();it!=allocated.rend();++it) {
        std::uint32_t actual=0u;
        context.download(it->gpu,it->bytes,&actual,sizeof(actual));
        require(actual==0xfedcba98u,
                "Original aggregate pair retirement wrote outside source GPU allocation");
        addresses.unregister_slice(it->addr);
        context.destroy_buffer(it->gpu);
    }
    require(addresses.live_allocations()==0u,
            "Original aggregate pair retirement leaked source GPU allocations");
}
} // namespace
int main(int argc,char** argv) {
    try {
        if(argc!=6)throw std::invalid_argument(
            "Expected three aggregate retirement and two admission GPU shaders");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,load(argv[1]),load(argv[2]),load(argv[3]),
            load(argv[4]),load(argv[5]));
        std::cout<<"Original PhysX aggregate GPU pair retirement and reuse PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Original PhysX aggregate GPU pair retirement and reuse FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
