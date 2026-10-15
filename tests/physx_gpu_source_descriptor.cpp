// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_newbounds_descriptor.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace fyzz::physx_gpu;
using namespace fyzz::compute;
namespace {
constexpr std::uint32_t kBoxCount = 3, kCapacity = 6;
constexpr std::uint32_t kProjections = 8; // (3 current + 1 removed)*2
constexpr std::uint32_t kRegionWork = 5;
constexpr std::uint32_t kPairs = 8;
constexpr std::uint32_t kPhysicalBytes = 32768;
constexpr std::uint32_t kArenaGrowthBytes = 65536;

void require(bool good, const char* why) {
    if (!good) throw std::runtime_error(why);
}
template<typename Fn>
void must_reject(Fn&& f, const char* message) {
    bool did = false;
    try { f(); }
    catch (const std::exception&) { did = true; }
    if (!did) throw std::runtime_error(message);
}
std::uint32_t align256(std::uint32_t n) { return (n + 255u) & ~255u; }

struct Allocation {
    CudaDeviceAddressMap::Address address;
    std::uint32_t offset;
    std::uint32_t bytes;
};

class SourceFixture {
public:
    SourceFixture(CudaDeviceAddressMap& map, BufferId backing)
        : map_(map), buffer_(backing) {
        descriptor = reserve(sizeof(physx::PxgBroadPhaseDesc));
        descriptor_address = map_.register_slice(backing, descriptor,
                                               sizeof(physx::PxgBroadPhaseDesc));
        tracked.push_back({descriptor_address,descriptor,
                           static_cast<std::uint32_t>(sizeof(physx::PxgBroadPhaseDesc))});
        source.regionAccumTotal = kRegionWork;
        source.numPreviousHandles = kBoxCount;
        source.max_found_lost_pairs = kPairs;
        source.numCreatedHandles = 1;
        source.sharedFoundPairIndex = 1;
        source.overlapChecksTotalRegion = 0;

        bind(source.newIntegerBounds, kCapacity*24u);
        bind(source.updateData_groups, kCapacity*4u);
        bind(source.updateData_envIDs, kCapacity*4u);
        bind(source.boxHandles[0][0], kProjections*4u);
        bind(source.boxSapBox1D[0], kCapacity*8u);
        bind(source.regionRange, kCapacity*32u);
        bind(source.overlapChecksRegion, kRegionWork*8u);
        bind(source.overlapChecksHandleRegiones, kRegionWork*8u);
        bind(source.activeRegionsHistogram, 64u*kProjections*4u);
        bind(source.startRegionsHistogram, 64u*kProjections*4u);
        bind(source.orderedActiveRegionHandles, 64u*kProjections*4u);
        bind(source.orderedStartRegionHandles, 64u*kProjections*4u);
        bind(source.foundPairReport, kPairs*8u);
        bind(source.regionAccum, kProjections*4u);
        bind(source.blockOverlapChecksRegion, 32u*8u);
        workDescriptor = reserve(sizeof(SapOverlapWorkArenaDesc));
        prefixDescriptor = reserve(sizeof(SapOverlapPrefixArenaDesc));
        pairDescriptor = reserve(sizeof(SapNewBoundsPairsArenaDesc));
    }

    std::uint32_t reserve(std::size_t bytes) {
        const auto at=align256(cursor_);
        if (bytes > kPhysicalBytes - at)
            throw std::runtime_error("GPU descriptor test arena exhausted");
        cursor_ = at + static_cast<std::uint32_t>(bytes);
        return at;
    }

    template<typename T>
    void bind(T*& dest, std::uint32_t bytes) {
        const auto offset=reserve(bytes);
        const auto addr=map_.register_slice(buffer_,offset,bytes);
        dest=reinterpret_cast<T*>(static_cast<std::uintptr_t>(addr));
        tracked.push_back({addr,offset,bytes});
    }

    SourceNewBoundsArenaLayout layout() const {
        return {buffer_,kPhysicalBytes,descriptor_address,kBoxCount,kCapacity,kProjections};
    }

    CudaDeviceAddressMap& map_;
    BufferId buffer_;
    std::uint32_t cursor_=256u;
    std::uint32_t descriptor=0u,workDescriptor=0u,
                  prefixDescriptor=0u,pairDescriptor=0u;
    CudaDeviceAddressMap::Address descriptor_address=0;
    physx::PxgBroadPhaseDesc source{};
    std::vector<Allocation> tracked;
};

std::uint32_t physical(const CudaDeviceAddressMap& map, const void* address,
                       std::uint32_t bytes) {
    return static_cast<std::uint32_t>(
        map.resolve(static_cast<CudaDeviceAddressMap::Address>(
            reinterpret_cast<std::uintptr_t>(address)), bytes).offset);
}

void verify_fields(const SourceFixture& f, const SapNewBoundsSequenceArenaDescriptors& d) {
    const auto& p=d.pairs;
    const auto& m=f.map_;
    require(p.new_integer_bounds_offset==physical(m,f.source.newIntegerBounds,kCapacity*24u),
            "Source integer bound pointer mapping");
    require(p.group_ids_offset==physical(m,f.source.updateData_groups,kCapacity*4u),
            "Source collision group pointer mapping");
    require(p.environment_ids_offset==physical(m,f.source.updateData_envIDs,kCapacity*4u),
            "Source environment pointer mapping");
    require(p.sorted_x_handles_offset==physical(m,f.source.boxHandles[0][0],kProjections*4u),
            "Source SAP tagged handle mapping");
    require(p.x_sap_positions_offset==physical(m,f.source.boxSapBox1D[0],kCapacity*8u),
            "Source SAP min/max mapping");
    require(p.region_ranges_offset==physical(m,f.source.regionRange,kCapacity*32u),
            "Source 2xuint4 region bounds mapping");
    require(p.overlap_prefix_offset==physical(m,f.source.overlapChecksRegion,kRegionWork*8u),
            "Source 64-bit overlap prefix mapping");
    require(p.handle_regions_offset==physical(m,f.source.overlapChecksHandleRegiones,kRegionWork*8u),
            "Source handle-region map");
    require(p.active_histograms_offset==physical(m,f.source.activeRegionsHistogram,64u*kProjections*4u) &&
            p.start_histograms_offset==physical(m,f.source.startRegionsHistogram,64u*kProjections*4u),
            "Source 64-region histogram mapping");
    require(p.ordered_active_handles_offset==physical(m,f.source.orderedActiveRegionHandles,64u*kProjections*4u) &&
            p.ordered_start_handles_offset==physical(m,f.source.orderedStartRegionHandles,64u*kProjections*4u),
            "Source 64-region list mapping");
    require(p.found_reports_offset==physical(m,f.source.foundPairReport,kPairs*8u),
            "Source found report mapping");
    require(d.work.region_prefix_offset==physical(m,f.source.regionAccum,kProjections*4u) &&
            d.prefix.block_counts_offset==physical(m,f.source.blockOverlapChecksRegion,32u*8u),
            "Source histogram scan scratch");
    require(p.region_work_total_offset==f.descriptor+
                offsetof(physx::PxgBroadPhaseDesc,regionAccumTotal) &&
            p.overlap_total_offset==f.descriptor+
                offsetof(physx::PxgBroadPhaseDesc,overlapChecksTotalRegion) &&
            p.found_counter_offset==f.descriptor+
                offsetof(physx::PxgBroadPhaseDesc,sharedFoundPairIndex),
            "Source GPU descriptor scalar slots were shadowed");
    require(p.current_boxes==kBoxCount && p.box_capacity==kCapacity &&
            p.max_found_lost_pairs==kPairs && p.has_environment_ids==1u,
            "Source scalar parameters changed");
    require(d.work.overlap_counts_offset==p.overlap_prefix_offset &&
            d.prefix.overlap_counts_offset==p.overlap_prefix_offset &&
            d.prefix.total_offset==p.overlap_total_offset,
            "Original descriptor GPU dependencies not aliased");
}

void verify_gpu(Context& ctx) {
    CudaDeviceAddressMap addresses;
    const auto initial=ctx.create_buffer(kPhysicalBytes,MemoryPlacement::DeviceLocal);
    SourceFixture f(addresses,initial);
    const auto layout=f.layout();
    const auto first=lower_source_newbounds_descriptor(f.source,addresses,layout);
    verify_fields(f,first);

    // The original native source descriptor still contains genuine 64-bit
    // CUDA-style virtual pointers. GPU scalar fields are inline at exactly
    // the original PxgBroadPhaseDesc offsets inside that same backing.
    std::vector<std::uint8_t> image(kPhysicalBytes,0xa5u);
    ctx.upload(initial,0u,image.data(),image.size());
    const std::vector<UploadSpan> uploads{
        {initial,f.descriptor,&f.source,sizeof(f.source)},
        {initial,f.workDescriptor,&first.work,sizeof(first.work)},
        {initial,f.prefixDescriptor,&first.prefix,sizeof(first.prefix)},
        {initial,f.pairDescriptor,&first.pairs,sizeof(first.pairs)}
    };
    const auto firstSubmission=ctx.submit_upload_batch(uploads);
    require(firstSubmission!=0u,"PhysX source descriptor upload wasn't submitted");

    // An original allocateCopyOldDataAsync-style growth uses ONE ordered
    // GPU copy of the persistent arena, not repacking each descriptor
    // before each of the four kernels. Move *one* logical slice to prove
    // a single source allocation can also be reallocated independently.
    const auto grown=ctx.create_buffer(kArenaGrowthBytes,MemoryPlacement::DeviceLocal);
    const auto oldGroups=physical(addresses,f.source.updateData_groups,kCapacity*4u);
    const auto newGroups=static_cast<std::uint32_t>(kPhysicalBytes+256u);
    const std::vector<BatchCommand> copies{
        CopyCommand{initial,0u,grown,0u,kPhysicalBytes},
        CopyCommand{initial,oldGroups,grown,newGroups,kCapacity*4u}
    };
    const auto copied=ctx.submit_batch(copies);
    require(copied!=0u,"PhysX async arena relocation wasn't submitted");

    for(const auto& a : f.tracked)
        addresses.relocate_slice(a.address,grown,a.offset);
    const auto groupAddress=static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(f.source.updateData_groups));
    addresses.relocate_slice(groupAddress,grown,newGroups);

    auto relocatedLayout=f.layout();
    relocatedLayout.backing=grown;
    relocatedLayout.backing_bytes=kArenaGrowthBytes;
    const auto moved=lower_source_newbounds_descriptor(f.source,addresses,relocatedLayout);
    verify_fields(f,moved);
    require(first.pairs.group_ids_offset!=moved.pairs.group_ids_offset,
            "Relocation didn't update GPU field offset");
    require(first.pairs.new_integer_bounds_offset==moved.pairs.new_integer_bounds_offset,
            "Unrelated PhysX GPU allocation moved");

    // Re-upload only the changed native descriptor at the source's
    // allocation/growth boundary. Kernel submission merely binds the
    // persistent arena and these already-resident views.
    const auto corrected=ctx.submit_upload_batch({
        {grown,f.workDescriptor,&moved.work,sizeof(moved.work)},
        {grown,f.prefixDescriptor,&moved.prefix,sizeof(moved.prefix)},
        {grown,f.pairDescriptor,&moved.pairs,sizeof(moved.pairs)}
    });
    ctx.retire_buffer(initial); // old storage remains alive through prior copies.
    ctx.wait_submission(corrected);
    ctx.wait_submission(copied);
    ctx.wait_submission(firstSubmission);

    std::vector<std::uint8_t> actual(kArenaGrowthBytes,0);
    ctx.download(grown,0u,actual.data(),actual.size());
    auto read=[&](std::uint32_t offset, const void* value, std::size_t size) {
        require(std::memcmp(actual.data()+offset,value,size)==0,
                "Asynchronous source descriptor/relocation GPU contents differ");
    };
    read(f.descriptor,&f.source,sizeof(f.source));
    read(f.workDescriptor,&moved.work,sizeof(moved.work));
    read(f.prefixDescriptor,&moved.prefix,sizeof(moved.prefix));
    read(f.pairDescriptor,&moved.pairs,sizeof(moved.pairs));
    for(std::uint32_t i=0;i<kCapacity*4u;++i)
        require(actual[newGroups+i]==0xa5u,
                "GPU source data was lost during allocateCopyOldDataAsync-style relocation");

    // Separate source buffers are not silently repacked each dispatch.
    const auto incompatible=ctx.create_buffer(4096u,MemoryPlacement::DeviceLocal);
    const auto incompatiblePtr=addresses.register_slice(incompatible,0,256u);
    auto other=f.source;
    other.updateData_groups=reinterpret_cast<decltype(other.updateData_groups)>(
        static_cast<std::uintptr_t>(incompatiblePtr));
    must_reject([&]{(void)lower_source_newbounds_descriptor(other,addresses,relocatedLayout);},
                "Disjoint PhysX allocations silently fell back to per-kernel repacking");
    other=f.source;
    other.updateData_envIDs=nullptr;
    const auto noEnv=lower_source_newbounds_descriptor(other,addresses,relocatedLayout);
    require(noEnv.pairs.has_environment_ids==0u && noEnv.pairs.environment_ids_offset==0u,
            "Absent optional source environment pointer ignored");
    auto wrong=relocatedLayout;
    wrong.current_boxes=kCapacity+1u;
    must_reject([&]{(void)lower_source_newbounds_descriptor(f.source,addresses,wrong);},
                "Invalid source PhysX frame capacity accepted");
    wrong=relocatedLayout;
    wrong.backing_bytes=f.descriptor+16u;
    must_reject([&]{(void)lower_source_newbounds_descriptor(f.source,addresses,wrong);},
                "Out-of-range source GPU descriptor silently accepted");

    const auto temp=addresses.unregister_slice(incompatiblePtr);
    require(temp.buffer==incompatible,"Source slice owner lost");
    must_reject([&]{(void)addresses.resolve(incompatiblePtr,4u);},
                "Freed original PhysX CUDA virtual address remained accessible");
    ctx.destroy_buffer(incompatible);
    for(const auto& a:f.tracked) {
        const auto physicalSlice=addresses.unregister_slice(a.address);
        require(physicalSlice.buffer==grown &&
                physicalSlice.offset==(a.address==groupAddress?newGroups:a.offset),
                "Logical PhysX CUDA free lost relocated GPU resource");
    }
    require(addresses.live_allocations()==0u,"Source GPU virtual allocation leak");
    ctx.destroy_buffer(grown);
}
} // namespace

int main() {
    try {
        Context ctx(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        verify_gpu(ctx);
        std::cout << "PhysX original broadphase descriptor addressing and async growth PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX source GPU descriptor addressing FAILED: " << e.what() << '\n';
        return 1;
    }
}
