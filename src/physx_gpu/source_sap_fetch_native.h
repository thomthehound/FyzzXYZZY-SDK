// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/sap_host_frame_roles.h"

#include "PxgBroadPhaseDesc.h"
#include "PxgBroadPhasePairReport.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace fyzz::physx_gpu {

// The original mFoundActorPairsMapped/mLostActorPairsMapped are owned by
// PxgCudaBroadPhaseSap. They are *destinations*, not new shadow pair lists.
// A native resource supplies the bytes that the source copyReports kernel
// previously wrote to mapped CUDA host memory.
struct SourceSapActorPairDestination {
    physx::PxgBroadPhasePair* data = nullptr;
    std::size_t capacity = 0; // records, not bytes
};

struct SourceSapFetchedCounts {
    std::uint32_t found_actors = 0;
    std::uint32_t lost_actors = 0;
    std::uint32_t pairs_needed = 0;
    std::uint32_t overlap_checks_total_region = 0;
    std::uint32_t start_region_accum_total = 0;
    std::uint32_t region_accum_total = 0;
    bool overflow = false;
};

// Lower the original gpuDMABack()/fetchBroadPhaseResults() readback step.
// The caller MUST invoke this from BroadPhaseCompletionGate's fetch callback,
// after the source stream has completed and before PhysX installs/releases
// the original found/lost pair task continuations.
//
// GPU descriptor and pair buffers remain at their original CUDA virtual
// addresses in the caller-owned map. No scene scheduler or staging heap is
// created here; Context::download is the native synchronous DtoH operation at
// the original fetch boundary. Copies into the caller's existing pinned pair
// arrays are performed only after validating both destinations and sources.
// The original seven buffer history families rotate after successful fetch,
// including the no-update path. Integer bounds swap separately in update().
inline SourceSapFetchedCounts fetch_source_sap_results(
    compute::Context& context,
    const CudaDeviceAddressMap& addresses,
    CudaDeviceAddressMap::Address descriptor_address,
    physx::PxgBroadPhaseDesc& host_descriptor,
    SourceSapActorPairDestination found,
    SourceSapActorPairDestination lost,
    bool abort_mode,
    SapFrameBufferRoles& roles) {

    const auto desc_source = addresses.resolve(
        descriptor_address, sizeof(physx::PxgBroadPhaseDesc));
    physx::PxgBroadPhaseDesc gpu_desc{};
    context.download(desc_source.buffer, desc_source.offset,
                     &gpu_desc, sizeof(gpu_desc));

    // The source asserts these invariants; the native boundary must also
    // reject corrupt GPU counts before an unsigned subtraction can wrap.
    const auto cap = gpu_desc.max_found_lost_pairs;
    if (gpu_desc.sharedFoundPairIndex < gpu_desc.sharedFoundAggPairIndex ||
        gpu_desc.sharedLostPairIndex < gpu_desc.sharedLostAggPairIndex ||
        gpu_desc.sharedFoundAggPairIndex > cap ||
        gpu_desc.sharedLostAggPairIndex > cap)
        throw std::invalid_argument("Invalid source GPU broadphase report counters");

    SourceSapFetchedCounts counts;
    counts.pairs_needed = std::max(gpu_desc.sharedFoundPairIndex,
                                  gpu_desc.sharedLostPairIndex);
    counts.overlap_checks_total_region = gpu_desc.overlapChecksTotalRegion;
    counts.start_region_accum_total = gpu_desc.startRegionAccumTotal;
    counts.region_accum_total = gpu_desc.regionAccumTotal;
    counts.overflow = gpu_desc.found_lost_pairs_overflow_flags != 0;
    if (!abort_mode) {
        counts.found_actors =
            std::min(cap, gpu_desc.sharedFoundPairIndex) -
            gpu_desc.sharedFoundAggPairIndex;
        counts.lost_actors =
            std::min(cap, gpu_desc.sharedLostPairIndex) -
            gpu_desc.sharedLostAggPairIndex;
    }

    auto checked_pair_source = [&](const physx::PxgBroadPhasePair* device,
                                   SourceSapActorPairDestination host,
                                   std::uint32_t count) {
        if (!count) return CudaDeviceAddressMap::Resolved{};
        if (!host.data || count > host.capacity ||
            count > std::numeric_limits<std::size_t>::max() /
                        sizeof(physx::PxgBroadPhasePair))
            throw std::out_of_range("PhysX source mapped actor-pair capacity exceeded");
        const auto address = static_cast<CudaDeviceAddressMap::Address>(
            reinterpret_cast<std::uintptr_t>(device));
        return addresses.resolve(address,
            std::size_t(count) * sizeof(physx::PxgBroadPhasePair));
    };

    // Both sources must validate BEFORE writing into either pinned host array.
    const auto found_source = checked_pair_source(
        gpu_desc.foundPairReportMap, found, counts.found_actors);
    const auto lost_source = checked_pair_source(
        gpu_desc.lostPairReportMap, lost, counts.lost_actors);
    if (counts.found_actors)
        context.download(found_source.buffer, found_source.offset, found.data,
                         std::size_t(counts.found_actors) *
                             sizeof(physx::PxgBroadPhasePair));
    if (counts.lost_actors)
        context.download(lost_source.buffer, lost_source.offset, lost.data,
                         std::size_t(counts.lost_actors) *
                             sizeof(physx::PxgBroadPhasePair));

    host_descriptor = gpu_desc;
    roles.swap_histories_after_fetch();
    return counts;
}

} // namespace fyzz::physx_gpu
