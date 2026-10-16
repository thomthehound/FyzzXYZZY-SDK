// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/broadphase_copy_reports_arena.h"
#include "physx_gpu/broadphase_report_classification_arena.h"
#include "physx_gpu/cuda_device_address_map.h"

#include "PxgBroadPhaseDesc.h"
#include "PxgBroadPhasePairReport.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace fyzz::physx_gpu {

struct SourceSapNativeReportLayout {
    compute::BufferId backing = 0;
    std::uint32_t backing_bytes = 0;
    CudaDeviceAddressMap::Address source_descriptor = 0;
    CudaDeviceAddressMap::Address classification_descriptor = 0;
    CudaDeviceAddressMap::Address copy_descriptor = 0;
    std::uint32_t volume_capacity = 0;
    std::uint32_t volume_stride = 0;
    std::uint32_t aggregate_member_offset = 0;
};

struct SourceSapNativeReportDescriptors {
    BroadphaseReportClassificationArenaDesc classification;
    BroadphaseCopyReportsArenaDesc copy;
};

// Source-preserving lowering for the original three GPU report launches
// (BP_ACCUMULATE_REPORT_STAGE_1, STAGE_2, BP_COPY_REPORTS). The existing
// GPU shaders use resolved byte offsets within a *single PhysX-owned heap
// root*. The source descriptor, counters, input/output report buffers and
// Bp::VolumeData array remain at their ORIGINAL virtual CUDA addresses;
// there are no copied/shadow pair lists, new heap or host-side classifier.
// Multiple native backing resources remain future work.
inline SourceSapNativeReportDescriptors lower_source_sap_report_descriptors(
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    const SourceSapNativeReportLayout& layout) {

    if (!layout.backing || !layout.backing_bytes ||
        layout.backing_bytes % 4u ||
        !layout.source_descriptor ||
        !layout.classification_descriptor || !layout.copy_descriptor ||
        !source.max_found_lost_pairs ||
        !source.max_found_lost_agg_pairs)
        throw std::invalid_argument("Incomplete source PhysX report heap and capacities");

    const auto native = [&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes, std::size_t alignment = 4u) {
        if (!address || !bytes || bytes > layout.backing_bytes)
            throw std::out_of_range("Missing source report GPU allocation");
        const auto span = addresses.resolve(address, bytes);
        if (span.buffer != layout.backing || span.offset % alignment ||
            span.offset > layout.backing_bytes ||
            bytes > layout.backing_bytes - span.offset ||
            span.offset > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Source report pointer crosses native heap root");
        return static_cast<std::uint32_t>(span.offset);
    };
    const auto bytes_for = [](std::uint64_t n, std::uint64_t stride) {
        if (!n || !stride || n > std::numeric_limits<std::uint32_t>::max() / stride)
            throw std::out_of_range("Source PhysX report allocation capacity overflow");
        return static_cast<std::size_t>(n * stride);
    };
    const auto pointer = [](auto* p) {
        return static_cast<CudaDeviceAddressMap::Address>(
            reinterpret_cast<std::uintptr_t>(p));
    };
    const auto d = native(layout.source_descriptor, sizeof(source),
                          alignof(physx::PxgBroadPhaseDesc));
    (void)native(layout.classification_descriptor,
                 sizeof(BroadphaseReportClassificationArenaDesc),
                 alignof(BroadphaseReportClassificationArenaDesc));
    (void)native(layout.copy_descriptor, sizeof(BroadphaseCopyReportsArenaDesc),
                 alignof(BroadphaseCopyReportsArenaDesc));

    const auto member = [&](std::size_t offset, std::size_t width) {
        if (offset > sizeof(source) || width > sizeof(source) - offset)
            throw std::logic_error("Source GPU descriptor scalar ABI changed");
        return native(layout.source_descriptor + offset, width);
    };
    // These source counters are consecutive in PxgBroadPhaseDesc.
    static_assert(offsetof(physx::PxgBroadPhaseDesc, sharedLostPairIndex) ==
                  offsetof(physx::PxgBroadPhaseDesc, sharedFoundPairIndex) + 4u,
                  "Original source found/lost report scalar layout changed");
    static_assert(offsetof(physx::PxgBroadPhaseDesc, sharedLostAggPairIndex) ==
                  offsetof(physx::PxgBroadPhaseDesc, sharedFoundAggPairIndex) + 4u,
                  "Original source aggregate scalar layout changed");
    static_assert(offsetof(physx::PxgBroadPhaseDesc, found_lost_pairs_overflow_flags) +
                  4u <= sizeof(physx::PxgBroadPhaseDesc),
                  "Source bool overflow word cannot be written safely");

    SourceSapNativeReportDescriptors out;
    auto& c = out.classification;
    c.raw_pair_counts_offset =
        member(offsetof(physx::PxgBroadPhaseDesc, sharedFoundPairIndex), 8u);
    c.aggregate_pair_counts_offset =
        member(offsetof(physx::PxgBroadPhaseDesc, sharedFoundAggPairIndex), 8u);
    const auto raw_capacity = bytes_for(source.max_found_lost_pairs,
                                         sizeof(physx::PxgBroadPhasePair));
    const auto agg_capacity = bytes_for(source.max_found_lost_agg_pairs,
                                         sizeof(physx::PxgBroadPhasePair));
    c.found_pairs_offset = native(pointer(source.foundPairReport), raw_capacity);
    c.lost_pairs_offset = native(pointer(source.lostPairReport), raw_capacity);
    c.found_aggregate_output_offset = native(pointer(source.foundAggPairReport), agg_capacity);
    c.lost_aggregate_output_offset = native(pointer(source.lostAggPairReport), agg_capacity);
    c.found_actor_output_offset = native(pointer(source.foundActorPairReport), raw_capacity);
    c.lost_actor_output_offset = native(pointer(source.lostActorPairReport), raw_capacity);
    c.found_aggregate_block_offset = native(pointer(source.aggReportBlock[0]), 32u * 4u);
    c.found_actor_block_offset = native(pointer(source.actorReportBlock[0]), 32u * 4u);
    c.lost_aggregate_block_offset = native(pointer(source.aggReportBlock[1]), 32u * 4u);
    c.lost_actor_block_offset = native(pointer(source.actorReportBlock[1]), 32u * 4u);
    c.raw_report_capacity = source.max_found_lost_pairs;
    c.aggregate_report_capacity = source.max_found_lost_agg_pairs;
    c.has_volume_data = source.aabbMngr_volumeData ? 1u : 0u;
    c.volume_capacity = layout.volume_capacity;
    if (c.has_volume_data) {
        if (!layout.volume_capacity || !layout.volume_stride ||
            layout.volume_stride % 4u ||
            layout.aggregate_member_offset % 4u ||
            layout.aggregate_member_offset > layout.volume_stride ||
            4u > layout.volume_stride - layout.aggregate_member_offset)
            throw std::invalid_argument("Invalid original Bp::VolumeData layout");
        c.volume_aggregate_tags_offset = native(
            pointer(source.aabbMngr_volumeData),
            bytes_for(layout.volume_capacity, layout.volume_stride));
        c.volume_stride = layout.volume_stride;
        c.aggregate_member_offset = layout.aggregate_member_offset;
    }

    auto& p = out.copy;
    p.raw_pair_counts_offset = c.raw_pair_counts_offset;
    p.aggregate_pair_counts_offset = c.aggregate_pair_counts_offset;
    p.found_actor_pairs_offset = c.found_actor_output_offset;
    p.lost_actor_pairs_offset = c.lost_actor_output_offset;
    // These are source CUDA-mapped host pointers. The single-root native
    // fixture maps them to GPU-visible output slices. Actual mapped host
    // interop must be supplied by the original scene's GPU allocator.
    p.found_map_offset = native(pointer(source.foundPairReportMap), raw_capacity);
    p.lost_map_offset = native(pointer(source.lostPairReportMap), raw_capacity);
    p.overflow_flag_offset = member(
        offsetof(physx::PxgBroadPhaseDesc, found_lost_pairs_overflow_flags), 4u);
    p.report_capacity = source.max_found_lost_pairs;

    // Ensure the source struct itself remains on the supplied heap.
    (void)d;
    return out;
}

} // namespace fyzz::physx_gpu
