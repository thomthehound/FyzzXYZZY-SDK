// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_newbounds_descriptor.h"

// The source GPU layout uses six uint32 integer-bound words, two uint32
// min/max projection indices, two uint4 region extents and two uint32
// pair handles. Keep the sizes explicit to avoid requiring CUDA headers
// in a Vulkan/D3D12 host-only translation unit.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace fyzz::physx_gpu {
namespace {
std::uint32_t checked_bytes(std::uint64_t n, std::uint64_t stride) {
    if (n && stride > std::numeric_limits<std::uint32_t>::max() / n)
        throw std::out_of_range("PhysX source GPU descriptor field range overflow");
    return static_cast<std::uint32_t>(n * stride);
}
std::uint32_t native_offset(
    CudaDeviceAddressMap::Address address,
    const CudaDeviceAddressMap& addresses,
    const SourceNewBoundsArenaLayout& layout,
    std::uint32_t bytes, std::size_t alignment = 4u) {
    if (!address || !bytes)
        throw std::invalid_argument("Missing required PhysX GPU descriptor pointer");
    const auto found = addresses.resolve(address, bytes);
    if (found.buffer != layout.backing || found.offset % alignment ||
        found.offset > layout.backing_bytes ||
        bytes > layout.backing_bytes - found.offset ||
        found.offset > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("PhysX source descriptor spans disjoint GPU backing");
    return static_cast<std::uint32_t>(found.offset);
}
template <typename T>
std::uint32_t native_pointer(
    const T* ptr, const CudaDeviceAddressMap& addresses,
    const SourceNewBoundsArenaLayout& layout, std::uint32_t bytes,
    std::size_t alignment = 4u) {
    return native_offset(static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(ptr)), addresses, layout, bytes, alignment);
}
std::uint32_t inline_scalar(
    const SourceNewBoundsArenaLayout& layout,
    std::uint32_t descriptor_offset, std::size_t member, std::size_t bytes) {
    const auto offset = std::uint64_t(descriptor_offset) + member;
    if (offset % bytes != 0u ||
        offset > std::numeric_limits<std::uint32_t>::max() ||
        offset > layout.backing_bytes || bytes > layout.backing_bytes - offset)
        throw std::out_of_range("Source broadphase descriptor scalar outside GPU arena");
    return static_cast<std::uint32_t>(offset);
}
} // namespace

SapNewBoundsSequenceArenaDescriptors lower_source_newbounds_descriptor(
    const physx::PxgBroadPhaseDesc& d, const CudaDeviceAddressMap& addresses,
    const SourceNewBoundsArenaLayout& layout) {
    if (!layout.backing || !layout.backing_bytes || !layout.current_boxes ||
        layout.box_capacity < layout.current_boxes ||
        layout.box_capacity < d.numPreviousHandles ||
        std::uint64_t(d.numPreviousHandles) + d.numCreatedHandles <
            d.numRemovedHandles ||
        std::uint64_t(d.numPreviousHandles) + d.numCreatedHandles -
            d.numRemovedHandles != layout.current_boxes ||
        std::uint64_t(layout.current_boxes + std::uint64_t(d.numRemovedHandles))*2u !=
            layout.projection_slots ||
        !layout.descriptor ||
        !layout.overlap_prefix_capacity ||
        !layout.handle_region_capacity ||
        d.max_found_lost_pairs == 0u)
        throw std::invalid_argument("Invalid source PhysX broadphase frame/arena");

    // Source mBPDescBuf remains resident in this same native backing. Its
    // scalar counters are actual GPU memory locations, not invented copies.
    const auto descriptor = native_offset(
        layout.descriptor, addresses, layout, sizeof(physx::PxgBroadPhaseDesc),
        alignof(physx::PxgBroadPhaseDesc));
    const auto projection_count = layout.projection_slots;
    const auto histogram_bytes = checked_bytes(projection_count, 64u * 4u);
    const auto handles_bytes = checked_bytes(projection_count, 4u);
    const auto sap_bytes = checked_bytes(layout.box_capacity, 8u);
    const auto bounds_bytes = checked_bytes(layout.box_capacity, 24u);
    const auto range_bytes = checked_bytes(layout.box_capacity, 32u);
    const auto pair_bytes = checked_bytes(d.max_found_lost_pairs, 8u);
    const auto overlap_bytes = checked_bytes(layout.overlap_prefix_capacity,
                                             sizeof(physx::regionOverlapType));
    const auto region_prefix_bytes = checked_bytes(projection_count, 4u);
    const auto block_bytes = checked_bytes(32u, sizeof(physx::regionOverlapType));

    SapNewBoundsPairsArenaDesc pair{};
    pair.new_integer_bounds_offset = native_pointer(d.newIntegerBounds, addresses, layout, bounds_bytes);
    pair.group_ids_offset = native_pointer(d.updateData_groups, addresses, layout,
                                          checked_bytes(layout.box_capacity, 4u));
    if (d.updateData_envIDs)
        pair.environment_ids_offset = native_pointer(d.updateData_envIDs, addresses, layout,
                                                    checked_bytes(layout.box_capacity, 4u));
    pair.sorted_x_handles_offset = native_pointer(d.boxHandles[0][0], addresses, layout, handles_bytes);
    pair.x_sap_positions_offset = native_pointer(d.boxSapBox1D[0], addresses, layout, sap_bytes);
    pair.region_ranges_offset = native_pointer(d.regionRange, addresses, layout, range_bytes);
    pair.overlap_prefix_offset = native_pointer(d.overlapChecksRegion, addresses, layout,
                                                overlap_bytes, alignof(physx::regionOverlapType));
    pair.region_work_total_offset = inline_scalar(layout, descriptor,
        offsetof(physx::PxgBroadPhaseDesc, regionAccumTotal), sizeof(d.regionAccumTotal));
    pair.overlap_total_offset = inline_scalar(layout, descriptor,
        offsetof(physx::PxgBroadPhaseDesc, overlapChecksTotalRegion), sizeof(d.overlapChecksTotalRegion));
    pair.handle_regions_offset = native_pointer(d.overlapChecksHandleRegiones, addresses, layout,
                                                checked_bytes(layout.handle_region_capacity, 8u));
    pair.active_histograms_offset = native_pointer(d.activeRegionsHistogram, addresses, layout, histogram_bytes);
    pair.start_histograms_offset = native_pointer(d.startRegionsHistogram, addresses, layout, histogram_bytes);
    pair.ordered_active_handles_offset = native_pointer(d.orderedActiveRegionHandles, addresses, layout, histogram_bytes);
    pair.ordered_start_handles_offset = native_pointer(d.orderedStartRegionHandles, addresses, layout, histogram_bytes);
    pair.found_reports_offset = native_pointer(d.foundPairReport, addresses, layout, pair_bytes);
    pair.found_counter_offset = inline_scalar(layout, descriptor,
        offsetof(physx::PxgBroadPhaseDesc, sharedFoundPairIndex), sizeof(d.sharedFoundPairIndex));
    pair.max_found_lost_pairs = d.max_found_lost_pairs;
    pair.current_boxes = layout.current_boxes;
    pair.box_capacity = layout.box_capacity;
    pair.has_environment_ids = d.updateData_envIDs ? 1u : 0u;

    const auto region_prefix = native_pointer(d.regionAccum, addresses, layout, region_prefix_bytes);
    const auto block = native_pointer(d.blockOverlapChecksRegion, addresses, layout,
                                      block_bytes, alignof(physx::regionOverlapType));
    return make_sap_newbounds_sequence_descriptors(pair, region_prefix, block);
}
} // namespace fyzz::physx_gpu
