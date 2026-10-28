// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapPostAxisBuffersNative.h"

#include "physx_port/gpubroadphase/include/PxgBroadPhaseKernelIndices.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace fyzz::physx_gpu {
namespace {
static_assert(sizeof(int) == sizeof(physx::PxU32),
              "Original SAP per-region allocations require 32-bit int");
// Original PxgIntegerAABB.h: PxgHandleRegion holds two PxU32;
// PxgIntegerRegion holds two CUDA uint4 (four PxU32 each).
// The original header also includes CUDA vector_types.h, so retain its
// documented storage ABI without pulling a CUDA dependency into the host.
constexpr physx::PxU64 kHandleRegionBytes = 2u * sizeof(physx::PxU32);
constexpr physx::PxU64 kIntegerRegionBytes = 8u * sizeof(physx::PxU32);
static_assert(kHandleRegionBytes == 8u);
static_assert(kIntegerRegionBytes == 32u);
}

void allocate_sap_post_axis_buffers_native(
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapPostAxisBuffers buffers) {
    // The same source PxU32-region limit is checked before committing
    // gpuDMAUp counters, and again here for direct allocator callers.
    const auto region_slots = checked_sap_region_slots_native(counts.projections);

    for (physx::PxU32 i = 0; i < 2u; ++i) {
        buffers.aggregate_report_blocks[i].allocate(32u * sizeof(physx::PxU32), PX_FL);
        buffers.actor_report_blocks[i].allocate(32u * sizeof(physx::PxU32), PX_FL);
    }

    buffers.active_regions.allocate(region_slots * sizeof(int), PX_FL);
    buffers.start_regions.allocate(region_slots * sizeof(int), PX_FL);
    buffers.ordered_active_handles.allocate(region_slots * sizeof(int), PX_FL);
    buffers.ordered_start_handles.allocate(region_slots * sizeof(int), PX_FL);

    buffers.overlap_checks.allocate(
        kSapRegionSlotsPerProjection * counts.current_boxes * sizeof(physx::PxU64), PX_FL);
    buffers.block_overlap_checks.allocate(
        physx::PxU64(physx::PxgBPKernelGridDim::BP_OUTPUT_OVERLAPCHECKS_HISTOGRAM)
            * sizeof(physx::PxU64), PX_FL);
    buffers.overlap_handle_regions.allocate(
        kSapRegionSlotsPerProjection * counts.current_boxes * kHandleRegionBytes, PX_FL);
    buffers.region_range.allocate(
        physx::PxU64(counts.capacity) * kIntegerRegionBytes, PX_FL);
    buffers.start_region_accum.allocate(
        physx::PxU64(counts.projections) * sizeof(int), PX_FL);
    buffers.block_start_region_accum.allocate(
        physx::PxU64(physx::PxgBPKernelGridDim::BP_OUTPUT_START_REGION_HISTOGRAM)
            * sizeof(int), PX_FL);
    buffers.region_accum.allocate(
        physx::PxU64(counts.projections) * sizeof(int), PX_FL);
    buffers.block_region_accum.allocate(
        physx::PxU64(physx::PxgBPKernelGridDim::BP_OUTPUT_REGION_HISTOGRAM)
            * sizeof(int), PX_FL);
}

void allocate_sap_new_integer_bounds_native(
    const SourceSapGpuDMAUpCounts& counts,
    physx::PxgCudaBuffer& new_integer_bounds) {
    // Source PxgIntegerAABB has six 32-bit integer coordinates.
    constexpr physx::PxU64 kIntegerAabbBytes = 6u * sizeof(physx::PxU32);
    static_assert(kIntegerAabbBytes == 24u);
    new_integer_bounds.allocate(
        physx::PxU64(counts.capacity) * kIntegerAabbBytes, PX_FL);
}

} // namespace fyzz::physx_gpu
