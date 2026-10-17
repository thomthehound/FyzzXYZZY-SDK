// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef FYZZXYZZY_PXG_SAP_SOURCE_BROADPHASE_DESCRIPTOR_NATIVE_H
#define FYZZXYZZY_PXG_SAP_SOURCE_BROADPHASE_DESCRIPTOR_NATIVE_H

// Import the real source descriptor FIRST: the pinned PhysX version
// defines SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU in this header.
// Descriptor ABI and the standalone updated-handle branch must agree.
#include "PxgBroadPhaseDesc.h"
#include "physx_port/gpucommon/PxgCudaBuffer.h"

namespace fyzz::physx_gpu {

// These values MUST originate from an existing PxgAABBManager's getters.
// No AABB manager or replacement bitmap is constructed by this adapter.
// A null pointer means the original standalone-BP branch (updated array).
struct SourceSapAabbManagerDescriptorView {
    CUdeviceptr changed_handles;
    physx::PxU32 changed_bitmap_word_count;
    CUdeviceptr added_handles;
    CUdeviceptr removed_handles;
    CUdeviceptr aggregated_bounds;
    CUdeviceptr volume_data;
};

// References to PhysX source-owned broadphase allocations. The host
// descriptor is a borrowed view, not an alternative owner of GPU memory.
// Grouped only for readability: all fields correspond to original
// PxgCudaBroadPhaseSap m*Buf members.
struct SourceSapBroadphaseDescriptorBuffers {
    const physx::PxgCudaBuffer& created;
    const physx::PxgCudaBuffer& removed;
    const physx::PxgCudaBuffer& updated;
    const physx::PxgCudaBuffer& old_integer_bounds;
    const physx::PxgCudaBuffer& new_integer_bounds;
    const physx::PxgCudaBuffer& fp_bounds;
    const physx::PxgCudaBuffer& contact_distances;
    const physx::PxgCudaBuffer& groups;
    const physx::PxgCudaBuffer& env_ids;

    const physx::PxgCudaBuffer& found_pairs;
    const physx::PxgCudaBuffer& lost_pairs;
    const physx::PxgCudaBuffer& found_aggregate;
    const physx::PxgCudaBuffer& lost_aggregate;
    const physx::PxgCudaBuffer& found_actor;
    const physx::PxgCudaBuffer& lost_actor;

    const physx::PxgCudaBufferN<3>& sap_boxes;
    const physx::PxgCudaBufferN<3>& new_sap_boxes;
    const physx::PxgCudaBufferN<3>& ranks;
    const physx::PxgCudaBufferN<3>& projections;
    const physx::PxgCudaBufferN<6>& projection_handles;
    const physx::PxgCudaBufferN<6>& total_end_histograms;
    const physx::PxgCudaBufferN<6>& block_total_end_histograms;
    const physx::PxgCudaBufferN<6>& end_histograms;
    const physx::PxgCudaBufferN<6>& block_end_histograms;
    const physx::PxgCudaBufferN<6>& start_histograms;
    const physx::PxgCudaBufferN<6>& block_start_histograms;
    const physx::PxgCudaBufferN<6>& end_handles;
    const physx::PxgCudaBufferN<6>& start_handles;
    const physx::PxgCudaBufferN<3>& incremental_comparisons;
    const physx::PxgCudaBufferN<3>& block_incremental_comparisons;

    const physx::PxgCudaBufferN<2>& aggregate_report_blocks;
    const physx::PxgCudaBufferN<2>& actor_report_blocks;
    const physx::PxgCudaBuffer& active_regions;
    const physx::PxgCudaBuffer& start_regions;
    const physx::PxgCudaBuffer& ordered_active_handles;
    const physx::PxgCudaBuffer& ordered_start_handles;
    const physx::PxgCudaBuffer& block_overlap_checks;
    const physx::PxgCudaBuffer& overlap_checks;
    const physx::PxgCudaBuffer& overlap_handle_regions;
    const physx::PxgCudaBuffer& region_range;
    const physx::PxgCudaBuffer& start_region_accum;
    const physx::PxgCudaBuffer& block_start_region_accum;
    const physx::PxgCudaBuffer& region_accum;
    const physx::PxgCudaBuffer& block_region_accum;
};

struct SourceSapBroadphaseDescriptorInputs {
    // Source computes previous from the live post-update count rather
    // than loading a separate mutable previous-box state.
    physx::PxU32 num_boxes;
    physx::PxU32 created_count;
    physx::PxU32 removed_count;
    physx::PxU32 updated_count;
    CUdeviceptr mapped_found_actor_pairs; // existing pinned-map GPU alias
    CUdeviceptr mapped_lost_actor_pairs;
    physx::PxU32 region_accum_total;
    physx::regionOverlapType overlap_checks_total;
    physx::PxU32 max_found_lost_pairs;
    physx::PxU32 max_found_lost_aggregate_pairs;
    const SourceSapAabbManagerDescriptorView* aabb_manager;
};

// Source PxgCudaBroadPhaseSap::updateDescriptor: fully reset the exact
// source struct, apply manager-vs-standalone branch, and populate every
// original field it writes, including 2x3 endpoint indexing and scalar
// reset semantics. DOES NOT upload; original gpuDMAUp uploads this first,
// after changed-handle HtoD and before both radix descriptor arrays.
void prepare_sap_broadphase_descriptor_native(
    physx::PxgBroadPhaseDesc& desc,
    SourceSapBroadphaseDescriptorBuffers buffers,
    const SourceSapBroadphaseDescriptorInputs& inputs);

} // namespace fyzz::physx_gpu
#endif
