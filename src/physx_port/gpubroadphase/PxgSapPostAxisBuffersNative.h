// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef FYZZXYZZY_PXG_SAP_POST_AXIS_BUFFERS_NATIVE_H
#define FYZZXYZZY_PXG_SAP_POST_AXIS_BUFFERS_NATIVE_H

#include "physx_port/gpubroadphase/PxgSapGpuDMAUpHandlesNative.h"

namespace fyzz::physx_gpu {

// References to original PxgCudaBroadPhaseSap members. All source buffers
// and their lifetime remain owned by PhysX; nothing is allocated by a
// second broadphase or by a replacement GPU-buffer controller.
struct SourceSapPostAxisBuffers {
    physx::PxgCudaBufferN<2>& aggregate_report_blocks;
    physx::PxgCudaBufferN<2>& actor_report_blocks;
    physx::PxgCudaBuffer& active_regions;
    physx::PxgCudaBuffer& start_regions;
    physx::PxgCudaBuffer& ordered_active_handles;
    physx::PxgCudaBuffer& ordered_start_handles;
    physx::PxgCudaBuffer& overlap_checks;
    physx::PxgCudaBuffer& block_overlap_checks;
    physx::PxgCudaBuffer& overlap_handle_regions;
    physx::PxgCudaBuffer& region_range;
    physx::PxgCudaBuffer& start_region_accum;
    physx::PxgCudaBuffer& block_start_region_accum;
    physx::PxgCudaBuffer& region_accum;
    physx::PxgCudaBuffer& block_region_accum;
};

// Source gpuDMAUp: execute immediately AFTER finishing the third
// per-axis prefix + remainder. This method contains the exact 2-element
// report loop and region/overlap/allocation sequence before allocating
// changed-handle buffers. Every buffer is scratch and uses non-copying
// PxgCudaBuffer::allocate (NOT allocateCopyOldDataAsync).
void allocate_sap_post_axis_buffers_native(
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapPostAxisBuffers buffers);

// Source gpuDMAUp: mNewIntegerBoundsBuf is allocated only AFTER the
// created/removed/(optional updated) changed-handle buffer allocations,
// and BEFORE updateDescriptor/updateRadixSortDesc.
void allocate_sap_new_integer_bounds_native(
    const SourceSapGpuDMAUpCounts& counts,
    physx::PxgCudaBuffer& new_integer_bounds);

} // namespace fyzz::physx_gpu
#endif
