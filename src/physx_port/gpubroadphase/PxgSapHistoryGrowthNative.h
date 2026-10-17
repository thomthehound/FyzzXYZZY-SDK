// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef FYZZXYZZY_PXG_SAP_HISTORY_GROWTH_NATIVE_H
#define FYZZXYZZY_PXG_SAP_HISTORY_GROWTH_NATIVE_H

#include "physx_port/gpubroadphase/PxgSapGpuDMAUpHandlesNative.h"

namespace fyzz::physx_gpu {
class SourceSapDmaNative;

// Source: the first copy-preserving allocation in
// PxgCudaBroadPhaseSap::gpuDMAUp, immediately after updating box counts.
// Binds the actual mOldIntegerBoundsBuf owner; no substitute GPU history.
void grow_sap_old_bounds_native(
    const SourceSapGpuDMAUpCounts& counts,
    physx::PxgCudaBuffer& old_integer_bounds,
    SourceSapDmaNative& stream);

// Source: the first six operations in EACH axis of gpuDMAUp's original
// three-axis loop. The caller must invoke
// grow_sap_history_axis_remainder_native(i) immediately afterwards,
// completing both inner j passes and per-axis scratch BEFORE advancing
// to the next axis. No cross-axis batching/reordering.
struct SourceSapHistoryAxisBuffers {
    physx::PxgCudaBuffer& box_sap_1d;
    physx::PxgCudaBuffer& new_box_sap_1d;
    physx::PxgCudaBuffer& projections;
    physx::PxgCudaBuffer& projection_ranks;
    physx::PxgCudaBuffer& temporary_projections;
    physx::PxgCudaBuffer& temporary_handles;
};

void grow_sap_history_axis_prefix_native(
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapHistoryAxisBuffers buffers,
    SourceSapDmaNative& stream);

// Remaining original per-axis loop body. Each N<6> index is j*3 + axis;
// N<3> indexes the outer axis directly. The source owner retains all
// arrays and their original buddy-heap/deferred-copy lifetime semantics.
struct SourceSapHistoryAxisRemainderBuffers {
    physx::PxgCudaBufferN<6>& projection_handles;
    physx::PxgCudaBufferN<6>& end_histograms;
    physx::PxgCudaBufferN<6>& block_end_histograms;
    physx::PxgCudaBufferN<6>& end_handles;
    physx::PxgCudaBufferN<6>& start_histograms;
    physx::PxgCudaBufferN<6>& block_start_histograms;
    physx::PxgCudaBufferN<6>& start_handles;
    physx::PxgCudaBufferN<3>& incremental_comparisons;
    physx::PxgCudaBufferN<3>& incremental_block_comparisons;
    physx::PxgCudaBufferN<6>& total_end_histograms;
    physx::PxgCudaBufferN<6>& block_total_end_histograms;
};

// Invoke AFTER grow_sap_history_axis_prefix_native for this axis and
// BEFORE the prefix of the next axis. Implements both original j=0,1
// seven-buffer sequences followed by the four per-axis scratch buffers.
void grow_sap_history_axis_remainder_native(
    const SourceSapGpuDMAUpCounts& counts,
    physx::PxU32 axis,
    SourceSapHistoryAxisRemainderBuffers buffers,
    SourceSapDmaNative& stream);

} // namespace fyzz::physx_gpu
#endif
