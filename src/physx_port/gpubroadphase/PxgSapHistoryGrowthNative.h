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
// three-axis loop. The caller must execute the original inner j=0,1
// histogram/handle updates and per-axis scratch allocations BEFORE
// calling this helper for the next axis. No cross-axis batching/reordering.
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

} // namespace fyzz::physx_gpu
#endif
