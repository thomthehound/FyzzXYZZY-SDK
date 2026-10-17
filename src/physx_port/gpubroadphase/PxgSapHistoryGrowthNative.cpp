// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapHistoryGrowthNative.h"
#include "physx_gpu/source_sap_dma_native.h"

#include "PxgSapBox1D.h"

#include <cstdint>

namespace fyzz::physx_gpu {
namespace {
// Upstream PxgIntegerAABB::mMinMax is precisely six PxU32 values.
// PxgIntegerAABB.h includes CUDA's vector_types.h for a second,
// unrelated region struct; the native host is intentionally CUDA-free.
// Keep the source's 24-byte GPU storage contract explicit here.
constexpr physx::PxU64 kIntegerAabbBytes = 6u * sizeof(physx::PxU32);
static_assert(kIntegerAabbBytes == 24u);
static_assert(sizeof(physx::PxgSapBox1D) == 2u * sizeof(physx::PxU32));
}

void grow_sap_old_bounds_native(
    const SourceSapGpuDMAUpCounts& counts,
    physx::PxgCudaBuffer& old_integer_bounds,
    SourceSapDmaNative& stream) {
    old_integer_bounds.allocateCopyOldDataAsync(
        physx::PxU64(counts.capacity) * kIntegerAabbBytes, stream, PX_FL);
}

void grow_sap_history_axis_prefix_native(
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapHistoryAxisBuffers buffers,
    SourceSapDmaNative& stream) {
    // Preserve the *source ordering* within each axis. In particular,
    // the old SAP boxes/projections use deferred-free GPU DtoD growth;
    // new SAP boxes, ranks and temporaries do NOT copy old bytes.
    buffers.box_sap_1d.allocateCopyOldDataAsync(
        physx::PxU64(counts.capacity) * sizeof(physx::PxgSapBox1D),
        stream, PX_FL);
    buffers.new_box_sap_1d.allocate(
        physx::PxU64(counts.capacity) * sizeof(physx::PxgSapBox1D), PX_FL);
    buffers.projections.allocateCopyOldDataAsync(
        physx::PxU64(counts.padded_projections) * sizeof(int),
        stream, PX_FL);
    buffers.projection_ranks.allocate(
        physx::PxU64(counts.padded_projections) * sizeof(int), PX_FL);
    buffers.temporary_projections.allocate(
        physx::PxU64(counts.padded_projections) * sizeof(int), PX_FL);
    buffers.temporary_handles.allocate(
        physx::PxU64(counts.padded_projections) * sizeof(int), PX_FL);
}

} // namespace fyzz::physx_gpu
