// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapHistoryGrowthNative.h"
#include "physx_gpu/source_sap_dma_native.h"

#include "PxgSapBox1D.h"
#include "PxgBroadPhaseKernelIndices.h"

#include <stdexcept>

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

void grow_sap_history_axis_remainder_native(
    const SourceSapGpuDMAUpCounts& counts,
    physx::PxU32 axis,
    SourceSapHistoryAxisRemainderBuffers buffers,
    SourceSapDmaNative& stream) {
    if (axis >= 3u)
        throw std::out_of_range("PhysX SAP history has exactly three spatial axes");
    static_assert(sizeof(int) == sizeof(physx::PxU32),
                  "Original SAP endpoint and radix buffers require 32-bit int");
    // Retain upstream's j*3+i layout and the seven original allocation/
    // GPU-copy calls PER j; batching all N<6> arrays by category would
    // reorder deferred frees and the original PxgCudaBuffer dependencies.
    for (physx::PxU32 j = 0; j < 2u; ++j) {
        const auto index = j * 3u + axis;
        buffers.projection_handles[index].allocateCopyOldDataAsync(
            physx::PxU64(counts.padded_projections) * sizeof(int), stream, PX_FL);
        buffers.end_histograms[index].allocateCopyOldDataAsync(
            physx::PxU64(counts.projections) * sizeof(int), stream, PX_FL);
        buffers.block_end_histograms[index].allocateCopyOldDataAsync(
            physx::PxU64(physx::PxgBPKernelGridDim::BP_OUTPUT_ENDPT_HISTOGRAM)
                * sizeof(int), stream, PX_FL);
        buffers.end_handles[index].allocateCopyOldDataAsync(
            physx::PxU64(counts.current_boxes) * sizeof(int), stream, PX_FL);
        buffers.start_histograms[index].allocateCopyOldDataAsync(
            physx::PxU64(counts.projections) * sizeof(int), stream, PX_FL);
        buffers.block_start_histograms[index].allocateCopyOldDataAsync(
            physx::PxU64(physx::PxgBPKernelGridDim::BP_OUTPUT_ENDPT_HISTOGRAM)
                * sizeof(int), stream, PX_FL);
        buffers.start_handles[index].allocateCopyOldDataAsync(
            physx::PxU64(counts.current_boxes) * sizeof(int), stream, PX_FL);
    }

    // Original per-axis scratch is allocated WITHOUT preserving old data.
    buffers.incremental_comparisons[axis].allocate(
        physx::PxU64(counts.projections) * sizeof(int), PX_FL);
    buffers.incremental_block_comparisons[axis].allocate(
        physx::PxU64(physx::PxgBPKernelGridDim::BP_COMPUTE_INCREMENTAL_CMP_COUNTS1)
            * sizeof(int), PX_FL);
    buffers.total_end_histograms[axis].allocate(
        physx::PxU64(counts.projections) * sizeof(physx::PxU32), PX_FL);
    buffers.block_total_end_histograms[axis].allocate(
        physx::PxU64(physx::PxgBPKernelGridDim::BP_OUTPUT_ENDPT_HISTOGRAM)
            * sizeof(physx::PxU32), PX_FL);
}

} // namespace fyzz::physx_gpu
