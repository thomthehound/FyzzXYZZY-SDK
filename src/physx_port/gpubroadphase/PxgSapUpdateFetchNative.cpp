// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapUpdateFetchNative.h"
#include "physx_port/lowlevelaabb/include/BpBroadPhaseUpdate.h"

#include <cstdint>

namespace fyzz::physx_gpu {

bool should_execute_sap_update_after_dma_up_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    bool& force_update) {
    // Exact source ordering: gpuDMAUp has ALREADY completed before this
    // test in PxgCudaBroadPhaseSap::update(). Do not skip the DMA work
    // for an idle frame; mForceUpdate exists to force one final pass
    // after activity so the previous/new bounds double buffer converges.
    bool forced_update = false;
    const bool gpu_state_changed = update.getGpuStateChanged();
    if (update.getNumCreatedHandles() + std::uint64_t(update.getNumRemovedHandles()) == 0 &&
        !update.getStateChanged() && !gpu_state_changed) {
        if (force_update)
            forced_update = true;
        else
            return false; // original early return retains false state
    }
    force_update = !forced_update;
    return true;
}

void swap_sap_integer_bounds_after_pairs_native(
    physx::PxgCudaBuffer& new_integer_bounds,
    physx::PxgCudaBuffer& old_integer_bounds) {
    physx::PxgCudaBuffer::swapBuffer(new_integer_bounds, old_integer_bounds);
}

void swap_sap_endpoint_histories_after_fetch_native(
    SourceSapFetchHistoryBuffers b) {
    // Preserve the exact order in pinned fetchBroadPhaseResults.
    // No allocations or device commands; only original mPtr/mSize
    // ownership swaps. The caller must first complete gpuDMABack.
    for (physx::PxU32 i = 0; i < 3u; ++i) {
        const physx::PxU32 other = i + 3u;
        physx::PxgCudaBuffer::swapBuffer(b.projection_handles[i], b.projection_handles[other]);
        physx::PxgCudaBuffer::swapBuffer(b.block_end_histograms[i], b.block_end_histograms[other]);
        physx::PxgCudaBuffer::swapBuffer(b.end_histograms[i], b.end_histograms[other]);
        physx::PxgCudaBuffer::swapBuffer(b.end_handles[i], b.end_handles[other]);
        physx::PxgCudaBuffer::swapBuffer(b.block_start_histograms[i], b.block_start_histograms[other]);
        physx::PxgCudaBuffer::swapBuffer(b.start_histograms[i], b.start_histograms[other]);
        physx::PxgCudaBuffer::swapBuffer(b.start_handles[i], b.start_handles[other]);
    }
}
} // namespace fyzz::physx_gpu
