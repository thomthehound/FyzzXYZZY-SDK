// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef FYZZXYZZY_PXG_SAP_UPDATE_FETCH_NATIVE_H
#define FYZZXYZZY_PXG_SAP_UPDATE_FETCH_NATIVE_H

#include "physx_port/gpucommon/PxgCudaBuffer.h"

namespace physx { namespace Bp { class BroadPhaseUpdateData; } }

namespace fyzz::physx_gpu {

// Source PxgCudaBroadPhaseSap::update() gate, invoked ONLY AFTER
// gpu_dma_up_sap_frame_native() has already completed the source
// allocation/descriptor/HtoD preparation. This only decides whether
// the owner runs its original GPU stages; it never dispatches them.
// force_update is a reference to the real mForceUpdate member, which
// the pinned source constructor initializes to true.
bool should_execute_sap_update_after_dma_up_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    bool& force_update);

// Source PxgCudaBroadPhaseSap::update() swaps the two integer-bound
// buffers after generateNewPairsKernel and BEFORE clearNewFlagKernel.
// This must not be deferred to fetchBroadPhaseResults.
void swap_sap_integer_bounds_after_pairs_native(
    physx::PxgCudaBuffer& new_integer_bounds,
    physx::PxgCudaBuffer& old_integer_bounds);

// Borrowed original seven N<6> double-buffered endpoint histories.
// The source fetchBroadPhaseResults swaps each axis i with i+3 only
// AFTER gpuDMABack has read back/published the descriptor and reported
// completion. This helper does not synchronize or own the source stream.
struct SourceSapFetchHistoryBuffers {
    physx::PxgCudaBufferN<6>& projection_handles;
    physx::PxgCudaBufferN<6>& block_end_histograms;
    physx::PxgCudaBufferN<6>& end_histograms;
    physx::PxgCudaBufferN<6>& end_handles;
    physx::PxgCudaBufferN<6>& block_start_histograms;
    physx::PxgCudaBufferN<6>& start_histograms;
    physx::PxgCudaBufferN<6>& start_handles;
};

void swap_sap_endpoint_histories_after_fetch_native(
    SourceSapFetchHistoryBuffers buffers);

} // namespace fyzz::physx_gpu
#endif
