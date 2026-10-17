// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef FYZZXYZZY_PXG_SAP_GPU_DMA_UP_HANDLES_NATIVE_H
#define FYZZXYZZY_PXG_SAP_GPU_DMA_UP_HANDLES_NATIVE_H

// The pinned PxgCudaBroadPhaseSap.h imports this source header BEFORE
// declaring conditional updated-handle members. It defines
// SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU and must also control the native
// member-view ABI, regardless of the including translation unit.
#include "PxgBroadPhaseDesc.h"
#include "physx_port/gpucommon/PxgCudaBuffer.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <vector>

namespace physx { namespace Bp { class BroadPhaseUpdateData; } }

namespace fyzz::physx_gpu {
class SourceSapDmaNative;

// These are references to the EXISTING PxgCudaBroadPhaseSap member fields;
// neither helper introduces a separate broadphase state or ownership model.
// Match the feature switch established by the pinned source descriptor
// header BEFORE defining these views. The AABB-manager path still owns the
// updated-array members when the feature is compiled, but its descriptor
// uses the manager's bitmap instead of the updated list.
struct SourceSapGpuDMAUpCountMembers {
    physx::PxU32& num_boxes;
    physx::PxU32& boxes_capacity;
    physx::PxU32& created_size;
    physx::PxU32& removed_size;
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
    physx::PxU32& updated_size;
#endif
};

struct SourceSapGpuDMAUpHandleBuffers {
    physx::PxgTypedCudaBuffer<physx::PxU32>& created;
    physx::PxgTypedCudaBuffer<physx::PxU32>& removed;
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
    physx::PxgTypedCudaBuffer<physx::PxU32>& updated;
#endif
};

struct SourceSapGpuDMAUpCounts {
    physx::PxU32 previous_boxes;
    physx::PxU32 current_boxes;
    physx::PxU32 capacity;
    physx::PxU32 created;
    physx::PxU32 removed;
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
    physx::PxU32 updated;
#endif
    physx::PxU32 projections;
    physx::PxU32 padded_projections;
};

// Pinned PxgCudaBroadPhaseSap::gpuDMAUp reserves 64 slots for each
// projection, storing the rounded totalNbProjectionRegions in PxU32.
// Keep this source constraint shared between count preflight and the
// post-axis allocation helper, so the frame never commits counts before
// discovering deterministic region-slot overflow.
inline constexpr physx::PxU64 kSapRegionSlotsPerProjection = 64u;
physx::PxU64 checked_sap_region_slots_native(physx::PxU64 projections);

// Source ordering is essential:
// 1. Derive/commit counts from original Bp::BroadPhaseUpdateData.
// 2. Grow old integer bounds, then all three per-axis prefix/remainder
//    buffer sequences via PxgSapHistoryGrowthNative.
// 3. Allocate original aggregate report and region scratch storage via
//    PxgSapPostAxisBuffersNative.
// 4. Allocate created/removed/(optional updated) handle arrays (this file).
// 5. Allocate new integer bounds via PxgSapPostAxisBuffersNative.
// 6. Build original broadphase/radix descriptors.
// 7. Queue handle uploads (this file), then broadphase, regular radix
//    and WOR radix descriptors (PxgSapDmaUpUploadsNative).
// The helpers are separated to preserve exact source allocation/copy order
// when integrated back into the owning PhysX SAP broadphase.
SourceSapGpuDMAUpCounts prepare_gpu_dma_up_counts_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapGpuDMAUpCountMembers members);

void allocate_gpu_dma_up_handles_native(
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapGpuDMAUpHandleBuffers buffers);

// Append to caller-owned frame HtoD list without submitting. Used by
// the real gpuDMAUp contiguous handle/descriptor transfer sequence; the
// legacy individual queue function remains valid for targeted tests.
void append_gpu_dma_up_handles_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapGpuDMAUpHandleBuffers buffers,
    std::vector<SourceSapUpload>& uploads);

void queue_gpu_dma_up_handles_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapGpuDMAUpHandleBuffers buffers,
    SourceSapDmaNative& stream);

} // namespace fyzz::physx_gpu
#endif
