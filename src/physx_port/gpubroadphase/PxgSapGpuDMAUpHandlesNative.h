// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef FYZZXYZZY_PXG_SAP_GPU_DMA_UP_HANDLES_NATIVE_H
#define FYZZXYZZY_PXG_SAP_GPU_DMA_UP_HANDLES_NATIVE_H

#include "physx_port/gpucommon/PxgCudaBuffer.h"

namespace physx { namespace Bp { class BroadPhaseUpdateData; } }

namespace fyzz::physx_gpu {
class SourceSapDmaNative;

// These are references to the EXISTING PxgCudaBroadPhaseSap member fields;
// neither helper introduces a separate broadphase state or ownership model.
// Match the upstream SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU feature switch:
// the default GPU/AABB-manager path has NO updated-handle field or buffer.
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

// Source ordering is essential:
// 1. Derive/commit counts from original Bp::BroadPhaseUpdateData.
// 2. Grow old integer bounds, then all three per-axis prefix/remainder
//    buffer sequences via PxgSapHistoryGrowthNative.
// 3. Allocate original aggregate report and region scratch storage via
//    PxgSapPostAxisBuffersNative.
// 4. Allocate created/removed/(optional updated) handle arrays (this file).
// 5. Allocate new integer bounds via PxgSapPostAxisBuffersNative.
// 6. Build original broadphase/radix descriptors (pending).
// 7. Queue handle uploads (this file), followed by descriptor uploads.
// The helpers are separated to preserve exact source allocation/copy order
// when integrated back into the owning PhysX SAP broadphase.
SourceSapGpuDMAUpCounts prepare_gpu_dma_up_counts_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapGpuDMAUpCountMembers members);

void allocate_gpu_dma_up_handles_native(
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapGpuDMAUpHandleBuffers buffers);

void queue_gpu_dma_up_handles_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapGpuDMAUpHandleBuffers buffers,
    SourceSapDmaNative& stream);

} // namespace fyzz::physx_gpu
#endif
