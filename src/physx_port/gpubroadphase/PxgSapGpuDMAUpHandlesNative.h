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

// Borrow the original PxgCudaBroadPhaseSap member fields: all lifetime,
// counters, and device buffers remain owned by that source broadphase.
// The result describes source projection sizing for the next gpuDMAUp slice.
struct SourceSapGpuDMAUpHandleMembers {
    physx::PxU32& num_boxes;
    physx::PxU32& boxes_capacity;
    physx::PxU32& created_size;
    physx::PxU32& removed_size;
    physx::PxU32& updated_size;
    physx::PxgTypedCudaBuffer<physx::PxU32>& created;
    physx::PxgTypedCudaBuffer<physx::PxU32>& removed;
    physx::PxgTypedCudaBuffer<physx::PxU32>& updated;
};

struct SourceSapGpuDMAUpCounts {
    physx::PxU32 previous_boxes;
    physx::PxU32 current_boxes;
    physx::PxU32 capacity;
    physx::PxU32 created;
    physx::PxU32 removed;
    physx::PxU32 updated;
    physx::PxU32 projections;
    physx::PxU32 padded_projections;
};

// Source: initial and final handle-array portions of
// PxgCudaBroadPhaseSap::gpuDMAUp. The caller inserts the original
// intermediate persistent SAP buffer allocation, updateDescriptor,
// and updateRadixSortDesc before the final source descriptor uploads.
// No independent scene state or altered PhysX scheduling is introduced.
SourceSapGpuDMAUpCounts gpu_dma_up_handles_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapGpuDMAUpHandleMembers members,
    SourceSapDmaNative& stream);

} // namespace fyzz::physx_gpu
#endif
