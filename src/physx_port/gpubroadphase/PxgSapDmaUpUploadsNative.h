// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef FYZZXYZZY_PXG_SAP_DMA_UP_UPLOADS_NATIVE_H
#define FYZZXYZZY_PXG_SAP_DMA_UP_UPLOADS_NATIVE_H

#include "PxgBroadPhaseDesc.h"
#include "PxgRadixSortDesc.h"
#include "physx_port/gpucommon/PxgCudaBuffer.h"

namespace fyzz::physx_gpu {
class SourceSapDmaNative;

// Existing PxgCudaBroadPhaseSap descriptor buffers, originally allocated
// in its constructor. The caller retains both their owning GPU memory
// and the pinned host descriptor records throughout this source stage.
struct SourceSapDmaUpDescriptorBuffers {
    const physx::PxgTypedCudaBuffer<physx::PxgBroadPhaseDesc>& broadphase;
    const physx::PxgTypedCudaBuffer<physx::PxgRadixSortDesc>& radix;
    const physx::PxgTypedCudaBuffer<physx::PxgRadixSortDesc>& radix_without_ranks;
};

// Source PxgCudaBroadPhaseSap::gpuDMAUp descriptor HtoD sequence. Must be
// called AFTER source queue_gpu_dma_up_handles_native on the SAME mStream,
// following updateDescriptor and updateRadixSortDesc. One native upload
// batch retains the exact source ordering: BP descriptor, six regular
// radix descriptors, then six without-ranks radix descriptors. This
// function allocates nothing and does not wait/poll for GPU completion.
void queue_sap_dma_up_descriptors_native(
    const physx::PxgBroadPhaseDesc& bp_desc,
    const physx::PxgRadixSortDesc* radix_descs,
    const physx::PxgRadixSortDesc* radix_wor_descs,
    SourceSapDmaUpDescriptorBuffers buffers,
    SourceSapDmaNative& stream);

} // namespace fyzz::physx_gpu
#endif
