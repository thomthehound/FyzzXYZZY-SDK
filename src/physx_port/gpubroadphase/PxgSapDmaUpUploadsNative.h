// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef FYZZXYZZY_PXG_SAP_DMA_UP_UPLOADS_NATIVE_H
#define FYZZXYZZY_PXG_SAP_DMA_UP_UPLOADS_NATIVE_H

#include "physx_port/gpubroadphase/include/PxgBroadPhaseDesc.h"
#include "physx_port/gpucommon/include/PxgRadixSortDesc.h"
#include "physx_port/gpucommon/PxgCudaBuffer.h"

#include <vector>

namespace fyzz::physx_gpu {
class SourceSapDmaNative;
struct SourceSapUpload;

// Existing PxgCudaBroadPhaseSap descriptor buffers, originally allocated
// in its constructor. The caller retains both their owning GPU memory
// and the pinned host descriptor records throughout this source stage.
struct SourceSapDmaUpDescriptorBuffers {
    const physx::PxgTypedCudaBuffer<physx::PxgBroadPhaseDesc>& broadphase;
    const physx::PxgTypedCudaBuffer<physx::PxgRadixSortDesc>& radix;
    const physx::PxgTypedCudaBuffer<physx::PxgRadixSortDesc>& radix_without_ranks;
};

// Original constructor descriptor buffers must exist BEFORE a frame
// mutates persistent SAP count state. Validate all three as one group.
void validate_sap_dma_up_descriptor_buffers_native(
    SourceSapDmaUpDescriptorBuffers buffers);

// Source PxgCudaBroadPhaseSap::gpuDMAUp descriptor HtoD sequence. Must be
// called AFTER source queue_gpu_dma_up_handles_native on the SAME mStream,
// following updateDescriptor and updateRadixSortDesc. One native upload
// batch retains the exact source ordering: BP descriptor, six regular
// radix descriptors, then six without-ranks radix descriptors. This
// function allocates nothing and does not wait/poll for GPU completion.
// Append original descriptor transfers to the same source frame HtoD
// list as handle uploads, preserving the source stream ordering.
void append_sap_dma_up_descriptors_native(
    const physx::PxgBroadPhaseDesc& bp_desc,
    const physx::PxgRadixSortDesc* radix_descs,
    const physx::PxgRadixSortDesc* radix_wor_descs,
    SourceSapDmaUpDescriptorBuffers buffers,
    std::vector<SourceSapUpload>& uploads);

void queue_sap_dma_up_descriptors_native(
    const physx::PxgBroadPhaseDesc& bp_desc,
    const physx::PxgRadixSortDesc* radix_descs,
    const physx::PxgRadixSortDesc* radix_wor_descs,
    SourceSapDmaUpDescriptorBuffers buffers,
    SourceSapDmaNative& stream);

} // namespace fyzz::physx_gpu
#endif
