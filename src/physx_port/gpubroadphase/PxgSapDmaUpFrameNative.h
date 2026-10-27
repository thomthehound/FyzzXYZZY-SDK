// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef FYZZXYZZY_PXG_SAP_DMA_UP_FRAME_NATIVE_H
#define FYZZXYZZY_PXG_SAP_DMA_UP_FRAME_NATIVE_H

#include "physx_port/gpubroadphase/PxgSapGpuDMAUpHandlesNative.h"
#include "physx_port/gpubroadphase/PxgSapHistoryGrowthNative.h"
#include "physx_port/gpubroadphase/PxgSapPostAxisBuffersNative.h"
#include "physx_port/gpubroadphase/PxgSapBroadphaseDescriptorNative.h"
#include "physx_port/gpubroadphase/PxgSapRadixDescriptorsNative.h"
#include "physx_port/gpubroadphase/PxgSapDmaUpUploadsNative.h"

#include <array>

namespace fyzz::physx_gpu {
class SourceSapDmaNative;

// Borrowed aliases to EXISTING PxgCudaBroadPhaseSap members. This is the
// original gpuDMAUp() call-site dependency bundle, not an alternative
// broadphase object: no scene state, heap, stream, buffers or reports are
// owned or allocated by this struct. The surrounding PhysX owner remains
// responsible for all member lifetimes, source task ordering and fetch.
struct SourceSapDmaUpFrameMembers {
    SourceSapGpuDMAUpCountMembers counts;
    SourceSapGpuDMAUpHandleBuffers handles;
    physx::PxgCudaBuffer& old_integer_bounds;
    std::array<SourceSapHistoryAxisBuffers, 3> axes;
    SourceSapHistoryAxisRemainderBuffers remainder;
    SourceSapPostAxisBuffers post_axis;
    physx::PxgCudaBuffer& new_integer_bounds;

    SourceSapBroadphaseDescriptorBuffers broadphase_buffers;
    SourceSapBroadphaseDescriptorInputs descriptor_inputs;
    SourceSapRadixDescriptorBuffers radix_buffers;
    SourceSapDmaUpDescriptorBuffers descriptor_gpu_buffers;

    physx::PxgBroadPhaseDesc& broadphase_descriptor;
    physx::PxgRadixSortDesc* radix_descriptors;
    physx::PxgRadixSortDesc* radix_wor_descriptors;
};

// Original PxgCudaBroadPhaseSap::gpuDMAUp() in a single source-ordered,
// native-compute-backed call. Inputs are an existing BroadPhaseUpdateData,
// source-owned GPU buffers and host descriptor storage, and the EXISTING
// source stream. This does NOT replace preBroadPhase(), the source GPU
// kernel/task graph, GPU/CPU result publication, or PxScene scheduling.
//
// The broadphase AABB-manager getter aliases, mapped pair-report aliases,
// capacities/totals and host descriptor pointers must already come from
// the source owner; no synthetic AABB manager is constructed here.
SourceSapGpuDMAUpCounts gpu_dma_up_sap_frame_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapDmaUpFrameMembers members,
    SourceSapDmaNative& stream);

// This is the contiguous beginning of the original
// PxgCudaBroadPhaseSap::update(): gpuDMAUp always runs BEFORE its
// forced/quiet update gate. The caller retains the original force-update
// member, allocations, stream and downstream kernel/fetch continuations.
// In particular, a rejected DMA preparation cannot consume mForceUpdate.
struct SourceSapUpdateAdmission {
    SourceSapGpuDMAUpCounts counts;
    bool execute_gpu_stages = false;
};

SourceSapUpdateAdmission begin_sap_update_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapDmaUpFrameMembers members,
    SourceSapDmaNative& stream,
    bool& force_update);

} // namespace fyzz::physx_gpu
#endif
