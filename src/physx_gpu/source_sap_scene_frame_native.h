// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_update_native.h"
#include "physx_gpu/source_sap_fetch_native.h"
#include "physx_gpu/source_sap_scene_bounds.h"
#include "physx_gpu/source_sap_scene_reports.h"

#include <span>
#include <vector>

namespace fyzz::physx_gpu {

// One borrowed PxgAABBManager/PxgCudaBroadPhaseSap scene frame. Handles,
// memory, buffers, GPU stream, task completion and pinned report
// destinations remain with the original PhysX owners. This adapter
// neither creates a broadphase registry nor advances a second clock.
struct SourceSapSceneFrameNativeMembers {
    std::span<const native::GpuBroadphaseBox> live_boxes;
    std::span<const physx::Bp::ShapeHandle> source_handles;
    SourceSapSceneBounds aabb_arrays;
    SourceSapSceneFrameChanges changes;
    const physx::Bp::BpFilter& filter;
    bool state_changed;
    bool gpu_state_changed;

    SourceSapDmaUpFrameMembers dma;
    SourceSapUpdateNativeFrame update;
    SourceSapFetchHistoryBuffers histories;
    SourceSapActorPairDestination found_pairs;
    SourceSapActorPairDestination lost_pairs;
    std::span<const native::BodyId> volume_to_actor;
    std::vector<native::Pair>& found;
    std::vector<native::Pair>& lost;

    SourceSapMappedReportView* mapped_reports = nullptr;
    bool abort_mode = false;
};

struct SourceSapSceneFrameNativeResult {
    SourceSapUpdateAdmission update;
    SourceSapFetchedCounts fetch;
};

// Called by the original source owner at its update/fetch boundary.
// preBroadPhase and the AABB-manager delta lists must already be prepared.
// This is one source stream through gpuDMAUp, all SAP kernels, fence
// completion, report publication and the seven fetch-time history swaps.
// Output spans remain owned by the source/Sc consumers.
SourceSapSceneFrameNativeResult advance_source_sap_scene_frame_native(
    SourceSapSceneFrameNativeMembers members,
    SourceSapSceneBoundsScratch& geometry_scratch,
    SourceSapDmaNative& stream,
    bool& force_update,
    const SourceSapUpdateNativeStages& stages,
    const CudaDeviceAddressMap& addresses,
    compute::Context& context);

} // namespace fyzz::physx_gpu
