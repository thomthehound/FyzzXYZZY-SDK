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
    // Original Bp::BroadPhaseUpdateData created by
    // prepare_source_sap_scene_update(), AFTER Sc has authored geometry
    // and BEFORE the source preBroadPhase task runs. Borrowed here after
    // that task's dependency has been satisfied by the source owner.
    const physx::Bp::BroadPhaseUpdateData& aabb_update;

    SourceSapDmaUpFrameMembers dma;
    SourceSapUpdateNativeFrame update;
    SourceSapFetchHistoryBuffers histories;
    SourceSapActorPairDestination found_pairs;
    SourceSapActorPairDestination lost_pairs;
    std::span<const BodyId> volume_to_actor;
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
// Sc geometry preparation and the AABB-manager preBroadPhase continuation
// MUST have run first, on the original PhysX task chain. This call joins
// GPU DMA, the SAP kernels, source fence completion, actor report
// publication and the seven fetch-time history swaps on that same stream.
// Output spans remain owned by the source/Sc consumers.
SourceSapSceneFrameNativeResult advance_source_sap_scene_frame_native(
    SourceSapSceneFrameNativeMembers members,
    SourceSapDmaNative& stream,
    bool& force_update,
    const SourceSapUpdateNativeStages& stages,
    const CudaDeviceAddressMap& addresses,
    compute::Context& context);

} // namespace fyzz::physx_gpu
