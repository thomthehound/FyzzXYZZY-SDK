// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_update_native.h"
#include "physx_gpu/source_sap_fetch_native.h"
#include "physx_gpu/source_sap_scene_bounds.h"
#include "physx_gpu/source_sap_scene_reports.h"
#include "physx_gpu/source_sap_volume_actor_lifetime_native.h"
#include "physx_gpu/source_sap_aabb_report_retire_native.h"
#include "physx_port/gpubroadphase/PxgAabbManagerPreUpdateNative.h"

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
    // The source AABB manager owns all bitmap and VolumeData buffers.
    // Its first-pass GPU upload precedes SAP's gpuDMAUp on this same stream.
    const SourceSapAabbManagerPreUpdateNative& aabb_manager;

    SourceSapDmaUpFrameMembers dma;
    SourceSapUpdateNativeFrame update;
    SourceSapFetchHistoryBuffers histories;
    SourceSapActorPairDestination found_pairs;
    SourceSapActorPairDestination lost_pairs;
    // Existing source volume-handle -> Sc actor identity stays alive for
    // removed handles through GPU lost-pair fetch; no second actor map.
    std::span<BodyId> volume_to_actor;
    std::vector<native::Pair>& found;
    std::vector<native::Pair>& lost;
    // PxgAABBManager::processFoundPairs/processLostPairs follow successful
    // fetch and clear these ORIGINAL host and device bitmap allocations.
    const SourceFillD32NativeStage& bitmap_fill;
    SourceSapAabbReportBitmapNative added_bitmap;
    SourceSapAabbReportBitmapNative removed_bitmap;

    SourceSapMappedReportView* mapped_reports = nullptr;
    bool abort_mode = false;
};

struct SourceSapSceneFrameNativeResult {
    SourceSapUpdateAdmission update;
    SourceSapFetchedCounts fetch;
};

// Called by the original source owner at its update/fetch boundary.
// Sc geometry preparation and the AABB manager's handle/bitmap first
// pass MUST have run first. This call performs its native GPU pre-update,
// SAP gpuDMAUp/update, source fence completion, actor report publication
// and seven fetch-time history swaps on the same original stream.
// Output spans remain owned by the source/Sc consumers.
SourceSapSceneFrameNativeResult advance_source_sap_scene_frame_native(
    SourceSapSceneFrameNativeMembers members,
    SourceSapDmaNative& stream,
    bool& force_update,
    const SourceSapUpdateNativeStages& stages,
    const CudaDeviceAddressMap& addresses,
    compute::Context& context);

} // namespace fyzz::physx_gpu
