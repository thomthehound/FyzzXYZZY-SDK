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
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {
class PxgAabbSourceMembersNative;
class PxgSapSourceMembersNative;
class SourceAggregateMemberHistoryNativeStage;
class SourceAggregateProjectionNativeStage;
class SourceAggregateSelfCollisionNativeStage;
class SourceAggregateReportCopyNativeStage;

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
    // When using the persistent source AABB owner, resolve its GPU
    // bitmap allocations AFTER preBpUpdate_GPU (which can grow them)
    // and complete its frame only after successful source GPU fetch.
    PxgAabbSourceMembersNative* aabb_owner = nullptr;
    // Optional original aggregate post-SAP kernel, supplied by the same
    // source GPU context when aggregate execution is enabled.
    const SourceAggregateMemberHistoryNativeStage* aggregate_member_history = nullptr;
    const SourceAggregateProjectionNativeStage* aggregate_projection = nullptr;
    const SourceAggregateSelfCollisionNativeStage* aggregate_self_collision = nullptr;
    const SourceAggregateReportCopyNativeStage* aggregate_report_copy = nullptr;
};

struct SourceSapSceneFrameNativeResult {
    SourceSapUpdateAdmission update;
    SourceSapFetchedCounts fetch;
};

// Both source-owner view aliases are verified before any GPU work. The
// references are original PhysX AABB manager allocations; no registry,
// lookup loop, or per-frame copy of the source identity table is required.
void validate_source_sap_owned_aabb_aliases_native(
    PxgAabbSourceMembersNative& owner,
    const SourceSapAabbManagerPreUpdateNative& pre,
    std::span<BodyId> volume_to_actor);

// Full source-owner boundary: admission, one GPU preBroadPhase / gpuDMAUp,
// source kernel update/fetch, bitmap retirement and AABB frame completion.
// The original Sc scene retains its element ID tracker until its own
// postReportsCleanup, after all Sc reports have been consumed.
struct SourceSapOwnedSceneFrameNativeInputs {
    std::span<const native::GpuBroadphaseBox> live_boxes;
    std::span<const physx::Bp::ShapeHandle> live_handles;
    const physx::Bp::BpFilter& filter;
    // An empty layout derives directly from the source GPU DMA result;
    // explicit layouts are reserved for targeted contract tests.
    const SourceSapUpdateNativeFrame& update;
    std::vector<native::Pair>& found;
    std::vector<native::Pair>& lost;
    const SourceFillD32NativeStage& bitmap_fill;
    SourceSapMappedReportView& mapped_reports;
    bool abort_mode=false;
};

SourceSapSceneFrameNativeResult advance_source_sap_owned_scene_frame_native(
    PxgAabbSourceMembersNative& aabb,
    PxgSapSourceMembersNative& sap,
    SourceSapOwnedSceneFrameNativeInputs inputs,
    SourceSapDmaNative& stream,
    const SourceSapUpdateNativeStages& stages,
    const CudaDeviceAddressMap& addresses,
    compute::Context& context);

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
