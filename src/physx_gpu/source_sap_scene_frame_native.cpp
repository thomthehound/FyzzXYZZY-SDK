// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_scene_frame_native.h"

#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
SourceSapSceneFrameNativeResult advance_source_sap_scene_frame_native(
    SourceSapSceneFrameNativeMembers members,
    SourceSapDmaNative& stream,
    bool& force_update,
    const SourceSapUpdateNativeStages& stages,
    const CudaDeviceAddressMap& addresses,
    compute::Context& context) {

    // The source manager's handle/bitmap first pass has already created
    // this frame's authoritative Bp update. Validate its GPU dirty flag
    // before queuing any source heap allocation or transfer.
    if(members.aabb_manager.gpu_state_changed!=
       members.aabb_update.getGpuStateChanged())
        throw std::invalid_argument(
            "Source AABB manager GPU dirty flag disagrees with Bp frame");

    // A removed source VolumeData is already reset, but its Sc actor ID
    // must survive until GPU lost-pair publication. Validate original
    // first-pass lists and retained identities BEFORE any GPU upload.
    const SourceSapVolumeActorLifetimeNative volume_lifetime{
        members.volume_to_actor,members.aabb_manager.volume_data};
    validate_source_sap_volume_actor_lifetime_native(
        volume_lifetime,members.aabb_update);

    // PxgAABBManager::preBpUpdate_GPU executes before SAP update().
    // Its original device bitmap and VolumeData buffers are used by
    // the SAP descriptor, not a parallel scene-owned history.
    pre_update_source_sap_aabb_manager_native(
        members.aabb_update,members.aabb_manager,stream);
    const auto manager_descriptor=
        source_sap_aabb_descriptor_native(members.aabb_manager);
    auto dma=members.dma;
    dma.descriptor_inputs.aabb_manager=&manager_descriptor;

    // Original PxgCudaBroadPhaseSap::update: gpuDMAUp always precedes
    // mForceUpdate gating. The same owner retains the stream and all
    // persistent allocations until its fetch continuation has completed.
    const auto admission=advance_source_sap_frame_native(
        members.aabb_update,dma,stream,force_update,stages,
        addresses,members.update);
    stream.wait_at_fetch();

    // Complete both source bitmap D32 view validations before reporting
    // success to Sc. Reserve the following GPU command queue now so
    // post-fetch retirement cannot allocate after history publication.
    auto bitmap_retirement=prepare_source_sap_aabb_report_retirement_native(
        stream,members.bitmap_fill,addresses,
        members.added_bitmap,members.removed_bitmap);
    auto& pending=stream.commands();
    pending.reserve(pending.size()+bitmap_retirement.commands.size());

    // Fetch publishes the ORIGINAL CPU destination arrays and rotates
    // all seven PxgCudaBufferN<6> endpoint histories exactly once.
    // Failed source-buffer validation does not publish either report.
    const auto fetched=fetch_source_sap_results_native(
        context,addresses,members.update.device_descriptor,
        members.dma.broadphase_descriptor,
        members.found_pairs,members.lost_pairs,
        members.abort_mode,
        [histories=members.histories]() noexcept {
            swap_sap_endpoint_histories_after_fetch_native(histories);
        },
        members.mapped_reports,
        [&](const SourceSapFetchedCounts& counts) {
            if(counts.overflow)
                throw std::overflow_error(
                    "Original GPU SAP found/lost report capacity overflow");
            const std::span<const physx::PxgBroadPhasePair> source_found{
                members.found_pairs.data,counts.found_actors};
            const std::span<const physx::PxgBroadPhasePair> source_lost{
                members.lost_pairs.data,counts.lost_actors};
            // GPU found reports require currently admitted source
            // VolumeData. Lost reports can still reference retired
            // handles via the original source-to-Sc identity mapping.
            validate_source_sap_found_volume_lifetime_native(
                source_found,volume_lifetime);
            lower_source_sap_actor_reports_to_sc(
                source_found,source_lost,members.volume_to_actor,
                members.found,members.lost);
        });
    // Source processFoundPairs/processLostPairs clear added/removed maps
    // only after the completed GPU report and Sc admission. These D32
    // dispatches are enqueued on the SAME stream; no fetch-time host wait.
    commit_source_sap_aabb_report_retirement_native(
        stream,std::move(bitmap_retirement));
    retire_source_sap_volume_actor_lifetime_native(
        volume_lifetime,members.aabb_update);
    return {admission,fetched};
}
} // namespace fyzz::physx_gpu
