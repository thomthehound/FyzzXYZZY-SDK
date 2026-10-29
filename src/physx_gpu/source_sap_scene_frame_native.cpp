// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_scene_frame_native.h"
#include "physx_port/gpubroadphase/PxgAabbSourceMembersNative.h"
#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
void append_source_sap_post_aggregate_member_history_native(
    const physx::PxgBroadPhaseDesc& descriptor,
    PxgAabbSourceMembersNative& aabb,
    const SourceAggregateMemberHistoryNativeStage& stage,
    SourceSapDmaNative& stream,
    const CudaDeviceAddressMap& addresses) {
    const auto gpu_address=[](auto pointer) {
        return static_cast<CudaDeviceAddressMap::Address>(
            reinterpret_cast<std::uintptr_t>(pointer));
    };
    // SAP swaps the PxgCudaBuffer owners while appending GPU commands.
    // Its descriptor's new/old pointers still identify the exact buffers
    // used by the current GPU update; owner member names may be reversed.
    const auto buffers=aabb.aggregate_member_history_buffers(
        gpu_address(descriptor.newIntegerBounds),
        gpu_address(descriptor.oldIntegerBounds));
    stage.append(stream,addresses,buffers);
}

void validate_source_sap_owned_aabb_aliases_native(
    PxgAabbSourceMembersNative& owner,
    const SourceSapAabbManagerPreUpdateNative& pre,
    std::span<BodyId> volume_to_actor) {
    const auto lifetime=owner.actor_lifetime();
    const auto added=owner.added_report_bitmap();
    const auto removed=owner.removed_report_bitmap();
    if(volume_to_actor.data()!=lifetime.volume_to_actor.data() ||
       volume_to_actor.size()!=lifetime.volume_to_actor.size() ||
       pre.volume_data.data()!=lifetime.volume_data.data() ||
       pre.volume_data.size()!=lifetime.volume_data.size() ||
       pre.added.words.data()!=added.host_words.data() ||
       pre.added.words.size()!=added.host_words.size() ||
       pre.removed.words.data()!=removed.host_words.data() ||
       pre.removed.words.size()!=removed.host_words.size())
        throw std::invalid_argument(
            "Source SAP frame does not borrow its original AABB manager owner");
}

SourceSapSceneFrameNativeResult advance_source_sap_scene_frame_native(
    SourceSapSceneFrameNativeMembers members,
    SourceSapDmaNative& stream,
    bool& force_update,
    const SourceSapUpdateNativeStages& stages,
    const CudaDeviceAddressMap& addresses,
    compute::Context& context) {

    if(members.aabb_owner)
        validate_source_sap_owned_aabb_aliases_native(
            *members.aabb_owner,members.aabb_manager,members.volume_to_actor);
    if(members.aabb_manager.has_aggregates &&
       (!members.aabb_owner || !members.aggregate_member_history ||
        !members.aggregate_projection))
        throw std::logic_error(
            "Original aggregate post-SAP requires AABB owner and GPU history stages");

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
    // Original PxgAABBManager::updateBPSecondPass postBPStage2 begins
    // after PxgCudaBroadPhaseSap::update. Newly admitted and retired
    // aggregate members must invalidate the descriptor's integer AABB
    // histories before projection/self/pair collision stages consume them.
    if(members.aabb_manager.has_aggregates) {
        append_source_sap_post_aggregate_member_history_native(
            dma.broadphase_descriptor,*members.aabb_owner,
            *members.aggregate_member_history,stream,addresses);
        const auto current_integer_address=
            static_cast<CudaDeviceAddressMap::Address>(
                reinterpret_cast<std::uintptr_t>(
                    dma.broadphase_descriptor.newIntegerBounds));
        const auto projections=members.aabb_owner->aggregate_projection_gpu_buffers(
            current_integer_address);
        members.aggregate_projection->append(stream,addresses,projections);
    }
    stream.wait_at_fetch();

    // Complete both source bitmap D32 view validations before reporting
    // success to Sc. Reserve the following GPU command queue now so
    // post-fetch retirement cannot allocate after history publication.
    // Original PxgCudaBuffer allocations may be resized by this very
    // frame's preBpUpdate_GPU. Fetch report bitmap pointers from the
    // persistent owner NOW, never from a view cached before GPU DMA.
    const auto added_bitmap=members.aabb_owner
        ? members.aabb_owner->added_report_bitmap()
        : members.added_bitmap;
    const auto removed_bitmap=members.aabb_owner
        ? members.aabb_owner->removed_report_bitmap()
        : members.removed_bitmap;
    auto bitmap_retirement=prepare_source_sap_aabb_report_retirement_native(
        stream,members.bitmap_fill,addresses,
        added_bitmap,removed_bitmap);
    auto& pending=stream.commands();
    pending.reserve(pending.size()+bitmap_retirement.commands.size());

    // Fetch publishes the ORIGINAL CPU destination arrays and rotates
    // all seven PxgCudaBufferN<6> endpoint histories exactly once.
    // Failed source-buffer validation does not publish either report.
    const auto fetched=fetch_source_sap_results_native(
        context,addresses,
        members.dma.descriptor_gpu_buffers.broadphase.getDevicePtr(),
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
    if(members.aabb_owner) {
        // PxgAABBManager's original report-processing and fetch boundary
        // retires added/removed/changed host maps and its dirty flags.
        // Leave this state untouched on any earlier failed GPU/Sc fetch.
        members.aabb_owner->complete_fetch(members.aabb_update);
    }else {
        retire_source_sap_volume_actor_lifetime_native(
            volume_lifetime,members.aabb_update);
    }
    return {admission,fetched};
}

SourceSapSceneFrameNativeResult advance_source_sap_owned_scene_frame_native(
    PxgAabbSourceMembersNative& aabb,
    PxgSapSourceMembersNative& sap,
    SourceSapOwnedSceneFrameNativeInputs inputs,
    SourceSapDmaNative& stream,
    const SourceSapUpdateNativeStages& stages,
    const CudaDeviceAddressMap& addresses,
    compute::Context& context) {

    // The original Sc/ElementSim handles and PxgAABBManager-owned
    // VolumeData, groups, bounds and delta bitmaps define this frame.
    // No SAP-local handle allocator or second geometry/identity history.
    const auto update=aabb.prepare(
        inputs.live_boxes,inputs.live_handles,inputs.filter);
    const auto lifetime=aabb.actor_lifetime();
    // SourceSapSceneFrameNativeMembers borrows this view by reference;
    // keep the borrowed original member aliases alive through GPU fetch.
    const auto pre_update=aabb.pre_update(sap.pre_broadphase_buffers());
    const SourceSapSceneFrameNativeMembers members{
        update,
        pre_update,
        sap.dma_up_members(nullptr,inputs.mapped_reports),
        inputs.update,
        sap.fetch_histories(),
        {sap.mapped_found_actor_pairs(),sap.max_found_lost_pairs()},
        {sap.mapped_lost_actor_pairs(),sap.max_found_lost_pairs()},
        lifetime.volume_to_actor,
        inputs.found,inputs.lost,
        inputs.bitmap_fill,
        {},{}, // bitmap device aliases resolve after preBpUpdate_GPU
        &inputs.mapped_reports,
        inputs.abort_mode,
        &aabb
    };
    return advance_source_sap_scene_frame_native(
        members,stream,sap.force_update(),stages,addresses,context);
}
} // namespace fyzz::physx_gpu
