// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_scene_frame_native.h"
#include "physx_gpu/source_aggregate_post_sap_member_history_native.h"
#include "physx_port/gpubroadphase/PxgAabbSourceMembersNative.h"
#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
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

namespace {

// Queue source SAP and aggregate work before blocking Sc report
// publication. Their device allocations remain owned through fetch.
SourceSapUpdateAdmission queue_source_sap_scene_frame_native(
    SourceSapSceneFrameNativeMembers members,
    SourceSapDmaNative& stream,
    bool& force_update,
    const SourceSapUpdateNativeStages& stages,
    const CudaDeviceAddressMap& addresses) {

    if(members.aabb_owner)
        validate_source_sap_owned_aabb_aliases_native(
            *members.aabb_owner,members.aabb_manager,members.volume_to_actor);
    if(members.aabb_manager.has_aggregates &&
       (!members.aabb_owner || !members.aggregate_member_history ||
        !members.aggregate_projection ||
        !members.aggregate_self_collision ||
        !members.aggregate_report_copy || !members.mapped_reports))
        throw std::logic_error(
            "Original aggregate post-SAP requires AABB owner and GPU history stages");

    if(members.aabb_owner && members.aabb_owner->aggregate_pair_capacity() &&
       (!members.aggregate_pair_admission || !members.aggregate_pair_collisions ||
        !members.aggregate_pair_retirement))
        throw std::logic_error(
            "Original aggregate pair pool requires GPU admission, persistent collisions and retirement");

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
        volume_lifetime,members.aabb_update,
        members.aabb_manager.has_aggregates);

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
        const auto address=[](auto pointer) {
            return static_cast<CudaDeviceAddressMap::Address>(
                reinterpret_cast<std::uintptr_t>(pointer));
        };
        const auto current=admission.execute_gpu_stages
            ? dma.old_integer_bounds.getDevicePtr()
            : dma.new_integer_bounds.getDevicePtr();
        const auto previous=admission.execute_gpu_stages
            ? dma.new_integer_bounds.getDevicePtr()
            : dma.old_integer_bounds.getDevicePtr();
        if(address(dma.broadphase_descriptor.newIntegerBounds)!=current ||
           address(dma.broadphase_descriptor.oldIntegerBounds)!=previous)
            throw std::logic_error(
                "Original SAP integer history descriptor differs from source owner after update");
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
        const auto self=members.aabb_owner->aggregate_self_collision_gpu_buffers(
            current_integer_address,
            address(dma.broadphase_descriptor.oldIntegerBounds),
            address(dma.broadphase_descriptor.updateData_groups));
        // The source previous-size buffer still refers to the old
        // aggregate projection slot here. Persistent collisions must
        // consume it BEFORE self-collision publishes this frame's size.
        if(members.aabb_owner->aggregate_pair_capacity()) {
            const auto pair_buffers=members.aabb_owner->aggregate_pair_admission_gpu_buffers(
                members.dma.descriptor_gpu_buffers.broadphase.getDevicePtr(),
                address(dma.broadphase_descriptor.foundAggPairReport),
                dma.broadphase_descriptor.max_found_lost_agg_pairs);
            members.aggregate_pair_admission->append(stream,addresses,pair_buffers);
            const auto collision=members.aabb_owner->aggregate_persistent_collision_gpu_buffers(
                current_integer_address,
                address(dma.broadphase_descriptor.oldIntegerBounds),
                members.aabb_manager.volume_data_device.getDevicePtr(),
                address(dma.broadphase_descriptor.updateData_groups),
                members.aabb_manager.removed.device.getDevicePtr());
            members.aggregate_pair_collisions->append(stream,addresses,collision);
        }
        members.aggregate_self_collision->append(stream,addresses,self);
        if(members.aabb_owner->aggregate_pair_capacity()) {
            const auto retirement=members.aabb_owner->aggregate_pair_retirement_gpu_buffers();
            members.aggregate_pair_retirement->append(stream,addresses,retirement);
        }
        // Original aggCopyReports follows aggregate pair processing on
        // this same stream and writes directly into the source mapped heap.
        const auto copy=members.aabb_owner->aggregate_report_copy_gpu_buffers(
            *members.mapped_reports);
        members.aggregate_report_copy->append(stream,addresses,copy);
        if(!members.abort_mode)
            members.aabb_owner->queue_aggregate_fetch_snapshot(
                stream,members.dma.descriptor_gpu_buffers.broadphase.getDevicePtr());
    }
    return admission;
}

// Sc publication is a separate boundary from GPU pair production. This
// waits only after all dependent GPU work on the source stream has been
// queued; the source frame's AABB and SAP allocations remain live here.
SourceSapSceneFrameNativeResult fetch_source_sap_scene_frame_native(
    SourceSapSceneFrameNativeMembers members,
    const SourceSapUpdateAdmission& admission,
    SourceSapDmaNative& stream,
    const CudaDeviceAddressMap& addresses,
    compute::Context& context) {
    const SourceSapVolumeActorLifetimeNative volume_lifetime{
        members.volume_to_actor,members.aabb_manager.volume_data};
    stream.wait_at_fetch();
    std::optional<PxgAabbSourceMembersNative::AggregateFetchSnapshot>
        aggregate_snapshot;
    std::array<SourceSapMappedReportSpan,2> aggregate_payloads{};
    if(members.aabb_manager.has_aggregates && !members.abort_mode) {
        aggregate_snapshot=members.aabb_owner->download_aggregate_fetch_snapshot(
            context,addresses);
        aggregate_payloads=members.aabb_owner->validated_aggregate_report_spans(
            aggregate_snapshot->counts,aggregate_snapshot->pair_overflow);
    }

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
            std::span<const physx::PxgBroadPhasePair> aggregate_found;
            std::span<const physx::PxgBroadPhasePair> aggregate_lost;
            if(members.aabb_manager.has_aggregates &&
               !members.abort_mode) {
                // These original mapped arrays have been written by the
                // source aggCopyReports GPU stage on the SAME source
                // stream. Fetch both only after SAP's own count/overflow
                // checks and before committing either Sc destination.
                const auto reports=members.aabb_owner->
                    complete_aggregate_mapped_reports(aggregate_snapshot->counts);
                aggregate_found=reports.found;
                aggregate_lost=reports.lost;
                for(const auto& pair:aggregate_found) {
                    if(pair.mVolA>=pair.mVolB ||
                       pair.mVolB>=volume_lifetime.volume_data.size())
                        throw std::out_of_range(
                            "Original aggregate GPU found an invalid source volume");
                    const auto& a=volume_lifetime.volume_data[pair.mVolA];
                    const auto& b=volume_lifetime.volume_data[pair.mVolB];
                    if(!a.getUserData() || !b.getUserData())
                        throw std::invalid_argument(
                            "Original aggregate GPU found a retired Sc shape");
                }
            }
            lower_source_sap_and_aggregate_reports_to_sc(
                source_found,source_lost,aggregate_found,aggregate_lost,
                members.volume_to_actor,members.found,members.lost);
        },
        aggregate_snapshot ? &aggregate_snapshot->broadphase : nullptr,
        aggregate_snapshot
            ? std::span<const SourceSapMappedReportSpan>(aggregate_payloads)
            : std::span<const SourceSapMappedReportSpan>{});
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

} // namespace

SourceSapSceneFrameNativeResult advance_source_sap_scene_frame_native(
    SourceSapSceneFrameNativeMembers members,
    SourceSapDmaNative& stream,
    bool& force_update,
    const SourceSapUpdateNativeStages& stages,
    const CudaDeviceAddressMap& addresses,
    compute::Context& context) {
    const auto admission=queue_source_sap_scene_frame_native(
        members,stream,force_update,stages,addresses);
    return fetch_source_sap_scene_frame_native(
        members,admission,stream,addresses,context);
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
    // An unavailable aggregate shader bundle cannot mutate even the
    // original host AABB arrays. Reject before source geometry staging.
    const auto original_pre_update=aabb.pre_update(sap.pre_broadphase_buffers());
    if(original_pre_update.has_aggregates && !inputs.aggregate_stages)
        throw std::logic_error(
            "Original aggregate Sc frame requires the complete source GPU kernel bundle");
    const auto update=aabb.prepare(
        inputs.live_boxes,inputs.live_handles,inputs.filter);
    const auto lifetime=aabb.actor_lifetime();
    // SourceSapSceneFrameNativeMembers borrows this view by reference;
    // keep the borrowed original member aliases alive through GPU fetch.
    const auto source_pre_update=aabb.pre_update(sap.pre_broadphase_buffers());
    const auto pre_update=source_pre_update.has_aggregates
        ? aabb.aggregate_pre_update(
            update,sap.pre_broadphase_buffers(),
            inputs.aggregate_stages->bounds,addresses,stream)
        : source_pre_update;
    const auto* aggregate=inputs.aggregate_stages;
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
        &aabb,
        aggregate?&aggregate->member_history:nullptr,
        aggregate?&aggregate->projection:nullptr,
        aggregate?&aggregate->self_collision:nullptr,
        aggregate?&aggregate->report_copy:nullptr,
        aggregate?&aggregate->pair_admission:nullptr,
        aggregate?&aggregate->pair_collisions:nullptr,
        aggregate?&aggregate->pair_retirement:nullptr
    };
    const auto admission=queue_source_sap_scene_frame_native(
        members,stream,sap.force_update(),stages,addresses);
    return fetch_source_sap_scene_frame_native(
        members,admission,stream,addresses,context);
}
} // namespace fyzz::physx_gpu
