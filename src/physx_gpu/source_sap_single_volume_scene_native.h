// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_scene_bounds.h"
#include "physx_port/gpubroadphase/PxgAabbManagerVolumeLifecycleNative.h"

#include <algorithm>
#include <bit>
#include <span>
#include <stdexcept>

namespace fyzz::physx_gpu {

// Lower the source AABB manager's real first-pass deltas into the scene
// bounds it owns, then construct the original Bp::BroadPhaseUpdateData.
// In GPU mode mUpdatedHandles is empty: mChangedHandleMap is the source
// of dirty bounds. This is the existing manager/Sc task boundary, not
// an independent handle registry or a second broadphase.
inline physx::Bp::BroadPhaseUpdateData
prepare_source_sap_single_volume_scene_native(
    SourceSapSingleVolumeMembersNative manager,
    SourceSapSingleVolumeFirstPassNative handles,
    std::span<const native::GpuBroadphaseBox> live_boxes,
    std::span<const physx::Bp::ShapeHandle> live_handles,
    std::span<const BodyId> source_volume_actors,
    SourceSapSceneBounds arrays,
    const physx::Bp::BpFilter& filter,
    bool bounds_changed,
    SourceSapSceneBoundsScratch& scratch,
    std::vector<SourceScBoxTransformInputNative>* gpu_records = nullptr,
    SourceSapDeviceCachedPoseNative* cached_poses = nullptr,
    bool validated_sc_snapshot = false) {
    // Source volume state and preBroadPhase must reference the exact
    // same original AABB-manager host arrays (not independently cloned
    // bounds, distance or group storage).
    if(arrays.groups.data()!=manager.groups.data() ||
       arrays.groups.size()!=manager.groups.size() ||
       arrays.contact_distances.data()!=manager.contact_distances.data() ||
       arrays.contact_distances.size()!=manager.contact_distances.size() ||
       arrays.bounds.size()!=manager.groups.size())
        throw std::invalid_argument(
            "Source SAP Sc bounds do not alias the original AABB manager arrays");

    // The original Sc shape admission installs filtering based on the
    // Sc rigid actor ID, which may differ from the native report BodyId.
    // Bounds writes must neither resurrect a retired source volume nor
    // silently change its static/kinematic/dynamic filter classification.
    if(live_boxes.size()!=live_handles.size())
        throw std::invalid_argument(
            "Source Sc volumes and live geometry counts disagree");
    if(!validated_sc_snapshot)for(std::size_t i=0;i<live_handles.size();++i) {
        const auto handle=live_handles[i];
        if(handle>=source_volume_actors.size() ||
           source_volume_actors[handle]!=live_boxes[i].id)
            throw std::invalid_argument(
                "Source GPU volume handle no longer belongs to the supplied Sc actor");
        if(handle>=manager.volume_data.size() ||
           handle>=manager.groups.size() ||
           manager.groups[handle]==physx::Bp::FilterGroup::eINVALID ||
           !manager.volume_data[handle].isSingleActor())
            throw std::invalid_argument(
                "Sc box references a volume not admitted by the source AABB manager");
        const auto group=manager.groups[handle];
        const auto motion=live_boxes[i].motion;
        const bool static_group=group==physx::Bp::getFilterGroup_Statics();
        const auto type=static_cast<physx::PxU32>(group) &
                        BP_FILTERING_TYPE_MASK;
        // ScShapeSimBase::getBPGroup intentionally treats kinematics
        // with forced notifications as DYNAMIC for broadphase filtering.
        // Both encodings remain authoritative when the body is kinematic.
        const bool compatible=
            motion==native::GpuBroadphaseBox::Motion::Static
                ? static_group
                : !static_group &&
                  (type==physx::Bp::FilterType::DYNAMIC ||
                   (motion==native::GpuBroadphaseBox::Motion::Kinematic &&
                    type==physx::Bp::FilterType::KINEMATIC));
        if(!compatible)
            throw std::invalid_argument(
                "Source Sc motion does not match admitted PhysX filter group");
    }

    collect_source_sap_single_volume_first_pass_native(manager,handles);
    const SourceSapSceneFrameChanges changes{
        {handles.created.begin(),handles.created.size()},
        {}, // PxgAABBManager::updateBPSecondPass always passes zero updated handles.
        {handles.removed.begin(),handles.removed.size()},
        {manager.changed.getWords(),manager.changed.getWordCount()}};
    if(validated_sc_snapshot) {
        // Sc has already validated and retained the sorted shape/handle roster.
        // Source created/changed bits select its actual dirty geometry without
        // scanning every unchanged actor for another SAP frame.
        auto& indices=scratch.dirty_indices;
        indices.clear();
        const auto select=[&](physx::Bp::ShapeHandle handle,bool required) {
            if(handle>=source_volume_actors.size() || handle>=arrays.bounds.size())
                throw std::out_of_range("Source SAP dirty volume exceeds Sc allocation");
            const auto actor=source_volume_actors[handle];
            const auto found=std::lower_bound(
                live_boxes.begin(),live_boxes.end(),actor,
                [](const native::GpuBroadphaseBox& box,BodyId id) {
                    return box.id<id;
                });
            if(found==live_boxes.end() || found->id!=actor) {
                if(required || manager.groups[handle]!=physx::Bp::FilterGroup::eINVALID)
                    throw std::invalid_argument(
                        "Source SAP dirty volume is missing from Sc geometry");
                return;
            }
            const auto index=static_cast<std::size_t>(found-live_boxes.begin());
            if(live_handles[index]!=handle)
                throw std::invalid_argument(
                    "Source SAP dirty volume has mismatched Sc handle");
            indices.push_back(index);
        };
        for(const auto handle:changes.created)select(handle,true);
        for(const auto handle:changes.updated)select(handle,true);
        for(std::size_t word=0;word<changes.changed_bitmap_words.size();++word) {
            auto bits=changes.changed_bitmap_words[word];
            while(bits) {
                const auto bit=static_cast<std::size_t>(std::countr_zero(bits));
                select(static_cast<physx::Bp::ShapeHandle>(word*32u+bit),false);
                bits&=bits-1u;
            }
        }
        std::sort(indices.begin(),indices.end());
        indices.erase(std::unique(indices.begin(),indices.end()),indices.end());
        scratch.selected_boxes.clear();
        scratch.selected_handles.clear();
        scratch.selected_boxes.reserve(indices.size());
        scratch.selected_handles.reserve(indices.size());
        for(const auto index:indices) {
            scratch.selected_boxes.push_back(live_boxes[index]);
            scratch.selected_handles.push_back(live_handles[index]);
        }
        // Geometry validation remains all-or-nothing over the changed set.
        // Removed handles are checked against that set by the same source
        // update constructor; they cannot reenter the numerical stream.
        return prepare_source_sap_scene_update(
            scratch.selected_boxes,scratch.selected_handles,arrays,changes,
            filter,manager.persistent_state_changed || bounds_changed,
            manager.gpu_state_changed,scratch,true,gpu_records,cached_poses);
    }
    return prepare_source_sap_scene_update(
        live_boxes,live_handles,arrays,changes,filter,
        manager.persistent_state_changed || bounds_changed,
        manager.gpu_state_changed,scratch,
        true,gpu_records,cached_poses); // mGroups is owned by original Sc shape admission.
}
} // namespace fyzz::physx_gpu
