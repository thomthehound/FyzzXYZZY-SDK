// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sc_element_lifecycle_native.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <stdexcept>
#include <optional>
#include <vector>

namespace fyzz::physx_gpu {

// Original Sc actor/element ID trackers and the one ElementSim identity
// associated with each admitted native box. The source AABB manager owns
// all GPU volume data, bitmaps and actor-report identities. The actor's
// element handle is never inferred from BodyId or GPU array position.
//
// Native Scene allocates monotonically increasing BodyIds; the per-actor
// element record has a stable address for the lifetime of the source
// scene, including while a removed shape has pending GPU lost reports.
// This record is the native single-shape ElementSim user-data identity,
// not an alternate broadphase volume or pair registry.
class SourceScSceneElementsNative final {
public:
    explicit SourceScSceneElementsNative(PxgAabbSourceMembersNative& aabb)
        : aabb_(aabb) {}

    // Sc::Scene allocates the aggregate root's ElementSim ID from the
    // SAME tracker as its shapes. Its stable user-data record is retained
    // until AABB aggregate destruction and postReportsCleanup.
    physx::Bp::AggregateHandle create_aggregate(
        physx::PxU32 max_shapes,physx::Bp::FilterGroup::Enum root_group,
        physx::PxU32 filter_hint=0u) {
        if(!max_shapes ||
           root_group==physx::Bp::FilterGroup::eINVALID)
            throw std::invalid_argument("Invalid original Sc aggregate root");
        const auto root=elements_.createID();
        try {
            aggregate_roots_.push_back({root,PX_INVALID_U32,true});
        }catch(...) {
            elements_.releaseID(root);
            throw;
        }
        try {
            const auto id=aabb_.create_aggregate(
                root,root_group,&aggregate_roots_.back(),
                max_shapes,PX_INVALID_U32,filter_hint);
            aggregate_roots_.back().id=id;
            return id;
        }catch(...) {
            aggregate_roots_.pop_back();
            elements_.releaseID(root);
            throw;
        }
    }

    void destroy_aggregate(physx::Bp::AggregateHandle id) {
        auto& root=aggregate_root(id);
        // Disabling simulation removes AABB membership but does not
        // remove the Sc::ElementSim. Such members still own this root
        // and must not acquire a different aggregate on ID recycling.
        for(const auto& member:records_)
            if(member.live && member.aggregate_id==id)
                throw std::logic_error(
                    "Cannot destroy original Sc aggregate with attached members");
        aabb_.destroy_aggregate(id);
        elements_.releaseID(root.root);
        root.live=false;
    }

    void admit(BodyId actor,native::GpuBroadphaseBox::Motion motion,
               float contact_distance) {
        admit_actor(actor,motion,contact_distance,std::nullopt);
    }

    void admit_aggregate_member(
        physx::Bp::AggregateHandle aggregate,BodyId actor,
        native::GpuBroadphaseBox::Motion motion,float contact_distance) {
        (void)aggregate_root(aggregate);
        admit_actor(actor,motion,contact_distance,aggregate);
    }

    void retire(BodyId actor) {
        auto& element=live(actor);
        const auto handle=handles_[actor];
        if(element.in_aabb) {
            if(element.aggregate_id!=PX_INVALID_U32) {
                aabb_.remove_aggregate_member(handle);
                elements_.releaseID(handle);
            }else
                SourceScElementLifecycleNative::remove(elements_,aabb_,handle);
        }else
            elements_.releaseID(handle);
        actor_ids_.releaseID(element.rigid_id);
        element.live=false;
        element.in_aabb=false;
    }

    void set_simulation_enabled(BodyId actor,bool enabled) {
        auto& element=live(actor);
        if(element.in_aabb==enabled)return;
        if(enabled) {
            // Sc::ElementSim's ID survives eSIMULATION_SHAPE refiltering.
            // An add before GPU fetch cancels a pending source removal;
            // an add after fetch restores the SAME still-live shape ID.
            const bool is_static=
                element.motion==native::GpuBroadphaseBox::Motion::Static;
            const auto group=physx::Bp::getFilterGroup(
                is_static,element.rigid_id,
                element.motion==native::GpuBroadphaseBox::Motion::Kinematic);
            const auto handle=handles_[actor];
            if(element.aggregate_id!=PX_INVALID_U32)
                aabb_.admit_aggregate_member(
                    element.aggregate_id,handle,actor,
                    element.contact_distance,group,&element);
            else {
                const bool canceled_removed=
                    aabb_.source_removed().boundedTest(handle)!=0;
                aabb_.admit(handle,actor,element.contact_distance,group,&element);
                // Re-admission of a previously published single volume
                // requires bounds refresh even when its remove was canceled.
                if(canceled_removed)
                    aabb_.mark_changed(handle,element.contact_distance);
            }
        }else {
            // This is a shape-flag change, NOT ElementSim destruction.
            // The Sc element ID survives both source owner paths.
            if(element.aggregate_id!=PX_INVALID_U32)
                aabb_.remove_aggregate_member(handles_[actor]);
            else
                aabb_.remove(handles_[actor]);
        }
        element.in_aabb=enabled;
    }

    void mark_changed(BodyId actor,float contact_distance) {
        auto& element=live(actor);
        // ScShapeCore retains contact distance across eSIMULATION_SHAPE
        // disablement. Only an admitted AABB volume can have a changed
        // bitmap bit; a disabled shape remembers the distance for re-add.
        if(element.in_aabb) {
            if(element.aggregate_id!=PX_INVALID_U32)
                aabb_.mark_aggregate_member_changed(
                    handles_[actor],contact_distance);
            else
                aabb_.mark_changed(handles_[actor],contact_distance);
        }
        element.contact_distance=contact_distance;
    }

    void set_motion(BodyId actor,native::GpuBroadphaseBox::Motion motion) {
        auto& element=live(actor);
        if(element.motion==motion)return;
        if(motion==native::GpuBroadphaseBox::Motion::Static ||
           element.motion==native::GpuBroadphaseBox::Motion::Static)
            throw std::invalid_argument(
                "Source Sc shape cannot switch between static and rigid");
        if(element.in_aabb && element.aggregate_id!=PX_INVALID_U32)
            throw std::logic_error(
                "Source aggregate member BP group change requires original fetch retirement");
        if(element.in_aabb) {
            // ScBodySim::switchTo{Kinematic,Dynamic} invokes
            // ShapeSimBase::updateBPGroup: setBPGroup then
            // reinsertBroadPhase for the SAME ElementSim ID.
            // Source bitmap cancellation alone does not upload bounds,
            // so mark that volume changed after the source reinsertion.
            const auto handle=handles_[actor];
            const auto group=physx::Bp::getFilterGroup(
                false,element.rigid_id,
                motion==native::GpuBroadphaseBox::Motion::Kinematic);
            aabb_.remove(handle);
            aabb_.admit(handle,actor,element.contact_distance,group,&element);
            aabb_.mark_changed(handle,element.contact_distance);
        }
        element.motion=motion;
    }

    void post_reports_cleanup() {
        // Sc::Scene::postReportsCleanup runs after GPU pair publication
        // and AABB-manager removal/bitmap retirement. Both original
        // ID trackers defer recycling until this barrier.
        SourceScElementLifecycleNative::post_reports_cleanup(elements_);
        actor_ids_.processPendingReleases();
        actor_ids_.clearDeletedIDMap();
    }

    const std::vector<physx::Bp::ShapeHandle>& handles() const noexcept {
        return handles_;
    }
    physx::Bp::ShapeHandle handle(BodyId actor) const {
        return handles_.at(actor);
    }
    bool element_deleted(physx::Bp::ShapeHandle id) const {
        return elements_.isDeletedID(id)!=0;
    }
    physx::Bp::ShapeHandle aggregate_root_handle(
        physx::Bp::AggregateHandle id) {
        return aggregate_root(id).root;
    }

private:
    struct Element {
        BodyId actor;
        physx::PxU32 rigid_id;
        native::GpuBroadphaseBox::Motion motion;
        float contact_distance;
        bool live,in_aabb;
        physx::Bp::AggregateHandle aggregate_id=PX_INVALID_U32;
    };
    struct AggregateRoot {
        physx::Bp::ShapeHandle root;
        physx::Bp::AggregateHandle id;
        bool live;
    };
    AggregateRoot& aggregate_root(physx::Bp::AggregateHandle id) {
        for(auto& root:aggregate_roots_)
            if(root.live && root.id==id)return root;
        throw std::out_of_range("Unknown or retired source Sc aggregate");
    }
    void admit_actor(
        BodyId actor,native::GpuBroadphaseBox::Motion motion,
        float contact_distance,
        std::optional<physx::Bp::AggregateHandle> aggregate) {
        if(actor!=records_.size() || actor!=handles_.size())
            throw std::logic_error(
                "Source Sc actor admission must follow native Scene actor order");
        if(handles_.size()==handles_.capacity()) {
            const auto capacity=handles_.capacity();
            handles_.reserve(capacity+std::max<std::size_t>(capacity,8));
        }
        const auto rigid_id=actor_ids_.createID();
        try {
            records_.push_back({
                actor,rigid_id,motion,contact_distance,true,true,
                aggregate.value_or(PX_INVALID_U32)});
        }catch(...) {
            actor_ids_.releaseID(rigid_id);
            throw;
        }
        try {
            if(aggregate) {
                const auto handle=elements_.createID();
                try {
                    const auto group=physx::Bp::getFilterGroup(
                        motion==native::GpuBroadphaseBox::Motion::Static,
                        rigid_id,
                        motion==native::GpuBroadphaseBox::Motion::Kinematic);
                    aabb_.admit_aggregate_member(
                        *aggregate,handle,actor,contact_distance,group,
                        &records_.back());
                }catch(...) {
                    elements_.releaseID(handle);
                    throw;
                }
                handles_.push_back(handle);
            }else {
                const auto handle=SourceScElementLifecycleNative::admit(
                    elements_,aabb_,actor,rigid_id,contact_distance,motion,
                    &records_.back());
                handles_.push_back(handle);
            }
        }catch(...) {
            records_.pop_back();
            actor_ids_.releaseID(rigid_id);
            throw;
        }
    }
    Element& live(BodyId actor) {
        if(actor>=records_.size() || !records_[actor].live)
            throw std::out_of_range("Unknown or retired source Sc element");
        return records_[actor];
    }

    PxgAabbSourceMembersNative& aabb_;
    physx::Sc::ObjectIDTracker elements_,actor_ids_;
    std::deque<Element> records_;
    std::deque<AggregateRoot> aggregate_roots_;
    std::vector<physx::Bp::ShapeHandle> handles_;
};

} // namespace fyzz::physx_gpu
