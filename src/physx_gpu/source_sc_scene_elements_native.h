// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sc_element_lifecycle_native.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <stdexcept>
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

    void admit(BodyId actor,native::GpuBroadphaseBox::Motion motion,
               float contact_distance) {
        if(actor!=records_.size() || actor!=handles_.size())
            throw std::logic_error(
                "Source Sc actor admission must follow native Scene actor order");
        if(handles_.size()==handles_.capacity()) {
            const auto capacity=handles_.capacity();
            handles_.reserve(capacity+std::max<std::size_t>(capacity,8));
        }
        const auto rigid_id=actor_ids_.createID();
        try {
            records_.push_back({actor,rigid_id,motion,contact_distance,true,true});
        }catch(...) {
            actor_ids_.releaseID(rigid_id);
            throw;
        }
        try {
            const auto handle=SourceScElementLifecycleNative::admit(
                elements_,aabb_,actor,rigid_id,contact_distance,motion,
                &records_.back());
            handles_.push_back(handle);
        }catch(...) {
            records_.pop_back();
            actor_ids_.releaseID(rigid_id);
            throw;
        }
    }

    void retire(BodyId actor) {
        auto& element=live(actor);
        const auto handle=handles_[actor];
        if(element.in_aabb)
            SourceScElementLifecycleNative::remove(elements_,aabb_,handle);
        else
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
            const bool canceled_removed=
                aabb_.source_removed().boundedTest(handle)!=0;
            aabb_.admit(handle,actor,element.contact_distance,group,&element);
            // If re-admission cancels a prior remove, there is no new
            // created handle to trigger a bounds upload. The actor may
            // have moved while the shape was disabled.
            if(canceled_removed)
                aabb_.mark_changed(handle,element.contact_distance);
        }else {
            // This is a shape-flag change, NOT ElementSim destruction.
            // Do not release its Sc shape ID or actor ID.
            aabb_.remove(handles_[actor]);
        }
        element.in_aabb=enabled;
    }

    void mark_changed(BodyId actor,float contact_distance) {
        auto& element=live(actor);
        // ScShapeCore retains contact distance across eSIMULATION_SHAPE
        // disablement. Only an admitted AABB volume can have a changed
        // bitmap bit; a disabled shape remembers the distance for re-add.
        if(element.in_aabb)
            aabb_.mark_changed(handles_[actor],contact_distance);
        element.contact_distance=contact_distance;
    }

    void set_motion(BodyId actor,native::GpuBroadphaseBox::Motion motion) {
        auto& element=live(actor);
        if(element.motion==motion)return;
        if(motion==native::GpuBroadphaseBox::Motion::Static ||
           element.motion==native::GpuBroadphaseBox::Motion::Static)
            throw std::invalid_argument(
                "Source Sc shape cannot switch between static and rigid");
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

private:
    struct Element {
        BodyId actor;
        physx::PxU32 rigid_id;
        native::GpuBroadphaseBox::Motion motion;
        float contact_distance;
        bool live,in_aabb;
    };
    Element& live(BodyId actor) {
        if(actor>=records_.size() || !records_[actor].live)
            throw std::out_of_range("Unknown or retired source Sc element");
        return records_[actor];
    }

    PxgAabbSourceMembersNative& aabb_;
    physx::Sc::ObjectIDTracker elements_,actor_ids_;
    std::deque<Element> records_;
    std::vector<physx::Bp::ShapeHandle> handles_;
};

} // namespace fyzz::physx_gpu
