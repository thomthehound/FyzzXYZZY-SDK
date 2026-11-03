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
        // The GPU member arena uses source 32-bit byte addresses.
        // Reject impossible reservations before reserving the Sc list.
        if(max_shapes>std::numeric_limits<physx::PxU32>::max()/8u)
            throw std::out_of_range(
                "Original Sc aggregate maxShapes exceeds GPU member address range");
        const auto root=elements_.createID();
        const bool recycled=!free_root_slots_.empty();
        const auto slot=recycled
            ? free_root_slots_.back():aggregate_roots_.size();
        const auto original_count=aggregate_roots_.size();
        try {
            if(recycled) {
                auto& record=aggregate_roots_[slot];
                // Preserve the source's stable user-data address. Increase
                // attachment reservation only when this root needs more.
                record.attached_actors.reserve(max_shapes);
                record.root=root;
                record.id=PX_INVALID_U32;
                record.max_shapes=max_shapes;
                record.live=true;
                free_root_slots_.pop_back();
            }else {
                aggregate_roots_.push_back(
                    {root,PX_INVALID_U32,true,max_shapes,{},slot});
                aggregate_roots_.back().attached_actors.reserve(max_shapes);
            }
        }catch(...) {
            if(!recycled && aggregate_roots_.size()!=original_count)
                aggregate_roots_.pop_back();
            elements_.releaseID(root);
            throw;
        }
        auto& record=aggregate_roots_[slot];
        try {
            const auto id=aabb_.create_aggregate(
                root,root_group,&record,
                max_shapes,PX_INVALID_U32,filter_hint);
            record.id=id;
            return id;
        }catch(...) {
            record.live=false;
            if(recycled)free_root_slots_.push_back(slot);
            else aggregate_roots_.pop_back();
            elements_.releaseID(root);
            throw;
        }
    }

    // PxAggregate::removeActor removes the ScShapeSim from its aggregate
    // and re-admits the actor to the scene. The previous ElementSim handle
    // remains retired through GPU fetch; its old reports retain actor data.
    // The surviving actor and Sc rigid ID do not change.
    void remove_actor_from_aggregate(
        physx::Bp::AggregateHandle id,BodyId actor) {
        remove_actor_from_root(aggregate_root(id),actor);
    }

    // PxAggregate::release detaches the actors into its Sc scene rather
    // than destroying their rigid bodies. Root retirement follows the
    // original AABB member-removal and ID-recycling barriers.
    void destroy_aggregate(physx::Bp::AggregateHandle id) {
        auto& root=aggregate_root(id);
        while(!root.attached_actors.empty())
            remove_actor_from_root(root,root.attached_actors.back());
        // AABB can still publish reports referencing this stable root
        // until the original postReportsCleanup fetch boundary.
        pending_root_slots_.reserve(pending_root_slots_.size()+1u);
        aabb_.destroy_aggregate(id);
        elements_.releaseID(root.root);
        root.live=false;
        pending_root_slots_.push_back(root.slot);
    }

    void admit(BodyId actor,native::GpuBroadphaseBox::Motion motion,
               float contact_distance) {
        admit_actor(actor,actor,motion,contact_distance,std::nullopt);
    }

    // Shape elements have independent source IDs; multiple elements may
    // reference one Sc rigid actor and therefore share its Bp filter group.
    void admit_shape(ShapeId shape,BodyId actor,
                     native::GpuBroadphaseBox::Motion motion,
                     float contact_distance) {
        admit_actor(shape,actor,motion,contact_distance,std::nullopt);
    }

    // Source Sc records the ElementSim and rigid actor IDs even when
    // the shape is an infinite plane with no finite broadphase volume.
    void admit_plane(BodyId actor) {
        if(actor!=records_.size() || actor!=handles_.size())
            throw std::logic_error(
                "Source Sc plane admission must follow native actor order");
        if(handles_.size()==handles_.capacity()) {
            const auto capacity=handles_.capacity();
            handles_.reserve(capacity+std::max<std::size_t>(capacity,8));
        }
        const auto rigid_id=acquire_rigid_id(actor);
        try {
            const auto handle=elements_.createID();
            try {
                records_.push_back({
                    actor,rigid_id,native::GpuBroadphaseBox::Motion::Static,
                    0.0f,true,false});
                try {
                    handles_.push_back(handle);
                }catch(...) {
                    records_.pop_back();
                    throw;
                }
            }catch(...) {
                elements_.releaseID(handle);
                throw;
            }
        }catch(...) {
            release_rigid_id(actor);
            throw;
        }
    }

    void admit_aggregate_member(
        physx::Bp::AggregateHandle aggregate,BodyId actor,
        native::GpuBroadphaseBox::Motion motion,float contact_distance) {
        auto& root=aggregate_root(aggregate);
        if(root.attached_actors.size()>=root.max_shapes)
            throw std::invalid_argument(
                "Original Sc aggregate actor capacity exceeded");
        admit_actor(actor,actor,motion,contact_distance,aggregate);
        // Reserved at aggregate creation; source Sc attachment is
        // distinct from the AABB manager's enabled-shape member array.
        records_[actor].aggregate_attachment_index=
            static_cast<physx::PxU32>(root.attached_actors.size());
        root.attached_actors.push_back(actor);
    }

    void retire(BodyId actor) {
        auto& element=live(actor);
        element.device_cached_pose_valid=false;
        const auto handle=handles_[actor];
        if(element.in_aabb) {
            if(element.aggregate_id!=PX_INVALID_U32) {
                aabb_.remove_aggregate_member(handle);
                elements_.releaseID(handle);
            }else
                SourceScElementLifecycleNative::remove(elements_,aabb_,handle);
        }else
            elements_.releaseID(handle);
        if(element.aggregate_id!=PX_INVALID_U32)
            unlink_attached_actor(
                aggregate_root(element.aggregate_id),element);
        release_rigid_id(element.actor);
        element.live=false;
        element.in_aabb=false;
    }

    void set_simulation_enabled(BodyId actor,bool enabled) {
        auto& element=live(actor);
        if(element.in_aabb==enabled)return;
        element.device_cached_pose_valid=false;
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
        element.device_cached_pose_valid=false;
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
        element.device_cached_pose_valid=false;
        if(motion==native::GpuBroadphaseBox::Motion::Static ||
           element.motion==native::GpuBroadphaseBox::Motion::Static)
            throw std::invalid_argument(
                "Source Sc shape cannot switch between static and rigid");
        if(element.in_aabb) {
            // ScBodySim::switchTo{Kinematic,Dynamic} re-inserts the
            // original ElementSim with its new Bp group. Aggregated
            // shapes re-enter the same PxgAggregate member list, never
            // the standalone SAP endpoint list. A pending GPU member
            // retirement is canceled for this same shape identity.
            const auto handle=handles_[actor];
            const auto group=physx::Bp::getFilterGroup(
                false,element.rigid_id,
                motion==native::GpuBroadphaseBox::Motion::Kinematic);
            if(element.aggregate_id!=PX_INVALID_U32) {
                aabb_.remove_aggregate_member(handle);
                aabb_.admit_aggregate_member(
                    element.aggregate_id,handle,actor,
                    element.contact_distance,group,&element);
                aabb_.mark_aggregate_member_changed(
                    handle,element.contact_distance);
            }else {
                aabb_.remove(handle);
                aabb_.admit(handle,actor,element.contact_distance,group,&element);
                aabb_.mark_changed(handle,element.contact_distance);
            }
        }
        element.motion=motion;
    }

    void post_reports_cleanup() {
        // Sc::Scene::postReportsCleanup runs after GPU pair publication
        // and AABB-manager removal/bitmap retirement. Both original
        // ID trackers defer recycling until this barrier.
        if(!pending_root_slots_.empty())
            free_root_slots_.reserve(
                free_root_slots_.size()+pending_root_slots_.size());
        SourceScElementLifecycleNative::post_reports_cleanup(elements_);
        actor_ids_.processPendingReleases();
        actor_ids_.clearDeletedIDMap();
        free_root_slots_.insert(free_root_slots_.end(),
            pending_root_slots_.begin(),pending_root_slots_.end());
        pending_root_slots_.clear();
    }

    void set_device_shape_pose_cache_enabled(BodyId shape,bool enabled) {
        auto& element=live(shape);
        element.device_pose_supported=enabled;
        if(!enabled)element.device_cached_pose_valid=false;
    }
    bool device_cache_eligible(BodyId actor) const noexcept {
        return actor<records_.size() && records_[actor].live &&
            records_[actor].device_pose_supported &&
            records_[actor].in_aabb &&
            records_[actor].aggregate_id==PX_INVALID_U32 &&
            records_[actor].motion==native::GpuBroadphaseBox::Motion::Dynamic;
    }
    physx::Bp::ShapeHandle device_shape_cache_handle(BodyId actor) {
        if(actor>=handles_.size())
            throw std::out_of_range("Source Sc GPU body has no ElementSim");
        if(!device_cache_eligible(actor))
            return std::numeric_limits<physx::Bp::ShapeHandle>::max();
        // This solver submission will overwrite the original device
        // shape cache. A later island deactivation can restore the Sc
        // pose, so prior-frame authorization cannot survive the write.
        records_[actor].device_cached_pose_valid=false;
        return handles_[actor];
    }
    bool has_authorized_device_cached_pose(BodyId actor) const noexcept {
        return device_cache_eligible(actor) &&
            records_[actor].device_cached_pose_valid;
    }
    void authorize_device_integrated_pose(BodyId actor) {
        if(!device_cache_eligible(actor))
            throw std::logic_error(
                "Source Sc integrated pose has no standalone dynamic ElementSim");
        records_[actor].device_cached_pose_valid=true;
    }
    void invalidate_device_cached_poses() noexcept {
        for(auto& record:records_)
            record.device_cached_pose_valid=false;
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
    std::size_t aggregate_root_storage_slots() const noexcept {
        return aggregate_roots_.size();
    }

private:
    struct Element {
        BodyId actor;
        physx::PxU32 rigid_id;
        native::GpuBroadphaseBox::Motion motion;
        float contact_distance;
        bool live,in_aabb;
        physx::Bp::AggregateHandle aggregate_id=PX_INVALID_U32;
        physx::PxU32 aggregate_attachment_index=PX_INVALID_U32;
        bool device_cached_pose_valid=false;
        bool device_pose_supported=true;
    };
    struct AggregateRoot {
        physx::Bp::ShapeHandle root;
        physx::Bp::AggregateHandle id;
        bool live;
        physx::PxU32 max_shapes;
        // Sc actor attachment survives eSIMULATION_SHAPE disablement;
        // PxgAggregate::updateBoundIndices stores enabled BP members.
        std::vector<BodyId> attached_actors;
        std::size_t slot=0u;
    };
    AggregateRoot& aggregate_root(physx::Bp::AggregateHandle id) {
        for(auto& root:aggregate_roots_)
            if(root.live && root.id==id)return root;
        throw std::out_of_range("Unknown or retired source Sc aggregate");
    }
    // NpAggregate/Sc actor attachments are indexed independently of
    // PxgAggregate's enabled-bounds array. Swap-last keeps actor removal
    // and aggregate release linear in the number of attached actors.
    void unlink_attached_actor(AggregateRoot& root,Element& element) {
        const auto index=element.aggregate_attachment_index;
        if(index>=root.attached_actors.size() ||
           root.attached_actors[index]!=element.actor)
            throw std::logic_error(
                "Original Sc aggregate actor attachment index is invalid");
        const BodyId tail=root.attached_actors.back();
        root.attached_actors[index]=tail;
        records_[tail].aggregate_attachment_index=index;
        root.attached_actors.pop_back();
        element.aggregate_attachment_index=PX_INVALID_U32;
    }
    void remove_actor_from_root(AggregateRoot& root,BodyId actor) {
        auto& element=live(actor);
        element.device_cached_pose_valid=false;
        if(element.aggregate_id!=root.id)
            throw std::invalid_argument(
                "Source Sc actor does not belong to the aggregate");
        const auto old_handle=handles_[actor];
        const auto new_handle=elements_.createID();
        try {
            if(element.in_aabb) {
                const auto group=physx::Bp::getFilterGroup(
                    element.motion==native::GpuBroadphaseBox::Motion::Static,
                    element.rigid_id,
                    element.motion==native::GpuBroadphaseBox::Motion::Kinematic);
                aabb_.admit(new_handle,actor,element.contact_distance,
                            group,&element);
            }
        }catch(...) {
            elements_.releaseID(new_handle);
            throw;
        }
        if(element.in_aabb)
            aabb_.remove_aggregate_member(old_handle);
        elements_.releaseID(old_handle);
        handles_[actor]=new_handle;
        unlink_attached_actor(root,element);
        element.aggregate_id=PX_INVALID_U32;
    }
    void admit_actor(
        ShapeId shape,BodyId actor,native::GpuBroadphaseBox::Motion motion,
        float contact_distance,
        std::optional<physx::Bp::AggregateHandle> aggregate) {
        if(shape!=records_.size() || shape!=handles_.size())
            throw std::logic_error(
                "Source Sc shape admission must follow ElementSim order");
        for(const auto& element:records_)
            if(element.live && element.actor==actor &&
               element.motion!=motion)
                throw std::invalid_argument(
                    "One Sc rigid actor cannot have conflicting shape motion");
        if(handles_.size()==handles_.capacity()) {
            const auto capacity=handles_.capacity();
            handles_.reserve(capacity+std::max<std::size_t>(capacity,8));
        }
        const auto rigid_id=acquire_rigid_id(actor);
        try {
            records_.push_back({
                actor,rigid_id,motion,contact_distance,true,true,
                aggregate.value_or(PX_INVALID_U32)});
        }catch(...) {
            release_rigid_id(actor);
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
                        *aggregate,handle,shape,contact_distance,group,
                        &records_.back());
                }catch(...) {
                    elements_.releaseID(handle);
                    throw;
                }
                handles_.push_back(handle);
            }else {
                const auto handle=SourceScElementLifecycleNative::admit(
                    elements_,aabb_,shape,rigid_id,contact_distance,motion,
                    &records_.back());
                handles_.push_back(handle);
            }
        }catch(...) {
            records_.pop_back();
            release_rigid_id(actor);
            throw;
        }
    }

    struct ActorIdentity {
        physx::PxU32 rigid_id=PX_INVALID_U32;
        std::uint32_t live_shapes=0u;
    };
    physx::PxU32 acquire_rigid_id(BodyId actor) {
        if(actor>=rigid_actor_ids_.size())
            rigid_actor_ids_.resize(static_cast<std::size_t>(actor)+1u);
        auto& identity=rigid_actor_ids_[actor];
        if(identity.live_shapes==std::numeric_limits<std::uint32_t>::max())
            throw std::out_of_range("Sc rigid actor shape count overflow");
        if(!identity.live_shapes)
            identity.rigid_id=actor_ids_.createID();
        ++identity.live_shapes;
        return identity.rigid_id;
    }
    void release_rigid_id(BodyId actor) {
        auto& identity=rigid_actor_ids_[actor];
        if(!--identity.live_shapes) {
            actor_ids_.releaseID(identity.rigid_id);
            identity.rigid_id=PX_INVALID_U32;
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
    std::vector<ActorIdentity> rigid_actor_ids_;
    std::deque<AggregateRoot> aggregate_roots_;
    // Recycle root user-data addresses only after original Sc/GPU fetch.
    // These are storage slots, not a second aggregate identity registry.
    std::vector<std::size_t> pending_root_slots_,free_root_slots_;
    std::vector<physx::Bp::ShapeHandle> handles_;
};

} // namespace fyzz::physx_gpu
