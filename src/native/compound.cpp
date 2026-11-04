// Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "native/scene.h"
#include "foundation/PxMathUtils.h"

namespace fyzz::native {

Transform Scene::validated_pose(Transform pose) {
    const auto p=pose.position;
    auto& q=pose.orientation;
    const float length=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
    if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
       !std::isfinite(length) || std::abs(length-1.0f)>=1.0e-2f)
        throw std::invalid_argument("Rigid transform must be finite and sane");
    q={q.x/length,q.y/length,q.z/length,q.w/length};
    return pose;
}

void Scene::refresh_shape_links(BodyId id) {
    auto& body=actors_[id];
    body.first_simulation_shape=invalid_shape;
    body.simulation_shape_count=0;
    ShapeId previous=invalid_shape;
    const auto link=[&](ShapeId shape,ShapeId next) {
        auto& value=shapes_[shape];
        if(value.next_simulation_shape==next)return;
        if(gpu_broadphase_)gpu_broadphase_->mark_shape_bounds_changed(shape,value.contact_offset);
        value.next_simulation_shape=next;
    };
    for(ShapeId i=body.first_shape;i!=invalid_shape;i=shapes_[i].next) {
        auto& shape=shapes_[i];
        if(!shape.simulation_shape_enabled || shape.geometry==Geometry::Plane || !shape.device_cache_supported)continue;
        if(previous==invalid_shape)body.first_simulation_shape=i;
        else link(previous,i);
        previous=i;
        ++body.simulation_shape_count;
    }
    if(previous!=invalid_shape)link(previous,invalid_shape);
}

void Scene::wake_shape_partners(ShapeId id) {
    const auto& shape=shapes_[id];
    for(const auto& manifold:manifolds_) {
        if(manifold.pair.a!=id && manifold.pair.b!=id)continue;
        const Pair bodies=body_pair(manifold.pair);
        const BodyId other=bodies.a==shape.body?bodies.b:bodies.a;
        auto& partner=actors_[other];
        if(partner.alive && partner.motion==Motion::Dynamic) {
            if(partner.sleeping && !partner.activating)queue_node_activation(other);
            partner.activating|=partner.sleeping;
            authored_activation_pending_|=partner.sleeping;
            partner.sleeping=false;
            partner.ready_for_sleep=false;
            partner.wake_counter=std::max(partner.wake_counter,0.4f);
            partner.sleep_linear_acc={};
            partner.sleep_angular_acc={};
        }
    }
}

void Scene::detach_shape(ShapeId id) {
    auto& shape=require_shape(id);
    auto& actor=actors_[shape.body];
    speculative_islands_.activating_nodes.reserve(actors_.size());
    accurate_islands_.activating_nodes.reserve(actors_.size());
    if(gpu_broadphase_)gpu_broadphase_->retire_shape(id);
    erase_shape_friction_history(id);
    wake_shape_partners(id);
    if(shape.previous!=invalid_shape)shapes_[shape.previous].next=shape.next;
    else actor.first_shape=shape.next;
    if(shape.next!=invalid_shape)shapes_[shape.next].previous=shape.previous;
    --actor.shape_count;
    if(rigid_node(actor.motion))--rigid_shape_count_;
    if(shape.geometry==Geometry::Plane && shape.simulation_shape_enabled)--active_plane_count_;
    shape.alive=false;
    refresh_shape_links(shape.body);
    std::erase(live_shape_ids_,id);
    broadphase_endpoint_membership_dirty_=true;
    gpu_pair_refilter_dirty_=true;
    gpu_broadphase_snapshot_current_=false;
    plane_pair_membership_dirty_=true;
    dirty_=true;
}
void Scene::set_shape_local_pose(ShapeId id,Transform pose) {
    pose=validated_pose(pose);
    auto& shape=require_shape(id);
    if(shape.geometry==Geometry::Plane)
        throw std::logic_error("Horizontal floor has a fixed local pose");
    speculative_islands_.activating_nodes.reserve(actors_.size());
    accurate_islands_.activating_nodes.reserve(actors_.size());
    if(gpu_broadphase_)gpu_broadphase_->mark_shape_bounds_changed(id,shape.contact_offset);
    erase_shape_friction_history(id);
    wake_shape_partners(id);
    for(EdgeIndex edge:active_edge_handles_)
        if(island_edges_[edge].shapes.a==id || island_edges_[edge].shapes.b==id)
            island_edges_[edge].pcm_cache={};
    shape.local_pose=pose;
    refresh_shape_pose(shape);
    shape.cached_bounds_current=false;
    gpu_broadphase_snapshot_current_=false;
    dirty_=true;
    if(actors_[shape.body].motion==Motion::Dynamic)
        set_velocity(shape.body,actors_[shape.body].velocity);
}

BodyMassProperties Scene::body_mass_properties(BodyId id) const {
    const auto& body=require(id);
    if(body.motion==Motion::Static)
        throw std::logic_error("Static body has no mass properties");
    return {body.inverse_mass,body.inverse_inertia_local,body.mass_local_pose};
}
void Scene::update_mass_properties(BodyId id,float density) {
    auto& body=require(id);
    if(body.motion==Motion::Static)
        throw std::logic_error("Static body has no mass properties");
    if(!std::isfinite(density) || density<=0.0f)
        throw std::invalid_argument("Mass density must be positive and finite");
    using namespace physx;
    // PxMassProperties: rotate each shape tensor, sum about the weighted
    // center of mass with the parallel-axis theorem, then diagonalize.
    const auto unit_mass=[](const Shape& shape) {
        if(shape.geometry==Geometry::Box)
            return shape.half_extents.x*shape.half_extents.y*shape.half_extents.z*8.0f;
        return (4.0f/3.0f)*PxPi*shape.radius*shape.radius*shape.radius;
    };
    float mass=0.0f;
    PxVec3 center(0.0f);
    for(ShapeId i=body.first_shape;i!=invalid_shape;i=shapes_[i].next) {
        const auto& shape=shapes_[i];
        if(!shape.simulation_shape_enabled)continue;
        const float m=unit_mass(shape)*density;
        mass+=m;
        center+=PxVec3(shape.local_pose.position.x,shape.local_pose.position.y,
                       shape.local_pose.position.z)*m;
    }
    if(!PxIsFinite(mass) || mass<=0.0f || !center.isFinite())
        throw std::invalid_argument("Compound mass outside supported range");
    center/=mass;
    PxMat33 tensor(PxZero);
    for(ShapeId i=body.first_shape;i!=invalid_shape;i=shapes_[i].next) {
        const auto& shape=shapes_[i];
        if(!shape.simulation_shape_enabled)continue;
        const float m=unit_mass(shape)*density;
        PxVec3 diagonal;
        if(shape.geometry==Geometry::Box) {
            const auto h=shape.half_extents;
            diagonal=PxVec3(h.y*h.y+h.z*h.z,h.x*h.x+h.z*h.z,h.x*h.x+h.y*h.y)*(m/3.0f);
        } else diagonal=PxVec3((2.0f/5.0f)*m*shape.radius*shape.radius);
        const auto q=shape.local_pose.orientation;
        const PxMat33 rotation(PxQuat(q.x,q.y,q.z,q.w));
        const PxVec3 d=PxVec3(shape.local_pose.position.x,shape.local_pose.position.y,
                               shape.local_pose.position.z)-center;
        tensor+=rotation*PxMat33::createDiagonal(diagonal)*rotation.getTranspose();
        tensor+=(PxMat33::createDiagonal(PxVec3(d.dot(d)))-PxMat33(d*d.x,d*d.y,d*d.z))*m;
    }
    if(!tensor.column0.isFinite() || !tensor.column1.isFinite() || !tensor.column2.isFinite())
        throw std::invalid_argument("Compound inertia outside supported range");
    PxQuat axes;
    const PxVec3 inertia=PxDiagonalize(tensor,axes);
    const PxVec3 inverse(1.0f/inertia.x,1.0f/inertia.y,1.0f/inertia.z);
    const float inverse_mass=1.0f/mass;
    if(!inverse.isFinite() || inverse.minElement()<=0.0f ||
       !PxIsFinite(inverse_mass) || inverse_mass<=0.0f || !axes.isFinite())
        throw std::invalid_argument("Compound inverse mass/inertia outside supported range");
    // Friction anchors are expressed in the old mass frames.
    erase_body_friction_history(id);
    body.inverse_mass=inverse_mass;
    body.inverse_inertia_local={inverse.x,inverse.y,inverse.z};
    body.mass_local_pose={{center.x,center.y,center.z},{axes.x,axes.y,axes.z,axes.w}};
    body.angular_inertia_available=true;
    body.density=density;
    dirty_=true;
}
} // namespace fyzz::native
