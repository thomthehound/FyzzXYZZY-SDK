// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "native/scene.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace fyzz::native {
namespace {

// Source transcription: PhysX BpBroadPhaseIntegerAABB.h::encodeFloat,
// IntegerAABB::encodeFloatMin/Max, with its four-bit conservative expansion.
std::uint32_t encode_float(float v) {
    const auto bits = std::bit_cast<std::uint32_t>(v);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
std::uint32_t encode_min(float v) {
    return ((encode_float(v) >> 4) - 1u) << 4;
}
std::uint32_t encode_max(float v) {
    return (((encode_float(v) >> 4) + 1u) << 4) | 4u;
}
// DyBodyCoreIntegrator::integrateCore locks motion/writeback axes without
// changing inverse inertia or the response of contact constraints.
void lock_linear_axes(Vec3& v,std::uint32_t flags) noexcept {
    if(flags & LockLinearX)v.x=0.0f;
    if(flags & LockLinearY)v.y=0.0f;
    if(flags & LockLinearZ)v.z=0.0f;
}
void lock_angular_axes(Vec3& v,std::uint32_t flags) noexcept {
    if(flags & LockAngularX)v.x=0.0f;
    if(flags & LockAngularY)v.y=0.0f;
    if(flags & LockAngularZ)v.z=0.0f;
}
bool finite(Vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
bool finite(Quat q) {
    return std::isfinite(q.x) && std::isfinite(q.y) &&
           std::isfinite(q.z) && std::isfinite(q.w);
}
// Equivalent to the original PxQuat::rotate and rotateInv using an
// orthonormal body pose; inertia is transformed into world space by
// R * diag(inverseInertia) * R^T, as in Cm::transformInertiaTensor.
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y,
            a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x};
}
Vec3 rotate(Quat q, Vec3 v) {
    const Vec3 u{q.x,q.y,q.z};
    const Vec3 c = cross(u,v);
    const Vec3 t{c.x*2.0f,c.y*2.0f,c.z*2.0f};
    const Vec3 d = cross(u,t);
    return {v.x + q.w*t.x + d.x,
            v.y + q.w*t.y + d.y,
            v.z + q.w*t.z + d.z};
}
Vec3 rotate_inverse(Quat q, Vec3 v) {
    return rotate({-q.x,-q.y,-q.z,q.w},v);
}
// DyRigidBodyToSolverBody.cpp::copyToSolverBodyData gyroscopic
// velocity update. Source computes the torque in body-local inertia
// coordinates, renormalizes angular momentum and rotates the resulting
// velocity correction back to world space before applying axis locks.
Vec3 gyroscopic_velocity(Vec3 angular, Vec3 inverse_inertia, Quat pose,
                         float dt) {
    const Vec3 inertia{
        inverse_inertia.x>0.0f?1.0f/inverse_inertia.x:0.0f,
        inverse_inertia.y>0.0f?1.0f/inverse_inertia.y:0.0f,
        inverse_inertia.z>0.0f?1.0f/inverse_inertia.z:0.0f};
    const Vec3 local=rotate_inverse(pose,angular);
    const Vec3 original{
        inertia.x*local.x,inertia.y*local.y,inertia.z*local.z};
    const Vec3 torque=cross(local,original);
    Vec3 next{original.x-torque.x*dt,
              original.y-torque.y*dt,
              original.z-torque.z*dt};
    const float old_magnitude=std::sqrt(
        original.x*original.x+original.y*original.y+original.z*original.z);
    const float new_magnitude=std::sqrt(
        next.x*next.x+next.y*next.y+next.z*next.z);
    const float ratio=new_magnitude>0.0f?old_magnitude/new_magnitude:0.0f;
    next={next.x*ratio,next.y*ratio,next.z*ratio};
    const Vec3 local_delta{
        inverse_inertia.x*next.x-local.x,
        inverse_inertia.y*next.y-local.y,
        inverse_inertia.z*next.z-local.z};
    const Vec3 delta=rotate(pose,local_delta);
    return {angular.x+delta.x,angular.y+delta.y,angular.z+delta.z};
}
Quat multiply(Quat a, Quat b) {
    // Source PxQuat::operator*: (x,y,z,w) Hamilton product.
    return {
        a.w*b.x + b.w*a.x + a.y*b.z - a.z*b.y,
        a.w*b.y + b.w*a.y + a.z*b.x - a.x*b.z,
        a.w*b.z + b.w*a.z + a.x*b.y - a.y*b.x,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    };
}
Quat integrate_quaternion(Quat pose, Vec3 angular, float dt) {
    // Direct translation of DyBodyCoreIntegrator.h::integrateCore's
    // closed-form post-solver quaternion step. The input already
    // includes position-iteration angular motion from the source PGS
    // contact Jacobians when the body participated in a constraint.
    float w = angular.x*angular.x + angular.y*angular.y +
              angular.z*angular.z;
    if (w != 0.0f) {
        w = std::sqrt(w);
        constexpr float max_w = 1.0e7f;
        if (w > max_w) {
            const float scale = max_w / w;
            angular = {angular.x*scale,angular.y*scale,angular.z*scale};
            w = max_w;
        }
        const float v = dt * w * 0.5f;
        const float s = std::sin(v) / w;
        const float cosine = std::cos(v);
        const Quat dq = multiply({
            angular.x*s,angular.y*s,angular.z*s,0.0f},pose);
        const Quat next{
            dq.x + pose.x*cosine, dq.y + pose.y*cosine,
            dq.z + pose.z*cosine, dq.w + pose.w*cosine};
        const float len = std::sqrt(next.x*next.x + next.y*next.y +
                                    next.z*next.z + next.w*next.w);
        if (!(len > 0.0f) || !std::isfinite(len))
            throw std::overflow_error("Native orientation normalization failed");
        pose = {next.x/len,next.y/len,next.z/len,next.w/len};
    }
    return pose;
}
void validate_box(Vec3 h, Vec3 p) {
    if (!finite(h) || !finite(p) ||
        h.x <= 0.0f || h.y <= 0.0f || h.z <= 0.0f)
        throw std::invalid_argument("Native scene requires positive finite box geometry");
}
Pair ordered_pair(BodyId a, BodyId b) {
    return {std::min(a, b), std::max(a, b)};
}
// CmPriorityQueue.h::PriorityQueueBase uses a binary min-heap whose
// comparator considers only the hop estimate. In particular, ties select
// the left child during pop and do not bubble during push. std::push_heap
// (especially with BodyId or discovery-order tie breaks) changes the
// mVisitedNodes ordering consumed by IslandSim::splitEdges.
template<class T, class HigherPriority>
void source_hop_push(std::vector<T>& heap, T value, HigherPriority higher) {
    std::size_t index=heap.size();
    heap.push_back(value);
    while(index) {
        const std::size_t parent=(index-1)/2;
        if(!higher(value,heap[parent]))break;
        heap[index]=heap[parent];
        index=parent;
    }
    heap[index]=value;
}
template<class T, class HigherPriority>
T source_hop_pop(std::vector<T>& heap, HigherPriority higher) {
    const T result=heap.front();
    const std::size_t remaining=heap.size()-1;
    const T last=heap.back();
    heap.pop_back();
    std::size_t index=0;
    for(;;) {
        std::size_t child=index*2+1;
        if(child>=remaining)break;
        const std::size_t right=child+1;
        if(right<remaining && higher(heap[right],heap[child]))
            child=right;
        if(higher(last,heap[child]))break;
        heap[index]=heap[child];
        index=child;
    }
    if(remaining)heap[index]=last;
    return result;
}
} // namespace

BodyId Scene::add(Actor actor) {
    if (actors_.size() >= std::numeric_limits<BodyId>::max())
        throw std::overflow_error("Native scene body identifier exhausted");
    const auto id = static_cast<BodyId>(actors_.size());
    actor.alive = true;
    const bool dynamic_node=actor.kind==Kind::DynamicBox;
    actors_.push_back(actor);
    if(dynamic_node)++live_dynamic_count_;
    dirty_ = true;
    // Static nodes belong to the shared world endpoint, not an island.
    // Their insertion changes SAP/Gu edges but not dynamic connectivity.
    island_topology_dirty_ |= dynamic_node;
    return id;
}
BodyId Scene::add_static_box(Vec3 h, Vec3 p) {
    validate_box(h, p);
    Actor a;
    a.kind = Kind::StaticBox;
    a.half_extents = h;
    a.position = p;
    return add(a);
}
BodyId Scene::add_static_floor(float height) {
    if (!std::isfinite(height))
        throw std::invalid_argument("Native scene floor must be finite");
    Actor a;
    a.kind = Kind::StaticPlane;
    a.position = {0.0f, height, 0.0f};
    return add(a);
}
BodyId Scene::add_dynamic_box(Vec3 h, Vec3 p, float density) {
    validate_box(h, p);
    if (!std::isfinite(density) || density <= 0.0f)
        throw std::invalid_argument("Native scene density must be finite and positive");
    Actor a;
    a.kind = Kind::DynamicBox;
    a.half_extents = h;
    a.position = p;
    a.density = density;
    // Cache source-derived inverse mass once, with a checked double
    // intermediate so extreme but finite dimensions cannot silently
    // turn a dynamic body into an infinite-mass actor.
    const double mass = 8.0 * static_cast<double>(h.x) * h.y * h.z * density;
    const double inverse = 1.0 / mass;
    if (!std::isfinite(inverse) || inverse <= 0.0 ||
        inverse > std::numeric_limits<float>::max() ||
        static_cast<float>(inverse) == 0.0f)
        throw std::invalid_argument("Native dynamic-box mass outside supported range");
    a.inverse_mass = static_cast<float>(inverse);
    // Original PhysX PxMassProperties(PxBoxGeometry) computes a
    // density-one mass of 8*hx*hy*hz and a principal-axis inertia
    // Ixx = mass*(hy^2+hz^2)/3 (and cyclic permutations). The rigid
    // body extension then scales both mass and inertia by density.
    // Keep the source's float products, rather than replacing
    // inertial properties with a made-up size-independent response.
    // Preserve the source's evaluation sequence: PxMassProperties
    // first builds a unit-density box tensor from squared extents and
    // (unit mass * 1) / 3, then scales each tensor component by density.
    // Scaling the common factor by density *before* multiplying the
    // diagonal can change float underflow/overflow and rounding.
    const float unit_mass = (h.x * h.y * h.z) * 8.0f;
    const Vec3 extent_squared{h.x*h.x, h.y*h.y, h.z*h.z};
    const float unit_scale = (unit_mass * 1.0f) / 3.0f;
    const Vec3 unit_inertia{
        (extent_squared.y + extent_squared.z) * unit_scale,
        (extent_squared.x + extent_squared.z) * unit_scale,
        (extent_squared.x + extent_squared.y) * unit_scale};
    const Vec3 inertia{
        unit_inertia.x * density,
        unit_inertia.y * density,
        unit_inertia.z * density};
    // The translational admission boundary is the representability of
    // inverse mass above, not of the inertia tensor. Valid boxes can have
    // finite inverse mass while their source PxReal inertia underflows,
    // overflows, or yields an unrepresentable inverse inertia. Retain
    // translation-only actors without inventing an angular tensor.
    if (finite(inertia) && inertia.x > 0.0f &&
        inertia.y > 0.0f && inertia.z > 0.0f) {
        const Vec3 inverse_inertia{
            1.0f/inertia.x, 1.0f/inertia.y, 1.0f/inertia.z};
        a.angular_inertia_available =
            finite(inverse_inertia) &&
            inverse_inertia.x > 0.0f && inverse_inertia.y > 0.0f &&
            inverse_inertia.z > 0.0f;
        if (a.angular_inertia_available)
            a.inverse_inertia_local = inverse_inertia;
    }
    // Keep the pre-existing tiny-mass TRANSLATION stress case admitted.
    // It is not an angular-body admission and must not silently run
    // with zero/infinite substitute inertia in the actual source solver.
    return add(a);
}
Scene::Actor& Scene::require(BodyId id) {
    if (id >= actors_.size() || !actors_[id].alive)
        throw std::out_of_range("Unknown or retired native body ID");
    return actors_[id];
}
const Scene::Actor& Scene::require(BodyId id) const {
    if (id >= actors_.size() || !actors_[id].alive)
        throw std::out_of_range("Unknown or retired native body ID");
    return actors_[id];
}
void Scene::set_position(BodyId id, Vec3 position) {
    if (!finite(position))
        throw std::invalid_argument("Native scene position must be finite");
    set_pose(id,position,require(id).orientation);
}
void Scene::set_pose(BodyId id, Vec3 position, Quat orientation) {
    // PxRigidActor::setGlobalPose accepts PxQuat::isSane and stores
    // getNormalized(). Validate before mutating wake or cache state.
    const float magnitude=std::sqrt(
        orientation.x*orientation.x+orientation.y*orientation.y+
        orientation.z*orientation.z+orientation.w*orientation.w);
    if(!finite(position) || !finite(orientation) ||
       !std::isfinite(magnitude) || std::abs(magnitude-1.0f)>=1.0e-2f)
        throw std::invalid_argument("Native rigid pose is not sane");
    Quat normalized{
        orientation.x/magnitude,orientation.y/magnitude,
        orientation.z/magnitude,orientation.w/magnitude};
    auto& actor=require(id);
    // A translation-only edit passes the stored orientation unchanged.
    // Avoid introducing roundoff-only rotation discontinuities while
    // preserving the existing idempotent-pose/anchor-cache contract.
    if(orientation.x==actor.orientation.x &&
       orientation.y==actor.orientation.y &&
       orientation.z==actor.orientation.z &&
       orientation.w==actor.orientation.w)
        normalized=actor.orientation;
    // The admitted plane's world normal is fixed to +Y; rotated
    // planes require the still-unported general plane pose geometry.
    if(actor.kind==Kind::StaticPlane) {
        if(normalized.x!=0.0f || normalized.y!=0.0f ||
           normalized.z!=0.0f || std::abs(normalized.w)!=1.0f)
            throw std::logic_error("Native horizontal floor cannot rotate");
        // q and -q encode the same rotation. Retain the fixed plane's
        // canonical logical identity instead of treating a sign flip
        // as an authored rotation or a cache discontinuity.
        normalized={0.0f,0.0f,0.0f,1.0f};
    }
    const bool changing_pose=
        actor.position.x!=position.x ||
        actor.position.y!=position.y || actor.position.z!=position.z ||
        actor.orientation.x!=normalized.x ||
        actor.orientation.y!=normalized.y ||
        actor.orientation.z!=normalized.z ||
        actor.orientation.w!=normalized.w;
    if(changing_pose) {
        speculative_islands_.activating_nodes.reserve(actors_.size());
        accurate_islands_.activating_nodes.reserve(actors_.size());
    }
    // NpRigidDynamic::setGlobalPose wakes even on an idempotent edit.
    if(actor.kind==Kind::DynamicBox) {
        if(actor.sleeping) {
            if(!actor.activating)queue_node_activation(id);
            actor.activating=true;
            authored_activation_pending_=true;
            actor.sleep_linear_acc={};
            actor.sleep_angular_acc={};
        }
        actor.wake_counter=std::max(actor.wake_counter,0.4f);
        actor.ready_for_sleep=false;
        actor.sleeping=false;
    }
    // ScRigidSim::notifyShapesOfTransformChange calls
    // ShapeInteraction::resetManagerCachedState for every existing
    // overlap interaction. That invalidates PxsContactManager's PCM
    // geometry cache even for a small (or idempotent) authored pose.
    // Dy::integrateCore never calls this authored-pose path: ordinary
    // solver-integrated motion retains the PCM manifold and refreshes
    // its cached box/plane-local witnesses on the next Gu pass.
    if(pcm_plane_box_enabled_) {
        bool invalidated=false;
        for(EdgeIndex edge_id:active_edge_handles_) {
            auto& edge=contact_edges_[edge_id];
            if(edge.pair.a==id || edge.pair.b==id) {
                edge.pcm_plane_box={};
                invalidated=true;
            }
        }
        // A source ShapeInteraction cache reset requires a fresh
        // narrowphase publication even for an idempotent authored pose,
        // including geometry-only advance_collision(false) queries.
        if(invalidated)dirty_=true;
    }
    if(!changing_pose)return;
    // Authored transforms are discontinuities: retained strong-friction
    // anchors must not bridge either a translation or an orientation edit.
    erase_shape_friction_history(id);
    actor.position=position;
    actor.orientation=normalized;
    actor.cached_bounds_current=false;
    // Preserve the established source contact-manager wake handoff:
    // relocating or rotating either endpoint wakes old touching peers
    // before their previous Gu managers can be retired.
    for(const auto& manifold:manifolds_) {
        if(manifold.pair.a!=id && manifold.pair.b!=id)continue;
        const BodyId other=manifold.pair.a==id
            ? manifold.pair.b:manifold.pair.a;
        auto& dynamic=actors_[other];
        if(!dynamic.alive || dynamic.kind!=Kind::DynamicBox ||
           (!dynamic.sleeping && !dynamic.ready_for_sleep))continue;
        if(!dynamic.activating)queue_node_activation(other);
        dynamic.activating=true;
        authored_activation_pending_=true;
        dynamic.sleeping=false;
        dynamic.ready_for_sleep=false;
        dynamic.wake_counter=std::max(dynamic.wake_counter,0.4f);
        dynamic.sleep_linear_acc={};
        dynamic.sleep_angular_acc={};
    }
    dirty_=true;
}
void Scene::erase_shape_friction_history(BodyId id) {
    // Explicit authored shape discontinuities compact only affected
    // cached anchors. Reindex remaining source manager spans after
    // this explicit edit; the normal PGS path rotates its own index.
    if(!std::erase_if(friction_patches_, [id](const FrictionPatch& patch) {
        return patch.pair.a==id || patch.pair.b==id;
    }))return;
    for(auto& span:friction_by_edge_)span.epoch=0;
    for(std::size_t cursor=0;cursor<friction_patches_.size();) {
        const Pair pair=friction_patches_[cursor].pair;
        const std::size_t first=cursor;
        do {++cursor;}
        while(cursor<friction_patches_.size() &&
              friction_patches_[cursor].pair==pair);
        const EdgeIndex edge=contact_edge_index(pair);
        if(edge==invalid_edge || edge>=friction_by_edge_.size())
            continue;
        const std::size_t count=cursor-first;
        if(count>32)
            throw std::logic_error(
                "PhysX contact manager has more than 32 friction patches");
        friction_by_edge_[edge]={pair,first,
            static_cast<std::uint8_t>(count),friction_cache_epoch_};
    }
}
void Scene::set_velocity(BodyId id, Vec3 velocity) {
    if (!finite(velocity))
        throw std::invalid_argument("Native scene velocity must be finite");
    auto& actor = require(id);
    if (actor.kind != Kind::DynamicBox)
        throw std::logic_error("Static actors have no linear velocity");
    // Publish the source wake before changing velocity; an allocator
    // failure must not commit half of an authored state change.
    if(actor.sleeping && !actor.activating)queue_node_activation(id);
    actor.velocity = velocity;
    // NpRigidDynamic::setLinearVelocity wakes a dynamic body and resets
    // its solver sleep preparation, not the authoritative island graph.
    // NpRigidDynamic::wakeUpInternalNoKinematicTest preserves the
    // accumulated Dy sleep filter of an already active actor. Only
    // genuine sleep-to-wake transitions begin with a cleared filter.
    if(actor.sleeping) {
        actor.activating=true;
        authored_activation_pending_=true;
        actor.sleep_linear_acc={};
        actor.sleep_angular_acc={};
    }
    actor.wake_counter=std::max(actor.wake_counter,0.4f);
    actor.ready_for_sleep=false;
    actor.sleeping=false;
    // This base solver has no inter-frame constraint cache to invalidate.
    // No contact scan or broadphase rebuild is needed for velocity edits.
}
void Scene::set_angular_velocity(BodyId id, Vec3 velocity) {
    if (!finite(velocity))
        throw std::invalid_argument("Native angular velocity must be finite");
    auto& actor = require(id);
    if (actor.kind != Kind::DynamicBox)
        throw std::logic_error("Static actors have no angular velocity");
    if (!actor.angular_inertia_available &&
        (velocity.x != 0.0f || velocity.y != 0.0f || velocity.z != 0.0f))
        throw std::logic_error(
            "Angular inertia is not representable in PhysX float body state");
    if(actor.sleeping && !actor.activating)queue_node_activation(id);
    actor.angular_velocity = velocity;
    // NpRigidDynamic::wakeUpInternalNoKinematicTest preserves the
    // accumulated Dy sleep filter of an already active actor. Only
    // genuine sleep-to-wake transitions begin with a cleared filter.
    if(actor.sleeping) {
        actor.activating=true;
        authored_activation_pending_=true;
        actor.sleep_linear_acc={};
        actor.sleep_angular_acc={};
    }
    actor.wake_counter=std::max(actor.wake_counter,0.4f);
    actor.ready_for_sleep=false;
    actor.sleeping=false;
}
void Scene::apply_torque_impulse(BodyId id, Vec3 impulse) {
    add_spatial_velocity_mod(id,impulse,ForceMode::Impulse,true);
}
void Scene::add_spatial_velocity_mod(BodyId id, Vec3 value,
                                     ForceMode mode, bool angular) {
    if(!finite(value))
        throw std::invalid_argument("Native spatial force must be finite");
    auto& actor=require(id);
    if(actor.kind!=Kind::DynamicBox)
        throw std::logic_error("Static actors cannot receive spatial forces");
    if(angular && !actor.angular_inertia_available)
        throw std::logic_error(
            "Angular velocity modification requires representable inertia");
    // NpRigidBodyTemplate::addSpatialForce converts eFORCE/eIMPULSE
    // with world inverse mass/inertia. eACCELERATION and
    // eVELOCITY_CHANGE are already the respective velocity changes.
    const bool per_second=
        mode==ForceMode::Force || mode==ForceMode::Acceleration;
    if(mode!=ForceMode::Force && mode!=ForceMode::Impulse &&
       mode!=ForceMode::Acceleration && mode!=ForceMode::VelocityChange)
        throw std::invalid_argument("Unknown PhysX force mode");
    Vec3 converted=value;
    if(mode==ForceMode::Force || mode==ForceMode::Impulse) {
        if(angular) {
            const Vec3 local=rotate_inverse(actor.orientation,value);
            converted=rotate(actor.orientation,{
                local.x*actor.inverse_inertia_local.x,
                local.y*actor.inverse_inertia_local.y,
                local.z*actor.inverse_inertia_local.z});
        } else {
            converted={value.x*actor.inverse_mass,
                       value.y*actor.inverse_mass,
                       value.z*actor.inverse_mass};
        }
    }
    Vec3& destination=angular
        ? (per_second ? actor.pending_angular_acceleration
                      : actor.pending_angular_velocity_delta)
        : (per_second ? actor.pending_linear_acceleration
                      : actor.pending_linear_velocity_delta);
    const Vec3 total{destination.x+converted.x,
                     destination.y+converted.y,
                     destination.z+converted.z};
    if(!finite(converted) || !finite(total))
        throw std::overflow_error("Native spatial velocity modification overflow");
    // Source BodySim::notifyDirtySpatialAcceleration/Velocity wakes
    // the existing actor/island without replacing solver ownership.
    if(actor.sleeping && !actor.activating)queue_node_activation(id);
    destination=total;
    if(actor.sleeping) {
        actor.activating=true;
        authored_activation_pending_=true;
        actor.sleep_linear_acc={};
        actor.sleep_angular_acc={};
    }
    actor.wake_counter=std::max(actor.wake_counter,0.4f);
    actor.ready_for_sleep=false;
    actor.sleeping=false;
}
void Scene::clear_spatial_velocity_mod(BodyId id, ForceMode mode,
                                       bool angular) {
    auto& actor=require(id);
    if(actor.kind!=Kind::DynamicBox)
        throw std::logic_error("Static actors have no spatial force state");
    if(mode!=ForceMode::Force && mode!=ForceMode::Impulse &&
       mode!=ForceMode::Acceleration && mode!=ForceMode::VelocityChange)
        throw std::invalid_argument("Unknown PhysX force mode");
    const bool per_second=
        mode==ForceMode::Force || mode==ForceMode::Acceleration;
    Vec3& destination=angular
        ? (per_second ? actor.pending_angular_acceleration
                      : actor.pending_angular_velocity_delta)
        : (per_second ? actor.pending_linear_acceleration
                      : actor.pending_linear_velocity_delta);
    destination={};
}
void Scene::set_shape_offsets(BodyId id, float contact_offset,
                              float rest_offset) {
    // NpShape::setContactOffset and setRestOffset require finite values,
    // contactOffset >= 0 and restOffset < contactOffset. The two fields
    // are published together, avoiding invalid intermediate authored
    // states when lowering to the source PxShape API.
    if (!std::isfinite(contact_offset) || !std::isfinite(rest_offset) ||
        contact_offset < 0.0f || rest_offset >= contact_offset)
        throw std::invalid_argument("Native PhysX shape offsets invalid");
    auto& actor=require(id);
    if(actor.contact_offset==contact_offset &&
       actor.rest_offset==rest_offset)return;
    // An authored shape change invalidates prior friction correlation;
    // retire only this shape's anchors in the existing scene-owned cache.
    erase_shape_friction_history(id);
    // ScShapeCore shape-parameter edits invalidate the native PCM
    // witnesses for this shape's active contact managers. Refreshing
    // only relative transforms would not generate candidates newly
    // admitted by a larger contact distance. Actor edits are explicit,
    // so walk current overlap handles, never historical retired slots.
    if(pcm_plane_box_enabled_)
        for(EdgeIndex edge_id:active_edge_handles_) {
            auto& edge=contact_edges_[edge_id];
            if(edge.pair.a==id || edge.pair.b==id)
                edge.pcm_plane_box.initialized=false;
        }
    actor.contact_offset=contact_offset;
    actor.rest_offset=rest_offset;
    actor.cached_bounds_current=false;
    dirty_=true;
}

void Scene::set_material(BodyId id, Material material) {
    // NpMaterial::setRestitution and PxsMaterialData accept negative
    // restitution as source compliant spring stiffness and nonnegative damping.
    if (!std::isfinite(material.static_friction) ||
        !std::isfinite(material.dynamic_friction) ||
        !std::isfinite(material.restitution) ||
        !std::isfinite(material.damping) ||
        material.static_friction<0.0f ||
        material.dynamic_friction<0.0f ||
        material.restitution>1.0f || material.damping<0.0f ||
        static_cast<unsigned>(material.friction_combine)>3 ||
        static_cast<unsigned>(material.restitution_combine)>3 ||
        static_cast<unsigned>(material.damping_combine)>3)
        throw std::invalid_argument(
            "Native PhysX contact material coefficients/modes invalid");
    require(id).material=material;
}
void Scene::set_dynamics(BodyId id, RigidBodyDynamics d) {
    auto& actor=require(id);
    if(actor.kind!=Kind::DynamicBox)
        throw std::logic_error("Static actors have no dynamic solver state");
    // ScBodyCore::setSolverIterationCounts requires at least one
    // position iteration, but permits ZERO requested velocity iterations.
    // DySolverControl still runs its mandatory final velocity/writeback
    // pass when that count is zero. The source stores squared speed
    // limits in PxReal. Validate the complete edit before publishing it.
    if(!std::isfinite(d.linear_damping) ||
       !std::isfinite(d.angular_damping) ||
       !std::isfinite(d.max_linear_velocity) ||
       !std::isfinite(d.max_angular_velocity) ||
       !std::isfinite(d.max_depenetration_velocity) ||
       !std::isfinite(d.max_contact_impulse) ||
       !std::isfinite(d.contact_slop_coefficient) ||
       d.max_depenetration_velocity<=0 ||
       d.max_contact_impulse<0 ||
       d.contact_slop_coefficient<0 ||
       d.linear_damping<0 || d.angular_damping<0 ||
       d.max_linear_velocity<0 || d.max_angular_velocity<0 ||
       !std::isfinite(d.max_linear_velocity*d.max_linear_velocity) ||
       !std::isfinite(d.max_angular_velocity*d.max_angular_velocity) ||
       d.position_iterations==0 ||
       (d.lock_flags & ~rigid_body_lock_mask)!=0)
        throw std::invalid_argument("Native PxsBodyCore dynamics invalid");
    actor.dynamics=d;
}

void Scene::set_contact_report_threshold(BodyId id, float force) {
    // ScBodyCore::contactReportThreshold and
    // DySolverConstraints::writeBackContact's finite threshold gate.
    // PX_MAX_REAL is the source disabled default; negative, NaN and
    // infinity are not valid report-threshold values.
    if (!std::isfinite(force) || force < 0.0f)
        throw std::invalid_argument(
            "Native contact report threshold must be nonnegative and finite");
    auto& actor=require(id);
    if (actor.kind!=Kind::DynamicBox)
        throw std::logic_error(
            "Source contact report threshold belongs to dynamic rigid bodies");
    actor.contact_report_threshold=force;
}
void Scene::remove(BodyId id) {
    auto& actor=require(id);
    const bool dynamic_node=actor.kind==Kind::DynamicBox;
    // Retirement may wake multiple sleeping Gu partners. Reserve both
    // graph-local event queues before touching partner motion flags.
    if(!manifolds_.empty()) {
        speculative_islands_.activating_nodes.reserve(actors_.size());
        accurate_islands_.activating_nodes.reserve(actors_.size());
    }
    // Sc actor retirement removes physical contact-manager edges. A
    // dormant or sleep-ready rigid body resting on the retired actor
    // cannot continue sleeping without its supporting contact: revoke
    // prior-frame readiness before third-pass deactivation and reenter
    // gravity/Gu contact generation on the next native step.
    // Traverse the published Gu pairs only on this explicit edit path.
    for(const auto& manifold:manifolds_) {
        if(manifold.pair.a!=id && manifold.pair.b!=id)continue;
        const BodyId other=manifold.pair.a==id
            ? manifold.pair.b:manifold.pair.a;
        auto& partner=actors_[other];
        if(!partner.alive || partner.kind!=Kind::DynamicBox ||
           (!partner.sleeping && !partner.ready_for_sleep))continue;
        if(!partner.activating)queue_node_activation(other);
        partner.activating=true;
        authored_activation_pending_=true;
        partner.sleeping=false;
        partner.ready_for_sleep=false;
        partner.wake_counter=std::max(partner.wake_counter,0.4f);
        partner.sleep_linear_acc={};
        partner.sleep_angular_acc={};
    }
    actor.alive = false;
    if(dynamic_node)--live_dynamic_count_;
    dirty_ = true;
    island_topology_dirty_ |= dynamic_node;
}
BodyState Scene::body_state(BodyId id) const {
    const auto& actor = require(id);
    return {actor.position, actor.velocity,
            actor.orientation, actor.angular_velocity, actor.sleeping};
}
void Scene::update_actor_bounds(Actor& actor) {
    // GuBounds::computeBounds(eBOX) followed by Sc cached-shape
    // publication. Construct every encoded endpoint before committing.
        const Vec3 p = actor.position, h = actor.half_extents;
        // Source: GuBounds.cpp::computeBounds(eBOX), basisExtentV.
        // The world AABB of a rotated box is the sum of absolute
        // world-space basis columns weighted by the half extents,
        // plus its shape contact offset. The identity fast path
        // avoids three quaternion rotations for common unrotated boxes.
        Vec3 extent = h;
        const Quat q = actor.orientation;
        if (q.x != 0.0f || q.y != 0.0f ||
            q.z != 0.0f || q.w != 1.0f) {
            const Vec3 x = rotate(q, {1.0f,0.0f,0.0f});
            const Vec3 y = rotate(q, {0.0f,1.0f,0.0f});
            const Vec3 z = rotate(q, {0.0f,0.0f,1.0f});
            extent = {
                std::abs(x.x)*h.x + std::abs(y.x)*h.y + std::abs(z.x)*h.z,
                std::abs(x.y)*h.x + std::abs(y.y)*h.y + std::abs(z.y)*h.z,
                std::abs(x.z)*h.x + std::abs(y.z)*h.y + std::abs(z.z)*h.z
            };
        }
        extent.x += actor.contact_offset;
        extent.y += actor.contact_offset;
        extent.z += actor.contact_offset;
        const std::array<float, 6> e{
            p.x-extent.x, p.y-extent.y, p.z-extent.z,
            p.x+extent.x, p.y+extent.y, p.z+extent.z};
        for (float v : e)
            if (!std::isfinite(v))
                throw std::overflow_error("Native scene AABB is not finite");
        std::array<std::uint32_t,6> encoded{};
        for(unsigned axis=0;axis<3;++axis) {
            encoded[axis]=encode_min(e[axis]);
            encoded[axis+3]=encode_max(e[axis+3]);
        }
        actor.encoded_bounds=encoded;
        actor.cached_bounds_current=true;
}
void Scene::update_bounds() {
    // Previous ScPipeline::afterIntegration already refreshed each
    // integrated rigid body's bounds. Authored shape edits invalidate
    // only their own cache entry, and first-time actors are uncached.
    endpoints_.clear();
    for(std::size_t index=0;index<actors_.size();++index) {
        Actor& actor=actors_[index];
        if(!actor.alive || actor.kind==Kind::StaticPlane)continue;
        if(!actor.cached_bounds_current)update_actor_bounds(actor);
        endpoints_.push_back({actor.encoded_bounds[0],
                              static_cast<BodyId>(index)});
    }
    std::sort(endpoints_.begin(),endpoints_.end(),
        [](Endpoint a,Endpoint b) {
            return a.min_x<b.min_x ||
                (a.min_x==b.min_x && a.id<b.id);
        });
}
void Scene::after_integration_cached_bounds() {
    // ScPipeline::afterIntegration::updateScBodyAndShapeSim refreshes
    // shape bounds from integrated poses. Consume the SAME task-owned
    // accurate island ranges that Dy integrated, without a scene-wide
    // actor scan, a duplicate AABB pass, or another shape registry.
    // Sleep-deactivated islands never entered those task ranges.
    for(const auto& task:solver_tasks_)
        for(std::size_t batch=task.first_batch;batch<task.batch_end;++batch) {
            const auto island=active_accurate_island_batches_[batch];
            for(std::size_t index=accurate_island_offsets_[island];
                index<accurate_island_offsets_[island+1];++index) {
                Actor& body=actors_[accurate_island_nodes_[index]];
                if(!body.sleeping)
                    update_actor_bounds(body);
            }
        }
}
void Scene::update_pairs() {
    pending_pairs_.clear();
    active_.clear();
    // Source-corresponding SAP: x-axis sweep and integer y/z overlap.
    // Retain ordered pair history across frames; exclude static/static.
    for (Endpoint ep : endpoints_) {
        const auto& actor = actors_[ep.id];
        const auto& a = actor.encoded_bounds;
        active_.erase(std::remove_if(active_.begin(), active_.end(),
            [&](BodyId id) { return actors_[id].encoded_bounds[3] < ep.min_x; }),
            active_.end());
        for (BodyId other_id : active_) {
            const auto& other = actors_[other_id];
            if (actor.kind != Kind::DynamicBox && other.kind != Kind::DynamicBox)
                continue;
            const auto& b = other.encoded_bounds;
            if (a[1] > b[4] || b[1] > a[4] ||
                a[2] > b[5] || b[2] > a[5]) continue;
            pending_pairs_.push_back(ordered_pair(ep.id, other_id));
        }
        active_.push_back(ep.id);
    }
    // The infinite plane is only a *candidate* broadphase shape, not a
    // resolved contact. Native narrowphase must reject noncontact candidates.
    for (std::size_t i = 0; i < actors_.size(); ++i) {
        if (!actors_[i].alive || actors_[i].kind != Kind::StaticPlane) continue;
        for (std::size_t j = 0; j < actors_.size(); ++j) {
            if (actors_[j].alive && actors_[j].kind == Kind::DynamicBox)
                pending_pairs_.push_back(ordered_pair(
                    static_cast<BodyId>(i), static_cast<BodyId>(j)));
        }
    }
    std::sort(pending_pairs_.begin(), pending_pairs_.end());
    pending_pairs_.erase(std::unique(pending_pairs_.begin(), pending_pairs_.end()),
                         pending_pairs_.end());
}
std::uint32_t Scene::contact_edge_index(Pair pair) const noexcept {
    const auto it=std::lower_bound(previous_pairs_.begin(),
                                   previous_pairs_.end(),pair);
    if(it==previous_pairs_.end() || !(*it==pair))return invalid_edge;
    return active_edge_handles_[static_cast<std::size_t>(
        it-previous_pairs_.begin())];
}
bool Scene::contact_edge_connected(Pair pair) const noexcept {
    const auto handle=contact_edge_index(pair);
    return handle!=invalid_edge && contact_edges_[handle].connected;
}
bool Scene::speculative_contact_edge_active(Pair pair) const noexcept {
    const auto id=contact_edge_index(pair);
    return id!=invalid_edge && (contact_edges_[id].speculative_state&1u);
}
bool Scene::accurate_contact_edge_active(Pair pair) const noexcept {
    const auto id=contact_edge_index(pair);
    return id!=invalid_edge && (contact_edges_[id].accurate_state&1u);
}
bool Scene::contact_interaction_active(Pair pair) const noexcept {
    const auto id=contact_edge_index(pair);
    return id!=invalid_edge && contact_edges_[id].interaction_active;
}
std::uint32_t Scene::allocate_contact_edge(Pair pair) {
    // SimpleIslandManager::addContactManager -> addEdge obtains the
    // edge handle once; accurate touch may later toggle independently.
    EdgeIndex id;
    if(!free_edge_handles_.empty()) {
        id=free_edge_handles_.back();
        step_checkpoint_.save_edge(*this,id);
        free_edge_handles_.pop_back();
        contact_edges_[id]={pair,true,false,false};
    } else {
        if(contact_edges_.size()>=invalid_edge)
            throw std::overflow_error("Native island edge handles exhausted");
        id=static_cast<EdgeIndex>(contact_edges_.size());
        // deactivateNodeInternal may publish a contact edge after the
        // geometry transaction commits. Reserve its bounded source
        // output before enlarging the authoritative handle arena.
        // As with the source's reusable event arrays, grow capacity
        // geometrically. Reserving the exact new watermark for every
        // edge would reallocate on every manager admission.
        const std::size_t needed=contact_edges_.size()+1;
        const auto reserve_events=[&](IslandGraph& graph) {
            auto& events=graph.deactivating_edges;
            if(events.capacity()>=needed)return;
            const std::size_t capacity=events.capacity();
            const std::size_t growth=capacity+
                std::max<std::size_t>(capacity,8);
            events.reserve(std::max(needed,growth));
        };
        reserve_events(speculative_islands_);
        reserve_events(accurate_islands_);
        const auto reserve_activation=[&](IslandGraph& graph) {
            auto& events=graph.activated_edges;
            if(events.capacity()>=needed)return;
            const std::size_t capacity=events.capacity();
            events.reserve(std::max(needed,capacity+
                std::max<std::size_t>(capacity,8)));
        };
        reserve_activation(speculative_islands_);
        reserve_activation(accurate_islands_);
        step_checkpoint_.activity_undo.reserve(needed);
        activity_undo_generations_.resize(needed,0);
        contact_edges_.push_back({pair,true,false,false});
    }
    return id;
}
// PxsIslandSim.cpp::IslandSim::addConnectionToGraph and
// removeConnectionInternal: two intrusive endpoint instances per
// EdgeIndex. The speculative and accurate IslandSim graphs retain
// separate lists, but share the authoritative contact-manager handle.
void Scene::link_contact_edge(EdgeIndex id, bool accurate) {
    auto& edge=contact_edges_[id];
    auto& heads=accurate?accurate_node_first_:speculative_node_first_;
    auto instance_at=[&](std::size_t index)->EdgeInstance& {
        auto& owner=contact_edges_[index/2];
        return (accurate?owner.accurate:owner.speculative)[index&1];
    };
    for(std::size_t side=0;side<2;++side) {
        const BodyId node=side==0?edge.pair.a:edge.pair.b;
        // The PhysX static world endpoint has no dynamic island node.
        if(actors_[node].kind!=Kind::DynamicBox)continue;
        auto& link=(accurate?edge.accurate:edge.speculative)[side];
        if(link.linked)continue;
        const std::size_t index=static_cast<std::size_t>(id)*2+side;
        const auto first=heads[node];
        // Record all overwritten source objects *before* any mutation.
        // Recording can allocate and throw; replay itself cannot.
        step_checkpoint_.save_edge(*this,id);
        if(first!=invalid_instance)
            step_checkpoint_.save_edge(*this,static_cast<EdgeIndex>(first/2));
        step_checkpoint_.save_head(*this,node,accurate);
        link.prev=invalid_instance;
        link.next=first;
        link.linked=true;
        if(first!=invalid_instance)
            instance_at(first).prev=index;
        heads[node]=index;
    }
}
// PxsIslandSim.cpp::addEdgeToIsland appends an indexed contact-manager
// edge to its accurate island. Links stay in the existing EdgeIndex slots;
// source island handles own only the ordered first/last indices.
void Scene::append_accurate_island_edge(EdgeIndex id, std::uint32_t owner) {
    auto& island=accurate_islands_.islands[owner];
    if(!island.live || !contact_edges_[id].connected ||
       island.first_contact==id ||
       contact_edges_[id].island_prev!=invalid_edge ||
       contact_edges_[id].island_next!=invalid_edge)
        throw std::logic_error("Invalid PhysX accurate island edge insertion");
    step_checkpoint_.save_graph_ownership(*this);
    step_checkpoint_.save_edge(*this,id);
    if(island.last_contact!=invalid_edge) {
        step_checkpoint_.save_edge(*this,island.last_contact);
        contact_edges_[island.last_contact].island_next=id;
    } else {
        island.first_contact=id;
    }
    contact_edges_[id].island_prev=island.last_contact;
    contact_edges_[id].island_next=invalid_edge;
    island.last_contact=id;
    ++island.contact_count;
}
// PxsIslandSim.cpp::removeEdgeFromIsland with
// IG_STORE_ISLAND_EDGES_IN_ARRAYS=1 replaces the removed element with the
// final array element. Rewire the final EdgeIndex into that slot in O(1).
// Retaining stable order here would be *different* from PhysX.
void Scene::remove_accurate_island_edge(EdgeIndex id) {
    const Pair pair=contact_edges_[id].pair;
    const BodyId node=actors_[pair.a].kind==Kind::DynamicBox
        ? pair.a:pair.b;
    const auto owner=accurate_islands_.node_island[node];
    if(owner==IslandGraph::invalid) return;
    auto& island=accurate_islands_.islands[owner];
    if(island.first_contact==invalid_edge || !island.contact_count)
        throw std::logic_error("PhysX accurate island edge list underflow");
    step_checkpoint_.save_graph_ownership(*this);
    const EdgeIndex last=island.last_contact;
    const EdgeIndex before=contact_edges_[id].island_prev;
    const EdgeIndex after=contact_edges_[id].island_next;
    step_checkpoint_.save_edge(*this,id);
    if(last!=id) {
        const EdgeIndex last_before=contact_edges_[last].island_prev;
        step_checkpoint_.save_edge(*this,last);
        if(last_before!=invalid_edge) {
            step_checkpoint_.save_edge(*this,last_before);
            contact_edges_[last_before].island_next=invalid_edge;
        }
        island.last_contact=last_before;
        const EdgeIndex successor=after==last?invalid_edge:after;
        if(before!=invalid_edge) {
            step_checkpoint_.save_edge(*this,before);
            contact_edges_[before].island_next=last;
        } else island.first_contact=last;
        if(successor!=invalid_edge) {
            step_checkpoint_.save_edge(*this,successor);
            contact_edges_[successor].island_prev=last;
        } else island.last_contact=last;
        contact_edges_[last].island_prev=before;
        contact_edges_[last].island_next=successor;
    } else {
        if(before!=invalid_edge) {
            step_checkpoint_.save_edge(*this,before);
            contact_edges_[before].island_next=invalid_edge;
        } else island.first_contact=invalid_edge;
        island.last_contact=before;
    }
    contact_edges_[id].island_prev=invalid_edge;
    contact_edges_[id].island_next=invalid_edge;
    --island.contact_count;
}
void Scene::unlink_contact_edge(EdgeIndex id, bool accurate) {
    // A removed speculative bridge can isolate an already sleeping node
    // when its awake neighbor is being retired. The full graph rebuild
    // runs in the first pass, so record the affected surviving node
    // directly from this source edge before it is disconnected.
    // Third-pass deactivation validates the rebuilt island; no island
    // inventory reconstruction or unrelated island scan is needed.
    const auto& retiring=contact_edges_[id];
    if(!accurate && (retiring.speculative_state&1u)) {
        for(BodyId candidate:{retiring.pair.a,retiring.pair.b}) {
            const BodyId other=(candidate==retiring.pair.a)
                ? retiring.pair.b:retiring.pair.a;
            if(actors_[candidate].alive &&
               actors_[candidate].kind==Kind::DynamicBox &&
               actors_[candidate].sleeping &&
               actors_[other].kind==Kind::DynamicBox &&
               !actors_[other].alive)
                speculative_split_sleep_candidates_.push_back(candidate);
        }
    }
    // Gu touch loss and manager retirement cancel activity without
    // fabricating an IslandSim sleep-deactivation event. Source Sc's
    // overlap removal destroys the interaction rather than publishing
    // an IslandSim sleep event. Retire that flag on the same EdgeIndex.
    if(!accurate && contact_edges_[id].interaction_active) {
        step_checkpoint_.save_activity(*this,id);
        contact_edges_[id].interaction_active=false;
        --active_contact_interaction_count_;
    }
    deactivate_contact_edge(id,accurate,false);
    // Contact manager retirement and touch loss both remove the same
    // accurate source edge before disconnecting its intrusive endpoints.
    if(accurate && contact_edges_[id].connected)
        remove_accurate_island_edge(id);
    auto& edge=contact_edges_[id];
    auto& heads=accurate?accurate_node_first_:speculative_node_first_;
    auto instance_at=[&](std::size_t index)->EdgeInstance& {
        auto& owner=contact_edges_[index/2];
        return (accurate?owner.accurate:owner.speculative)[index&1];
    };
    for(std::size_t side=0;side<2;++side) {
        const BodyId node=side==0?edge.pair.a:edge.pair.b;
        auto& link=(accurate?edge.accurate:edge.speculative)[side];
        if(!link.linked)continue;
        // IslandSim::removeConnectionFromGraph marks both live rigid
        // endpoints dirty and clears a fast route that used the removed
        // edge. Record the pending route work before changing adjacency.
        const BodyId other=side==0?edge.pair.b:edge.pair.a;
        // Static contacts cannot split an all-awake dynamic island.
        // PhysX still tracks static touch counts for sleeping; that
        // lifecycle is outside this admitted contact-only subset.
        if(actors_[node].alive && actors_[other].alive &&
           actors_[node].kind==Kind::DynamicBox &&
           actors_[other].kind==Kind::DynamicBox) {
            auto& dirty=accurate?accurate_dirty_nodes_:speculative_dirty_nodes_;
            auto& dirty_map=accurate?accurate_dirty_map_:speculative_dirty_map_;
            // IslandSim::mDirtyMap is a set: each endpoint enters the
            // pending source worklist at most once per graph/pass.
            dirty_map.resize(actors_.size(),0);
            if(!dirty_map[node]) {
                dirty.push_back(node);
                dirty_map[node]=1;
            }
            auto& fast=accurate?accurate_fast_route_:speculative_fast_route_;
            if(node<fast.size() && fast[node]==other) {
                step_checkpoint_.save_hint(*this,node,accurate);
                fast[node]=std::numeric_limits<BodyId>::max();
            }
        }
        step_checkpoint_.save_edge(*this,id);
        if(link.prev==invalid_instance)
            step_checkpoint_.save_head(*this,node,accurate);
        else
            step_checkpoint_.save_edge(*this,
                static_cast<EdgeIndex>(link.prev/2));
        if(link.next!=invalid_instance)
            step_checkpoint_.save_edge(*this,
                static_cast<EdgeIndex>(link.next/2));
        if(link.prev==invalid_instance)
            heads[node]=link.next;
        else
            instance_at(link.prev).next=link.next;
        if(link.next!=invalid_instance)
            instance_at(link.next).prev=link.prev;
        link={};
    }
}
void Scene::retire_destroyed_contact_edges() {
    // Source PostThirdPassTask frees mDestroyedEdges *after* speculative
    // and accurate lost-edge passes have joined. A removed handle cannot
    // be reused by an overlap born earlier in that same publication.
    // Reserve before changing any slots so allocation failure cannot
    // leave a partially retired handle pool in collision-only queries.
    if(retired_edge_handles_.empty())return;
    step_checkpoint_.save_handle_lists(*this);
    free_edge_handles_.reserve(
        free_edge_handles_.size()+retired_edge_handles_.size());
    for(EdgeIndex id:retired_edge_handles_) {
        step_checkpoint_.save_edge(*this,id);
        auto& edge=contact_edges_[id];
        edge={};
        free_edge_handles_.push_back(id);
    }
    retired_edge_handles_.clear();
}
void Scene::publish_pairs() {
    // Source SAP's sorted prior/new overlap streams drive one linear
    // merge. Edges and found/lost events are published together; do not
    // perform binary searches per pair or construct a second graph map.
    next_found_.clear();
    next_lost_.clear();
    next_edge_handles_.clear();
    next_edge_handles_.reserve(pending_pairs_.size());
    // Grow before linking any edge. List insertion/removal then has no
    // allocation, including in the source edge-retirement continuation.
    speculative_node_first_.resize(actors_.size(),invalid_instance);
    accurate_node_first_.resize(actors_.size(),invalid_instance);
    std::size_t old_index=0,new_index=0;
    while(old_index<previous_pairs_.size() ||
          new_index<pending_pairs_.size()) {
        if(new_index==pending_pairs_.size() ||
           (old_index<previous_pairs_.size() &&
            previous_pairs_[old_index]<pending_pairs_[new_index])) {
            step_checkpoint_.save_handle_lists(*this);
            const EdgeIndex id=active_edge_handles_[old_index];
            unlink_contact_edge(id,true);
            unlink_contact_edge(id,false);
            step_checkpoint_.save_edge(*this,id);
            contact_edges_[id].retiring=true;
            contact_edges_[id].connected=false;
            retired_edge_handles_.push_back(id);
            next_lost_.push_back(previous_pairs_[old_index++]);
        } else if(old_index==previous_pairs_.size() ||
                  pending_pairs_[new_index]<previous_pairs_[old_index]) {
            step_checkpoint_.save_handle_lists(*this);
            const Pair pair=pending_pairs_[new_index++];
            const EdgeIndex id=allocate_contact_edge(pair);
            // ScNPhaseCore::shouldSwapBodies is the source authority
            // for the contact manager's persistent oriented shape pair:
            // static second; two nonkinematic dynamics in descending
            // ActorID order (the admitted BodyIds track creation order).
            // Never use broadphase endpoint order or canonical pair
            // sorting to select Gu's reference/incident box face.
            const bool a_dynamic=
                actors_[pair.a].kind==Kind::DynamicBox;
            const bool b_dynamic=
                actors_[pair.b].kind==Kind::DynamicBox;
            contact_edges_[id].gu_order_reversed =
                (a_dynamic && b_dynamic) || (!a_dynamic && b_dynamic);
            link_contact_edge(id,false);
            next_edge_handles_.push_back(id);
            next_found_.push_back(pair);
        } else {
            next_edge_handles_.push_back(active_edge_handles_[old_index]);
            ++old_index;
            ++new_index;
        }
    }
    // Contact managers removed above remain allocated until the
    // post-third-pass join. Even an overlap discovered in this same
    // publication cannot recycle its retired handle prematurely.
    found_.swap(next_found_);
    lost_.swap(next_lost_);
    previous_pairs_.swap(pending_pairs_);
    active_edge_handles_.swap(next_edge_handles_);
    ++pair_epoch_;
}
std::uint8_t Scene::generate_pcm_plane_box(
    const Actor& plane, const Actor& box, float contact_distance,
    PcmPlaneBoxCache& cache, std::span<ContactPoint> output) {
    // GuPCMContactPlaneBox.cpp::pcmContactPlaneBox and
    // GuPersistentContactManifold.h::refreshContactPoints. The SDK
    // floor's +Y normal is original PhysX plane-local +X. The cache
    // stores the source box-local point A and projected plane-local
    // point B, not legacy Pxc midpoint-reconstructed contacts.
    // Gu::CalculatePCMBoxMargin uses BOX_MARGIN_RATIO=0.15 and
    // PxTolerancesScale.length=1; the plane breaking ratio is 0.2.
    const Vec3 relative{
        box.position.x-plane.position.x,
        box.position.y-plane.position.y,
        box.position.z-plane.position.z};
    const float margin=std::min(
        0.15f*std::min({box.half_extents.x,
                        box.half_extents.y,box.half_extents.z}),0.15f);
    const float project_break=margin*0.2f;
    const float project_break_sq=project_break*project_break;
    const auto sq=[](Vec3 v) {
        return v.x*v.x+v.y*v.y+v.z*v.z;
    };
    const auto diff=[](Vec3 a,Vec3 b) {
        return Vec3{a.x-b.x,a.y-b.y,a.z-b.z};
    };
    const auto dot=[](Vec3 a,Vec3 b) {
        return a.x*b.x+a.y*b.y+a.z*b.z;
    };
    // Source refresh visits backwards and replaces a removed cached
    // point with the last element. It removes tangentially displaced
    // anchors even when they remain within the speculative distance.
    const auto previous_count=cache.count;
    for(std::size_t i=cache.count;i>0;--i) {
        auto& point=cache.points[i-1];
        const Vec3 rotation=rotate(box.orientation,point.box_local);
        const Vec3 box_in_plane{
            relative.x+rotation.x,relative.y+rotation.y,
            relative.z+rotation.z};
        // Box-local normal A->B is the plane +Y axis in this
        // specialization; mLocalPointB sits on the plane (y=0).
        const Vec3 v=diff(box_in_plane,point.plane_local);
        const float distance=v.y;
        const Vec3 projected{box_in_plane.x,
                             box_in_plane.y-distance,box_in_plane.z};
        const Vec3 tangent=diff(point.plane_local,projected);
        if(sq(tangent)>project_break_sq)
            cache.points[i-1]=cache.points[--cache.count];
        else
            point.separation=distance;
    }
    const float max_delta=cache.initialized
        ? std::max({std::abs(relative.x-cache.relative_position.x),
                    std::abs(relative.y-cache.relative_position.y),
                    std::abs(relative.z-cache.relative_position.z)})
        : std::numeric_limits<float>::infinity();
    const Quat& q=box.orientation;
    const Quat& old=cache.relative_rotation;
    const float rotation_dot=q.x*old.x+q.y*old.y+q.z*old.z+q.w*old.w;
    const bool regenerate=!cache.initialized ||
        cache.count!=previous_count ||
        max_delta>project_break || rotation_dot<0.99996f;
    if(regenerate) {
        // Source plane-box generation emits corners in this precise
        // sign order (x,y,z), (x,y,-z), (x,-y,z), ..., (-x,-y,-z).
        // Unlike legacy Gu, the stored second contact position is
        // projected onto the plane, and each source manifold is reduced
        // to its four persistent witnesses before Dy sees it.
        cache.relative_position=relative;
        cache.relative_rotation=q;
        cache.initialized=true;
        cache.count=0;
        std::array<PcmPlaneContact,8> candidates{};
        std::size_t count=0;
        for(unsigned corner=0;corner<8;++corner) {
            const Vec3 local{
                (corner&4)?-box.half_extents.x:box.half_extents.x,
                (corner&2)?-box.half_extents.y:box.half_extents.y,
                (corner&1)?-box.half_extents.z:box.half_extents.z};
            const Vec3 r=rotate(q,local);
            const Vec3 in_plane{relative.x+r.x,relative.y+r.y,
                                relative.z+r.z};
            if(in_plane.y>=contact_distance)continue;
            candidates[count++]={local,
                {in_plane.x,0.0f,in_plane.z},in_plane.y};
        }
        if(count<=4) {
            for(std::size_t i=0;i<count;++i)
                cache.points[i]=candidates[i];
            cache.count=static_cast<std::uint8_t>(count);
        } else {
            // GuPersistentContactManifold.cpp::
            // reduceBatchContactsCluster (four-contact manifold). Keep
            // source deepest/farthest/segment-extrema ordering and then
            // cluster residual corners around those four witnesses.
            std::array<bool,8> chosen{};
            std::array<std::size_t,4> indices{};
            std::size_t first=0;
            for(std::size_t i=1;i<count;++i)
                if(candidates[i].separation<candidates[first].separation)
                    first=i;
            indices[0]=first;
            chosen[first]=true;
            std::size_t farthest=0;
            float max_distance=sq(diff(candidates[0].plane_local,
                                       candidates[first].plane_local));
            for(std::size_t i=1;i<count;++i) {
                const float distance=sq(diff(candidates[i].plane_local,
                                               candidates[first].plane_local));
                if(distance>max_distance) {
                    max_distance=distance;
                    farthest=i;
                }
            }
            indices[1]=farthest;
            chosen[farthest]=true;
            const Vec3 segment=diff(candidates[farthest].plane_local,
                                    candidates[first].plane_local);
            Vec3 sideways{-segment.z,0.0f,segment.x};
            const float norm_sq=sq(sideways);
            if(norm_sq>0.0f) {
                const float inv=1.0f/std::sqrt(norm_sq);
                sideways={sideways.x*inv,0.0f,sideways.z*inv};
            } else sideways={0.0f,1.0f,0.0f};
            float largest=-std::numeric_limits<float>::max();
            float smallest=std::numeric_limits<float>::max();
            std::size_t third=0,fourth=0;
            for(std::size_t i=0;i<count;++i) {
                if(chosen[i])continue;
                const float d=dot(diff(candidates[i].plane_local,
                                         candidates[first].plane_local),sideways);
                if(d>largest){largest=d;third=i;}
                if(d<smallest){smallest=d;fourth=i;}
            }
            indices[2]=third;
            chosen[third]=true;
            if(smallest*largest>0.0f) {
                largest=-std::numeric_limits<float>::max();
                for(std::size_t i=0;i<count;++i) {
                    if(chosen[i])continue;
                    const float d=dot(diff(candidates[i].plane_local,
                                             candidates[first].plane_local),
                                      sideways);
                    if(d>largest){largest=d;fourth=i;}
                }
            }
            indices[3]=fourth;
            chosen[fourth]=true;
            // Source clusters every remaining contact around the
            // ORIGINAL four geometric witnesses. Replacement changes
            // only the chosen contact index, not the cluster center.
            std::array<PcmPlaneContact,4> cluster_centers{};
            for(std::size_t k=0;k<4;++k)
                cluster_centers[k]=candidates[indices[k]];
            for(std::size_t i=0;i<count;++i) {
                if(chosen[i])continue;
                float nearest=std::numeric_limits<float>::max();
                std::size_t slot=0;
                for(std::size_t j=0;j<4;++j) {
                    const float d=sq(diff(candidates[i].plane_local,
                                           cluster_centers[j].plane_local));
                    if(d<nearest){nearest=d;slot=j;}
                }
                if(candidates[i].separation<
                   candidates[indices[slot]].separation)
                    indices[slot]=i;
            }
            for(std::size_t k=0;k<4;++k)
                cache.points[k]=candidates[indices[k]];
            cache.count=4;
        }
    }
    // Source addManifoldContactsToContactBuffer emits from cached B
    // positions, not A box vertices, filtering with <= contactDist.
    std::uint8_t emitted=0;
    for(std::size_t k=0;k<cache.count;++k) {
        const auto& pt=cache.points[k];
        if(pt.separation>contact_distance)continue;
        if(emitted>=output.size())
            throw std::logic_error("PCM plane-box output arena too small");
        const ContactPoint contact{
            {plane.position.x+pt.plane_local.x,plane.position.y,
             plane.position.z+pt.plane_local.z},
            {0.0f,-1.0f,0.0f},pt.separation};
        if(!finite(contact.position) || !std::isfinite(contact.separation))
            throw std::overflow_error("PhysX PCM plane-box contact overflow");
        output[emitted++]=contact;
    }
    return emitted;
}
void Scene::update_contacts() {
    // Gu narrowphase consumes the sum of both PxShapeCore contact offsets.
    next_manifolds_.clear();
    next_contact_points_.clear();
    // Original PxcNpCacheStreamPair retains source-generated geometry,
    // distinct from this frame's reconstructed midpoint contact output.
    next_gu_cache_records_.clear();
    next_gu_cached_points_.clear();
    // Build the Gu-offset lookup alongside this real contact-manager
    // publication; no second O(all Gu manifolds) pass during PGS.
    // Generation stamps invalidate untouched/retired EdgeIndex slots
    // without clearing the historical handle watermark each frame.
    next_manifold_by_edge_.resize(contact_edges_.size());
    if(++indexed_manifold_generation_==0) {
        for(auto& item:next_manifold_by_edge_)item.generation=0;
        for(auto& item:manifold_by_edge_)item.generation=0;
        indexed_manifold_generation_=1;
    }
    const auto new_lookup_generation=indexed_manifold_generation_;
    std::size_t old_cursor = 0;
    std::size_t old_cache_cursor=0;
    std::size_t pair_index = 0;
    for (const Pair pair : previous_pairs_) {
        const EdgeIndex edge=active_edge_handles_[pair_index++];
        // The sorted source contact-manager stream retains its collision
        // axis across successive physical-touch frames.
        while (old_cursor < manifolds_.size() &&
               manifolds_[old_cursor].pair < pair)
            ++old_cursor;
        const Actor& a = actors_[pair.a];
        const Actor& b = actors_[pair.b];
        const float contact_distance=a.contact_offset+b.contact_offset;
        if(!std::isfinite(contact_distance))
            throw std::overflow_error("Native Gu contact distance overflow");
        ContactManifold manifold{};
        manifold.pair = pair;
        manifold.edge_handle=edge;
        manifold.contact_offset = next_contact_points_.size();
        if (old_cursor < manifolds_.size() &&
            manifolds_[old_cursor].pair == pair)
            manifold.collision_axis = manifolds_[old_cursor].collision_axis;

        // PxcNpBatch::discreteNarrowPhase sorts geometry types (plane
        // before box) independently of ScShapeInteraction actor order.
        // Gu box/box uses the source-oriented Sc contact manager.
        const bool plane_pair=a.kind==Kind::StaticPlane ||
                              b.kind==Kind::StaticPlane;
        const bool reversed=!plane_pair &&
            contact_edges_[edge].gu_order_reversed;
        const Actor& first=plane_pair
            ? (a.kind==Kind::StaticPlane?a:b)
            : (reversed?b:a);
        const Actor& second=plane_pair
            ? (a.kind==Kind::StaticPlane?b:a)
            : (reversed?a:b);
        while(old_cache_cursor<gu_cache_records_.size() &&
              gu_cache_records_[old_cache_cursor].pair<pair)
            ++old_cache_cursor;
        const GuCacheRecord* old=
            old_cache_cursor<gu_cache_records_.size() &&
            gu_cache_records_[old_cache_cursor].pair==pair
            ? &gu_cache_records_[old_cache_cursor] : nullptr;
        GuCacheRecord cache{};
        cache.pair=pair;
        cache.offset=next_gu_cached_points_.size();
        cache.pose0={first.position,first.orientation};
        cache.pose1={second.position,second.orientation};
        cache.contact_distance=contact_distance;
        cache.collision_axis=old?old->collision_axis:manifold.collision_axis;
        manifold.collision_axis=cache.collision_axis;
        const bool pcm_plane=pcm_plane_box_enabled_ && plane_pair;
        bool cache_hit=false;
        if(pcm_plane) {
            // Persistent PCM cache belongs to the SAME contact-manager
            // slot as the speculative/accurate island touch flags.
            // Journal before the first mutation so rejected Gu/PGS
            // frames and collision-only queries restore both owners.
            step_checkpoint_.save_edge(*this,edge);
            const Actor& plane=a.kind==Kind::StaticPlane?a:b;
            const Actor& box=a.kind==Kind::StaticPlane?b:a;
            if(box.kind!=Kind::DynamicBox)
                throw std::logic_error(
                    "PCM plane-box requires the admitted dynamic box");
            manifold.count=generate_pcm_plane_box(
                plane,box,contact_distance,contact_edges_[edge].pcm_plane_box,
                std::span<ContactPoint>(box_contact_scratch_));
            next_contact_points_.insert(next_contact_points_.end(),
                box_contact_scratch_.begin(),
                box_contact_scratch_.begin()+manifold.count);
        }
        if(!pcm_plane && old &&
           old->contact_distance==contact_distance &&
           old->offset<=gu_cached_points_.size() &&
           old->count<=gu_cached_points_.size()-old->offset) {
            // PxcCacheLocalContacts compares both components of the
            // source transforms in the SECOND shape's local frame.
            const auto relative=[](const CachedPose& p0,
                                   const CachedPose& p1) {
                const Vec3 delta{
                    p0.position.x-p1.position.x,
                    p0.position.y-p1.position.y,
                    p0.position.z-p1.position.z};
                return CachedPose{
                    rotate_inverse(p1.orientation,delta),
                    multiply({-p1.orientation.x,-p1.orientation.y,
                              -p1.orientation.z,p1.orientation.w},
                             p0.orientation)};
            };
            const auto previous=relative(old->pose0,old->pose1);
            const auto current=relative(cache.pose0,cache.pose1);
            constexpr float translation_epsilon=0.01f; // 1% PxTolerancesScale.length=1
            constexpr float quaternion_epsilon=0.01f;
            const auto in_range=[](float a,float b,float threshold) {
                return std::abs(a-b)<threshold;
            };
            cache_hit=
                in_range(previous.position.x,current.position.x,translation_epsilon) &&
                in_range(previous.position.y,current.position.y,translation_epsilon) &&
                in_range(previous.position.z,current.position.z,translation_epsilon) &&
                in_range(previous.orientation.x,current.orientation.x,quaternion_epsilon) &&
                in_range(previous.orientation.y,current.orientation.y,quaternion_epsilon) &&
                in_range(previous.orientation.z,current.orientation.z,quaternion_epsilon) &&
                in_range(previous.orientation.w,current.orientation.w,quaternion_epsilon);
            if(cache_hit) {
                // The source copies the same original cached byte payload
                // to the next cache stream. Only the emitted Gu contacts
                // are updated; cached poses are NOT advanced on a hit.
                cache.pose0=old->pose0;
                cache.pose1=old->pose1;
                cache.count=old->count;
                for(std::size_t k=0;k<old->count;++k) {
                    const ContactPoint& prior=gu_cached_points_[old->offset+k];
                    const auto project=[&](const CachedPose& from,
                                           const CachedPose& to) {
                        Vec3 local=rotate_inverse(from.orientation,{
                            prior.position.x-from.position.x,
                            prior.position.y-from.position.y,
                            prior.position.z-from.position.z});
                        Vec3 v=rotate(to.orientation,local);
                        return Vec3{v.x+to.position.x,
                                    v.y+to.position.y,v.z+to.position.z};
                    };
                    const Vec3 p0=project(old->pose0,
                                          {first.position,first.orientation});
                    const Vec3 p1=project(old->pose1,
                                          {second.position,second.orientation});
                    const Vec3 difference{
                        p0.x-p1.x,p0.y-p1.y,p0.z-p1.z};
                    const Vec3 n=prior.normal;
                    const ContactPoint out{{
                        (p0.x+p1.x)*0.5f,(p0.y+p1.y)*0.5f,
                        (p0.z+p1.z)*0.5f},n,
                        prior.separation+
                            difference.x*n.x+difference.y*n.y+difference.z*n.z};
                    if(!finite(out.position) ||
                       !std::isfinite(out.separation))
                        throw std::overflow_error("Native Pxc cached contact reconstruction overflow");
                    next_contact_points_.push_back(out);
                    next_gu_cached_points_.push_back(prior);
                }
                manifold.count=old->count;
            }
        }
        if(!pcm_plane && !cache_hit && (a.kind == Kind::StaticPlane || b.kind == Kind::StaticPlane)) {
            // Direct source order/geometry from
            // GuContactPlaneBox.cpp::contactPlaneBox. The SDK plane's
            // logical normal is +Y, corresponding to PhysX plane-local
            // +X. This computes boxToPlane.m.column[k].x times each
            // half-extent, constructs the eight source depth tests, and
            // publishes world-space rotated vertices. Do not replace
            // this with identity-only AABB corners.
            const Actor& plane = a.kind == Kind::StaticPlane ? a : b;
            const Actor& box = a.kind == Kind::StaticPlane ? b : a;
            if (box.kind != Kind::DynamicBox)
                continue;
            // GuContactPlaneBox always emits the plane-first normal.
            // PxcCacheLocalContacts owns that raw Gu value; the subsequent
            // PxcNpBatch::flipContacts changes only the published contacts
            // when Sc's actor order has the box before the plane.
            const Vec3 normal{0.0f, -1.0f, 0.0f};
            // Identity boxes dominate the admitted floor-contact path.
            // Preserve the source-derived rotated basis only when needed;
            // this avoids three quaternion rotations per plane/box pair.
            Vec3 x{box.half_extents.x, 0.0f, 0.0f};
            Vec3 y{0.0f, box.half_extents.y, 0.0f};
            Vec3 z{0.0f, 0.0f, box.half_extents.z};
            const Quat q = box.orientation;
            if (q.x != 0.0f || q.y != 0.0f ||
                q.z != 0.0f || q.w != 1.0f) {
                const Vec3 basis_x = rotate(q, {1.0f,0.0f,0.0f});
                const Vec3 basis_y = rotate(q, {0.0f,1.0f,0.0f});
                const Vec3 basis_z = rotate(q, {0.0f,0.0f,1.0f});
                x = {basis_x.x * box.half_extents.x,
                     basis_x.y * box.half_extents.x,
                     basis_x.z * box.half_extents.x};
                y = {basis_y.x * box.half_extents.y,
                     basis_y.y * box.half_extents.y,
                     basis_y.z * box.half_extents.y};
                z = {basis_z.x * box.half_extents.z,
                     basis_z.y * box.half_extents.z,
                     basis_z.z * box.half_extents.z};
            }
            const float limit = contact_distance -
                (box.position.y - plane.position.y);
            const float depths[8]{
                x.y + y.y + z.y - limit,
                x.y + y.y - z.y - limit,
                x.y - y.y + z.y - limit,
                x.y - y.y - z.y - limit,
               -x.y + y.y + z.y - limit,
               -x.y + y.y - z.y - limit,
               -x.y - y.y + z.y - limit,
               -x.y - y.y - z.y - limit
            };
            for (unsigned corner = 0; corner < 8; ++corner) {
                if (depths[corner] >= 0.0f) continue;
                const float sx = (corner & 4u) ? -1.0f : 1.0f;
                const float sy = (corner & 2u) ? -1.0f : 1.0f;
                const float sz = (corner & 1u) ? -1.0f : 1.0f;
                const Vec3 point{
                    box.position.x + sx*x.x + sy*y.x + sz*z.x,
                    box.position.y + sx*x.y + sy*y.y + sz*z.y,
                    box.position.z + sx*x.z + sy*y.z + sz*z.z};
                next_contact_points_.push_back({
                    point, normal, depths[corner] + contact_distance});
                ++manifold.count;
            }
        } else if(!pcm_plane && !cache_hit) {
            // Original GuContactBoxBox SAT, face/edge candidate generation,
            // six-axis separation cache, and source contact ordering.
            // Its contact buffer owns up to the original 255 contacts.
            // The Gu reference/incident face and cached six-axis hint
            // use ScNPhaseCore::shouldSwapBodies orientation. Retain
            // this SAME frame through Dy contact/friction preparation;
            // only the outward Sc report translates to canonical IDs.
            manifold.count = source_box_box_contacts(
                first.half_extents, first.position, first.orientation,
                second.half_extents, second.position, second.orientation,
                contact_distance, manifold.collision_axis, box_contact_scratch_);
            next_contact_points_.insert(next_contact_points_.end(),
                box_contact_scratch_.begin(),
                box_contact_scratch_.begin() + manifold.count);
        }
        if(!pcm_plane && !cache_hit) {
            // PxcCacheLocalContacts caches the RAW source Gu stream on a
            // miss, even for zero-contact overlaps. Source mPairData is
            // retained only when the contact generator actually ran.
            cache.count=manifold.count;
            cache.collision_axis=manifold.collision_axis;
            // The Gu contact arena and the legacy Pxc cache arena
            // have distinct ownership and offsets once PCM pairs are
            // admitted. Cached source points come from THIS manager's
            // output window, not cache.offset (which indexes only
            // legacy cached points and omits preceding PCM contacts).
            // Dy still indexes the full Gu stream by contact_offset.
            next_gu_cached_points_.insert(next_gu_cached_points_.end(),
                next_contact_points_.begin()+manifold.contact_offset,
                next_contact_points_.begin()+manifold.contact_offset+
                    manifold.count);
        }
        if(!pcm_plane)next_gu_cache_records_.push_back(cache);
        // PxcNpBatch::flipContacts runs AFTER PxcCacheLocalContacts.
        // The cached points above must retain the raw plane-first Gu
        // normal, so a subsequent cache hit's motion.dot(normal) keeps
        // the source separation sign. ScShapeInteraction orders the
        // dynamic body before the plane regardless of creation order;
        // only its published contact stream receives the flip.
        if(plane_pair) {
            for(std::size_t k=0;k<manifold.count;++k) {
                Vec3& n=next_contact_points_[manifold.contact_offset+k].normal;
                n={-n.x,-n.y,-n.z};
            }
        }
        if (manifold.count > 0) {
            next_manifold_by_edge_[edge]={
                next_manifolds_.size(),new_lookup_generation};
            next_manifolds_.push_back(manifold);
        }
    }

    // BP pair found/lost and narrowphase touch found/lost are distinct
    // source events. Report changed touch pairs exactly once per frame.
    next_found_contacts_.clear();
    next_lost_contacts_.clear();
    const auto pair_less = [](const ContactManifold& lhs,
                              const ContactManifold& rhs) {
        return lhs.pair < rhs.pair;
    };
    // Sorted broadphase candidate iteration preserves manifold order.
    auto next = next_manifolds_.begin();
    auto old = manifolds_.begin();
    while (next != next_manifolds_.end() && old != manifolds_.end()) {
        if (pair_less(*next, *old)) {
            next_found_contacts_.push_back(next->pair);
            ++next;
        } else if (pair_less(*old, *next)) {
            next_lost_contacts_.push_back(old->pair);
            ++old;
        } else {
            ++next; ++old;
        }
    }
    for (; next != next_manifolds_.end(); ++next)
        next_found_contacts_.push_back(next->pair);
    for (; old != manifolds_.end(); ++old)
        next_lost_contacts_.push_back(old->pair);
    // Assign all pointers only after the arena has stopped growing.
    // Both old and new contact arenas swap together with their manifolds.
    for (auto& manifold : next_manifolds_)
        manifold.points = std::span<const ContactPoint>(
            next_contact_points_.data() + manifold.contact_offset,
            manifold.count);
    // Source SimpleIslandManager::setEdgeConnected/Disconnected acts on
    // the persistent contact-manager handle. Physical touch loss must
    // NOT destroy its speculative overlap edge or allocate a new handle.
    for(Pair pair:next_found_contacts_) {
        const auto id=contact_edge_index(pair);
        if(id!=invalid_edge) {
            link_contact_edge(id,true);
            step_checkpoint_.save_edge(*this,id);
            contact_edges_[id].connected=true;
        }
    }
    for(Pair pair:next_lost_contacts_) {
        const auto id=contact_edge_index(pair);
        if(id!=invalid_edge) {
            unlink_contact_edge(id,true);
            step_checkpoint_.save_edge(*this,id);
            contact_edges_[id].connected=false;
        }
    }
    manifolds_.swap(next_manifolds_);
    contact_points_.swap(next_contact_points_);
    gu_cache_records_.swap(next_gu_cache_records_);
    gu_cached_points_.swap(next_gu_cached_points_);
    manifold_by_edge_.swap(next_manifold_by_edge_);
    manifold_lookup_generation_=new_lookup_generation;
    found_contacts_.swap(next_found_contacts_);
    lost_contacts_.swap(next_lost_contacts_);
    ++contact_epoch_;
}
// IG::SimpleIslandManager::firstPassIslandGen/secondPassIslandGen:
// speculative edges come from the scene's published SAP candidates,
// whereas accurate edges come only from the resolved physical touches.
// The admitted all-awake rigid island owner now preserves source-style
// persistent handles, node membership and per-graph static-touch counts.
// Topology rebuild is skipped for unchanged dynamic connectivity; source
// sleep/deactivation, task continuations and local split budgets remain
// N1 requirements. Static touch transitions alter counters but do not
// manufacture dynamic-island topology changes.
// IslandSim::processLostEdges calls findRoute on dirty endpoints before
// splitting an island. For the admitted all-awake dynamic contact subset,
// a lost edge can only split its component if its endpoints are no longer
// connected in the *surviving* graph. Traverse the persistent source
// EdgeInstance adjacency owned by the corresponding contact-manager
// handles. Root-directed fast-route and hop hints are cached in the
// all-awake subset; stable island identities and deactivation remain open.
// IslandSim::findRoute traverses per-node mFirstEdgeIndex and the
// paired EdgeInstance indices, rather than rebuilding an adjacency
// graph every time a contact disappears. The persistent source slot
// lists and static-touch counters follow independent SAP/Gu publications.
// IslandSim::processLostEdges processes nodes dirtied by removed graph
// links. Its findRoute first tries mFastRoute to the island root and
// otherwise traverses surviving mFirstEdgeIndex adjacency, rebuilding
// hop/route estimates. The all-awake rigid subset has no sleeping or
// deferred dirty-work budget; a failed root route proves a real split.
bool Scene::lost_island_edges_retain_routes(bool accurate) {
    auto& dirty=accurate?accurate_dirty_nodes_:speculative_dirty_nodes_;
    if(dirty.empty())return true;
    // PxsIslandSim::processLostEdges traverses mDirtyMap by ascending
    // node index. Our deduplicated vector is filled in lost-edge
    // endpoint order, which need not match the bitmap iteration.
    // Sort that existing scratch worklist in place: earlier successful
    // routes mutate hop estimates used by later dirty-node searches.
    std::sort(dirty.begin(),dirty.end());
    const auto& first=accurate?accurate_node_first_:speculative_node_first_;
    const auto& graph=accurate?accurate_islands_:speculative_islands_;
    auto& fast=accurate?accurate_fast_route_:speculative_fast_route_;
    auto& fast_edge=accurate?accurate_fast_edge_:speculative_fast_edge_;
    auto& hops=accurate?accurate_hops_:speculative_hops_;
    auto& stamps=accurate?accurate_hint_stamps_:speculative_hint_stamps_;
    const auto epoch=accurate?accurate_hint_epoch_:speculative_hint_epoch_;
    const BodyId invalid=std::numeric_limits<BodyId>::max();
    fast.resize(actors_.size(),invalid);
    fast_edge.resize(actors_.size(),invalid_edge);
    // IslandSim::addNode initializes mHopCounts to zero. A node with
    // no established cached route is not assigned infinite hop cost.
    hops.resize(actors_.size(),0);
    stamps.resize(actors_.size(),0);
    // Retain visited marks across publications; only grow on actor
    // admission, rather than clearing O(all actors) on a lost-edge pass.
    island_route_seen_.resize(actors_.size(),0);
    island_route_predecessor_.resize(actors_.size(),invalid);
    island_route_predecessor_edge_.resize(actors_.size(),invalid_edge);
    // PhysX mVisitedState is shared by every dirty-node search in this
    // third-pass IslandSim, so nodes assigned to an earlier split are
    // never searched again against their newly promoted root. Preserve
    // the worklist witness state without clearing O(all actors) per pass.
    island_route_confirmed_.resize(actors_.size(),0);
    if(++island_route_confirmed_stamp_==0) {
        std::fill(island_route_confirmed_.begin(),
                  island_route_confirmed_.end(),0);
        island_route_confirmed_stamp_=1;
    }
    const auto confirmed_stamp=island_route_confirmed_stamp_;
    bool split_occurred=false;

    // A cached source EdgeIndex is stable until retirement, and never
    // aliases an unrelated route just because its slot is later reused:
    // both endpoints and this graph's intrusive instances must still
    // match. One edge-slot check per hop replaces linear neighbor scans.
    const auto live_hop=[&](BodyId from,BodyId to,EdgeIndex id) {
        if(id==invalid_edge || id>=contact_edges_.size())return false;
        const auto& edge=contact_edges_[id];
        if(!edge.live || edge.retiring ||
           (accurate && !edge.connected))return false;
        const bool forward=edge.pair.a==from && edge.pair.b==to;
        const bool reverse=edge.pair.a==to && edge.pair.b==from;
        if(!forward && !reverse)return false;
        const auto& links=accurate?edge.accurate:edge.speculative;
        return links[0].linked && links[1].linked;
    };
    const auto root_for=[&](BodyId node) {
        return graph.islands[graph.node_island[node]].root;
    };
    // processLostEdges visits mDirtyMap in index order and splits a
    // disconnected component *inside* that iteration. Later dirty
    // nodes must observe the new island owner and the shared witnesses.
    for(BodyId node:dirty) {
        if(!actors_[node].alive || actors_[node].kind!=Kind::DynamicBox ||
           island_route_confirmed_[node]==confirmed_stamp)
            continue;
        ++island_dirty_route_checks_;
        const BodyId root=root_for(node);
        if(root==node) {
            island_route_confirmed_[node]=confirmed_stamp;
            continue;
        }

        // Source tryFastPath: only accept a cached path when every
        // individual link is still present and hop estimates decrease.
        // Never trust the cache merely because two nodes retain the
        // same pre-split island identifier.
        BodyId probe=node;
        bool fast_found=false;
        island_split_members_.clear();
        for(std::size_t depth=0;depth<actors_.size();++depth) {
            if(probe==root ||
               (island_route_confirmed_[probe]==confirmed_stamp &&
                graph.node_island[probe]==graph.node_island[node])) {
                fast_found=true;
                break;
            }
            if(stamps[probe]!=epoch)break;
            const BodyId next=fast[probe];
            if(next==invalid || next>=actors_.size() ||
               !actors_[next].alive || stamps[next]!=epoch ||
               hops[next]>=hops[probe] ||
               !live_hop(probe,next,fast_edge[probe]))
                break;
            island_split_members_.push_back(probe);
            probe=next;
        }
        if(fast_found) {
            ++island_fast_route_hits_;
            // Source tryFastPath retains valid witnesses along every
            // root-directed step, not just at the starting dirty node.
            for(BodyId visited:island_split_members_)
                island_route_confirmed_[visited]=confirmed_stamp;
            continue;
        }

        // IslandSim::findRoute falls back to a priority queue ordered
        // by estimated hops to the root, rather than FIFO traversal.
        // These estimates can be stale: the priority changes only search
        // order, never reachability or the requirement for a live route.
        // A successful route refreshes mFastRoute and mHopCounts hints.
        const auto higher=[&](BodyId a,BodyId b) {
            return hops[a]<hops[b];
        };
        if(++island_route_visit_stamp_==0) {
            std::fill(island_route_seen_.begin(),island_route_seen_.end(),0);
            island_route_visit_stamp_=1;
        }
        const auto visit_stamp=island_route_visit_stamp_;
        island_route_queue_.clear();
        island_split_members_.clear();
        island_split_members_.push_back(node);
        source_hop_push(island_route_queue_,node,higher);
        island_route_seen_[node]=visit_stamp;
        island_route_predecessor_[node]=invalid;
        island_route_predecessor_edge_[node]=invalid_edge;
        bool found=false;
        BodyId witness=root;
        // The reusable queue is a min-heap by estimated hop count.
        // The source's Cm::PriorityQueue has no secondary tie-break:
        // equal-hop promotion follows its exact binary-heap child rules.
        while(!island_route_queue_.empty() && !found) {
            const BodyId from=source_hop_pop(island_route_queue_,higher);
            for(std::size_t i=first[from];i!=invalid_instance;) {
                const ContactEdge& edge=contact_edges_[i/2];
                const auto& link=(accurate?edge.accurate:
                                  edge.speculative)[i&1];
                const BodyId to=(i&1)?edge.pair.a:edge.pair.b;
                const std::size_t i_edge=i/2;
                i=link.next;
                if(!actors_[to].alive ||
                   actors_[to].kind!=Kind::DynamicBox ||
                   island_route_seen_[to]==visit_stamp)
                    continue;
                island_route_seen_[to]=visit_stamp;
                island_route_predecessor_[to]=from;
                island_route_predecessor_edge_[to]=
                    static_cast<EdgeIndex>(i_edge);
                // Source findRoute can terminate on an earlier pass's
                // visited, confirmed island member as well as the root.
                // This reuses mVisitedState's witness instead of
                // rediscovering its complete fast route in this search.
                if(to==root ||
                   (island_route_confirmed_[to]==confirmed_stamp &&
                    graph.node_island[to]==graph.node_island[node])) {
                    found=true;
                    witness=to;
                    break;
                }
                // mVisitedNodes is ordered by queue insertion, not pop.
                // A failed search has already collected the entire
                // disconnected component in its exact source visit order.
                island_split_members_.push_back(to);
                source_hop_push(island_route_queue_,to,higher);
            }
        }
        if(!found) {
            split_lost_island(accurate,node);
            for(BodyId member:island_split_members_)
                island_route_confirmed_[member]=confirmed_stamp;
            split_occurred=true;
            continue;
        }

        if(witness==root) {
            step_checkpoint_.save_hint(*this,root,accurate);
            fast[root]=invalid;
            fast_edge[root]=invalid_edge;
            hops[root]=0;
            stamps[root]=epoch;
        }
        // Upstream unwindRoute repairs the critical root/witness path
        // backward before processLostEdges fills the remaining
        // discovered branches in their mVisitedNodes insertion order.
        for(BodyId current=witness;current!=node;) {
            const BodyId predecessor=island_route_predecessor_[current];
            step_checkpoint_.save_hint(*this,predecessor,accurate);
            fast[predecessor]=current;
            fast_edge[predecessor]=island_route_predecessor_edge_[current];
            hops[predecessor]=hops[current]+1;
            stamps[predecessor]=epoch;
            island_route_confirmed_[predecessor]=confirmed_stamp;
            current=predecessor;
        }
        for(BodyId visited:island_split_members_) {
            if(island_route_confirmed_[visited]==confirmed_stamp)
                continue;
            const BodyId predecessor=island_route_predecessor_[visited];
            if(predecessor==invalid)
                throw std::logic_error(
                    "Source successful route lost its traversal parent");
            step_checkpoint_.save_hint(*this,visited,accurate);
            fast[visited]=predecessor;
            fast_edge[visited]=island_route_predecessor_edge_[visited];
            hops[visited]=hops[predecessor]+1;
            stamps[visited]=epoch;
            island_route_confirmed_[visited]=confirmed_stamp;
        }
    }
    return !split_occurred;
}

// PxsIslandSim.cpp::processLostEdges: after failed findRoute, migrate
// precisely the visited component out of its previous island. The
// retained root's handle and all unrelated island chains remain intact.
// The source's heap insertion order determines the new node chain and
// indexed contact-manager splitEdges order. This is the admitted rigid
// graph; kinematic witnesses and bounded dirty budgets remain separate.
void Scene::split_lost_island(bool accurate,BodyId seed) {
    auto& graph=accurate?accurate_islands_:speculative_islands_;
    auto& hops=accurate?accurate_hops_:speculative_hops_;
    auto& routes=accurate?accurate_fast_route_:speculative_fast_route_;
    auto& route_edges=accurate?accurate_fast_edge_:speculative_fast_edge_;
    auto& stamps=accurate?accurate_hint_stamps_:speculative_hint_stamps_;
    const auto epoch=accurate?accurate_hint_epoch_:speculative_hint_epoch_;
    const BodyId invalid=std::numeric_limits<BodyId>::max();
    if(!actors_[seed].alive ||
       actors_[seed].kind!=Kind::DynamicBox)
        throw std::logic_error("Source split seed is not dynamic");
    const auto owner=graph.node_island[seed];
    if(owner==IslandGraph::invalid || !graph.islands[owner].live)
        throw std::logic_error("Source split seed lacks an island");
    const BodyId root=graph.islands[owner].root;
    // findRoute has ALREADY visited every reachable node and found no
    // route to root. Reuse its mVisitedNodes and parent/depth records:
    // repeating the heap traversal wastes work and can change ordering.
    if(root==seed || island_split_members_.empty() ||
       island_split_members_.front()!=seed ||
       island_route_seen_[root]==island_route_visit_stamp_ ||
       island_split_members_.size()>=graph.islands[owner].count)
        throw std::logic_error("Source failed-route split invariant lost");
    step_checkpoint_.save_graph_ownership(*this);

        // Source removeEdgeFromIsland uses swap-last on the old indexed
        // contact-manager array. Remove exactly once, in visited-node
        // adjacency order, before changing each node's island identity.
        island_split_edges_.clear();
        if(accurate) {
            for(BodyId node:island_split_members_)
                for(std::size_t i=accurate_node_first_[node];
                    i!=invalid_instance;) {
                    const EdgeIndex edge_index=
                        static_cast<EdgeIndex>(i/2);
                    const auto& edge=contact_edges_[edge_index];
                    const auto& link=edge.accurate[i&1];
                    i=link.next;
                    if(!edge.connected ||
                       (node!=edge.pair.a &&
                        actors_[edge.pair.a].kind==Kind::DynamicBox))
                        continue;
                    remove_accurate_island_edge(edge_index);
                    island_split_edges_.push_back(edge_index);
                }
        }

        // HandleManager::getHandle obtains a genuinely new island
        // handle, leaving the retained root component's identity in
        // place. The graph's existing free list is authoritative.
        IslandGraph::IslandId created;
        if(!graph.free_handles.empty()) {
            created=graph.free_handles.back();
            graph.free_handles.pop_back();
        } else {
            if(graph.islands.size()>=IslandGraph::invalid)
                throw std::overflow_error("Source split handles exhausted");
            created=static_cast<IslandGraph::IslandId>(
                graph.islands.size());
            graph.islands.push_back({});
        }
        // Detach exactly the source visited nodes from their old
        // intrusive node chain. No unrelated graph node is visited or
        // remapped, and the surviving old root/tail remain owned by
        // the original handle.
        std::size_t static_touches=0;
        for(BodyId node:island_split_members_) {
            const BodyId before=graph.node_prev[node];
            const BodyId after=graph.node_next[node];
            if(before!=invalid)graph.node_next[before]=after;
            else graph.islands[owner].root=after;
            if(after!=invalid)graph.node_prev[after]=before;
            else graph.islands[owner].tail=before;
            --graph.islands[owner].count;
            const auto count=graph.node_static_touches[node];
            static_touches+=count;
            graph.islands[owner].static_touches-=count;
        }

        auto& island=graph.islands[created];
        island={};
        island.live=true;
        island.root=seed;
        island.tail=island_split_members_.back();
        island.count=island_split_members_.size();
        island.static_touches=static_touches;
        // PhysX processLostEdges creates an active detached island when
        // the original island was active. Preserve this event order now,
        // before another split can consume a recycled island handle.
        if(graph.islands[owner].active_index!=invalid_instance) {
            island.active_index=graph.active_islands.size();
            graph.active_islands.push_back(created);
        }
        for(std::size_t i=0;i<island_split_members_.size();++i) {
            const BodyId node=island_split_members_[i];
            const BodyId previous=i?island_split_members_[i-1]:invalid;
            const BodyId next=i+1<island_split_members_.size()
                ?island_split_members_[i+1]:invalid;
            graph.node_prev[node]=previous;
            graph.node_next[node]=next;
            graph.node_island[node]=created;
            step_checkpoint_.save_hint(*this,node,accurate);
            routes[node]=previous==invalid?invalid:
                island_route_predecessor_[node];
            route_edges[node]=previous==invalid?invalid_edge:
                island_route_predecessor_edge_[node];
            hops[node]=previous==invalid?0:
                hops[island_route_predecessor_[node]]+1;
            stamps[node]=epoch;
        }
        // Source splitEdges appends the collected edge sequence,
        // not an order reconstructed from canonical Gu pairs.
        if(accurate)
            for(EdgeIndex id:island_split_edges_)
                append_accurate_island_edge(id,created);
        if(accurate)++accurate_island_count_;
        else {
            ++speculative_island_count_;
            // Source processLostEdges evaluates deactivation after local
            // split repair. An awake speculative island can lose its
            // final awake peer without ANY new accurate-node sleep.
            // Both the retained and detached roots may now be inactive.
            speculative_split_sleep_candidates_.push_back(root);
            speculative_split_sleep_candidates_.push_back(seed);
        }
        ++island_local_split_count_;
}

void Scene::update_island_connectivity(bool repair_lost_edges) {
    // PxsIslandSim::IslandSim's node-to-island IDs and persistent island
    // handle slots are now the authoritative connectivity owner. The
    // speculative and accurate graphs share source contact-manager edge
    // slots, not a second actor registry or a provisional union/find map.
    const BodyId invalid=std::numeric_limits<BodyId>::max();
    using Graph=IslandGraph;
    const auto dynamic_node=[&](BodyId node) {
        return actors_[node].alive && actors_[node].kind==Kind::DynamicBox;
    };
    const auto dynamic_edge=[&](Pair pair) {
        return dynamic_node(pair.a) && dynamic_node(pair.b);
    };

    bool rebuild=island_topology_dirty_;
    bool local_split=false;
    if(!rebuild && repair_lost_edges) {
        // Both source IslandSim owners must consume their dirty maps.
        // A speculative split cannot skip accurate-root hint repair.
        const bool speculative_intact=
            lost_island_edges_retain_routes(false);
        const bool accurate_intact=
            lost_island_edges_retain_routes(true);
        // A failed route was split immediately in source dirty-map
        // order, not deferred until after searching every old root.
        local_split=!speculative_intact || !accurate_intact;
    }
    if(!rebuild && repair_lost_edges && !local_split)
        return;
    if(!rebuild && !repair_lost_edges) {
        // IslandSim::addConnectionToGraph/removeConnectionInternal update
        // per-node and per-island static-touch counts without merging
        // dynamic islands. SAP candidates and Gu touches are independent.
        // Rebuilds instead count the final intrusive static links once.
        const auto change_static=[&](Graph& graph,Pair pair,bool added) {
            const bool da=dynamic_node(pair.a);
            const bool db=dynamic_node(pair.b);
            if(da==db)return;
            const BodyId node=da?pair.a:pair.b;
            const auto id=graph.node_island[node];
            if(id==Graph::invalid || id>=graph.islands.size())
                throw std::logic_error(
                    "PhysX static edge lacks dynamic island node");
            step_checkpoint_.save_graph_ownership(*this);
            auto& count=graph.node_static_touches[node];
            auto& total=graph.islands[id].static_touches;
            if(added) {
                if(count==std::numeric_limits<std::uint32_t>::max())
                    throw std::overflow_error(
                        "PhysX island node static-touch count overflow");
                ++count;
                ++total;
            } else {
                if(!count || !total)
                    throw std::logic_error(
                        "PhysX static-touch counter underflow");
                --count;
                --total;
            }
        };
        for(Pair pair:found_)
            change_static(speculative_islands_,pair,true);
        for(Pair pair:lost_)
            change_static(speculative_islands_,pair,false);
        for(Pair pair:found_contacts_)
            change_static(accurate_islands_,pair,true);
        for(Pair pair:lost_contacts_)
            change_static(accurate_islands_,pair,false);
    }
    if(!rebuild) {
        // No newly connected dynamic edges means no island membership,
        // handle or solver batch change. Preserve the source's quiet
        // publication contract without touching island revision/epochs.
        bool dynamic_found=false;
        for(Pair pair:found_)
            if(dynamic_edge(pair)) {dynamic_found=true;break;}
        if(!dynamic_found)
            for(Pair pair:found_contacts_)
                if(dynamic_edge(pair)) {dynamic_found=true;break;}
        if(!dynamic_found && !local_split) {
            // Even a static-only touch is a source contact-manager edge
            // in its awake dynamic island; it cannot be skipped merely
            // because dynamic connectivity and handles remain unchanged.
            if(!repair_lost_edges)
                for(Pair pair:found_contacts_) {
                    const auto id=contact_edge_index(pair);
                    const BodyId node=dynamic_node(pair.a)?pair.a:pair.b;
                    if(id!=invalid_edge && dynamic_node(node))
                        append_accurate_island_edge(
                            id,accurate_islands_.node_island[node]);
                }
            return;
        }
    }

    // Process source node admission and processNewEdges in their
    // original order, *before* rebuilding membership from the final
    // adjacency. That final traversal accounts for lost edges and actor
    // retirement, but cannot alter the identity chosen by an earlier
    // size-accumulating merge. It runs only on topology-dirty first pass
    // or on a proven third-pass split, never on quiet frames.
    const auto merge=[&](Graph& graph,Pair pair) {
        if(!dynamic_edge(pair))return false;
        const bool accurate=&graph==&accurate_islands_;
        auto& hops=accurate?accurate_hops_:speculative_hops_;
        auto& fast=accurate?accurate_fast_route_:speculative_fast_route_;
        auto& fast_edge=accurate?accurate_fast_edge_:speculative_fast_edge_;
        auto& stamps=accurate?accurate_hint_stamps_:speculative_hint_stamps_;
        const auto epoch=accurate?accurate_hint_epoch_:speculative_hint_epoch_;
        const auto a=graph.node_island[pair.a];
        const auto b=graph.node_island[pair.b];
        const EdgeIndex connecting_edge=contact_edge_index(pair);
        if(connecting_edge==invalid_edge)
            throw std::logic_error("Source island merge lacks contact edge");
        if(a==b) {
            // PxsIslandSim::processNewEdges retains an existing island,
            // but a new chord may shorten its root-directed hop estimate.
            // The source updates *only* the endpoint with a shorter path.
            const auto ha=hops[pair.a],hb=hops[pair.b];
            if(ha+1<hb) {
                step_checkpoint_.save_hint(*this,pair.b,accurate);
                hops[pair.b]=ha+1;
                fast[pair.b]=pair.a;
                fast_edge[pair.b]=connecting_edge;
                stamps[pair.b]=epoch;
            } else if(hb+1<ha) {
                step_checkpoint_.save_hint(*this,pair.a,accurate);
                hops[pair.a]=hb+1;
                fast[pair.a]=pair.b;
                fast_edge[pair.a]=connecting_edge;
                stamps[pair.a]=epoch;
            }
            return false;
        }
        // IslandSim::mergeIslands preserves the larger island handle,
        // remapping only the smaller island-node chain, then freeing
        // the subsumed handle. Ties select the second island.
        step_checkpoint_.save_graph_ownership(*this);
        const bool keep_a=graph.islands[a].count>graph.islands[b].count;
        const auto retained=keep_a?a:b;
        const auto retired=keep_a?b:a;
        auto& winner=graph.islands[retained];
        auto& loser=graph.islands[retired];
        const BodyId winner_node=keep_a?pair.a:pair.b;
        const BodyId loser_node=keep_a?pair.b:pair.a;
        // Source mergeIslandsInternal adds this conservative path
        // estimate to every node in the subsumed island before setting
        // the connecting endpoint to winner.hops + 1. This is already
        // an O(loser.count) node-remap walk, not a new global pass.
        const std::uint32_t extra_path=
            hops[winner_node]+hops[loser_node]+1;
        for(BodyId node=loser.root;node!=invalid;
            node=graph.node_next[node]) {
            step_checkpoint_.save_hint(*this,node,accurate);
            hops[node]+=extra_path;
            graph.node_island[node]=retained;
        }
        step_checkpoint_.save_hint(*this,loser_node,accurate);
        hops[loser_node]=hops[winner_node]+1;
        fast[loser_node]=winner_node;
        fast_edge[loser_node]=connecting_edge;
        // The connecting route is valid immediately. Existing paths in
        // both islands remain eligible where their live edges and hop
        // ordering still lead to the retained root.
        stamps[loser_node]=epoch;
        graph.node_prev[loser.root]=winner.tail;
        graph.node_next[winner.tail]=loser.root;
        winner.tail=loser.tail;
        winner.count+=loser.count;
        winner.static_touches+=loser.static_touches;
        if(&graph==&accurate_islands_ &&
           loser.first_contact!=invalid_edge) {
            // PxsIslandSim.cpp::mergeEdges appends the losing island's
            // ordered contact-manager array to the retained island.
            if(winner.last_contact!=invalid_edge) {
                step_checkpoint_.save_edge(*this,winner.last_contact);
                step_checkpoint_.save_edge(*this,loser.first_contact);
                contact_edges_[winner.last_contact].island_next=
                    loser.first_contact;
                contact_edges_[loser.first_contact].island_prev=
                    winner.last_contact;
            } else winner.first_contact=loser.first_contact;
            winner.last_contact=loser.last_contact;
            winner.contact_count+=loser.contact_count;
        }
        // Source mergeIslands removes the subsumed island from the
        // dense mActiveIslands array with markIslandInactive's swap-last
        // operation. Do this BEFORE the freed handle can be recycled by
        // a lost-edge split in the same publication.
        const bool losing_active=loser.active_index!=invalid_instance;
        if(losing_active) {
            const auto index=loser.active_index;
            const auto last=graph.active_islands.back();
            graph.active_islands[index]=last;
            graph.islands[last].active_index=index;
            graph.active_islands.pop_back();
        }
        if(losing_active && winner.active_index==invalid_instance) {
            winner.active_index=graph.active_islands.size();
            graph.active_islands.push_back(retained);
        }
        loser={};
        graph.free_handles.push_back(retired);
        return true;
    };
    // Source addNode admits new dynamic rigid bodies as singleton islands.
    // Preserve *all* existing island node chains until processNewEdges
    // has performed its ordered merges, even when contacts were lost in
    // the same publication. Removing lost edges from intrusive adjacency
    // is not permission to split the old membership before those merges.
    const auto admit_nodes=[&](Graph& graph) {
        // Source IslandSim::addNode sets mFastRoute=INVALID and
        // mHopCounts=0 on each newly admitted rigid node. Preserve the
        // existing graph's estimates on reconnect; never reset all hints
        // because an unrelated actor joined this frame. The aligned
        // slots are reused by subsequent dirty-route traversal.
        auto& fast=(&graph==&accurate_islands_)?
            accurate_fast_route_:speculative_fast_route_;
        auto& fast_edge=(&graph==&accurate_islands_)?
            accurate_fast_edge_:speculative_fast_edge_;
        auto& hops=(&graph==&accurate_islands_)?
            accurate_hops_:speculative_hops_;
        auto& stamps=(&graph==&accurate_islands_)?
            accurate_hint_stamps_:speculative_hint_stamps_;
        fast.resize(actors_.size(),invalid);
        fast_edge.resize(actors_.size(),invalid_edge);
        hops.resize(actors_.size(),0);
        stamps.resize(actors_.size(),0);
        graph.node_island.resize(actors_.size(),Graph::invalid);
        graph.node_next.resize(actors_.size(),invalid);
        graph.node_prev.resize(actors_.size(),invalid);
        graph.node_static_touches.resize(actors_.size(),0);
        graph.active_reference_counts.resize(actors_.size(),0);
        graph.activating_seen.resize(actors_.size(),0);
        for(std::size_t i=0;i<actors_.size();++i) {
            const BodyId id=static_cast<BodyId>(i);
            if(!dynamic_node(id) ||
               graph.node_island[id]!=Graph::invalid)continue;
            Graph::IslandId handle;
            if(!graph.free_handles.empty()) {
                handle=graph.free_handles.back();
                graph.free_handles.pop_back();
                graph.islands[handle]={id,id,1,true};
            } else {
                if(graph.islands.size()>=Graph::invalid)
                    throw std::overflow_error("Source island handles exhausted");
                handle=static_cast<Graph::IslandId>(graph.islands.size());
                graph.islands.push_back({id,id,1,true});
            }
            graph.node_island[id]=handle;
            graph.node_next[id]=invalid;
            graph.node_prev[id]=invalid;
            // Source addNode(isActive=true) queues the admitted rigid
            // body for its graph's active-node publication. Admission
            // already traverses actor IDs; avoid another whole-scene
            // pass in every subsequent solver frame.
            if(!actors_[id].sleeping)
                graph.activating_nodes.push_back(id);
        }
    };

    // A source island keeps its handle when its original root remains in
    // the surviving component. If a lost bridge splits one component,
    // only the new component needs a handle; unrelated live handles
    // cannot be renumbered. This bounded whole-graph reconstruction is
    // retained for actor admission/retirement only. Ordinary lost-edge
    // splits now detach source-visited components locally instead of
    // taking this O(V+E) path; quiet frames never take either path.
    const auto rebuild_graph=[&](Graph& graph,bool accurate) {
        Graph next;
        next.islands.resize(graph.islands.size());
        next.node_island.assign(actors_.size(),Graph::invalid);
        next.node_next.assign(actors_.size(),invalid);
        next.node_prev.assign(actors_.size(),invalid);
        next.node_static_touches.assign(actors_.size(),0);
        // Source-owned active-index slots and third-pass output survive
        // on the authoritative graph. Rebuild prepares enough capacity
        // for no-throw deactivation after a geometry-only publication.
        next.active_node_index.assign(actors_.size(),invalid_instance);
        next.active_nodes.reserve(actors_.size());
        next.active_islands.reserve(actors_.size());
        next.activating_nodes=graph.activating_nodes;
        next.activating_nodes.reserve(actors_.size());
        next.activating_seen.assign(actors_.size(),0);
        next.active_island_seen.resize(next.islands.size(),0);
        next.nodes_to_deactivate.reserve(actors_.size());
        // Full admission/retirement rebuilds replace graph handles but
        // do not recreate contact managers or reset their active flags.
        next.active_edge_count=graph.active_edge_count;
        // The original EdgeIndex slots, not reconstructed island
        // membership, determine each node's active references.
        next.active_reference_counts=graph.active_reference_counts;
        next.active_reference_counts.resize(actors_.size(),0);
        next.activated_edges.reserve(contact_edges_.size());
        next.deactivating_edges.reserve(contact_edges_.size());
        // Keep the existing active-node permutation and the original
        // prefix through actor admission, retirement and component
        // splits. Source addNode does not reset mActiveNodes.
        for(std::size_t i=0;i<graph.active_nodes.size();++i) {
            const BodyId id=graph.active_nodes[i];
            if(!dynamic_node(id))continue;
            next.active_node_index[id]=next.active_nodes.size();
            next.active_nodes.push_back(id);
            if(i<graph.initial_active_node_count)
                ++next.initial_active_node_count;
        }
        next.free_handles=graph.free_handles;
        // PxsIslandSim::processLostEdges selects roots from the
        // mDirtyMap's node-index iteration when findRoute fails.
        // This bitmap guides dirty-root selection when actor admission
        // or retirement forces full membership reconciliation; ordinary
        // lost-edge splits use the local source visited-node chain.
        std::vector<std::uint8_t> dirty_member(actors_.size(),0);
        for(BodyId id:(accurate?accurate_dirty_nodes_:
                                 speculative_dirty_nodes_))
            if(id<dirty_member.size())dirty_member[id]=1;
        std::vector<std::uint32_t> component(actors_.size(),0);
        std::vector<BodyId> members;
        members.reserve(actors_.size());
        // Source splitEdges consumes mVisitedNodes in discovery order,
        // not global BodyId or Gu pair order. Build that visitation from
        // the graph's dirty nodes and existing intrusive adjacency once
        // the replacement components have been identified.
        std::vector<BodyId> split_visit_nodes;
        if(accurate)split_visit_nodes.reserve(actors_.size());
        const auto& first=accurate?accurate_node_first_:
                                   speculative_node_first_;
        std::uint32_t component_number=0;
        std::size_t islands=0;
        for(std::size_t index=0;index<actors_.size();++index) {
            const BodyId start=static_cast<BodyId>(index);
            if(!dynamic_node(start) || component[start])continue;
            ++component_number;
            members.clear();
            members.push_back(start);
            component[start]=component_number;
            std::size_t static_touches=0;
            for(std::size_t cursor=0;cursor<members.size();++cursor) {
                const BodyId from=members[cursor];
                std::size_t node_static_touches=0;
                for(std::size_t i=first[from];i!=invalid_instance;) {
                    const auto& edge=contact_edges_[i/2];
                    const auto& link=(accurate?edge.accurate:
                                      edge.speculative)[i&1];
                    const BodyId to=(i&1)?edge.pair.a:edge.pair.b;
                    i=link.next;
                    if(!dynamic_node(to)) {
                        // In each graph a static contact manager has one
                        // intrusive instance on its dynamic endpoint.
                        if(actors_[to].alive &&
                           actors_[to].kind!=Kind::DynamicBox)
                            ++node_static_touches;
                        continue;
                    }
                    if(component[to])continue;
                    component[to]=component_number;
                    members.push_back(to);
                }
                if(node_static_touches>
                    std::numeric_limits<std::uint32_t>::max())
                    throw std::overflow_error(
                        "PhysX island node static-touch count overflow");
                next.node_static_touches[from]=
                    static_cast<std::uint32_t>(node_static_touches);
                static_touches+=node_static_touches;
            }
            Graph::IslandId handle=Graph::invalid;
            // Preserve the winning pre-split source handle when its root
            // survives this component. Found-edge merge sizes and tie
            // ordering were already applied to graph before this scan.
            // An orphaned old-root component needs a new handle.
            for(BodyId member:members) {
                if(member>=graph.node_island.size())continue;
                const auto prior=graph.node_island[member];
                if(prior>=graph.islands.size() ||
                   !graph.islands[prior].live ||
                   next.islands[prior].live)continue;
                const BodyId root=graph.islands[prior].root;
                if(root<component.size() &&
                   component[root]==component_number &&
                   (handle==Graph::invalid ||
                    graph.islands[prior].count>
                        graph.islands[handle].count ||
                    (graph.islands[prior].count==
                         graph.islands[handle].count && prior>handle)))
                    handle=prior;
            }
            // A retired root has no surviving root component. Keep its
            // handle on the first surviving component where possible.
            if(handle==Graph::invalid) {
                for(BodyId member:members) {
                    if(member>=graph.node_island.size())continue;
                    const auto prior=graph.node_island[member];
                    if(prior>=graph.islands.size() ||
                       !graph.islands[prior].live ||
                       next.islands[prior].live)continue;
                    const BodyId root=graph.islands[prior].root;
                    if(root>=component.size() ||
                       !dynamic_node(root)) {
                        handle=prior;
                        break;
                    }
                }
            }
            if(handle==Graph::invalid) {
                if(!next.free_handles.empty()) {
                    handle=next.free_handles.back();
                    next.free_handles.pop_back();
                } else {
                    if(next.islands.size()>=Graph::invalid)
                        throw std::overflow_error("Source island handles exhausted");
                    handle=static_cast<Graph::IslandId>(next.islands.size());
                    next.islands.push_back({});
                }
            }
            BodyId root=members.front();
            bool retained_root=false;
            if(handle<graph.islands.size()) {
                const BodyId oldroot=graph.islands[handle].root;
                if(oldroot<component.size() &&
                   component[oldroot]==component_number) {
                    root=oldroot;
                    retained_root=true;
                }
            }
            if(!retained_root) {
                // A component that cannot reach the original root
                // takes the *dirty* route-start node as its new root
                // (mVisitedNodes[0]), not its smallest BodyId. Dirty
                // bitmap iteration in PhysX is ascending node index.
                bool found_dirty=false;
                for(BodyId node:members)
                    if(dirty_member[node] &&
                       (!found_dirty || node<root)) {
                        root=node;
                        found_dirty=true;
                    }
            }
            auto& island=next.islands[handle];
            island.root=root;
            island.tail=root;
            island.count=members.size();
            island.static_touches=static_touches;
            island.live=true;
            next.node_island[root]=handle;
            for(BodyId member:members) {
                if(member==root)continue;
                next.node_island[member]=handle;
                next.node_prev[member]=island.tail;
                next.node_next[island.tail]=member;
                island.tail=member;
            }
            ++islands;
        }
        next.free_handles.clear();
        for(std::size_t i=0;i<next.islands.size();++i)
            if(!next.islands[i].live)
                next.free_handles.push_back(static_cast<Graph::IslandId>(i));
        // IslandSim::addNode leaves existing mFastRoute and mHopCounts
        // intact. A full membership reconstruction must likewise
        // preserve hints for nodes with the same source island root.
        // Only a split or retired root invalidates a cached root route;
        // removed individual edges are already checked on every use.
        auto& stamps=accurate?accurate_hint_stamps_:
                              speculative_hint_stamps_;
        for(std::size_t i=0;i<actors_.size();++i) {
            const BodyId node=static_cast<BodyId>(i);
            if(!dynamic_node(node) ||
               node>=graph.node_island.size() ||
               graph.node_island[node]==Graph::invalid ||
               next.node_island[node]==Graph::invalid)
                continue;
            const BodyId previous_root=
                graph.islands[graph.node_island[node]].root;
            const BodyId current_root=
                next.islands[next.node_island[node]].root;
            if(previous_root!=current_root && stamps[node]!=0) {
                step_checkpoint_.save_hint(*this,node,accurate);
                stamps[node]=0;
            }
        }
        // Preserve the source active-island permutation across a
        // full actor admission/retirement rebuild for surviving handles.
        // Deleted handles and newly created components reconcile from
        // the active-node list at the source wake/third-pass handoff.
        for(const Graph::IslandId id:graph.active_islands)
            if(id<next.islands.size() && next.islands[id].live &&
               graph.islands[id].root==next.islands[id].root &&
               next.islands[id].active_index==invalid_instance) {
                next.islands[id].active_index=next.active_islands.size();
                next.active_islands.push_back(id);
            }
        next.active_island_seen.resize(next.islands.size(),0);
        if(accurate) {
            // Original processLostEdges selects the first dirty node
            // from mDirtyMap (ascending node index), then findRoute
            // discovers nodes through the source EdgeInstance adjacency
            // with a min-priority queue of cached mHopCounts estimates.
            // mVisitedNodes is recorded when each node enters that
            // queue, NOT when the queued node is subsequently popped.
            // This distinction controls splitEdges' manager permutation.
            // Reuse the component map above, retaining the bounded
            // node/handle reconstruction only for actor-lifecycle
            // reconciliation. The ordinary lost-edge path is now
            // source-local and never calls this bounded reconstruction.
            std::vector<BodyId> first_node(component_number+1,invalid);
            std::vector<BodyId> first_dirty(component_number+1,invalid);
            for(BodyId node=0;node<actors_.size();++node)
                if(component[node] && first_node[component[node]]==invalid)
                    first_node[component[node]]=node;
            for(BodyId node:accurate_dirty_nodes_)
                if(node<component.size() && component[node] &&
                   (first_dirty[component[node]]==invalid ||
                    node<first_dirty[component[node]]))
                    first_dirty[component[node]]=node;
            struct RouteEntry {
                BodyId node;
                std::uint32_t hops;
            };
            const auto higher=[](const RouteEntry& a,const RouteEntry& b) {
                return a.hops<b.hops;
            };
            std::vector<RouteEntry> pending;
            pending.reserve(actors_.size());
            std::vector<std::uint32_t> seen(actors_.size(),0);
            // Original mDirtyMap enumerates *node indices*, and each
            // failed findRoute immediately removes the orphan's edges
            // from the same old island array. Visiting components by
            // lowest BodyId instead would change replaceWithLast order
            // when several orphan components split in one pass.
            std::vector<std::uint32_t> visit_components;
            visit_components.reserve(component_number);
            for(std::uint32_t comp=1;comp<=component_number;++comp)
                visit_components.push_back(comp);
            std::sort(visit_components.begin(),visit_components.end(),
                [&](std::uint32_t a,std::uint32_t b) {
                    const bool ad=first_dirty[a]!=invalid;
                    const bool bd=first_dirty[b]!=invalid;
                    if(ad!=bd)return ad;
                    const BodyId an=ad?first_dirty[a]:first_node[a];
                    const BodyId bn=bd?first_dirty[b]:first_node[b];
                    return an!=bn?an<bn:a<b;
                });
            for(std::uint32_t comp:visit_components) {
                const BodyId start=first_dirty[comp]!=invalid
                    ? first_dirty[comp]:first_node[comp];
                if(start==invalid)continue;
                pending.clear();
                seen[start]=comp;
                split_visit_nodes.push_back(start);
                const auto estimate=[&](BodyId node) {
                    return node<accurate_hops_.size()
                        ? accurate_hops_[node]
                        : std::numeric_limits<std::uint32_t>::max();
                };
                source_hop_push(pending,RouteEntry{start,estimate(start)},
                                higher);
                while(!pending.empty()) {
                    const BodyId from=source_hop_pop(pending,higher).node;
                    for(std::size_t i=accurate_node_first_[from];
                        i!=invalid_instance;) {
                        const ContactEdge& edge=contact_edges_[i/2];
                        const auto& link=edge.accurate[i&1];
                        const BodyId to=(i&1)?edge.pair.a:edge.pair.b;
                        i=link.next;
                        if(to>=component.size() || component[to]!=comp ||
                           seen[to]==comp)continue;
                        seen[to]=comp;
                        split_visit_nodes.push_back(to);
                        source_hop_push(pending,RouteEntry{to,estimate(to)},
                                        higher);
                    }
                }
            }
            // PxsIslandSim::processLostEdges walks visited component
            // nodes, removing each incident manager only once (at its
            // first dynamic endpoint, or its sole static-touch endpoint).
            // Every removal uses removeEdgeFromIsland's replaceWithLast:
            // the original/root island's remaining edges do NOT retain
            // their pre-split order when a middle edge migrates.
            //
            // This actor-lifecycle fallback preserves source dirty-node
            // seed, hop-priority queue and enqueue-order mVisitedNodes
            // migration, but still reconstructs the full node roster.
            // Lost-contact splits use the local source node-chain path.
            struct MigratedManager {
                EdgeIndex edge;
                Graph::IslandId owner;
            };
            std::vector<MigratedManager> migrated;
            migrated.reserve(manifolds_.size());
            const auto prior_owner_retained=[&](BodyId node) {
                if(!dynamic_node(node))return false;
                const auto old=graph.node_island[node];
                const auto fresh=next.node_island[node];
                return old!=Graph::invalid && old==fresh &&
                    old<graph.islands.size() &&
                    old<next.islands.size() &&
                    graph.islands[old].live &&
                    next.islands[old].live &&
                    graph.islands[old].root==next.islands[old].root;
            };
            for(BodyId node:split_visit_nodes) {
                if(prior_owner_retained(node))continue;
                for(std::size_t i=accurate_node_first_[node];
                    i!=invalid_instance;) {
                    const EdgeIndex edge_index=static_cast<EdgeIndex>(i/2);
                    const auto& edge=contact_edges_[edge_index];
                    const auto& link=edge.accurate[i&1];
                    i=link.next;
                    if(!edge.connected)continue;
                    // The first endpoint owns dynamic/dynamic entries.
                    // A world/static first endpoint cannot be visited,
                    // so admit that entry from its only dynamic endpoint.
                    const BodyId first_node=edge.pair.a;
                    if((node!=first_node && dynamic_node(first_node)) ||
                       (node!=edge.pair.a && node!=edge.pair.b))
                        continue;
                    const auto target=next.node_island[node];
                    if(target==Graph::invalid)
                        throw std::logic_error(
                            "Source split manager lost its destination island");
                    // Every migrated manager belongs to precisely one
                    // old accurate array and is visited once.
                    remove_accurate_island_edge(edge_index);
                    migrated.push_back({edge_index,target});
                }
            }
            // Old (retained-root) island arrays have now been compacted
            // with source replaceWithLast semantics. Copy their remaining
            // order before publishing replacement component handles.
            std::vector<MigratedManager> retained;
            retained.reserve(manifolds_.size());
            for(const auto& island:graph.islands)
                for(EdgeIndex id=island.first_contact;id!=invalid_edge;
                    id=contact_edges_[id].island_next) {
                    const Pair pair=contact_edges_[id].pair;
                    const BodyId node=dynamic_node(pair.a)?pair.a:pair.b;
                    if(!dynamic_node(node) || !contact_edges_[id].connected)
                        continue;
                    const auto owner=next.node_island[node];
                    if(owner==Graph::invalid)
                        throw std::logic_error(
                            "Source retained manager lost its destination island");
                    retained.push_back({id,owner});
                }
            graph=std::move(next);
            // All links are journaled on the existing source EdgeIndex
            // slots. Rebuild first from the compacted retained arrays,
            // then append splitEdges in their collected node order.
            for(const auto& item:retained) {
                step_checkpoint_.save_edge(*this,item.edge);
                contact_edges_[item.edge].island_prev=invalid_edge;
                contact_edges_[item.edge].island_next=invalid_edge;
            }
            for(const auto& item:retained)
                append_accurate_island_edge(item.edge,item.owner);
            for(const auto& item:migrated)
                append_accurate_island_edge(item.edge,item.owner);
        } else graph=std::move(next);
        return islands;
    };


    bool order_changed=rebuild || local_split;
    if(local_split) {
        // Only dirty, disconnected source components were migrated.
        // Existing actor and indexed-manager streams remain authoritative.
    } else if(rebuild) {
        step_checkpoint_.save_graph_ownership(*this);
        ++island_full_rebuild_count_;
        if(island_topology_dirty_ && !repair_lost_edges) {
            // Existing source islands still own pre-loss membership.
            // Admit newly added nodes, then merge *all* found edges
            // against accumulated island sizes, prior to any split.
            admit_nodes(speculative_islands_);
            admit_nodes(accurate_islands_);
            for(Pair pair:found_)
                merge(speculative_islands_,pair);
            for(Pair pair:found_contacts_) {
                merge(accurate_islands_,pair);
                const auto id=contact_edge_index(pair);
                const BodyId node=dynamic_node(pair.a)?pair.a:pair.b;
                if(id!=invalid_edge && dynamic_node(node))
                    append_accurate_island_edge(
                        id,accurate_islands_.node_island[node]);
            }
        }
        // Reconcile lost edges and actor retirement with the authoritative
        // intrusive graph. Already-selected source handles survive on the
        // connected component that retains their original root.
        speculative_island_count_=rebuild_graph(speculative_islands_,false);
        accurate_island_count_=rebuild_graph(accurate_islands_,true);
        accurate_island_nodes_.clear();
        for(std::size_t i=0;i<actors_.size();++i)
            if(dynamic_node(static_cast<BodyId>(i)))
                accurate_island_nodes_.push_back(static_cast<BodyId>(i));
    } else {
        for(Pair pair:found_)
            if(merge(speculative_islands_,pair))
                --speculative_island_count_;
        for(Pair pair:found_contacts_) {
            if(merge(accurate_islands_,pair)) {
                --accurate_island_count_;
                order_changed=true;
            }
            const auto id=contact_edge_index(pair);
            const BodyId node=dynamic_node(pair.a)?pair.a:pair.b;
            if(id!=invalid_edge && dynamic_node(node))
                append_accurate_island_edge(
                    id,accurate_islands_.node_island[node]);
        }
    }
    if(order_changed) {
        // Dy::DynamicsContext::iterateIslandsNodes consumes contiguous
        // *accurate* island batches, driven directly by source island
        // handles, not by a second union/find component representation.
        std::sort(accurate_island_nodes_.begin(),
                  accurate_island_nodes_.end(),
            [&](BodyId a,BodyId b) {
                const auto ia=accurate_islands_.node_island[a];
                const auto ib=accurate_islands_.node_island[b];
                return ia<ib || (ia==ib && a<b);
            });
        accurate_island_offsets_.clear();
        accurate_island_offsets_.push_back(0);
        for(std::size_t i=1;i<accurate_island_nodes_.size();++i)
            if(accurate_islands_.node_island[accurate_island_nodes_[i-1]]!=
               accurate_islands_.node_island[accurate_island_nodes_[i]])
                accurate_island_offsets_.push_back(i);
        if(!accurate_island_nodes_.empty())
            accurate_island_offsets_.push_back(accurate_island_nodes_.size());
    }
    // New source split handles may extend the visited-stamp arrays.
    // Allocate here, inside the failure-atomic topology transaction,
    // before no-throw wake/sleep publication updates mActiveIslands.
    speculative_islands_.active_island_seen.resize(
        speculative_islands_.islands.size(),0);
    speculative_pre_retirement_index_.resize(
        speculative_islands_.islands.size());
    accurate_islands_.active_island_seen.resize(
        accurate_islands_.islands.size(),0);
    speculative_islands_.active_islands.reserve(actors_.size());
    accurate_islands_.active_islands.reserve(actors_.size());
    speculative_islands_.activating_nodes.reserve(actors_.size());
    accurate_islands_.activating_nodes.reserve(actors_.size());
    speculative_islands_.activating_seen.resize(actors_.size(),0);
    accurate_islands_.activating_seen.resize(actors_.size(),0);
    // Root changes in rebuild_graph invalidate only affected node hints.
    // Ordinary merges and independent actor admissions retain live routes
    // in both graphs; each attempted route validates its full edge chain.
    island_topology_dirty_=false;
    ++island_topology_revision_;
}
// PxsIslandSim.cpp::markEdgeActive/removeEdgeFromActivatingList:
// the shared EdgeIndex has independent activity in the two IslandSims.
void Scene::activate_contact_edge(EdgeIndex id, bool accurate) noexcept {
    auto& edge=contact_edges_[id];
    if(!edge.live || edge.retiring ||
       (accurate && !edge.connected))return;
    auto& state=accurate?edge.accurate_state:edge.speculative_state;
    if(state&1u)return; // Edge::eACTIVE
    step_checkpoint_.save_activity(*this,id);
    if(actors_[edge.pair.a].kind==Kind::DynamicBox &&
       actors_[edge.pair.b].kind==Kind::DynamicBox) {
        step_checkpoint_.save_reference(*this,edge.pair.a);
        step_checkpoint_.save_reference(*this,edge.pair.b);
    }
    auto& graph=accurate?accurate_islands_:speculative_islands_;
    state|=3u; // Edge::eACTIVE | Edge::eACTIVATING
    graph.activated_edges.push_back(id);
    ++graph.active_edge_count;
    // Source IslandSim::markEdgeActive counts references only for
    // edges with two valid node endpoints. Static world edges do not
    // contribute to either rigid node's active reference count.
    if(actors_[edge.pair.a].kind==Kind::DynamicBox &&
       actors_[edge.pair.b].kind==Kind::DynamicBox) {
        ++graph.active_reference_counts[edge.pair.a];
        ++graph.active_reference_counts[edge.pair.b];
    }
}
void Scene::deactivate_contact_edge(EdgeIndex id, bool accurate,
                                    bool publish) noexcept {
    auto& edge=contact_edges_[id];
    auto& state=accurate?edge.accurate_state:edge.speculative_state;
    if(!(state&3u))return;
    step_checkpoint_.save_activity(*this,id);
    if(actors_[edge.pair.a].kind==Kind::DynamicBox &&
       actors_[edge.pair.b].kind==Kind::DynamicBox) {
        step_checkpoint_.save_reference(*this,edge.pair.a);
        step_checkpoint_.save_reference(*this,edge.pair.b);
    }
    auto& graph=accurate?accurate_islands_:speculative_islands_;
    if(state&2u) {
        auto& pending=graph.activated_edges;
        for(std::size_t i=0;i<pending.size();++i)
            if(pending[i]==id) {
                pending[i]=pending.back();
                pending.pop_back();
                break;
            }
    }
    const bool was_active=(state&1u)!=0;
    state=0;
    if(was_active) {
        --graph.active_edge_count;
        // Source removeEdgeFromActivatingList releases both valid
        // endpoint references on every real active-to-inactive edge
        // transition, including retirement of an actor no longer alive.
        if(actors_[edge.pair.a].kind==Kind::DynamicBox &&
           actors_[edge.pair.b].kind==Kind::DynamicBox) {
            --graph.active_reference_counts[edge.pair.a];
            --graph.active_reference_counts[edge.pair.b];
        }
        if(publish)graph.deactivating_edges.push_back(id);
    }
}
void Scene::clear_activated_edges(bool checkpointed) noexcept {
    // IslandSim::wakeIslands retires eACTIVATING from the previous
    // transition list, preserving persistent eACTIVE and active count.
    for(const bool accurate:{false,true}) {
        auto& graph=accurate?accurate_islands_:speculative_islands_;
        // Checkpointed publications swapped the previous output into
        // the rollback owner. Traverse it without changing its storage:
        // a rejected publication must restore the original span.
        auto& prior=checkpointed ?
            (accurate?step_checkpoint_.accurate_islands.activated_edges:
                      step_checkpoint_.speculative_islands.activated_edges):
            graph.activated_edges;
        for(EdgeIndex id:prior) {
            auto& state=accurate?contact_edges_[id].accurate_state:
                                 contact_edges_[id].speculative_state;
            if(state&2u) {
                step_checkpoint_.save_activity(*this,id);
                state&=static_cast<std::uint8_t>(~2u);
            }
        }
        if(!checkpointed)prior.clear();
    }
}
void Scene::activate_node_contact_edges(BodyId node) noexcept {
    // activateNodeInternal walks existing source intrusive edges.
    for(const bool accurate:{false,true}) {
        const auto& heads=accurate?accurate_node_first_:speculative_node_first_;
        for(std::size_t instance=heads[node];
            instance!=invalid_instance;) {
            const EdgeIndex id=static_cast<EdgeIndex>(instance/2);
            const auto& edge=contact_edges_[id];
            const auto& link=(accurate?edge.accurate:
                                       edge.speculative)[instance&1u];
            instance=link.next;
            activate_contact_edge(id,accurate);
        }
    }
}
void Scene::queue_node_activation(BodyId id) {
    // Reserve both owners before appending either: allocation failure
    // cannot leave a half-published authored wake or modified body.
    auto& speculative=speculative_islands_.activating_nodes;
    auto& accurate=accurate_islands_.activating_nodes;
    speculative.reserve(speculative.size()+1);
    accurate.reserve(accurate.size()+1);
    speculative.push_back(id);
    accurate.push_back(id);
}
void Scene::activate_published_islands() noexcept {
    // Quiet publications without source wake/connection events never
    // visit graph islands or allocate per-node activation records.
    if(!authored_activation_pending_ && found_contacts_.empty() &&
       found_.empty())return;
    const auto dynamic_node=[&](BodyId id) {
        return actors_[id].alive && actors_[id].kind==Kind::DynamicBox;
    };
    // A single generation per publication deduplicates each graph's
    // mActivatingNodes independently. This bounds the event list by
    // actor count, which was reserved before noexcept publication.
    const auto begin=[&](IslandGraph& graph) {
        if(++graph.activating_generation==0) {
            std::fill(graph.activating_seen.begin(),
                      graph.activating_seen.end(),0);
            graph.activating_generation=1;
        }
        for(BodyId id:graph.activating_nodes)
            if(id<graph.activating_seen.size())
                graph.activating_seen[id]=graph.activating_generation;
    };
    begin(speculative_islands_);
    begin(accurate_islands_);
    const auto append=[&](IslandGraph& graph,BodyId id) {
        if(!dynamic_node(id))return;
        if(graph.activating_seen[id]==graph.activating_generation)return;
        graph.activating_seen[id]=graph.activating_generation;
        graph.activating_nodes.push_back(id);
    };
    // Source first-pass SAP new-edge events precede second-pass Gu
    // touches. Each graph can receive a different subset and order.
    for(Pair pair:found_) {
        if(!dynamic_node(pair.a) || !dynamic_node(pair.b))continue;
        if(!actors_[pair.a].sleeping || !actors_[pair.b].sleeping) {
            append(speculative_islands_,pair.a);
            append(speculative_islands_,pair.b);
            append(accurate_islands_,pair.a);
            append(accurate_islands_,pair.b);
            step_checkpoint_.save_motion(*this,pair.a);
            step_checkpoint_.save_motion(*this,pair.b);
            actors_[pair.a].activating=true;
            actors_[pair.b].activating=true;
        }
    }
    for(Pair pair:found_contacts_) {
        for(BodyId id:{pair.a,pair.b}) {
            if(!dynamic_node(id))continue;
            append(accurate_islands_,id);
            append(speculative_islands_,id);
            step_checkpoint_.save_motion(*this,id);
            actors_[id].activating=true;
            actors_[id].ready_for_sleep=false;
        }
    }
    constexpr BodyId invalid=std::numeric_limits<BodyId>::max();
    // PxsIslandSim::wakeIslandsInternal consumes mActivatingNodes,
    // not the full island inventory. The graph's existing visit stamps
    // ensure each linked island is propagated once per publication,
    // even when several event nodes share one component.
    auto& graph=speculative_islands_;
    if(++graph.active_island_visit_stamp==0) {
        std::fill(graph.active_island_seen.begin(),
                  graph.active_island_seen.end(),0);
        graph.active_island_visit_stamp=1;
    }
    const auto wake_stamp=graph.active_island_visit_stamp;
    for(BodyId waking:graph.activating_nodes) {
        if(waking>=graph.node_island.size())continue;
        const auto owner=graph.node_island[waking];
        if(owner==IslandGraph::invalid ||
           owner>=graph.islands.size() ||
           !graph.islands[owner].live ||
           graph.active_island_seen[owner]==wake_stamp)continue;
        graph.active_island_seen[owner]=wake_stamp;
        for(BodyId id=graph.islands[owner].root;id!=invalid;
            id=graph.node_next[id]) {
            Actor& body=actors_[id];
            step_checkpoint_.save_motion(*this,id);
            body.activating=false;
            if(body.sleeping) {
                // A speculative-only edge can wake accurate components
                // that have no newly found accurate contact. Propagate
                // this genuine node activation into the accurate
                // IslandSim's independently ordered mActivatingNodes.
                append(accurate_islands_,id);
                body.sleeping=false;
                body.ready_for_sleep=false;
                body.wake_counter=std::max(body.wake_counter,0.4f);
                body.sleep_linear_acc={};
                body.sleep_angular_acc={};
            }
            activate_node_contact_edges(id);
        }
    }
    // addConnectionToGraph activates newly linked edges whenever
    // either endpoint is active, including static-world contacts.
    for(Pair pair:found_) {
        const EdgeIndex id=contact_edge_index(pair);
        if(id!=invalid_edge &&
           ((actors_[pair.a].kind==Kind::DynamicBox &&
             !actors_[pair.a].sleeping) ||
            (actors_[pair.b].kind==Kind::DynamicBox &&
             !actors_[pair.b].sleeping)))
            activate_contact_edge(id,false);
    }
    for(Pair pair:found_contacts_) {
        const EdgeIndex id=contact_edge_index(pair);
        if(id!=invalid_edge &&
           ((actors_[pair.a].kind==Kind::DynamicBox &&
             !actors_[pair.a].sleeping) ||
            (actors_[pair.b].kind==Kind::DynamicBox &&
             !actors_[pair.b].sleeping)))
            activate_contact_edge(id,true);
    }
    authored_activation_pending_=false;
}
void Scene::third_pass_island_gen() {
    // Both original IslandSim::processLostEdges third-pass tasks depend
    // on lost-contact management and join into postThirdPassIslandGen,
    // potentially overlapping the numerical solver. This synchronous
    // source-subset execution is scheduled once those collision products
    // are ready, *not* as a mandatory post-solver serial stage.
    // Rigid contact-manager edge activity is persistent; task
    // continuations and other edge types still need source admission.
    // Rebuild candidates from the first pass must survive until the
    // third-pass deactivation decision, alongside local lost-edge splits.
    update_island_connectivity(true);
    // Keep retired contact-manager handles allocated until the source
    // lost-contact/third-pass join reaches its consumer-completion edge.
    // The frame's Dy and Sc stages can still hold those manager handles
    // after island route repair. Do not return them to the free list here.
    // The source dirty map is consumed by the third-pass repair. No
    // duplicate route work should spill into an unchanged next frame.
    for(BodyId node:speculative_dirty_nodes_)
        speculative_dirty_map_[node]=0;
    for(BodyId node:accurate_dirty_nodes_)
        accurate_dirty_map_[node]=0;
    speculative_dirty_nodes_.clear();
    accurate_dirty_nodes_.clear();
}
// PxsIslandSim.cpp::IslandSim::processLostEdges(allowDeactivation) tests
// *previously published* Node::isReadyForSleeping flags in its third pass.
// Dy::sleepCheck updates those flags only after the current solver step,
// so a newly ready island cannot sleep until the next island pass.
// The source may run the two tasks concurrently and then repair a body
// integrated after deactivation in Sc::Scene::afterIntegration. Our serial
// CPU lowering establishes the same observable dependency by deactivating
// before the solver: no concurrently integrated pose needs restoration.
// Both accurate and speculative contact graphs retain their existing
// persistent handles and node membership while bodies are inactive.
void Scene::deactivate_ready_islands() noexcept {
    constexpr BodyId invalid=std::numeric_limits<BodyId>::max();
    // PxsSimpleIslandManager::thirdPassIslandGen clears the previous
    // accurate rigid deactivations. A now-ready node belongs to this
    // stage's output only when its accurate island actually deactivates.
    // Rebuild-time capacity was reserved before collision commit.
    accurate_islands_.nodes_to_deactivate.clear();
    speculative_islands_.deactivating_edges.clear();
    accurate_islands_.deactivating_edges.clear();
    // PhysX uses accurate-island deactivations for rigid-body state in
    // ScPipeline::afterIntegration. The speculative third-pass list
    // controls contact interaction activity, not rigid sleep admission.
    // PxsIslandSim::processLostEdges walks mActiveIslands in reverse,
    // after wakeIslands has published the active-index changes. A long-
    // sleeping scene must not revisit every dormant historical handle
    // on each solver frame, nor reverse the source deactivation order.
    for(std::size_t index=accurate_islands_.active_islands.size();
        index>0;--index) {
        const auto handle=accurate_islands_.active_islands[index-1];
        if(handle>=accurate_islands_.islands.size())continue;
        const auto& island=accurate_islands_.islands[handle];
        if(!island.live)continue;
        bool ready=true;
        bool has_active_node=false;
        for(BodyId node=island.root;node!=invalid;
            node=accurate_islands_.node_next[node]) {
            const Actor& actor=actors_[node];
            if(!actor.ready_for_sleep) {
                ready=false;
                break;
            }
            has_active_node |= !actor.sleeping;
        }
        if(!ready || !has_active_node)continue;
        for(BodyId node=island.root;node!=invalid;
            node=accurate_islands_.node_next[node]) {
            Actor& actor=actors_[node];
            if(actor.sleeping)continue;
            step_checkpoint_.save_motion(*this,node);
            accurate_islands_.nodes_to_deactivate.push_back(node);
            // ScBodySim::deactivate clears velocities and pending motion.
            // The surviving Gu contact managers are not retired here.
            actor.sleeping=true;
            actor.wake_counter=0.0f;
            actor.velocity={};
            actor.angular_velocity={};
            // ScBodySim::deactivate::setForcesToDefaults and
            // BodySim::putToSleep::simStateClearVelMod retire BOTH the
            // per-step velocity changes and per-second accelerations.
            // Leaving an old linear or angular force on an inactive
            // body would replay it after IslandSim wakes that body.
            actor.pending_linear_velocity_delta={};
            actor.pending_angular_velocity_delta={};
            actor.pending_linear_acceleration={};
            actor.pending_angular_acceleration={};
            actor.sleep_linear_acc={};
            actor.sleep_angular_acc={};
            // Accurate IslandSim::deactivateNodeInternal visits only
            // this newly inactive rigid node. Emit its manager when
            // the second endpoint is already inactive or the world.
            for(std::size_t i=accurate_node_first_[node];
                i!=invalid_instance;) {
                const auto& edge=contact_edges_[i/2];
                const auto& link=edge.accurate[i&1];
                const BodyId other=(i&1)?edge.pair.a:edge.pair.b;
                const EdgeIndex edge_id=static_cast<EdgeIndex>(i/2);
                i=link.next;
                if(edge.live && !edge.retiring && edge.connected &&
                   (actors_[other].kind!=Kind::DynamicBox ||
                    actors_[other].sleeping))
                    deactivate_contact_edge(edge_id,true,true);
            }
        }
    }
    if(accurate_islands_.nodes_to_deactivate.empty() &&
       speculative_split_sleep_candidates_.empty())return;
    // Speculative IslandSim can join accurate islands via SAP-only
    // contacts. It must NOT emit sleep edges while any member of its
    // larger speculative island remains active. Process only candidate
    // islands containing a newly inactive rigid actor, once per island.
    // A stamped root deduplicates candidates without a global scan.
    if(++speculative_sleep_visit_stamp_==0) {
        std::fill(speculative_sleep_seen_.begin(),
                  speculative_sleep_seen_.end(),0);
        speculative_sleep_visit_stamp_=1;
    }
    const auto stamp=speculative_sleep_visit_stamp_;
    speculative_sleep_order_.clear();
    const auto collect_candidate=[&](BodyId candidate) {
        if(candidate>=speculative_islands_.node_island.size())return;
        const auto handle=speculative_islands_.node_island[candidate];
        if(handle==IslandGraph::invalid ||
           handle>=speculative_islands_.islands.size())return;
        const auto& island=speculative_islands_.islands[handle];
        if(!island.live || island.root==invalid)return;
        if(speculative_sleep_seen_[island.root]==stamp)return;
        speculative_sleep_seen_[island.root]=stamp;
        speculative_sleep_order_.push_back({
            handle,speculative_sleep_order_.size()});
    };
    // Accurate deactivations and speculative split/retirement produce
    // independent candidate streams. Do not publish in their encounter
    // order: PxsIslandSim::processLostEdges iterates each graph's own
    // mActiveIslands backwards. Order only the affected island handles,
    // without scanning all dormant or still-active speculative islands.
    for(BodyId candidate:accurate_islands_.nodes_to_deactivate)
        collect_candidate(candidate);
    for(BodyId candidate:speculative_split_sleep_candidates_)
        collect_candidate(candidate);
    std::sort(speculative_sleep_order_.begin(),
              speculative_sleep_order_.end(),
        [&](const SpeculativeSleepCandidate& a,
            const SpeculativeSleepCandidate& b) {
            const auto source_index=[&](IslandGraph::IslandId handle) {
                const auto& saved=speculative_pre_retirement_index_[handle];
                return saved.generation==speculative_pre_retirement_generation_
                    ? saved.index
                    : speculative_islands_.islands[handle].active_index;
            };
            const auto ai=source_index(a.handle);
            const auto bi=source_index(b.handle);
            // A full actor-retirement rebuild can leave a newly
            // detached, already-sleeping component outside the current
            // active list; its retained loss candidate must still
            // retire its active edges. Existing source active handles
            // always precede these exceptional detached candidates.
            if(ai==invalid_instance || bi==invalid_instance) {
                if(ai==invalid_instance && bi==invalid_instance)
                    return a.discovery<b.discovery;
                return bi==invalid_instance;
            }
            return ai>bi;
        });
    for(const auto candidate:speculative_sleep_order_) {
        const auto& island=speculative_islands_.islands[candidate.handle];
        const BodyId root=island.root;
        bool all_sleeping=true;
        for(BodyId node=root;node!=invalid;
            node=speculative_islands_.node_next[node]) {
            if(!actors_[node].sleeping) {
                all_sleeping=false;
                break;
            }
        }
        if(!all_sleeping)continue;
        // Source deactivateIsland walks the original intrusive node
        // chain; deactivateNodeInternal publishes each active edge on
        // visiting its last inactive endpoint. Reuse the same stamp as
        // the candidate deduplication for the visited node prefix.
        for(BodyId node=root;node!=invalid;
            node=speculative_islands_.node_next[node]) {
            speculative_sleep_seen_[node]=stamp;
            for(std::size_t i=speculative_node_first_[node];
                i!=invalid_instance;) {
                const auto& edge=contact_edges_[i/2];
                const auto& link=edge.speculative[i&1];
                const BodyId other=(i&1)?edge.pair.a:edge.pair.b;
                const EdgeIndex edge_id=static_cast<EdgeIndex>(i/2);
                i=link.next;
                if(edge.live && !edge.retiring &&
                   (actors_[other].kind!=Kind::DynamicBox ||
                    speculative_sleep_seen_[other]==stamp))
                    deactivate_contact_edge(edge_id,false,true);
            }
        }
    }
}
// ScPipeline.cpp::activateEdgesInternal(eCONTACT_MANAGER) consumes the
// speculative graph's mActivatedEdges, then postThirdPassIslandGen
// consumes mDeactivatingEdges after both lost-edge tasks have joined.
// The admitted rigid contact manager is already the single EdgeIndex
// owner, so retain Sc::InteractionFlag::eIS_ACTIVE in that same slot.
// This serial execution preserves activation-before-deactivation order
// without inventing a second interaction registry or task dispatcher.
void Scene::consume_contact_interaction_activity() noexcept {
    for(EdgeIndex id:speculative_islands_.activated_edges) {
        auto& edge=contact_edges_[id];
        if(!edge.live || edge.retiring || edge.interaction_active ||
           !(edge.speculative_state&1u))continue;
        step_checkpoint_.save_activity(*this,id);
        edge.interaction_active=true;
        ++active_contact_interaction_count_;
    }
    // No accurate-island deactivation is consumed here. Sc explicitly
    // uses the speculative graph so an accurate component can sleep
    // while its speculative-only bridge remains interaction-active.
    for(EdgeIndex id:speculative_islands_.deactivating_edges) {
        auto& edge=contact_edges_[id];
        if(!edge.live || edge.retiring || !edge.interaction_active ||
           (edge.speculative_state&1u))continue;
        step_checkpoint_.save_activity(*this,id);
        edge.interaction_active=false;
        --active_contact_interaction_count_;
    }
}
// Source PxsIslandSim::mActiveIslands is distinct from the complete
// island inventory. Build active batch indices from the already-owned
// contiguous accurate island-node sequence after source wake/deactivation.
// An island is active while any member is awake; the later per-node
// sleeping checks still protect a transitional mixed-state island.
// Unlike copying a second body-ID array, only one index per active
// island is retained, with reusable allocation across frames.
void Scene::synchronize_active_island_nodes(
    bool capture_speculative_order) noexcept {
    // The source third pass iterates each IslandSim's active list in
    // reverse. Capture its pre-reconciliation positions before the
    // active-node sweep can drop a detached sleeping split or swap
    // surviving slots. Only live active handles are touched here.
    if(capture_speculative_order) {
        if(++speculative_pre_retirement_generation_==0) {
            for(auto& record:speculative_pre_retirement_index_)
                record.generation=0;
            speculative_pre_retirement_generation_=1;
        }
        for(std::size_t i=0;i<speculative_islands_.active_islands.size();++i) {
            const auto handle=speculative_islands_.active_islands[i];
            auto& record=speculative_pre_retirement_index_[handle];
            record.index=i;
            record.generation=speculative_pre_retirement_generation_;
        }
    }
    // Source PxsIslandSim::markInactive uses swap-last plus a reverse
    // active-node index; markActive appends. Preserve that lifecycle
    // across quiet frames and incremental merges instead of clearing and
    // recreating all active IDs. A component rebuild starts a new source
    // index map in update_island_connectivity's replacement graph.
    //
    // The rebuild reserves all live actor slots before publication.
    // Actor admission cannot reach this path without rebuilding, so
    // neither active-list append nor index lookup allocates here. This
    // matters for the failure-atomic geometry-only commit boundary.
    // Original SimpleIslandManager has two independent IslandSim owners.
    // The currently admitted rigid actors share the Sc sleep state,
    // but each graph retains its own source active-list permutation.
    for(IslandGraph* owner:{&speculative_islands_,&accurate_islands_}) {
    auto& graph=*owner;
    for(std::size_t index=0;index<graph.active_nodes.size();) {
        const BodyId id=graph.active_nodes[index];
        if(actors_[id].alive && !actors_[id].sleeping &&
           graph.node_island[id]!=IslandGraph::invalid) {
            ++index;
            continue;
        }
        // PxsIslandSim::markInactive first relocates an initially active
        // node to the tail of the initial prefix, shrinking that prefix
        // before removing it with the ordinary swap-last operation.
        // This keeps getNodesToActivate() a genuine source suffix when
        // a previously active node sleeps alongside new waking nodes.
        if(index<graph.initial_active_node_count) {
            const auto tail=graph.initial_active_node_count-1;
            const BodyId previous=graph.active_nodes[tail];
            graph.active_nodes[index]=previous;
            graph.active_node_index[previous]=index;
            graph.active_nodes[tail]=id;
            graph.active_node_index[id]=tail;
            --graph.initial_active_node_count;
        }
        // Retire the inactive node at its (possibly relocated) index.
        // The original traversal index now contains the previous
        // initial-prefix tail. It may itself be dormant, so do not
        // advance to the tail: recheck that compacted original slot on
        // the next iteration. This also covers repeated swap-last
        // removals whose replacement came from the activated suffix.
        const std::size_t retired=graph.active_node_index[id];
        const BodyId replacement=graph.active_nodes.back();
        graph.active_nodes[retired]=replacement;
        graph.active_node_index[replacement]=retired;
        graph.active_nodes.pop_back();
        graph.active_node_index[id]=invalid_instance;
    }
    // Source wakeIslands processes mActivatingNodes in event order,
    // independently for each graph. Append the woken island's linked
    // rigid members before adding newly admitted (not awakened) actors.
    // The reverse active-node index suppresses repeated event entries.
    for(BodyId waking:graph.activating_nodes) {
        if(waking>=graph.node_island.size())continue;
        const auto handle=graph.node_island[waking];
        if(handle==IslandGraph::invalid ||
           handle>=graph.islands.size() ||
           !graph.islands[handle].live)continue;
        for(BodyId id=graph.islands[handle].root;
            id!=std::numeric_limits<BodyId>::max();
            id=graph.node_next[id]) {
            if(!actors_[id].alive || actors_[id].sleeping ||
               graph.active_node_index[id]!=invalid_instance)continue;
            graph.active_node_index[id]=graph.active_nodes.size();
            graph.active_nodes.push_back(id);
        }
    }
    graph.activating_nodes.clear();
    // Node admission, authored wakes, SAP edges, Gu contacts and
    // speculative wake propagation all enqueue their source node once.
    // No need to scan the complete accurate island-node inventory
    // for previously dormant, unmodified bodies on every PGS step.
    // Original IslandSim::mActiveIslands / Island::mActiveIndex use
    // replace-with-last on a dormant island and append on activation.
    // Reconcile only the already-owned active rigid-node stream, never
    // the full set of dormant island handles or candidate edges.
    if(++graph.active_island_visit_stamp==0) {
        std::fill(graph.active_island_seen.begin(),
                  graph.active_island_seen.end(),0);
        graph.active_island_visit_stamp=1;
    }
    const auto generation=graph.active_island_visit_stamp;
    for(BodyId id:graph.active_nodes) {
        const auto owner_id=graph.node_island[id];
        if(owner_id!=IslandGraph::invalid)
            graph.active_island_seen[owner_id]=generation;
    }
    for(std::size_t index=0;index<graph.active_islands.size();) {
        const auto id=graph.active_islands[index];
        if(id<graph.islands.size() && graph.islands[id].live &&
           graph.active_island_seen[id]==generation) {
            graph.islands[id].active_index=index;
            ++index;
            continue;
        }
        if(id<graph.islands.size())
            graph.islands[id].active_index=invalid_instance;
        const auto moved=graph.active_islands.back();
        graph.active_islands[index]=moved;
        graph.active_islands.pop_back();
        if(index<graph.active_islands.size())
            graph.islands[moved].active_index=index;
    }
    for(BodyId id:graph.active_nodes) {
        const auto handle=graph.node_island[id];
        if(handle==IslandGraph::invalid)continue;
        auto& island=graph.islands[handle];
        if(island.active_index!=invalid_instance)continue;
        island.active_index=graph.active_islands.size();
        graph.active_islands.push_back(handle);
    }
    }
}
void Scene::prepare_active_accurate_island_batches() {
    // wakeIslands and third-pass retirement already reconciled the
    // authoritative graph lists before Dy consumes these batches.
    // Dy::DynamicsContext::iterateIslandsNodes consumes source
    // IslandSim::mActiveIslands, never all dormant island-node ranges.
    // The contiguous source node ranges remain sorted by island handle,
    // so resolve each active handle by binary search over their offsets.
    // This visits only active islands, in their persistent source order.
    const auto& graph=accurate_islands_;
    active_accurate_island_batches_.clear();
    if(accurate_island_offsets_.size()<2)return;
    const auto end=accurate_island_offsets_.end()-1;
    for(const auto handle:graph.active_islands) {
        const auto found=std::lower_bound(
            accurate_island_offsets_.begin(),end,handle,
            [&](std::size_t offset, IslandGraph::IslandId target) {
                return graph.node_island[accurate_island_nodes_[offset]]<target;
            });
        if(found==end ||
           graph.node_island[accurate_island_nodes_[*found]]!=handle)
            continue;
        active_accurate_island_batches_.push_back(
            static_cast<std::size_t>(found-accurate_island_offsets_.begin()));
    }
}
// Active native CPU stages remain within the one Scene owner:
//   DyBodyCoreIntegrator: unconstrained velocity/force preparation
//   DyConstraintPartition: ordered rigid-body contact manager batches
//   DyContactPrep: point-normal and correlated friction-anchor constraints
//   DySolverConstraints/DySolverControl: four position + one velocity pass
//   DyBodyCoreIntegrator: pose integration from saved position-phase motion
// The all-awake rigid island-node order is active here. Incremental island
// retirement, sleep/wake, task joins, joints, CCD, and exact packed ABI
// remain separate N1 requirements.
void Scene::prepare_solver_bodies() {
    // DyDynamics::preIntegrationParallel -> Dy::copyToSolverBodyData.
    // Retain capacity across frames. Prepare each contact participant once;
    // free-flight actors stay on the existing source body integrator.
    // This is the sole scene-owned transient solver state.
    solver_rows_.resize(actors_.size());
    // Keep a distinct monotonic generation: StepCheckpoint rolls back
    // frame_index_, but rejected solver rows remain in reusable storage.
    // Reusing a frame index would otherwise reactivate stale contact rows
    // after geometry changes on a retry.
    if(++solver_preparation_generation_==0) {
        for(auto& row:solver_rows_)row.prepared_generation=0;
        solver_preparation_generation_=1;
    }
    solver_active_ids_.clear();
    world_solver_row_ = {};
    // DyDynamicsContext::iterateIslandsContactEdges gives pre-integration
    // the accurate ACTIVE islands' ordered contact-manager EdgeIndices.
    // Consume the same intrusive stream as setupDescTask: retained Gu
    // managers in dormant islands are not solver-body admission work.
    // A zero maxContactImpulse still retains the accurate edge but
    // DyContactPrepShared::extractContacts admits no response rows.
    for(std::size_t batch:active_accurate_island_batches_) {
        const BodyId root=accurate_island_nodes_[
            accurate_island_offsets_[batch]];
        const auto handle=accurate_islands_.node_island[root];
        const auto& island=accurate_islands_.islands[handle];
        std::size_t visited=0;
        for(EdgeIndex edge=island.first_contact;edge!=invalid_edge;
            edge=contact_edges_[edge].island_next) {
            if(++visited>island.contact_count)
                throw std::logic_error(
                    "PhysX active contact-manager chain contains a cycle");
            const auto& manager=contact_edges_[edge];
            if(!manager.live || !manager.connected || manager.retiring)
                throw std::logic_error(
                    "PhysX active island contains an inactive contact manager");
            const Pair pair=manager.pair;
            const Actor& first=actors_[pair.a];
            const Actor& second=actors_[pair.b];
            if(std::min(first.dynamics.max_contact_impulse,
                        second.dynamics.max_contact_impulse)==0.0f)
                continue;
            for(BodyId id:{pair.a,pair.b})
                if(actors_[id].kind==Kind::DynamicBox &&
                   !actors_[id].sleeping)
                    solver_rows_[id].prepared_generation=
                        solver_preparation_generation_;
        }
        if(visited!=island.contact_count)
            throw std::logic_error(
                "PhysX active contact-manager chain is truncated");
    }
    // Dy::DynamicsContext::iterateIslandsNodes hands contiguous accurate
    // island nodes to preIntegrationParallel. Free-flight bodies retain the
    // source no-constraint integration path without preparing unused rows.
    for(std::size_t island:active_accurate_island_batches_) {
        for(std::size_t index=accurate_island_offsets_[island];
            index<accurate_island_offsets_[island+1];++index) {
            const BodyId id=accurate_island_nodes_[index];
            auto& row=solver_rows_[id];
            if(row.prepared_generation!=solver_preparation_generation_) {
                // DyConstraintPartition::initSolverProgress initializes
                // all active partition inputs, not only bodies with
                // nonzero Dy contact rows. A zero-capped Gu manager
                // still owns a source constraint descriptor, so its
                // lightweight partition fields must be reset every
                // frame even though extractContacts omits every row.
                // Piggyback on the existing active-island traversal:
                // no dormant-actor scan and no unused solver-body setup.
                row.solver.partition_mask=0;
                row.solver.max_dynamic_partition=0;
                row.solver.static_constraint_count=0;
                continue;
            }
            row={};
            row.prepared_generation=solver_preparation_generation_;
            solver_active_ids_.push_back(id);
            const Actor& actor=actors_[id];
            row.data.position=actor.position;
            row.data.orientation=actor.orientation;
            row.data.linear_velocity=actor.velocity;
            row.data.angular_velocity=actor.angular_velocity;
            // DyRigidBodyToSolverBody::copyToSolverBodyData masks
            // angular baseline motion but retains sqrtInvInertia columns.
            lock_angular_axes(row.data.angular_velocity,
                              actor.dynamics.lock_flags);
            row.data.inverse_mass=actor.inverse_mass;
            // The provisional translation-only mass admission does not
            // authorize a rotation-locked rigid contact. PhysX requires
            // the source body's real angular response for its per-point
            // contact Jacobians; an unavailable inverse inertia tensor
            // cannot silently become a zero-inertia solver body.
            if(!actor.angular_inertia_available)
                throw std::logic_error(
                    "Native rigid contact requires representable angular inertia");

            // CmUtils.h::transformInertiaTensor(sqrt(invInertia), pose.q).
            // Reuse the six unique terms of the symmetric world tensor.
            const Vec3 local{
                std::sqrt(actor.inverse_inertia_local.x),
                std::sqrt(actor.inverse_inertia_local.y),
                std::sqrt(actor.inverse_inertia_local.z)};
            const Quat q=actor.orientation;
            if(q.x==0.0f && q.y==0.0f && q.z==0.0f && q.w==1.0f) {
                row.data.sqrt_inverse_inertia={
                    Vec3{local.x,0.0f,0.0f},Vec3{0.0f,local.y,0.0f},
                    Vec3{0.0f,0.0f,local.z}};
                continue;
            }
            const Vec3 rx=rotate(q,{1.0f,0.0f,0.0f});
            const Vec3 ry=rotate(q,{0.0f,1.0f,0.0f});
            const Vec3 rz=rotate(q,{0.0f,0.0f,1.0f});
            const float xx=local.x*rx.x*rx.x + local.y*ry.x*ry.x +
                           local.z*rz.x*rz.x;
            const float yy=local.x*rx.y*rx.y + local.y*ry.y*ry.y +
                           local.z*rz.y*rz.y;
            const float zz=local.x*rx.z*rx.z + local.y*ry.z*ry.z +
                           local.z*rz.z*rz.z;
            const float xy=local.x*rx.x*rx.y + local.y*ry.x*ry.y +
                           local.z*rz.x*rz.y;
            const float xz=local.x*rx.x*rx.z + local.y*ry.x*ry.z +
                           local.z*rz.x*rz.z;
            const float yz=local.x*rx.y*rx.z + local.y*ry.y*ry.z +
                           local.z*rz.y*rz.z;
            row.data.sqrt_inverse_inertia={
                Vec3{xx,xy,xz},Vec3{xy,yy,yz},Vec3{xz,yz,zz}};
        }
    }
}

void Scene::partition_constraints() {
    // DyConstraintPartition.cpp::partitionContactConstraints, rigid-body
    // classification, and the source's static-contact reservation rule.
    // Contact managers are indivisible descriptors; their per-point normal
    // and friction rows execute together in the original owner. This
    // serial dispatch does not invent parallel task execution or islands.
    //
    // The source assigns conflict-free dynamic/dynamic constraints from
    // successive 32-bit partition masks; overflow starts a new wave.
    // A dynamic/static contact is then placed after that dynamic body's
    // last dynamic/dynamic partition, one static edge per partition.
    constexpr std::size_t partition_bits = 32;
    // Retry-safe generation labels avoid writing an unassigned
    // sentinel to every retained dormant Gu manager on each PGS step.
    constraint_partitions_.resize(manifolds_.size());
    constraint_partition_stamps_.resize(manifolds_.size(),0);
    if(++constraint_partition_generation_==0) {
        std::fill(constraint_partition_stamps_.begin(),
                  constraint_partition_stamps_.end(),0);
        constraint_partition_generation_=1;
    }
    const auto partition_generation=constraint_partition_generation_;
    ordered_constraints_.clear();
    partition_ends_.clear();
    partition_cursors_.clear();
    // DyConstraintPartition.cpp::classifyConstraintDesc queues only the
    // unassigned dynamic descriptors for each overflow wave; compact the
    // same reusable descriptor buffer in place to avoid an O(waves*all
    // manifolds) rescan for high-degree constraint graphs.
    // DyDynamicsBase::iterateIslandsContactEdges enumerates the
    // accurate active islands' eCONTACT_MANAGER edges. Gu manifolds
    // remain pair-sorted for source collision caches and writeback,
    // but that order is *not* the source constraint descriptor input.
    // Map handle -> original Gu index once, without a second contact
    // registry or a binary search for every island edge.
    // The preceding Gu narrowphase publication bound every live
    // manifold index to its persistent contact-manager EdgeIndex. Quiet
    // steps do not touch this lookup; topology/handle changes publish
    // the next one atomically with their new contact arena.
    // The generation test below rejects stale slots after touch loss,
    // contact-manager retirement and source handle reuse.
    indexed_managers_.clear();
    solver_tasks_.clear();
    std::size_t task_body_count=0;
    std::size_t task_constraint_count=0;
    std::size_t task_first_batch=0;
    std::size_t prepared_solver_bodies=0;
    std::uint8_t task_pos_iters=0, task_vel_iters=0;
    // Source DyDynamicsContext::updatePostKinematic combines consecutive
    // accurate active islands into solver task chains. With the SDK's
    // default rigid-only scene, mSolverBatchSize is 128 bodies and the
    // source minimumConstraintCount is 1. The grouped stream directly
    // determines partition/solve descriptor ordering below.
    constexpr std::size_t source_solver_batch_size=128;
    for(std::size_t batch_index=0;
        batch_index<active_accurate_island_batches_.size();++batch_index) {
        const auto batch=active_accurate_island_batches_[batch_index];
        const std::size_t offset=accurate_island_offsets_[batch];
        const BodyId root=accurate_island_nodes_[offset];
        const auto handle=accurate_islands_.node_island[root];
        const auto& island=accurate_islands_.islands[handle];
        std::size_t count=0;
        for(EdgeIndex edge=island.first_contact;edge!=invalid_edge;
            edge=contact_edges_[edge].island_next) {
            if(edge>=manifold_by_edge_.size() ||
               manifold_by_edge_[edge].generation!=
                   manifold_lookup_generation_ ||
               manifold_by_edge_[edge].index>=manifolds_.size() ||
               !(manifolds_[manifold_by_edge_[edge].index].pair==
                 contact_edges_[edge].pair))
                throw std::logic_error(
                    "Active PhysX island has an unindexed contact manager");
            indexed_managers_.push_back(manifold_by_edge_[edge].index);
            if(++count>island.contact_count)
                throw std::logic_error("PhysX island contact list cycle");
        }
        if(count!=island.contact_count)
            throw std::logic_error("PhysX island contact list truncated");
        task_body_count+=island.count;
        task_constraint_count+=island.contact_count;
        // DyDynamics.cpp::preIntegrationParallel accumulates the
        // maximum PxsBodyCore solverIterationCounts word over precisely
        // this source task's node array, including unconstrained bodies.
        for(std::size_t node=accurate_island_offsets_[batch];
            node<accurate_island_offsets_[batch+1];++node) {
            const BodyId id=accurate_island_nodes_[node];
            const auto& d=actors_[id].dynamics;
            task_pos_iters=std::max(task_pos_iters,d.position_iterations);
            task_vel_iters=std::max(task_vel_iters,d.velocity_iterations);
            if(solver_rows_[id].prepared_generation==
               solver_preparation_generation_)
                ++prepared_solver_bodies;
        }
        // DyDynamicsContext::updatePostKinematic breaks after EVERY
        // accurate island with eENABLE_ENHANCED_DETERMINISM; the
        // regular path retains source 128-body/min-one-contact batching.
        // Non-contact islands must also form their own task chains.
        if(enhanced_determinism_ ||
           (task_body_count>=source_solver_batch_size &&
            task_constraint_count>=1)) {
            solver_tasks_.push_back({task_first_batch,batch_index+1,
                indexed_managers_.size(),0,0,prepared_solver_bodies,
                task_pos_iters,task_vel_iters});
            task_first_batch=batch_index+1;
            task_body_count=0;
            task_constraint_count=0;
            task_pos_iters=task_vel_iters=0;
        }
    }
    if(task_body_count)
        solver_tasks_.push_back({task_first_batch,
            active_accurate_island_batches_.size(),
            indexed_managers_.size(),0,0,prepared_solver_bodies,
            task_pos_iters,task_vel_iters});
    // The admitted non-compound rigid PGS descriptor path in
    // DyDynamics.cpp::setupDescTask consumes this indexed-manager
    // sequence directly. The compound contact reducer's separate
    // two-key solver-body sort remains an explicit ABI admission gate.
    // Each source task chain partitions only its own island batch.
    // Use disjoint global partition ranges when concatenating the serial
    // CPU lowering; the original source can execute these chains in
    // parallel, but none may share a solver body across task groups.
    std::size_t manager_begin=0;
    std::size_t partition_base=0;
    pending_dynamic_constraints_.clear();
    for(auto& task:solver_tasks_) {
        const std::size_t manager_end=task.manager_end;
        pending_dynamic_constraints_.clear();
        for(std::size_t j=manager_begin;j<manager_end;++j) {
            const std::size_t i=indexed_managers_[j];
            const Pair pair=manifolds_[i].pair;
            if(actors_[pair.a].kind==Kind::DynamicBox &&
               actors_[pair.b].kind==Kind::DynamicBox &&
               !actors_[pair.a].sleeping && !actors_[pair.b].sleeping)
                pending_dynamic_constraints_.push_back(i);
        }
        std::size_t next_partition_base=partition_base;
        for(std::size_t wave=0;!pending_dynamic_constraints_.empty();
            wave+=partition_bits) {
            // After a 32-partition overflow, only still-pending dynamic
            // descriptors participate in the next wave. Resetting every
            // active body in the scene would multiply work by unrelated
            // islands and by source task-group count.
            if(wave)
                for(std::size_t i:pending_dynamic_constraints_) {
                    const Pair pair=manifolds_[i].pair;
                    solver_rows_[pair.a].solver.partition_mask=0;
                    solver_rows_[pair.b].solver.partition_mask=0;
                }
            std::size_t overflow_count=0;
            for(std::size_t k=0;k<pending_dynamic_constraints_.size();++k) {
                const std::size_t i=pending_dynamic_constraints_[k];
                const Pair pair=manifolds_[i].pair;
                auto& a=solver_rows_[pair.a].solver;
                auto& b=solver_rows_[pair.b].solver;
                const std::uint32_t available=~(a.partition_mask |
                                                 b.partition_mask);
                if(!available) {
                    pending_dynamic_constraints_[overflow_count++]=i;
                    continue;
                }
                const unsigned bit=std::countr_zero(available);
                const std::uint32_t mask=std::uint32_t{1}<<bit;
                a.partition_mask|=mask;
                b.partition_mask|=mask;
                const std::size_t local_partition=wave+bit;
                a.max_dynamic_partition=std::max(
                    a.max_dynamic_partition,local_partition+1);
                b.max_dynamic_partition=std::max(
                    b.max_dynamic_partition,local_partition+1);
                if(partition_base>std::numeric_limits<std::size_t>::max()-
                                  local_partition)
                    throw std::overflow_error(
                        "Native source task partition offset overflow");
                const std::size_t global_partition=
                    partition_base+local_partition;
                constraint_partitions_[i]=global_partition;
                constraint_partition_stamps_[i]=partition_generation;
                next_partition_base=std::max(next_partition_base,
                                             global_partition+1);
            }
            if(overflow_count==pending_dynamic_constraints_.size())
                throw std::logic_error(
                    "Native source constraint partition made no progress");
            pending_dynamic_constraints_.resize(overflow_count);
            if(overflow_count &&
               wave>std::numeric_limits<std::size_t>::max()-partition_bits)
                throw std::overflow_error(
                    "Native source constraint partition index overflow");
        }

        for(std::size_t j=manager_begin;j<manager_end;++j) {
            const std::size_t i=indexed_managers_[j];
            if(constraint_partition_stamps_[i]==partition_generation)
                continue;
            const Pair pair=manifolds_[i].pair;
            const bool dynamic_a=actors_[pair.a].kind==Kind::DynamicBox;
            const bool dynamic_b=actors_[pair.b].kind==Kind::DynamicBox;
            if((dynamic_a && actors_[pair.a].sleeping) ||
               (dynamic_b && actors_[pair.b].sleeping))continue;
            if(!dynamic_a && !dynamic_b)continue;
            auto& active=solver_rows_[dynamic_a?pair.a:pair.b].solver;
            const std::size_t local_partition=
                active.max_dynamic_partition+
                active.static_constraint_count++;
            if(local_partition<active.max_dynamic_partition ||
               partition_base>std::numeric_limits<std::size_t>::max()-
                                  local_partition)
                throw std::overflow_error(
                    "Native source static constraint partition overflow");
            const std::size_t global_partition=
                partition_base+local_partition;
            constraint_partitions_[i]=global_partition;
            constraint_partition_stamps_[i]=partition_generation;
            next_partition_base=std::max(next_partition_base,
                                         global_partition+1);
        }
        partition_base=next_partition_base;
        task.partition_end=partition_base;
        manager_begin=manager_end;
    }

    // The source accumulates counts into ordered descriptor offsets, then
    // stable-writes each descriptor by partition. This preserves the
    // input contact-manager order *within* each non-conflicting batch.
    std::size_t count=0;
    for (std::size_t i:indexed_managers_) {
        if(constraint_partition_stamps_[i]!=partition_generation)
            continue;
        const std::size_t p=constraint_partitions_[i];
        if (p>=partition_ends_.size()) partition_ends_.resize(p+1,0);
        ++partition_ends_[p];
        ++count;
    }
    std::size_t cumulative=0;
    for (auto& end : partition_ends_) {
        cumulative+=end;
        end=cumulative;
    }
    for(auto& task:solver_tasks_)
        task.constraint_end=task.partition_end
            ? partition_ends_[task.partition_end-1] : 0;
    partition_cursors_.resize(partition_ends_.size());
    for (std::size_t p=0;p<partition_cursors_.size();++p)
        partition_cursors_[p]=p ? partition_ends_[p-1] : 0;
    ordered_constraints_.resize(count);
    for (std::size_t i:indexed_managers_) {
        if(constraint_partition_stamps_[i]==partition_generation) {
            const std::size_t partition=constraint_partitions_[i];
            ordered_constraints_[partition_cursors_[partition]++]=i;
        }
    }
}

void Scene::prepare_constraints(float seconds) {
    // DyContactPrepShared.h::constructContactConstraint, rigid-body
    // noncompliant branch. Every Gu contact is a distinct constraint row.
    // Contiguous source Gu contact runs are correlated into the same
    // friction-patch linked lists before normal/friction row preparation.
    // Solver rows consume the patch order; Gu offsets remain writeback
    // indices. Multi-material contact patches and packed block ABI remain
    // separate source admission requirements.
    // DyCpuGpuBiasCoefficient.h::BiasCoefficientCollection::set(false,
    // false, ...): rigidContact = 0.8f for the CPU PGS solver. The same
    // source coefficient is used by DyContactPrep for penetration bias
    // and for friction-row impulse relaxation (not a 0.2 substitute).
    constexpr float rigid_contact_bias = 0.8f;
    // DyContactPrep uses max(body0.penBiasClamp,body1.penBiasClamp)
    // from authoritative PxsBodyCore. The default -1e32f remains
    // represented through each actor's initial max depenetration value.
    constexpr float bounce_threshold = -2.0f;
    // PxsMaterialCombiner.h::PxsCombineMaterials: the source takes the
    // higher precedence combine mode (Average, Min, Multiply, Max) of
    // both shapes and promotes static friction to at least dynamic.
    // PhysX PxsMaterialCombiner also gives compliant spring/damping
    // combinations precise priority before contact patch correlation.
    const auto combine=[](float a,float b,MaterialCombineMode mode) {
        switch(mode) {
        case MaterialCombineMode::Average: return 0.5f*(a+b);
        case MaterialCombineMode::Minimum: return std::min(a,b);
        case MaterialCombineMode::Multiply: return a*b;
        case MaterialCombineMode::Maximum: return std::max(a,b);
        }
        throw std::logic_error("Unrecognized source material combine mode");
    };
    const auto higher=[](MaterialCombineMode a,MaterialCombineMode b) {
        return static_cast<unsigned>(a)>static_cast<unsigned>(b)?a:b;
    };
    const float inv_dt = 1.0f / seconds;
    const auto dot = [](Vec3 u, Vec3 v) {
        return u.x*v.x + u.y*v.y + u.z*v.z;
    };
    const auto sub = [](Vec3 u, Vec3 v) {
        return Vec3{u.x-v.x,u.y-v.y,u.z-v.z};
    };
    // DyContactPrepShared and DyContactPrep mask each Jacobian and
    // friction positional-error component only when slop > abs(value).
    const auto suppress_slop=[](Vec3 v,float slop) {
        return Vec3{std::abs(v.x)<slop?0.0f:v.x,
                    std::abs(v.y)<slop?0.0f:v.y,
                    std::abs(v.z)<slop?0.0f:v.z};
    };
    const auto transform = [](const SolverBodyData& d, Vec3 v) {
        const auto& c = d.sqrt_inverse_inertia;
        return Vec3{c[0].x*v.x+c[1].x*v.y+c[2].x*v.z,
                    c[0].y*v.x+c[1].y*v.y+c[2].y*v.z,
                    c[0].z*v.x+c[1].z*v.y+c[2].z*v.z};
    };
    solver_contact_points_.clear();
    solver_contact_order_.clear();
    solver_friction_rows_.clear();
    next_friction_patches_.clear();
    // PXC_SAME_NORMAL and DyCorrelationBuffer::MAX_FRICTION_PATCHES.
    constexpr float same_normal = 0.999f;
    constexpr float friction_offset_threshold = 0.04f;
    constexpr float correlation_distance = 0.025f;
    const auto local_point = [](const Actor& actor, Vec3 world) {
        const Quat& q = actor.orientation;
        return rotate(Quat{-q.x,-q.y,-q.z,q.w},
            {world.x-actor.position.x,
             world.y-actor.position.y,
             world.z-actor.position.z});
    };
    const auto local_normal = [](const Actor& actor, Vec3 world_normal) {
        const Quat& q = actor.orientation;
        return rotate(Quat{-q.x,-q.y,-q.z,q.w}, world_normal);
    };
    const auto world_point = [](const Actor& actor, Vec3 local) {
        Vec3 r = rotate(actor.orientation,local);
        return Vec3{r.x+actor.position.x,
                    r.y+actor.position.y,
                    r.z+actor.position.z};
    };
    const auto squared_length = [](Vec3 v) {
        return v.x*v.x + v.y*v.y + v.z*v.z;
    };
    // Source DyContactPrep consumes each active island task's indexed
    // contact managers. Gu storage and its original point offsets remain
    // pair-sorted for collision and force writeback, but preparation
    // follows the true source solver descriptor sequence.
    if(friction_cache_epoch_==std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("Native friction cache epoch exhausted");
    const auto next_cache_epoch=friction_cache_epoch_+1;
    next_friction_by_edge_.resize(contact_edges_.size());
    for(std::size_t manager_index:indexed_managers_) {
        auto& manifold=manifolds_[manager_index];
        const EdgeIndex edge=manifold.edge_handle;
        if(edge>=contact_edges_.size() ||
           !(contact_edges_[edge].pair==manifold.pair))
            throw std::logic_error(
                "Dy source manager preparation has a mismatched island edge");
        step_checkpoint_.save_manifold_solver_metadata(*this,manager_index);
        manifold.prepared_friction_epoch=next_cache_epoch;
        // An inactive accurate island retains its Gu contact manager
        // but the source does not prepare or solve constraints for it.
        // Keep the Gu-indexed zero-impulse result slots without paying
        // for contact Jacobians, friction correlation or solver rows.
        const Actor& actor_a=actors_[manifold.pair.a];
        const Actor& actor_b=actors_[manifold.pair.b];
        const bool active_dynamic=
            (actor_a.kind==Kind::DynamicBox && !actor_a.sleeping) ||
            (actor_b.kind==Kind::DynamicBox && !actor_b.sleeping);
        if(!active_dynamic) {
            manifold.solver_offset=solver_contact_points_.size();
            manifold.friction_patch_offset=next_friction_patches_.size();
            manifold.friction_patch_count=0;
            // No Dy contact rows are admitted for dormant Gu managers.
            // The independently Gu-indexed writeback provides zeros.
            continue;
        }
        // Preserve ScShapeInteraction's actual ordered actor pair
        // through DyContactPrep. A canonical pair key is only used
        // for Gu arena identity, and must not reorder the two
        // momentum/Jacobian inputs after the source Gu generator.
        // PxcNpBatch sorts plane-box Gu geometry plane-first and
        // flips its returned normal back to ScShapeInteraction order.
        // ScNPhaseCore::shouldSwapBodies puts static shapes SECOND,
        // even when the floor was created before its dynamic partner.
        // The persistent source-order flag applies to every shape pair.
        const bool reversed=contact_edges_[edge].gu_order_reversed;
        const BodyId id0=reversed?manifold.pair.b:manifold.pair.a;
        const BodyId id1=reversed?manifold.pair.a:manifold.pair.b;
        const auto& a = actors_[id0].kind==Kind::DynamicBox
            ? solver_rows_[id0].data : world_solver_row_.data;
        const auto& b = actors_[id1].kind==Kind::DynamicBox
            ? solver_rows_[id1].data : world_solver_row_.data;
        manifold.solver_offset = solver_contact_points_.size();
        manifold.friction_patch_offset = next_friction_patches_.size();
        manifold.friction_patch_count = 0;
        const Actor& actor0 = actors_[id0];
        const Actor& actor1 = actors_[id1];
        // ScShapeInteraction::initializeLowLevelContactManager stores
        // max(body0.offsetSlop,body1.offsetSlop) in PxcNpWorkUnit.
        const float pair_slop=std::max(
            actor0.dynamics.contact_slop_coefficient,
            actor1.dynamics.contact_slop_coefficient);
        // DyContactPrepShared::constructContactConstraint subtracts
        // contactDesc.restDistance from each Gu surface separation.
        const float rest_distance=actor0.rest_offset+actor1.rest_offset;
        if(!std::isfinite(rest_distance) ||
           !std::isfinite(rest_distance+friction_offset_threshold))
            throw std::overflow_error("Native Dy rest distance overflow");
        // DyContactPrepShared::extractContacts applies the minimum
        // body-level maxContactImpulse to each Gu point. With the
        // admitted single-material shape, a zero cap suppresses ALL
        // solver contacts in this manager but leaves Gu touch/edge
        // publication authoritative. Preserve Gu force offsets as
        // explicit zero slots, without inventing contact response.
        const float pair_max_impulse=std::min(
            actor0.dynamics.max_contact_impulse,
            actor1.dynamics.max_contact_impulse);
        if(pair_max_impulse==0.0f) {
            // DyContactPrepShared::extractContacts omits these before
            // filling ANY contact/patch solver descriptors. Subsequent
            // live managers pack immediately after prior live rows.
            continue;
        }
        // PxRigidBody::setMaxDepenetrationVelocity stores the NEGATIVE
        // value in PxsBodyCore::penBiasClamp. DyContactPrep uses max of
        // both signed clamps, i.e. the smaller positive depenetration cap.
        const float max_penetration_bias=-std::min(
            actor0.dynamics.max_depenetration_velocity,
            actor1.dynamics.max_depenetration_velocity);
        const Material& m0=actor0.material;
        const Material& m1=actor1.material;
        const auto friction_mode=higher(m0.friction_combine,
                                        m1.friction_combine);
        // PxsMaterialCombiner::PxsCombineMaterials: a compliant material
        // wins over an ordinary one. An acceleration spring wins over
        // a force spring when both materials are compliant.
        const bool c0=m0.restitution<0.0f;
        const bool c1=m1.restitution<0.0f;
        const bool both_compliant=c0 && c1;
        const bool exactly_one_compliant=c0 != c1;
        const bool acceleration_override=both_compliant &&
            (m0.compliant_acceleration_spring !=
             m1.compliant_acceleration_spring);
        const bool acceleration_spring=
            m0.compliant_acceleration_spring ||
            m1.compliant_acceleration_spring;
        float restitution;
        float damping;
        if(acceleration_override) {
            restitution=m0.compliant_acceleration_spring
                ? m0.restitution : m1.restitution;
            damping=m0.compliant_acceleration_spring
                ? m0.damping : m1.damping;
        } else {
            const auto restitution_mode=exactly_one_compliant
                ? MaterialCombineMode::Minimum
                : higher(m0.restitution_combine,m1.restitution_combine);
            const auto damping_mode=exactly_one_compliant
                ? MaterialCombineMode::Maximum
                : higher(m0.damping_combine,m1.damping_combine);
            // Multiply two negative spring stiffnesses, then restore
            // the negative sign of the combined compliant stiffness.
            const float sign=both_compliant &&
                restitution_mode==MaterialCombineMode::Multiply
                ? -1.0f:1.0f;
            restitution=sign*combine(m0.restitution,m1.restitution,
                                     restitution_mode);
            damping=combine(m0.damping,m1.damping,damping_mode);
        }
        const bool disable_friction=m0.disable_friction || m1.disable_friction;
        const bool disable_strong_friction=disable_friction ||
            m0.disable_strong_friction || m1.disable_strong_friction;
        const float dynamic_friction=disable_friction ? 0.0f :
            std::max(0.0f,combine(m0.dynamic_friction,m1.dynamic_friction,
                                  friction_mode));
        const float static_friction=disable_friction ? 0.0f :
            std::max(dynamic_friction,
                     combine(m0.static_friction,m1.static_friction,
                             friction_mode));
        if (!std::isfinite(restitution) ||
            !std::isfinite(damping) ||
            !std::isfinite(static_friction) ||
            !std::isfinite(dynamic_friction))
            throw std::overflow_error("Native Pxs material combination overflow");

        // Original DyContactPrep::createContactPatches preserves
        // CONTIGUOUS Gu runs of compatible normal/material contacts.
        // DyFrictionCorrelation::correlatePatches then links each run
        // onto an existing friction patch (new or strong-friction
        // history), prepending to that patch's correlation list.
        // Construct those original two passes before preparing normal
        // rows; simply grouping individual normals loses linked-list
        // order and fails to prioritize valid previous-frame anchors.
        contact_patch_scratch_.clear();
        contact_patch_scratch_.reserve(manifold.count);
        for(std::size_t j=0;j<manifold.count;++j) {
            const auto& point=manifold.points[j];
            if(contact_patch_scratch_.empty() ||
               dot(contact_patch_scratch_.back().normal,point.normal)<
                   same_normal) {
                ContactPatchData cp{};
                cp.start=j;
                cp.count=1;
                cp.normal=point.normal;
                cp.bounds_min=cp.bounds_max=point.position;
                contact_patch_scratch_.push_back(cp);
            } else {
                auto& cp=contact_patch_scratch_.back();
                ++cp.count;
                cp.bounds_min={
                    std::min(cp.bounds_min.x,point.position.x),
                    std::min(cp.bounds_min.y,point.position.y),
                    std::min(cp.bounds_min.z,point.position.z)};
                cp.bounds_max={
                    std::max(cp.bounds_max.x,point.position.x),
                    std::max(cp.bounds_max.y,point.position.y),
                    std::max(cp.bounds_max.z,point.position.z)};
            }
        }
        constexpr std::size_t max_source_patches=32;
        std::array<std::size_t,max_source_patches> contact_heads{};
        contact_heads.fill(invalid_instance);
        std::array<std::size_t,255> contact_patch_ids{};
        // DyContactPrepShared::getFrictionPatches admits prior anchors
        // FIRST, only with a valid relative normal, separation and
        // strong-friction state. Fresh contact patches correlate against
        // this source-ordered cache before creating any new patch.
        if(!disable_strong_friction && edge<friction_by_edge_.size()) {
            const auto& history=friction_by_edge_[edge];
            if(history.epoch==friction_cache_epoch_ &&
               history.count && history.pair==manifold.pair &&
               history.offset<=friction_patches_.size() &&
               history.count<=friction_patches_.size()-history.offset)
            for(std::size_t previous=history.offset;
                previous<history.offset+history.count;++previous) {
                const auto& old=friction_patches_[previous];
                if(old.broken || old.disable_strong_friction ||
                   old.anchor_count==0)continue;
                const Vec3 old_normal_1_world=
                    rotate(actor1.orientation,old.body1_normal);
                const Vec3 old_normal_1_in_0=
                    local_normal(actor0,old_normal_1_world);
                if(dot(old.body0_normal,old_normal_1_in_0)<=same_normal)
                    continue;
                bool separated=false;
                for(std::size_t k=0;k<old.anchor_count;++k) {
                    const Vec3 p1_in0=local_point(actor0,
                        world_point(actor1,old.body1_anchors[k]));
                    if(std::abs(dot(sub(old.body0_anchors[k],p1_in0),
                                    old.body0_normal))>=correlation_distance) {
                        separated=true;
                        break;
                    }
                }
                if(separated)continue;
                if(manifold.friction_patch_count==max_source_patches)
                    throw std::overflow_error(
                        "Native source cached friction exceeds 32 patches");
                FrictionPatch copied=old;
                copied.normal=rotate(actor0.orientation,old.body0_normal);
                copied.contact_count=0;
                copied.contact_ids.fill(0xffff);
                copied.bounds_min={
                    std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max()};
                copied.bounds_max={
                    -std::numeric_limits<float>::max(),
                    -std::numeric_limits<float>::max(),
                    -std::numeric_limits<float>::max()};
                next_friction_patches_.push_back(copied);
                ++manifold.friction_patch_count;
            }
        }
        for(std::size_t cp_index=0;
            cp_index<contact_patch_scratch_.size();++cp_index) {
            auto& cp=contact_patch_scratch_[cp_index];
            std::size_t local_patch=0;
            for(;local_patch<manifold.friction_patch_count;++local_patch) {
                const auto& p=next_friction_patches_[
                    manifold.friction_patch_offset+local_patch];
                if(dot(p.normal,cp.normal)>=same_normal &&
                   p.restitution==restitution &&
                   p.damping==damping &&
                   p.acceleration_spring==acceleration_spring &&
                   p.static_friction==static_friction &&
                   p.dynamic_friction==dynamic_friction)
                    break;
            }
            if(local_patch==manifold.friction_patch_count) {
                if(manifold.friction_patch_count==max_source_patches)
                    throw std::overflow_error(
                        "Native source friction correlation exceeds 32 patches");
                FrictionPatch patch{};
                patch.pair=manifold.pair;
                patch.normal=cp.normal;
                patch.static_friction=static_friction;
                patch.dynamic_friction=dynamic_friction;
                patch.restitution=restitution;
                patch.damping=damping;
                patch.acceleration_spring=acceleration_spring;
                patch.disable_friction=disable_friction;
                patch.disable_strong_friction=disable_strong_friction;
                patch.body0_normal=local_normal(actor0,cp.normal);
                patch.body1_normal=local_normal(actor1,cp.normal);
                patch.contact_ids.fill(0xffff);
                next_friction_patches_.push_back(patch);
                ++manifold.friction_patch_count;
            }
            auto& patch=next_friction_patches_[
                manifold.friction_patch_offset+local_patch];
            if(patch.contact_count==0) {
                patch.bounds_min=cp.bounds_min;
                patch.bounds_max=cp.bounds_max;
            } else {
                patch.bounds_min={
                    std::min(patch.bounds_min.x,cp.bounds_min.x),
                    std::min(patch.bounds_min.y,cp.bounds_min.y),
                    std::min(patch.bounds_min.z,cp.bounds_min.z)};
                patch.bounds_max={
                    std::max(patch.bounds_max.x,cp.bounds_max.x),
                    std::max(patch.bounds_max.y,cp.bounds_max.y),
                    std::max(patch.bounds_max.z,cp.bounds_max.z)};
            }
            patch.contact_count+=static_cast<std::uint16_t>(cp.count);
            cp.next=contact_heads[local_patch];
            contact_heads[local_patch]=cp_index;
            for(std::size_t j=cp.start;j<cp.start+cp.count;++j)
                contact_patch_ids[j]=manifold.friction_patch_offset+local_patch;
        }
        // Source writes only friction patches with correlated contacts;
        // unused cached patches are never returned to the next frame.
        // Compact the same cache arena and remap Gu point/linked-list
        // indices together without allocating a second patch registry.
        std::array<std::size_t,max_source_patches> remap{};
        std::array<std::size_t,max_source_patches> used_heads{};
        std::size_t kept=0;
        for(std::size_t j=0;j<manifold.friction_patch_count;++j) {
            const std::size_t src=manifold.friction_patch_offset+j;
            if(next_friction_patches_[src].contact_count==0)continue;
            remap[j]=kept;
            used_heads[kept]=contact_heads[j];
            if(kept!=j)
                next_friction_patches_[manifold.friction_patch_offset+kept]=
                    next_friction_patches_[src];
            ++kept;
        }
        for(std::size_t j=0;j<manifold.count;++j)
            contact_patch_ids[j]=manifold.friction_patch_offset+
                remap[contact_patch_ids[j]-manifold.friction_patch_offset];
        manifold.friction_patch_count=static_cast<std::uint8_t>(kept);
        next_friction_patches_.resize(manifold.friction_patch_offset+kept);
        next_friction_by_edge_[edge]={manifold.pair,
            manifold.friction_patch_offset,manifold.friction_patch_count,
            next_cache_epoch};
        // DyContactPrep.cpp::setupFinalizeSolverConstraints selects the
        // first Gu normal of each reverse correlation-list head ONCE for
        // its SolverContactHeader. That one normal governs *all* contact
        // Jacobians, effective masses, restitution projections and both
        // friction tangents in the header. It is distinct from a cached
        // FrictionPatch::body0Normal/body1Normal, which belong to the
        // older strong-friction anchor correlation lifecycle.
        std::array<Vec3,max_source_patches> header_normals{};
        for(std::size_t j=0;j<kept;++j) {
            const auto head=used_heads[j];
            if(head==invalid_instance)
                throw std::logic_error(
                    "Source correlated friction patch has no contact head");
            header_normals[j]=
                manifold.points[contact_patch_scratch_[head].start].normal;
        }
        // Preserve source reverse correlation-list visitation in the
        // actual PGS solver while keeping Gu-indexed rows/force writeback.
        for(std::size_t j=0;j<kept;++j)
            for(std::size_t cp=used_heads[j];cp!=invalid_instance;
                cp=contact_patch_scratch_[cp].next)
                for(std::size_t c=contact_patch_scratch_[cp].start;
                    c<contact_patch_scratch_[cp].start+
                      contact_patch_scratch_[cp].count;++c)
                    solver_contact_order_.push_back(manifold.solver_offset+c);
        if(solver_contact_order_.size()!=
           manifold.solver_offset+manifold.count)
            throw std::logic_error(
                "Source correlation omitted a Gu contact descriptor");
        for(std::size_t j=0;j<manifold.count;++j) {
            const ContactPoint& contact=manifold.points[j];
            const std::size_t patch_id=contact_patch_ids[j];
            const Vec3 normal=header_normals[
                patch_id-manifold.friction_patch_offset];
            Vec3 ra_x_n = cross(sub(contact.position,a.position),normal);
            Vec3 rb_x_n = cross(sub(contact.position,b.position),normal);
            const float linear_vrel =
                dot(normal,sub(a.linear_velocity,b.linear_velocity));
            // DyContactPrepShared scales the slop by
            // max(vRelAngular/vRelLinear,1). A zero linear velocity uses
            // PX_MAX_REAL as the source's ratio fallback.
            if(pair_slop>0.0f) {
                const float raw_angular_vrel=
                    dot(ra_x_n,a.angular_velocity)-
                    dot(rb_x_n,b.angular_velocity);
                const float ratio=linear_vrel==0.0f
                    ? std::numeric_limits<float>::max()
                    : raw_angular_vrel/linear_vrel;
                const float normal_slop=pair_slop*std::max(ratio,1.0f);
                ra_x_n=suppress_slop(ra_x_n,normal_slop);
                rb_x_n=suppress_slop(rb_x_n,normal_slop);
            }
            const Vec3 ra = transform(a,ra_x_n);
            const Vec3 rb = transform(b,rb_x_n);
            const float angular_vrel =
                dot(ra_x_n,a.angular_velocity)-dot(rb_x_n,b.angular_velocity);
            const float vrel = linear_vrel+angular_vrel;
            const double response = static_cast<double>(a.inverse_mass) +
                static_cast<double>(b.inverse_mass) +
                static_cast<double>(dot(ra,ra)) + dot(rb,rb);
            const double vel_multiplier = response > 0.0
                ? 1.0 / response : 0.0;
            const float penetration = contact.separation-rest_distance;
            const float penetration_inv_dt = penetration*inv_dt;
            const bool separated = penetration >= 0.0f;
            const bool closing = -vrel > penetration_inv_dt;
            const bool bouncing = restitution > 0.0f &&
                bounce_threshold > vrel && closing;
            const float target_velocity = bouncing
                ? -vrel*restitution : 0.0f;
            SolverContactPoint row{};
            row.normal=normal;
            row.ra_x_n_sqrt_inertia=ra;
            row.rb_x_n_sqrt_inertia=rb;
            row.friction_patch=patch_id;
            row.max_impulse=pair_max_impulse;
            const double target=static_cast<double>(target_velocity-vrel);
            if(restitution<0.0f) {
                // DyContactPrepShared::computeCompliantContactCoefficients
                // disables damping for separated nonclosing contacts.
                // Force springs use unitResponse; acceleration springs
                // use the reciprocal response as a mass factor.
                const float enabled_damping=
                    separated && !closing ? 0.0f : damping;
                const double nrdt=static_cast<double>(seconds)*restitution;
                const double a_coeff=static_cast<double>(seconds)*
                    (static_cast<double>(enabled_damping)-nrdt);
                const double mass_factor=acceleration_spring
                    ? vel_multiplier : 1.0;
                const double x=1.0/(1.0+a_coeff*
                    (acceleration_spring ? 1.0 : response));
                const double bias=x*(-nrdt*penetration*mass_factor);
                row.velocity_multiplier=x*a_coeff*mass_factor;
                row.impulse_multiplier=1.0-x;
                row.biased_error=target*row.velocity_multiplier-bias;
                row.unbiased_error=row.biased_error;
            } else {
                // Normal restitution keeps a distinct unbiased
                // velocity-pass error after the biased position passes.
                const float penetration_inv_dt_scaled=separated
                    ? penetration_inv_dt
                    : penetration*(inv_dt*rigid_contact_bias);
                const double scaled_bias=bouncing ? 0.0 :
                    vel_multiplier*std::max(
                        max_penetration_bias,penetration_inv_dt_scaled);
                row.velocity_multiplier=vel_multiplier;
                row.biased_error=target*vel_multiplier-scaled_bias;
                row.unbiased_error=target*vel_multiplier-
                    (bouncing?0.0:std::max(scaled_bias,0.0));
            }
            if (!std::isfinite(row.biased_error) ||
                !std::isfinite(row.unbiased_error) ||
                !std::isfinite(row.impulse_multiplier) ||
                !std::isfinite(row.velocity_multiplier))
                throw std::overflow_error("Native Dy contact preparation overflow");
            solver_contact_points_.push_back(row);
        }
        // DyFrictionCorrelation::growPatches: retain up to two anchors
        // spread across each friction patch's contact bounds; the anchors
        // persist in the body-local frames, not as world-space contacts.
        for (std::size_t patch_id=manifold.friction_patch_offset;
             patch_id<next_friction_patches_.size(); ++patch_id) {
            auto& patch = next_friction_patches_[patch_id];
            // PxsCombineMaterials::eDISABLE_FRICTION produces zero
            // tangent constraints. Do not grow anchors or compute
            // local-to-world error/Jacobians for disabled patches.
            if (patch.disable_friction) {
                patch.anchor_count=0;
                patch.row_offset=solver_friction_rows_.size();
                continue;
            }
            const Vec3 extent=sub(patch.bounds_max,patch.bounds_min);
            const float diagonal_sq=squared_length(extent);
            bool keep_pair=false;
            if(patch.anchor_count==2) {
                const Vec3 diff=sub(patch.body0_anchors[0],
                                    patch.body0_anchors[1]);
                // DyFrictionCorrelation::growPatches retains a cached
                // two-anchor pair only when its spacing still covers
                // at least one-quarter of the current patch diagonal.
                // The source skips the anchor search in this case.
                keep_pair=squared_length(diff)*4.0f>=diagonal_sq;
                if(!keep_pair)patch.anchor_count=0;
            }
            std::array<Vec3,2> anchors{};
            // DyFrictionCorrelation::growPatches selects replacement
            // anchors as well as newly appended ones. A retained slot
            // must remain untouched unless the search replaces it.
            // The upstream j=oldCount publication misses replacements
            // of slot zero after a cached one-anchor patch grows.
            std::array<bool,2> changed{};
            std::uint8_t count=patch.anchor_count;
            if(!keep_pair) {
                if(count==1)
                    anchors[0]=world_point(actor0,patch.body0_anchors[0]);
                float point_dist_sq=0.0f;
                // The correlation linked list prepends each contact
                // patch, so source growPatches sees its runs in reverse
                // admission order, not in canonical Gu contact order.
                const std::size_t local=patch_id-
                    manifold.friction_patch_offset;
                for(std::size_t cp=used_heads[local];
                    cp!=invalid_instance;
                    cp=contact_patch_scratch_[cp].next) {
                    const auto& run=contact_patch_scratch_[cp];
                    for(std::size_t j=run.start;j<run.start+run.count;++j) {
                        const auto& point=manifold.points[j];
                        if(point.separation>=
                           friction_offset_threshold+rest_distance)
                            continue;
                        const Vec3 w=point.position;
                        if(count==0) {
                            patch.contact_ids[0]=static_cast<std::uint16_t>(j);
                            anchors[0]=w;
                            changed[0]=true;
                            ++count;
                        } else if(count==1) {
                            const float d=squared_length(sub(w,anchors[0]));
                            if(d>1.0e-8f) {
                                patch.contact_ids[1]=static_cast<std::uint16_t>(j);
                                anchors[1]=w;
                                changed[1]=true;
                                point_dist_sq=d;
                                ++count;
                            }
                        } else {
                            const float d0=squared_length(sub(w,anchors[0]));
                            const float d1=squared_length(sub(w,anchors[1]));
                            if(d0>d1 && d0>point_dist_sq) {
                                patch.contact_ids[1]=static_cast<std::uint16_t>(j);
                                anchors[1]=w;
                                changed[1]=true;
                                point_dist_sq=d0;
                            } else if(d1>=d0 && d1>point_dist_sq) {
                                patch.contact_ids[0]=static_cast<std::uint16_t>(j);
                                anchors[0]=w;
                                changed[0]=true;
                                point_dist_sq=d1;
                            }
                        }
                    }
                }
                for(std::size_t k=0;k<count;++k) {
                    if(!changed[k])continue;
                    patch.body0_anchors[k]=local_point(actor0,anchors[k]);
                    patch.body1_anchors[k]=local_point(actor1,anchors[k]);
                }
                if(count==0) {
                    patch.body0_anchors[0]={};
                    patch.body1_anchors[0]={};
                }
            }
            patch.anchor_count=count;
            patch.row_offset=solver_friction_rows_.size();
            if (count==0) continue;
            // DyContactPrep.cpp constructs tangents from relative linear
            // velocity, with the same 0.7071 normal-axis fallback and
            // 1e-4 squared-tangent threshold as the source.
            const Vec3 n=header_normals[
                patch_id-manifold.friction_patch_offset];
            const Vec3 rel=sub(a.linear_velocity,b.linear_velocity);
            Vec3 t0=sub(rel,{n.x*dot(rel,n),
                              n.y*dot(rel,n),n.z*dot(rel,n)});
            if (squared_length(t0)<=0.0001f) {
                t0=std::abs(n.x)<0.70710678f
                    ? Vec3{0.0f,-n.z,n.y}
                    : Vec3{-n.y,n.x,0.0f};
            }
            const float inv_l=1.0f/std::sqrt(squared_length(t0));
            t0={t0.x*inv_l,t0.y*inv_l,t0.z*inv_l};
            const Vec3 t1=cross(n,t0);
            for (std::size_t k=0;k<count;++k) {
                const Vec3 ra=rotate(actor0.orientation,patch.body0_anchors[k]);
                const Vec3 rb=rotate(actor1.orientation,patch.body1_anchors[k]);
                const Vec3 error=suppress_slop(sub(
                    {ra.x+actor0.position.x,ra.y+actor0.position.y,
                     ra.z+actor0.position.z},
                    {rb.x+actor1.position.x,rb.y+actor1.position.y,
                     rb.z+actor1.position.z}),pair_slop);
                for (const Vec3 tangent : {t0,t1}) {
                    const Vec3 ax0=suppress_slop(cross(ra,tangent),pair_slop);
                    const Vec3 ax1=suppress_slop(cross(rb,tangent),pair_slop);
                    const Vec3 j0=transform(a,ax0);
                    const Vec3 j1=transform(b,ax1);
                    const double response=static_cast<double>(a.inverse_mass)
                        +b.inverse_mass+dot(j0,j0)+dot(j1,j1);
                    SolverContactFriction fr{};
                    fr.tangent=tangent;
                    fr.ra_x_t_sqrt_inertia=j0;
                    fr.rb_x_t_sqrt_inertia=j1;
                    fr.velocity_multiplier=response>0.0
                        ? rigid_contact_bias/response : 0.0;
                    fr.bias=dot(tangent,error)*inv_dt;
                    fr.target_velocity=-(dot(tangent,rel)
                        +dot(ax0,a.angular_velocity)
                        -dot(ax1,b.angular_velocity));
                    if (!std::isfinite(fr.velocity_multiplier) ||
                        !std::isfinite(fr.bias) ||
                        !std::isfinite(fr.target_velocity))
                        throw std::overflow_error(
                            "Native source friction row preparation overflow");
                    solver_friction_rows_.push_back(fr);
                }
            }
        }
    }
    // Swap only after the new rows and patch anchors are fully prepared.
    // Old shape-local anchors remain available throughout correlation.
    friction_patches_.swap(next_friction_patches_);
    friction_by_edge_.swap(next_friction_by_edge_);
    friction_cache_epoch_=next_cache_epoch;
}
void Scene::solve_constraints() {
    // DySolverConstraintsShared.h::solveDynamicContacts /
    // solveStaticContacts and DySolverConstraints.cpp friction patch solve,
    // with the PhysX default of four position
    // iterations and one final velocity/writeback iteration. The final
    // writeback is itself a solve pass, not a second velocity sweep.
    // Wide prepared errors and impulses remain for the independently
    // admitted extreme translational mass range. Ordinary source PhysX
    // PxReal row precision is a separate strict-parity acceptance gate.
    // Source task groups carry their own maximum of constituent
    // PxsBodyCore solverIterationCounts. In particular, an unrelated
    // high-iteration island must never increase another group's solve.
    constexpr double max_float =
        static_cast<double>(std::numeric_limits<float>::max());
    const auto dot = [](Vec3 u, Vec3 v) {
        return u.x*v.x + u.y*v.y + u.z*v.z;
    };
    std::size_t task_constraint_begin=0;
    std::size_t task_solver_begin=0;
    for(const auto& task:solver_tasks_) {
    const unsigned position_iterations=task.position_iterations;
    // DySolverControl::solveV_Blocks executes max(requested, 1):
    // a zero requested velocity count skips optional velocity passes,
    // never the mandatory final solve/writeback pass. Preserve the raw
    // requested count in the source task descriptor.
    const unsigned velocity_iterations=std::max<unsigned>(
        task.velocity_iterations,1u);
    for (unsigned iter = 0;
         iter < position_iterations + velocity_iterations; ++iter) {
        // DyConstraintPartition's ordered descriptors, rather than
        // raw broadphase pair order, govern every position/velocity pass.
        // Within each descriptor, PhysX consumes the correlated
        // friction patches in source order: each patch's normal rows
        // followed immediately by its friction rows. Gu force-buffer
        // offsets remain independent from this solver permutation.
        for(std::size_t desc=task_constraint_begin;
            desc<task.constraint_end;++desc) {
            const auto index=ordered_constraints_[desc];
            const auto& manifold = manifolds_[index];
            // Dy descriptors retain Sc's source-oriented body order,
            // including the signs of the contact/friction Jacobians.
            const bool reversed=contact_edges_[
                manifold.edge_handle].gu_order_reversed;
            const BodyId id0=reversed?manifold.pair.b:manifold.pair.a;
            const BodyId id1=reversed?manifold.pair.a:manifold.pair.b;
            auto& a = actors_[id0].kind == Kind::DynamicBox
                ? solver_rows_[id0] : world_solver_row_;
            auto& b = actors_[id1].kind == Kind::DynamicBox
                ? solver_rows_[id1] : world_solver_row_;
            // A zero source maxContactImpulse is omitted by
            // DyContactPrepShared::extractContacts; there is no packed
            // solver header to execute. Gu touch state and its indexed
            // zero force-buffer slots remain owned by this Scene.
            if(manifold.friction_patch_count==0)
                continue;
            std::size_t order_cursor=manifold.solver_offset;
            for(std::size_t p=manifold.friction_patch_offset;
                p<manifold.friction_patch_offset+
                  manifold.friction_patch_count;++p) {
                auto& patch=friction_patches_[p];
                double normal_impulse=0.0;
                for(std::size_t i=0;i<patch.contact_count;++i) {
                // DyContactPrep concatenates this patch's reversed
                // contact-run list, then writes Gu-indexed force slots.
                auto& row=solver_contact_points_[
                    solver_contact_order_[order_cursor++]];
                const Vec3 n = row.normal;
                const Vec3 va{
                    a.solver.linear_velocity_delta.x,
                    a.solver.linear_velocity_delta.y,
                    a.solver.linear_velocity_delta.z};
                const Vec3 vb{
                    b.solver.linear_velocity_delta.x,
                    b.solver.linear_velocity_delta.y,
                    b.solver.linear_velocity_delta.z};
                // PxSolverBody holds only velocity deltas; pre-solver
                // velocity is already included in row.biased_error.
                const float normal_velocity = dot(n,va)-dot(n,vb) +
                    dot(a.solver.angular_state,row.ra_x_n_sqrt_inertia) -
                    dot(b.solver.angular_state,row.rb_x_n_sqrt_inertia);
                const double error = iter < position_iterations
                    ? row.biased_error : row.unbiased_error;
                const double delta_candidate = std::max(
                    -static_cast<double>(normal_velocity) *
                        row.velocity_multiplier + error,
                    -row.applied_force);
                // DySolverConstraintsShared::solveDynamicContacts
                // caps EACH normal row's accumulated impulse, not
                // the manager total and not the body velocity delta.
                // DySolverConstraintsShared multiplies the old force
                // by impulseMultiplier for every compliant PGS solve.
                // Noncompliant rows retain a multiplier of exactly 1.
                const double next_impulse=std::min(row.max_impulse,
                    row.impulse_multiplier*row.applied_force+delta_candidate);
                const double delta = next_impulse-row.applied_force;
                const double da = delta*a.data.inverse_mass;
                const double db = delta*b.data.inverse_mass;
                const std::array<double,12> results{
                    static_cast<double>(va.x)+n.x*da,
                    static_cast<double>(va.y)+n.y*da,
                    static_cast<double>(va.z)+n.z*da,
                    static_cast<double>(vb.x)-n.x*db,
                    static_cast<double>(vb.y)-n.y*db,
                    static_cast<double>(vb.z)-n.z*db,
                    static_cast<double>(a.solver.angular_state.x)+row.ra_x_n_sqrt_inertia.x*delta,
                    static_cast<double>(a.solver.angular_state.y)+row.ra_x_n_sqrt_inertia.y*delta,
                    static_cast<double>(a.solver.angular_state.z)+row.ra_x_n_sqrt_inertia.z*delta,
                    static_cast<double>(b.solver.angular_state.x)-row.rb_x_n_sqrt_inertia.x*delta,
                    static_cast<double>(b.solver.angular_state.y)-row.rb_x_n_sqrt_inertia.y*delta,
                    static_cast<double>(b.solver.angular_state.z)-row.rb_x_n_sqrt_inertia.z*delta};
                std::array<float,12> commit{};
                for (std::size_t k=0;k<results.size();++k) {
                    if (!std::isfinite(results[k]) ||
                        std::abs(results[k]) > max_float)
                        throw std::overflow_error(
                            "Native Dy contact velocity delta exceeds float range");
                    commit[k] = static_cast<float>(results[k]);
                }
                a.solver.linear_velocity_delta = {commit[0],commit[1],commit[2]};
                b.solver.linear_velocity_delta = {commit[3],commit[4],commit[5]};
                a.solver.angular_state = {commit[6],commit[7],commit[8]};
                b.solver.angular_state = {commit[9],commit[10],commit[11]};
                row.applied_force = next_impulse;
                normal_impulse+=next_impulse;
                }
                // DySolverConstraints::solveContact and _BStatic walk
                // one packed SolverContactHeader at a time: each
                // header solves its normal rows before friction rows.
                // The per-patch impulse sum is already available from
                // the current normal pass; no repeated O(patches*points)
                // scan of other contact groups is necessary.
                // DySolverControl.cpp selects friction on the LAST
                // THREE position iterations, regardless of this task's
                // maximum position iteration count. The default 4+1
                // skips its first position pass, while 1+N must solve
                // friction on its only position pass. Velocity passes
                // retain the enabled friction state from conclude.
                if ((iter>=position_iterations ||
                     position_iterations-iter<=3) &&
                    patch.anchor_count) {
                    const double coefficient=patch.anchor_count==2
                        ? 0.5 : 1.0;
                    const double max_static=
                        patch.static_friction*coefficient*normal_impulse;
                    const double max_dynamic=
                        patch.dynamic_friction*coefficient*normal_impulse;
                    // DySolverConstraints.cpp::solveContact and
                    // solveContact_BStatic create a fresh local 'broken'
                    // value on EVERY solve call. Only the final
                    // writeBackIteration publishes it to the friction
                    // patch's cached broken byte. Position-pass
                    // saturation must not invalidate an anchor that has
                    // converged during the unbiased velocity pass.
                    bool broken_this_iteration=false;
                    for (std::size_t k=patch.row_offset;
                         k<patch.row_offset+patch.anchor_count*2; ++k) {
                        auto& fr=solver_friction_rows_[k];
                        const Vec3 n=fr.tangent;
                        const Vec3 va=a.solver.linear_velocity_delta;
                        const Vec3 vb=b.solver.linear_velocity_delta;
                        const float tangent_delta=
                            dot(n,va)-dot(n,vb)+
                            dot(a.solver.angular_state,fr.ra_x_t_sqrt_inertia)-
                            dot(b.solver.angular_state,fr.rb_x_t_sqrt_inertia);
                        // concludeContact zeroes friction bias on the
                        // final position pass, before motion-velocity
                        // publication and the final velocity iteration.
                        const double bias=iter < position_iterations
                            ? fr.bias : 0.0;
                        const double candidate=fr.applied_force+
                            (fr.target_velocity-bias-tangent_delta)*
                                fr.velocity_multiplier;
                        const bool clamped=std::abs(candidate)>max_static;
                        // Preserve the source's signed FMin/FMax
                        // expression without invoking std::clamp on
                        // negative, unordered compliant patch bounds.
                        const double next=source_contact_friction_limit(
                            candidate,max_dynamic,clamped);
                        const double delta=next-fr.applied_force;
                        const double da=delta*a.data.inverse_mass;
                        const double db=delta*b.data.inverse_mass;
                        const auto& j0=fr.ra_x_t_sqrt_inertia;
                        const auto& j1=fr.rb_x_t_sqrt_inertia;
                        const std::array<double,12> results{
                            static_cast<double>(va.x)+n.x*da,
                            static_cast<double>(va.y)+n.y*da,
                            static_cast<double>(va.z)+n.z*da,
                            static_cast<double>(vb.x)-n.x*db,
                            static_cast<double>(vb.y)-n.y*db,
                            static_cast<double>(vb.z)-n.z*db,
                            static_cast<double>(a.solver.angular_state.x)+j0.x*delta,
                            static_cast<double>(a.solver.angular_state.y)+j0.y*delta,
                            static_cast<double>(a.solver.angular_state.z)+j0.z*delta,
                            static_cast<double>(b.solver.angular_state.x)-j1.x*delta,
                            static_cast<double>(b.solver.angular_state.y)-j1.y*delta,
                            static_cast<double>(b.solver.angular_state.z)-j1.z*delta};
                        std::array<float,12> commit{};
                        for (std::size_t v=0;v<results.size();++v) {
                            if (!std::isfinite(results[v]) ||
                                std::abs(results[v]) > max_float)
                                throw std::overflow_error(
                                    "Native Dy friction velocity delta exceeds float range");
                            commit[v]=static_cast<float>(results[v]);
                        }
                        a.solver.linear_velocity_delta={
                            commit[0],commit[1],commit[2]};
                        b.solver.linear_velocity_delta={
                            commit[3],commit[4],commit[5]};
                        a.solver.angular_state={
                            commit[6],commit[7],commit[8]};
                        b.solver.angular_state={
                            commit[9],commit[10],commit[11]};
                        fr.applied_force=next;
                        broken_this_iteration |= clamped;
                    }
                    if (iter + 1 ==
                        position_iterations + velocity_iterations)
                        patch.broken=broken_this_iteration;
                }
            }
            if(order_cursor!=manifold.solver_offset+manifold.count)
                throw std::logic_error(
                    "Native source contact solver patch stream truncated");
        }
        if (iter + 1 == position_iterations) {
            // DySolverControl.cpp::saveMotionVelocities occurs immediately
            // after the last biased/conclude position iteration, BEFORE
            // unbiased velocity iterations alter the solver deltas.
            for(std::size_t body=task_solver_begin;
                body<task.solver_body_end;++body) {
                const BodyId id=solver_active_ids_[body];
                auto& row = solver_rows_[id];
                row.motion_linear_delta = row.solver.linear_velocity_delta;
                row.motion_angular_state = row.solver.angular_state;
            }
        }
    }
    task_constraint_begin=task.constraint_end;
    task_solver_begin=task.solver_body_end;
    }
}
void Scene::publish_contact_events() {
    if(!contact_notifications_)return;
    // ScNPhaseCore::processPersistentContactEvents reports touching
    // interactions only while at least one rigid body is active.
    // The found/lost lists and sorted Gu managers belong to this very
    // scene: no shadow contact-pair history or global event registry.
    next_contact_events_.clear();
    next_contact_event_points_.clear();
    // Source ScNPhaseCore reports only active touching interactions
    // (or a newly found touch), not every retained Gu manager in a
    // sleeping accurate island. Collect the same Dy manager inputs and
    // all Found touches, then restore canonical Gu-index order before
    // combining with the independently ordered Lost stream.
    contact_report_managers_.clear();
    contact_report_managers_.insert(contact_report_managers_.end(),
        indexed_managers_.begin(),indexed_managers_.end());
    for(Pair pair:found_contacts_) {
        const EdgeIndex edge=contact_edge_index(pair);
        if(edge==invalid_edge || edge>=manifold_by_edge_.size() ||
           manifold_by_edge_[edge].generation!=manifold_lookup_generation_ ||
           manifold_by_edge_[edge].index>=manifolds_.size() ||
           !(manifolds_[manifold_by_edge_[edge].index].pair==pair))
            throw std::logic_error(
                "Sc found touch has no published Gu contact manager");
        contact_report_managers_.push_back(manifold_by_edge_[edge].index);
    }
    std::sort(contact_report_managers_.begin(),
              contact_report_managers_.end());
    contact_report_managers_.erase(
        std::unique(contact_report_managers_.begin(),
                    contact_report_managers_.end()),
        contact_report_managers_.end());
    // Active manager order comes from islands, not Gu. Its original
    // contact-point windows and the Sc event merge use Gu indices.
    next_contact_events_.reserve(
        contact_report_managers_.size()+lost_contacts_.size());
    std::size_t report_points=0;
    for(std::size_t index:contact_report_managers_)
        report_points+=manifolds_[index].count;
    next_contact_event_points_.reserve(report_points);
    std::size_t lost=0,found=0;
    for(std::size_t index:contact_report_managers_) {
        const auto& manifold=manifolds_[index];
        while(lost<lost_contacts_.size() &&
              lost_contacts_[lost]<manifold.pair) {
            const Pair pair=lost_contacts_[lost++];
            next_contact_events_.push_back({
                pair.a,pair.b,ContactEventKind::Lost,0,
                next_contact_event_points_.size(),0.0});
        }
        while(found<found_contacts_.size() &&
              found_contacts_[found]<manifold.pair)++found;
        const bool is_new=found<found_contacts_.size() &&
            found_contacts_[found]==manifold.pair;
        const auto& first=actors_[manifold.pair.a];
        const auto& second=actors_[manifold.pair.b];
        const bool active=
            (first.kind==Kind::DynamicBox && !first.sleeping) ||
            (second.kind==Kind::DynamicBox && !second.sleeping);
        if(!is_new && !active)continue;
        if(manifold.contact_offset>normal_impulse_writeback_.size() ||
           manifold.count>normal_impulse_writeback_.size()-
                              manifold.contact_offset)
            throw std::logic_error(
                "Contact notification references unpublished Gu forces");
        double impulse=0.0;
        const auto offset=next_contact_event_points_.size();
        for(std::size_t i=0;i<manifold.count;++i) {
            const std::size_t index=manifold.contact_offset+i;
            const auto& point=contact_points_[index];
            const double normal_force=normal_impulse_writeback_[index];
            impulse+=normal_force;
            Vec3 outward=point.normal;
            if(contact_edges_[manifold.edge_handle].gu_order_reversed)
                outward={-outward.x,-outward.y,-outward.z};
            next_contact_event_points_.push_back({
                point.position,outward,point.separation,normal_force});
        }
        if(!std::isfinite(impulse))
            throw std::overflow_error(
                "Contact notification impulse is not finite");
        next_contact_events_.push_back({
            manifold.pair.a,manifold.pair.b,
            is_new?ContactEventKind::Found:ContactEventKind::Persists,
            manifold.count,offset,impulse});
    }
    for(;lost<lost_contacts_.size();++lost) {
        const Pair pair=lost_contacts_[lost];
        next_contact_events_.push_back({
            pair.a,pair.b,ContactEventKind::Lost,0,
            next_contact_event_points_.size(),0.0});
    }
    // Publish only after every Sc/Dy stage and this event preparation
    // succeeds. A rejected step leaves the previous returned span and
    // backing storage untouched; the scratch is reused on retry.
    contact_event_points_.swap(next_contact_event_points_);
    contact_events_.swap(next_contact_events_);
}
void Scene::write_back_contacts() {
    // DySolverConstraints.cpp::writeBackContact copies final forceBuffer
    // impulses after the last velocity/writeback solve. Gu contact order,
    // not the reordered Dy partition list, owns output offsets. Here the
    // source force buffer is represented by SolverContactPoint::applied_force
    // and the output retains SDK-wide impulses for admitted extreme masses.
    // PhysX's Gu force buffer remains indexed by the original contact
    // arena even though Dy executes island-ordered contact managers.
    // A single contiguous clear establishes the zero-impulse result for
    // every dormant or zero-response Gu contact. Do not scan the dormant
    // contact-manager inventory merely to clear each contact window.
    normal_impulse_writeback_.assign(contact_points_.size(),0.0);
    friction_impulse_writeback_.resize(friction_patches_.size());
    // The checkpoint owns the prior successful Dy manager sequence.
    // Only those managers could still carry exposed Dy offsets when an
    // accurate island falls asleep without Gu republication. Source
    // publication that rotated the Gu arena constructs fresh, zeroed
    // metadata, so unrelated historical indices are skipped safely.
    for(std::size_t previous:step_checkpoint_.indexed_managers) {
        if(previous>=manifolds_.size())continue;
        auto& old=manifolds_[previous];
        if(old.prepared_friction_epoch==friction_cache_epoch_ ||
           (old.solver_offset==0 && old.friction_patch_offset==0 &&
            old.friction_patch_count==0 &&
            old.prepared_friction_epoch==0))
            continue;
        step_checkpoint_.save_manifold_solver_metadata(*this,previous);
        old.solver_offset=0;
        old.friction_patch_offset=0;
        old.friction_patch_count=0;
        old.prepared_friction_epoch=0;
    }
    // Both the original contact-force offsets and patch ownership are
    // checked only for managers actually admitted to current Dy tasks.
    // This is O(active contact points + prior active managers) besides
    // the unavoidable dense Gu output zero-fill.
    for(std::size_t manifold_index:indexed_managers_) {
        const auto& manifold=manifolds_[manifold_index];
        if(manifold.prepared_friction_epoch!=friction_cache_epoch_ ||
           manifold.contact_offset>contact_points_.size() ||
           manifold.count>contact_points_.size()-manifold.contact_offset)
            throw std::logic_error(
                "Active Dy writeback references invalid Gu contact owner");
        if(manifold.friction_patch_count==0)continue;
        if(manifold.solver_offset > solver_contact_points_.size() ||
           manifold.count > solver_contact_points_.size()-manifold.solver_offset)
            throw std::logic_error(
                "Native Dy contact writeback offset exceeds solver rows");
        for (std::size_t i=0;i<manifold.count;++i) {
            const double impulse=solver_contact_points_[
                manifold.solver_offset+i].applied_force;
            // Source DySolverConstraintsShared::solveDynamicContacts
            // clamps the compliant recurrence from above, but does NOT
            // clamp it to zero after multiplying the previous impulse
            // by a coefficient below one. DySolverConstraints::writeBackContact
            // copies that signed PxReal force buffer without rejecting it.
            // Reject only nonfinite state: clipping here or in the solver
            // would replace the source's actual spring recurrence.
            if (!std::isfinite(impulse))
                throw std::overflow_error(
                    "Native Dy contact writeback has nonfinite normal impulse");
            normal_impulse_writeback_[manifold.contact_offset+i]=impulse;
        }
        if (manifold.friction_patch_offset>friction_patches_.size() ||
            manifold.friction_patch_count>
                friction_patches_.size()-manifold.friction_patch_offset)
            throw std::logic_error(
                "Native Dy friction writeback exceeds correlated patches");
        for (std::size_t p=manifold.friction_patch_offset;
             p<manifold.friction_patch_offset+
                 manifold.friction_patch_count; ++p) {
            const auto& source=friction_patches_[p];
            if (!(source.pair==manifold.pair) || source.anchor_count>2 ||
                source.row_offset>solver_friction_rows_.size() ||
                source.anchor_count*2>
                    solver_friction_rows_.size()-source.row_offset)
                throw std::logic_error(
                    "Native Dy friction writeback patch ownership invalid");
            FrictionPatchWriteback& result=friction_impulse_writeback_[p];
            result={};
            result.pair=source.pair;
            result.anchor_count=source.anchor_count;
            result.broken=source.broken;
            // DyFrictionCorrelation stores body0 anchors in Sc's
            // source-ordered frame, not the canonical Gu report pair.
            const BodyId body0=contact_edges_[manifold.edge_handle]
                .gu_order_reversed ? source.pair.b : source.pair.a;
            const auto& actor=actors_[body0];
            for (std::size_t k=0;k<source.anchor_count;++k) {
                const Vec3 local=source.body0_anchors[k];
                const Vec3 rotated=rotate(actor.orientation,local);
                result.anchor_positions[k]={
                    actor.position.x+rotated.x,
                    actor.position.y+rotated.y,
                    actor.position.z+rotated.z};
                // DyContactPrep.cpp::writeBackContactFriction reconstructs
                // world impulse by summing both tangent vectors scaled
                // by each source SolverContactFriction::appliedForce.
                const auto& x=solver_friction_rows_[source.row_offset+2*k];
                const auto& y=solver_friction_rows_[source.row_offset+2*k+1];
                result.anchor_impulses[k]={
                    static_cast<double>(x.tangent.x)*x.applied_force+
                        static_cast<double>(y.tangent.x)*y.applied_force,
                    static_cast<double>(x.tangent.y)*x.applied_force+
                        static_cast<double>(y.tangent.y)*y.applied_force,
                    static_cast<double>(x.tangent.z)*x.applied_force+
                        static_cast<double>(y.tangent.z)*y.applied_force};
                const auto& v=result.anchor_impulses[k];
                if (!finite(result.anchor_positions[k]) ||
                    !std::isfinite(v.x) || !std::isfinite(v.y) ||
                    !std::isfinite(v.z))
                    throw std::overflow_error(
                        "Native Dy friction impulse writeback is not finite");
            }
        }
    }
}
void Scene::update_threshold_stream(float seconds) {
    // DySolverConstraints.cpp::writeBackContact emits a threshold element
    // only for a supported rigid-body manager with a nonzero final
    // normal impulse and an enabled reportThreshold on either body.
    // DyThresholdTable groups all static contacts against a dynamic body
    // using the SAME source world solver node. Do not key by the Gu
    // contact pair: distinct static shapes must sum before comparison.
    // A dynamic/dynamic contact retains its canonical body-pair key.
    //
    // DyThresholdTable::build aggregates impulses by rigid-body pair and
    // PxsForceThresholdTask::createForceChangeThresholdStream compares
    // aggregate impulse against threshold*dt. Never divide impulse by
    // dt and narrow to PxReal: this SDK also admits extreme mass cases.
    threshold_current_.clear();
    threshold_changes_.clear();
    // PxsForceThresholdTask consumes Dy writeback from the CURRENT
    // source solve tasks, not historical Gu managers in dormant islands.
    // Old exceeded-force records are reconciled below independently,
    // so a once-active pair still emits LOST when it stops being solved.
    // Keep each authoritative Gu contact_offset for impulse indexing.
    for(std::size_t index:indexed_managers_) {
        const auto& manifold=manifolds_[index];
        const Pair pair=manifold.pair;
        const bool dynamic_a=actors_[pair.a].kind==Kind::DynamicBox;
        const bool dynamic_b=actors_[pair.b].kind==Kind::DynamicBox;
        if (!dynamic_a && !dynamic_b)
            continue;
        const bool static_world=!(dynamic_a && dynamic_b);
        const ForceThresholdKey key=static_world
            ? ForceThresholdKey{dynamic_a ? pair.a : pair.b,0,true}
            : ForceThresholdKey{pair.a,pair.b,false};
        const float threshold=static_world
            ? actors_[key.actor0].contact_report_threshold
            : std::min(actors_[pair.a].contact_report_threshold,
                       actors_[pair.b].contact_report_threshold);
        if (threshold==std::numeric_limits<float>::max())
            continue;
        if (manifold.contact_offset>normal_impulse_writeback_.size() ||
            manifold.count>normal_impulse_writeback_.size()-
                manifold.contact_offset)
            throw std::logic_error(
                "Native Dy threshold stream exceeds source force writeback");
        double sum=0.0;
        for (std::size_t i=0;i<manifold.count;++i)
            sum+=normal_impulse_writeback_[manifold.contact_offset+i];
        if (!std::isfinite(sum))
            throw std::overflow_error(
                "Native Dy threshold normal impulse accumulation overflow");
        if (sum==0.0)
            continue;
        threshold_current_.push_back({key,sum,threshold});
    }
    // Map Gu manager pairs onto the shared static-world identity BEFORE
    // DyThresholdTable aggregation. The mapped sequence is no longer
    // necessarily ordered (different static actors interleave), so sort
    // only these active threshold candidates, in the reusable arena.
    // Do not allocate a parallel hash table or reorder Gu contacts.
    std::sort(threshold_current_.begin(),threshold_current_.end(),
        [](const ThresholdPair& a, const ThresholdPair& b) {
            return a.key<b.key;
        });
    std::size_t write=0;
    for (const auto& row:threshold_current_) {
        if (write && threshold_current_[write-1].key==row.key) {
            threshold_current_[write-1].impulse+=row.impulse;
            threshold_current_[write-1].threshold=std::min(
                threshold_current_[write-1].threshold,row.threshold);
        } else {
            threshold_current_[write++]=row;
        }
    }
    threshold_current_.resize(write);
    // Source tests accumulated per-pair impulse *strictly greater*
    // than reportThreshold * dt. This runs after final force writeback.
    write=0;
    for (const auto& row:threshold_current_) {
        if (row.impulse>static_cast<double>(row.threshold)*seconds)
            threshold_current_[write++]=row;
    }
    threshold_current_.resize(write);
    // Previous/current exceeded-force stream merge. The source also
    // includes persistent exceeded pairs in its force-change stream;
    // LOST events have zero accumulated force, while PERSISTS uses the
    // preceding element's force (as in PxsForceThresholdTask).
    std::size_t old_index=0,new_index=0;
    while (old_index<threshold_previous_.size() ||
           new_index<threshold_current_.size()) {
        const bool has_old=old_index<threshold_previous_.size();
        const bool has_new=new_index<threshold_current_.size();
        if (has_old &&
            (!has_new || threshold_previous_[old_index].key <
                         threshold_current_[new_index].key)) {
            const auto& old=threshold_previous_[old_index++];
            threshold_changes_.push_back({
                old.key,0.0,old.threshold,ForceThresholdChange::Lost});
        } else if (has_new &&
                   (!has_old || threshold_current_[new_index].key <
                                threshold_previous_[old_index].key)) {
            const auto& now=threshold_current_[new_index++];
            threshold_changes_.push_back({
                now.key,now.impulse,now.threshold,
                ForceThresholdChange::Found});
        } else {
            const auto& old=threshold_previous_[old_index++];
            ++new_index;
            threshold_changes_.push_back({
                old.key,old.impulse,old.threshold,
                ForceThresholdChange::Persists});
        }
    }
    threshold_previous_.swap(threshold_current_);
    ++threshold_cache_epoch_;
}
void Scene::publish_solver_bodies() {
    // The source retains pre-solver velocity independently and publishes
    // solved motion velocity after the iteration/conclude/writeback sequence.
    // The source's rigid-body normal-contact rows now update both linear
    // and angular velocity state. Friction/correlation/writeback parity
    // remains a separate N1 acceptance requirement.
    for (BodyId id : solver_active_ids_) {
        auto& actor = actors_[id];
        if (!actor.alive || actor.kind != Kind::DynamicBox) continue;
        const auto& row = solver_rows_[id];
        const Vec3& d = row.solver.linear_velocity_delta;
        Vec3 linear{
            row.data.linear_velocity.x+d.x,
            row.data.linear_velocity.y+d.y,
            row.data.linear_velocity.z+d.z};
        lock_linear_axes(linear,actor.dynamics.lock_flags);
        // DyBodyCoreIntegrator::integrateCore publishes angular solver
        // state through the *world* square-root inverse inertia tensor,
        // rather than treating PxSolverBody::angularState as angular velocity.
        Vec3 angular_state=row.solver.angular_state;
        // integrateCore masks solver-space angular components before
        // multiplication by the world sqrt-inverse-inertia tensor.
        lock_angular_axes(angular_state,actor.dynamics.lock_flags);
        const auto& i = row.data.sqrt_inverse_inertia;
        const Vec3 angular{
            row.data.angular_velocity.x + i[0].x*angular_state.x +
                i[1].x*angular_state.y + i[2].x*angular_state.z,
            row.data.angular_velocity.y + i[0].y*angular_state.x +
                i[1].y*angular_state.y + i[2].y*angular_state.z,
            row.data.angular_velocity.z + i[0].z*angular_state.x +
                i[1].z*angular_state.y + i[2].z*angular_state.z};
        if (!finite(linear) || !finite(angular))
            throw std::overflow_error("Native solver publication is not finite");
        step_checkpoint_.save_motion(*this,id);
        actor.velocity = linear;
        actor.angular_velocity = angular;
    }
}
void Scene::StepCheckpoint::save_motion(Scene& scene, BodyId id) noexcept {
    if(!includes_solver_state || !scene.activity_checkpoint_active_)return;
    if(scene.motion_undo_generations_[id]==scene.motion_undo_generation_)
        return;
    const Actor& body=scene.actors_[id];
    if(!body.alive || body.kind!=Kind::DynamicBox)return;
    scene.motion_undo_generations_[id]=scene.motion_undo_generation_;
    // capture() already reserved one record for each live dynamic body;
    // neither a speculative wake nor source deactivation can allocate.
    motions.push_back({id,body.position,body.velocity,body.angular_velocity,
        body.orientation,body.pending_angular_velocity_delta,
        body.pending_linear_velocity_delta,body.pending_linear_acceleration,
        body.pending_angular_acceleration,
        body.wake_counter,body.sleep_linear_acc,body.sleep_angular_acc,
        body.ready_for_sleep,body.sleeping,body.activating,
        body.encoded_bounds,body.cached_bounds_current});
}
void Scene::StepCheckpoint::save_manifold_solver_metadata(
    Scene& scene, std::size_t index) {
    // Gu arena rotation already restores these fields for a dirty step.
    // Only the quiet, previously published arena needs an undo record.
    // Generation stamps ensure multiple writes to one live manager save
    // exactly its pre-step descriptor, not intermediate Dy offsets.
    if(!scene.activity_checkpoint_active_ || !includes_solver_state ||
       !quiet_manifold_undo_enabled ||
       scene.contact_epoch_!=contact_epoch)return;
    if(scene.manifold_undo_generations_[index]==
       scene.manifold_undo_generation_)return;
    const auto& manifold=scene.manifolds_[index];
    quiet_manifold_metadata.push_back({
        index,manifold.solver_offset,manifold.friction_patch_offset,
        manifold.friction_patch_count,manifold.prepared_friction_epoch});
    scene.manifold_undo_generations_[index]=
        scene.manifold_undo_generation_;
}
void Scene::StepCheckpoint::save_edge(const Scene& scene, EdgeIndex id) {
    // Entries are recorded immediately before mutation. Multiple writes
    // to the same slot are permitted; reverse replay restores its original
    // contents without a search, a dirty bitmap or per-frame clearing of
    // a high-watermark array. Newly allocated slots are simply discarded.
    if(id<initial_edges)
        edge_undo.push_back({id,scene.contact_edges_[id]});
}
void Scene::StepCheckpoint::save_activity(
    Scene& scene, EdgeIndex id) noexcept {
    if(!scene.activity_checkpoint_active_ || id>=initial_edges)return;
    // Generation-deduplicated and pre-reserved before publication.
    if(scene.activity_undo_generations_[id]==
       scene.activity_undo_generation_)return;
    scene.activity_undo_generations_[id]=scene.activity_undo_generation_;
    const auto& edge=scene.contact_edges_[id];
    activity_undo.push_back({id,edge.speculative_state,edge.accurate_state,
                             edge.interaction_active});
}
void Scene::StepCheckpoint::save_reference(
    Scene& scene, BodyId node) noexcept {
    if(!scene.activity_checkpoint_active_ ||
       node>=initial_speculative_references ||
       node>=initial_accurate_references)return;
    if(scene.reference_undo_generations_[node]==
       scene.reference_undo_generation_)return;
    scene.reference_undo_generations_[node]=
        scene.reference_undo_generation_;
    reference_undo.push_back({node,
        scene.speculative_islands_.active_reference_counts[node],
        scene.accurate_islands_.active_reference_counts[node]});
}
void Scene::StepCheckpoint::save_hint(
    const Scene& scene, BodyId node, bool accurate) {
    const auto& routes=accurate?scene.accurate_fast_route_:
                                 scene.speculative_fast_route_;
    const auto& edges=accurate?scene.accurate_fast_edge_:
                                scene.speculative_fast_edge_;
    const auto& hops=accurate?scene.accurate_hops_:
                               scene.speculative_hops_;
    const auto& stamps=accurate?scene.accurate_hint_stamps_:
                                 scene.speculative_hint_stamps_;
    const auto initial=accurate?initial_accurate_hints:
                                initial_speculative_hints;
    // Record before the first store. A failing journal allocation leaves
    // the previous hint untouched. Slots added after capture are truncated.
    if(node<initial)
        hint_undo.push_back({node,routes[node],edges[node],
                             hops[node],stamps[node],accurate});
}
void Scene::StepCheckpoint::save_head(
    const Scene& scene, BodyId node, bool accurate) {
    const auto& heads=accurate ? scene.accurate_node_first_ :
                                scene.speculative_node_first_;
    const auto old_size=accurate ? initial_accurate_heads :
                                   initial_speculative_heads;
    if(node<old_size)
        head_undo.push_back({node,accurate,heads[node]});
}
void Scene::StepCheckpoint::save_handle_lists(const Scene& scene) {
    if(handles_saved)return;
    // Existing source handle arrays are copied only when overlap
    // membership or deferred retirement actually changes. Gu-only
    // accurate touch edits need only the small per-edge undo journal.
    active_handles.assign(scene.active_edge_handles_.begin(),
                          scene.active_edge_handles_.end());
    retired_handles.assign(scene.retired_edge_handles_.begin(),
                           scene.retired_edge_handles_.end());
    free_handles.assign(scene.free_edge_handles_.begin(),
                        scene.free_edge_handles_.end());
    handles_saved=true;
}
void Scene::StepCheckpoint::save_graph_ownership(Scene& scene) {
    if(graph_ownership_saved)return;
    // Snapshot the single authoritative source island owner only before
    // its first mutation; quiet frames should never copy node chains.
    const auto save=[](IslandGraph& saved,const IslandGraph& live) {
        saved.islands.assign(live.islands.begin(),live.islands.end());
        saved.node_island.assign(live.node_island.begin(),live.node_island.end());
        saved.node_next.assign(live.node_next.begin(),live.node_next.end());
        saved.node_prev.assign(live.node_prev.begin(),live.node_prev.end());
        saved.node_static_touches.assign(live.node_static_touches.begin(),
                                          live.node_static_touches.end());
        saved.free_handles.assign(live.free_handles.begin(),
                                   live.free_handles.end());
        // A rejected structural rebuild must also restore the stamp
        // storage to the PREVIOUS island-handle watermark. Otherwise a
        // later no-throw active-node handoff may index a shorter array.
        saved.active_island_seen.assign(live.active_island_seen.begin(),
                                        live.active_island_seen.end());
        saved.active_island_visit_stamp=live.active_island_visit_stamp;
    };
    save(speculative_islands,scene.speculative_islands_);
    save(accurate_islands,scene.accurate_islands_);
    accurate_nodes.assign(scene.accurate_island_nodes_.begin(),
                           scene.accurate_island_nodes_.end());
    accurate_offsets.assign(scene.accurate_island_offsets_.begin(),
                             scene.accurate_island_offsets_.end());
    graph_ownership_saved=true;
    ++scene.island_graph_checkpoint_count_;
}
void Scene::StepCheckpoint::capture(Scene& scene, bool capture_solver_state) {
    // Reuse capacities across frames. The checkpoint contains only state
    // that persists beyond a step or is exposed by Scene's read-only APIs.
    // It intentionally excludes rebuilt solver constraints and scratch.
    motions.clear();
    if(capture_solver_state) {
        // Activation can reach actors outside the previous active list.
        // Reserve before entering the noexcept island publication, then
        // journal each mutated dynamic body at most once this frame.
        if(++scene.motion_undo_generation_==0) {
            std::fill(scene.motion_undo_generations_.begin(),
                      scene.motion_undo_generations_.end(),0);
            scene.motion_undo_generation_=1;
        }
        scene.motion_undo_generations_.resize(scene.actors_.size(),0);
        motions.reserve(scene.live_dynamic_count_);
    }
    // Source EdgeInstance writes are journaled only when the contact
    // manager graph actually changes. Snapshotting the slot watermark
    // would duplicate every historical intrusive edge four times on
    // every PGS step even when all connections remain unchanged.
    edge_undo.clear();
    activity_undo.clear();
    reference_undo.clear();
    if(++scene.reference_undo_generation_==0) {
        std::fill(scene.reference_undo_generations_.begin(),
                  scene.reference_undo_generations_.end(),0);
        scene.reference_undo_generation_=1;
    }
    scene.reference_undo_generations_.resize(scene.actors_.size(),0);
    reference_undo.reserve(scene.actors_.size());
    initial_speculative_references=
        scene.speculative_islands_.active_reference_counts.size();
    initial_accurate_references=
        scene.accurate_islands_.active_reference_counts.size();
    if(++scene.activity_undo_generation_==0) {
        std::fill(scene.activity_undo_generations_.begin(),
                  scene.activity_undo_generations_.end(),0);
        scene.activity_undo_generation_=1;
    }
    scene.activity_undo_generations_.resize(scene.contact_edges_.size(),0);
    activity_undo.reserve(scene.contact_edges_.size());
    speculative_islands.activated_edges.reserve(
        scene.contact_edges_.size());
    accurate_islands.activated_edges.reserve(
        scene.contact_edges_.size());
    hint_undo.clear();
    head_undo.clear();
    initial_edges=scene.contact_edges_.size();
    initial_active_interactions=scene.active_contact_interaction_count_;
    initial_speculative_hints=scene.speculative_fast_route_.size();
    initial_accurate_hints=scene.accurate_fast_route_.size();
    speculative_hint_epoch=scene.speculative_hint_epoch_;
    accurate_hint_epoch=scene.accurate_hint_epoch_;
    initial_speculative_heads=scene.speculative_node_first_.size();
    initial_accurate_heads=scene.accurate_node_first_.size();
    handles_saved=false;
    graph_ownership_saved=false;
    // Source sleep-edge publication is noexcept after successful Gu
    // commit. Resize the generation scratch before any scene mutation.
    scene.speculative_sleep_seen_.resize(scene.actors_.size(),0);
    scene.speculative_sleep_order_.reserve(scene.actors_.size());
    scene.speculative_pre_retirement_index_.resize(
        scene.speculative_islands_.islands.size());
    // Collision-only advancement does not modify body motion, retained
    // friction patches, threshold history, or solver partition ends.
    // Avoid snapshotting any of these on editor/geometry-only queries.
    quiet_manifold_metadata.clear();
    quiet_manifold_undo_enabled=capture_solver_state && !scene.dirty_;
    if(quiet_manifold_undo_enabled) {
        if(++scene.manifold_undo_generation_==0) {
            std::fill(scene.manifold_undo_generations_.begin(),
                      scene.manifold_undo_generations_.end(),0);
            scene.manifold_undo_generation_=1;
        }
        scene.manifold_undo_generations_.resize(scene.manifolds_.size(),0);
    }
    if(capture_solver_state) {
        partition_ends.assign(scene.partition_ends_.begin(),
                              scene.partition_ends_.end());
    }
    // Active-index publication is observable even on quiet frames, so
    // retain its exact permutation. Membership and handles are lazily
    // snapshotted by save_graph_ownership() only when changed.
    const auto copy_active=[](IslandGraph& saved,const IslandGraph& live) {
        saved.active_nodes.assign(live.active_nodes.begin(),
                                   live.active_nodes.end());
        saved.active_node_index.assign(live.active_node_index.begin(),
                                        live.active_node_index.end());
        saved.active_islands.assign(live.active_islands.begin(),
                                     live.active_islands.end());
        saved.activating_nodes.assign(live.activating_nodes.begin(),
                                      live.activating_nodes.end());
        saved.nodes_to_deactivate.assign(live.nodes_to_deactivate.begin(),
                                          live.nodes_to_deactivate.end());
        // Per-edge reference mutations are restored through the sparse
        // endpoint journal. No O(all actors) counter-array snapshot is
        // needed for a normal rigid PGS frame.
        saved.active_edge_count=live.active_edge_count;
        saved.initial_active_node_count=live.initial_active_node_count;
    };
    copy_active(speculative_islands,scene.speculative_islands_);
    copy_active(accurate_islands,scene.accurate_islands_);
    // Copies preserve vector elements, not spare capacity. A successful
    // checkpoint must hand off full per-actor capacity to the scene:
    // third-pass deactivation and geometry-only active-node reconciliation
    // are noexcept AFTER the collision transaction commits. Reserving
    // here is still inside the throwing, pre-publication checkpoint.
    // Capacity is reused over successive frames, including failed steps.
    speculative_islands.active_nodes.reserve(scene.actors_.size());
    accurate_islands.active_nodes.reserve(scene.actors_.size());
    // A copied vector's capacity is only its current count. Both
    // source mActiveIslands owners must retain full no-throw wake
    // capacity after checkpoint publication or rejected rollback.
    speculative_islands.active_islands.reserve(scene.actors_.size());
    accurate_islands.active_islands.reserve(scene.actors_.size());
    speculative_islands.activating_nodes.reserve(scene.actors_.size());
    accurate_islands.activating_nodes.reserve(scene.actors_.size());
    // Both dedup stamp arrays are scratch and remain in the scene;
    // resize them before any noexcept wake publication is possible.
    scene.speculative_islands_.activating_seen.resize(
        scene.actors_.size(),0);
    scene.accurate_islands_.activating_seen.resize(
        scene.actors_.size(),0);
    // Reserve only the working copies, never the published arrays:
    // rejected frames restore the exact original active-island storage.
    accurate_islands.nodes_to_deactivate.reserve(scene.actors_.size());
    speculative_islands.deactivating_edges.reserve(
        scene.contact_edges_.size());
    accurate_islands.deactivating_edges.reserve(
        scene.contact_edges_.size());
    speculative_count=scene.speculative_island_count_;
    accurate_count=scene.accurate_island_count_;
    topology_revision=scene.island_topology_revision_;
    full_rebuild_count=scene.island_full_rebuild_count_;
    local_split_count=scene.island_local_split_count_;
    graph_checkpoint_count=scene.island_graph_checkpoint_count_;
    dirty_route_checks=scene.island_dirty_route_checks_;
    fast_route_hits=scene.island_fast_route_hits_;
    frame_index=scene.frame_index_;
    solved_result_frame=scene.solved_result_frame_;
    pair_epoch=scene.pair_epoch_;
    contact_epoch=scene.contact_epoch_;
    manifold_lookup_generation=scene.manifold_lookup_generation_;
    quiet_event_epoch=scene.quiet_event_epoch_;
    topology_dirty=scene.island_topology_dirty_;
    dirty=scene.dirty_;
    authored_activation_pending=scene.authored_activation_pending_;
    includes_solver_state=capture_solver_state;
    friction_cache_epoch=scene.friction_cache_epoch_;
    threshold_cache_epoch=scene.threshold_cache_epoch_;
    // The prior published buffers are moved into the checkpoint only
    // after all potentially throwing copies have completed. Successful
    // solves write directly into the reusable alternate allocations.
    // Rollback reverses these swaps without copying or invalidating
    // previously published report spans.
    normal_impulses.clear();
    friction_impulses.clear();
    threshold_changes.clear();
    scene.normal_impulse_writeback_.swap(normal_impulses);
    scene.friction_impulse_writeback_.swap(friction_impulses);
    scene.threshold_changes_.swap(threshold_changes);
    // The copied arrays become the mutable frame's working arrays.
    // The checkpoint retains the actual *published* vector allocations,
    // including retained spans into their contents. A rejected frame
    // swaps these originals back without invalidating exposed spans.
    // The working deactivation output retains actors_.size() capacity
    // for the subsequent noexcept third-pass sleep decision.
    scene.speculative_islands_.active_nodes.swap(
        speculative_islands.active_nodes);
    scene.accurate_islands_.active_nodes.swap(accurate_islands.active_nodes);
    scene.speculative_islands_.active_islands.swap(
        speculative_islands.active_islands);
    scene.accurate_islands_.active_islands.swap(
        accurate_islands.active_islands);
    scene.speculative_islands_.activating_nodes.swap(
        speculative_islands.activating_nodes);
    scene.accurate_islands_.activating_nodes.swap(
        accurate_islands.activating_nodes);
    scene.accurate_islands_.nodes_to_deactivate.swap(
        accurate_islands.nodes_to_deactivate);
    scene.speculative_islands_.deactivating_edges.swap(
        speculative_islands.deactivating_edges);
    scene.accurate_islands_.deactivating_edges.swap(
        accurate_islands.deactivating_edges);
    scene.speculative_islands_.activated_edges.swap(
        speculative_islands.activated_edges);
    scene.accurate_islands_.activated_edges.swap(
        accurate_islands.activated_edges);
    // The checkpoint owns the prior published lists; its former scratch
    // buffers must start empty before the next activation publication.
    scene.speculative_islands_.activated_edges.clear();
    scene.accurate_islands_.activated_edges.clear();
    if(capture_solver_state) {
        // The previous active batch view is observable through the
        // source island instrumentation, so rejected solves restore it.
        // Keep both buffers for allocation reuse rather than copying it.
        scene.active_accurate_island_batches_.swap(active_accurate_batches);
        scene.active_accurate_island_batches_.clear();
        // Preserve any externally retained indexed-manager span on a
        // failed frame, just like the source active-batch view.
        scene.indexed_managers_.swap(indexed_managers);
        scene.indexed_managers_.clear();
    }
    // No throwing operation remains in capture. All journaling is scoped
    // to the transaction and disabled again at commit/rollback.
    scene.activity_checkpoint_active_=true;
}

void Scene::StepCheckpoint::rollback(Scene& scene) noexcept {
    scene.activity_checkpoint_active_=false;
    // Restore the exact pre-publication route cache and epochs so retries
    // reuse valid prior paths. Only written hint slots were journaled;
    // replay and shrinking newly admitted slots cannot allocate.
    for(auto it=hint_undo.rbegin();it!=hint_undo.rend();++it) {
        auto& routes=it->accurate?scene.accurate_fast_route_:
                                   scene.speculative_fast_route_;
        auto& edges=it->accurate?scene.accurate_fast_edge_:
                                  scene.speculative_fast_edge_;
        auto& hops=it->accurate?scene.accurate_hops_:
                                 scene.speculative_hops_;
        auto& stamps=it->accurate?scene.accurate_hint_stamps_:
                                   scene.speculative_hint_stamps_;
        routes[it->node]=it->next;
        edges[it->node]=it->edge;
        hops[it->node]=it->hops;
        stamps[it->node]=it->stamp;
    }
    scene.speculative_fast_route_.resize(initial_speculative_hints);
    scene.speculative_fast_edge_.resize(initial_speculative_hints);
    scene.speculative_hops_.resize(initial_speculative_hints);
    scene.speculative_hint_stamps_.resize(initial_speculative_hints);
    scene.accurate_fast_route_.resize(initial_accurate_hints);
    scene.accurate_fast_edge_.resize(initial_accurate_hints);
    scene.accurate_hops_.resize(initial_accurate_hints);
    scene.accurate_hint_stamps_.resize(initial_accurate_hints);
    scene.speculative_hint_epoch_=speculative_hint_epoch;
    scene.accurate_hint_epoch_=accurate_hint_epoch;
    for(BodyId node:scene.speculative_dirty_nodes_)
        scene.speculative_dirty_map_[node]=0;
    for(BodyId node:scene.accurate_dirty_nodes_)
        scene.accurate_dirty_map_[node]=0;
    scene.speculative_dirty_nodes_.clear();
    scene.accurate_dirty_nodes_.clear();
    // Rewind source edge instances and node heads in reverse mutation
    // order, without allocation. Only touched pre-existing records were
    // copied, so unchanged history never incurs a full watermark copy.
    for(auto it=edge_undo.rbegin();it!=edge_undo.rend();++it)
        scene.contact_edges_[it->id]=it->previous;
    // Structural edge records can be taken after an activity change.
    // The compact activity journal is authoritative for the original
    // eACTIVE/eACTIVATING state and must replay LAST for each old slot.
    for(auto it=activity_undo.rbegin();it!=activity_undo.rend();++it) {
        scene.contact_edges_[it->id].speculative_state=it->speculative;
        scene.contact_edges_[it->id].accurate_state=it->accurate;
        scene.contact_edges_[it->id].interaction_active=
            it->interaction_active;
    }
    while(scene.contact_edges_.size()>initial_edges)
        scene.contact_edges_.pop_back();
    for(auto it=head_undo.rbegin();it!=head_undo.rend();++it) {
        auto& heads=it->accurate ? scene.accurate_node_first_ :
                                  scene.speculative_node_first_;
        heads[it->node]=it->previous;
    }
    while(scene.speculative_node_first_.size()>initial_speculative_heads)
        scene.speculative_node_first_.pop_back();
    while(scene.accurate_node_first_.size()>initial_accurate_heads)
        scene.accurate_node_first_.pop_back();
    if(handles_saved) {
        scene.active_edge_handles_.swap(active_handles);
        scene.retired_edge_handles_.swap(retired_handles);
        scene.free_edge_handles_.swap(free_handles);
    }
    // This merge buffer is scratch, cleared before every publication.
    scene.next_edge_handles_.clear();
    // Swaps cannot allocate or fail with the standard vector allocator.
    // In particular, pending angular impulses and the authoritative
    // previous-frame friction/threshold streams survive failed solves.
    if(includes_solver_state) {
        for(const Motion& motion:motions) {
            auto& a=scene.actors_[motion.id];
            a.position=motion.position;
            a.velocity=motion.velocity;
            a.angular_velocity=motion.angular_velocity;
            a.orientation=motion.orientation;
            a.pending_angular_velocity_delta=motion.pending_angular_velocity_delta;
            a.pending_linear_velocity_delta=motion.pending_linear_velocity_delta;
            a.pending_linear_acceleration=motion.pending_linear_acceleration;
            a.pending_angular_acceleration=motion.pending_angular_acceleration;
            a.wake_counter=motion.wake_counter;
            a.sleep_linear_acc=motion.sleep_linear_acc;
            a.sleep_angular_acc=motion.sleep_angular_acc;
            a.ready_for_sleep=motion.ready_for_sleep;
            a.sleeping=motion.sleeping;
            a.activating=motion.activating;
            a.encoded_bounds=motion.encoded_bounds;
            a.cached_bounds_current=motion.cached_bounds_current;
        }
    }
    // Swap back the exact original arena allocations. This preserves the
    // lifetime of contact spans returned before the unsuccessful step.
    const bool quiet_events_rotated=
        scene.quiet_event_epoch_!=quiet_event_epoch;
    if(scene.pair_epoch_!=pair_epoch) {
        scene.previous_pairs_.swap(scene.pending_pairs_);
        scene.found_.swap(scene.next_found_);
        scene.lost_.swap(scene.next_lost_);
    } else if(quiet_events_rotated) {
        // Quiet publication swapped event streams only, not pair state.
        scene.found_.swap(scene.next_found_);
        scene.lost_.swap(scene.next_lost_);
    }
    scene.pair_epoch_=pair_epoch;
    if(scene.contact_epoch_!=contact_epoch) {
        scene.manifolds_.swap(scene.next_manifolds_);
        scene.contact_points_.swap(scene.next_contact_points_);
        scene.gu_cache_records_.swap(scene.next_gu_cache_records_);
        scene.gu_cached_points_.swap(scene.next_gu_cached_points_);
        scene.manifold_by_edge_.swap(scene.next_manifold_by_edge_);
        scene.found_contacts_.swap(scene.next_found_contacts_);
        scene.lost_contacts_.swap(scene.next_lost_contacts_);
    } else if(quiet_events_rotated) {
        // Retain the prior contact-manifold arena and its published spans.
        scene.found_contacts_.swap(scene.next_found_contacts_);
        scene.lost_contacts_.swap(scene.next_lost_contacts_);
    }
    scene.contact_epoch_=contact_epoch;
    scene.manifold_lookup_generation_=manifold_lookup_generation;
    scene.quiet_event_epoch_=quiet_event_epoch;
    // For a quiet solve, the Gu arena never rotated. Restore its
    // externally exposed Dy offsets without copying contact points,
    // replacing the published span or journaling ordinary moving frames.
    if(includes_solver_state && !quiet_manifold_metadata.empty()) {
        for(const auto& prior:quiet_manifold_metadata) {
            auto& manifold=scene.manifolds_[prior.index];
            manifold.solver_offset=prior.solver_offset;
            manifold.friction_patch_offset=prior.friction_patch_offset;
            manifold.friction_patch_count=prior.friction_patch_count;
            manifold.prepared_friction_epoch=prior.prepared_friction_epoch;
        }
    }
    // The pre-step friction cache remains in the other source-owned
    // buffer after the complete new correlation stream is published.
    // Reverse the swap only if that boundary was actually crossed.
    if(includes_solver_state &&
       scene.friction_cache_epoch_!=friction_cache_epoch) {
        scene.friction_patches_.swap(scene.next_friction_patches_);
        scene.friction_by_edge_.swap(scene.next_friction_by_edge_);
    }
    scene.friction_cache_epoch_=friction_cache_epoch;
    scene.normal_impulse_writeback_.swap(normal_impulses);
    scene.friction_impulse_writeback_.swap(friction_impulses);
    // Previous exceeded-force pairs likewise remain in the reused
    // current buffer after the source force-change stream rotates.
    if(includes_solver_state &&
       scene.threshold_cache_epoch_!=threshold_cache_epoch)
        scene.threshold_previous_.swap(scene.threshold_current_);
    scene.threshold_cache_epoch_=threshold_cache_epoch;
    scene.threshold_changes_.swap(threshold_changes);
    if(includes_solver_state)
        scene.partition_ends_.swap(partition_ends);
    if(graph_ownership_saved) {
        scene.speculative_islands_.islands.swap(speculative_islands.islands);
        scene.speculative_islands_.node_island.swap(speculative_islands.node_island);
        scene.speculative_islands_.node_next.swap(speculative_islands.node_next);
        scene.speculative_islands_.node_prev.swap(speculative_islands.node_prev);
        scene.speculative_islands_.node_static_touches.swap(
            speculative_islands.node_static_touches);
        scene.speculative_islands_.free_handles.swap(speculative_islands.free_handles);
        scene.speculative_islands_.active_island_seen.swap(
            speculative_islands.active_island_seen);
        scene.speculative_islands_.active_island_visit_stamp=
            speculative_islands.active_island_visit_stamp;
    }
    scene.speculative_islands_.active_nodes.swap(speculative_islands.active_nodes);
    scene.speculative_islands_.active_node_index.swap(
        speculative_islands.active_node_index);
    // The structural island snapshot restores mActiveIndex itself.
    // A quiet step has no such snapshot: clear its current live slots,
    // then restore the previous ordered active list and reverse slots.
    if(!graph_ownership_saved)
        for(auto handle:scene.speculative_islands_.active_islands)
            if(handle<scene.speculative_islands_.islands.size())
                scene.speculative_islands_.islands[handle].active_index=
                    invalid_instance;
    scene.speculative_islands_.active_islands.swap(
        speculative_islands.active_islands);
    scene.speculative_islands_.activating_nodes.swap(
        speculative_islands.activating_nodes);
    if(!graph_ownership_saved)
        for(std::size_t i=0;i<scene.speculative_islands_.active_islands.size();++i)
            scene.speculative_islands_.islands[
                scene.speculative_islands_.active_islands[i]].active_index=i;
    scene.speculative_islands_.nodes_to_deactivate.swap(
        speculative_islands.nodes_to_deactivate);
    scene.speculative_islands_.deactivating_edges.swap(
        speculative_islands.deactivating_edges);
    scene.speculative_islands_.activated_edges.swap(
        speculative_islands.activated_edges);
    scene.speculative_islands_.active_edge_count=
        speculative_islands.active_edge_count;
    scene.speculative_islands_.initial_active_node_count=
        speculative_islands.initial_active_node_count;
    if(graph_ownership_saved) {
        scene.accurate_islands_.islands.swap(accurate_islands.islands);
        scene.accurate_islands_.node_island.swap(accurate_islands.node_island);
        scene.accurate_islands_.node_next.swap(accurate_islands.node_next);
        scene.accurate_islands_.node_prev.swap(accurate_islands.node_prev);
        scene.accurate_islands_.node_static_touches.swap(
            accurate_islands.node_static_touches);
        scene.accurate_islands_.free_handles.swap(accurate_islands.free_handles);
        scene.accurate_islands_.active_island_seen.swap(
            accurate_islands.active_island_seen);
        scene.accurate_islands_.active_island_visit_stamp=
            accurate_islands.active_island_visit_stamp;
    }
    scene.accurate_islands_.active_nodes.swap(accurate_islands.active_nodes);
    scene.accurate_islands_.active_node_index.swap(
        accurate_islands.active_node_index);
    if(!graph_ownership_saved)
        for(auto handle:scene.accurate_islands_.active_islands)
            if(handle<scene.accurate_islands_.islands.size())
                scene.accurate_islands_.islands[handle].active_index=
                    invalid_instance;
    scene.accurate_islands_.active_islands.swap(
        accurate_islands.active_islands);
    scene.accurate_islands_.activating_nodes.swap(
        accurate_islands.activating_nodes);
    if(!graph_ownership_saved)
        for(std::size_t i=0;i<scene.accurate_islands_.active_islands.size();++i)
            scene.accurate_islands_.islands[
                scene.accurate_islands_.active_islands[i]].active_index=i;
    scene.accurate_islands_.nodes_to_deactivate.swap(
        accurate_islands.nodes_to_deactivate);
    scene.accurate_islands_.deactivating_edges.swap(
        accurate_islands.deactivating_edges);
    scene.accurate_islands_.activated_edges.swap(
        accurate_islands.activated_edges);
    scene.accurate_islands_.active_edge_count=
        accurate_islands.active_edge_count;
    scene.accurate_islands_.initial_active_node_count=
        accurate_islands.initial_active_node_count;
    // The reference arrays belong to their original graph owners even
    // after a full island rebuild. Rewind only changed endpoint slots
    // after structural ownership and edge-state undo have completed.
    for(auto it=reference_undo.rbegin();it!=reference_undo.rend();++it) {
        scene.speculative_islands_.active_reference_counts[it->node]=
            it->speculative;
        scene.accurate_islands_.active_reference_counts[it->node]=
            it->accurate;
    }
    scene.speculative_islands_.active_reference_counts.resize(
        initial_speculative_references);
    scene.accurate_islands_.active_reference_counts.resize(
        initial_accurate_references);
    if(graph_ownership_saved) {
        scene.accurate_island_nodes_.swap(accurate_nodes);
        scene.accurate_island_offsets_.swap(accurate_offsets);
    }
    if(includes_solver_state)
        scene.active_accurate_island_batches_.swap(active_accurate_batches);
    if(includes_solver_state)
        scene.indexed_managers_.swap(indexed_managers);
    scene.speculative_island_count_=speculative_count;
    scene.accurate_island_count_=accurate_count;
    scene.island_topology_revision_=topology_revision;
    scene.island_full_rebuild_count_=full_rebuild_count;
    scene.island_local_split_count_=local_split_count;
    scene.island_graph_checkpoint_count_=graph_checkpoint_count;
    scene.island_dirty_route_checks_=dirty_route_checks;
    scene.island_fast_route_hits_=fast_route_hits;
    scene.frame_index_=frame_index;
    scene.solved_result_frame_=solved_result_frame;
    scene.active_contact_interaction_count_=initial_active_interactions;
    scene.island_topology_dirty_=topology_dirty;
    scene.dirty_=dirty;
    scene.authored_activation_pending_=authored_activation_pending;
 }

void Scene::RigidStepJoin::finish_lost_contacts() {
    // Sc::Scene::processLostContacts3 completes after the lost-touch
    // and third-pass consumers have finished. The source continuation
    // contributes one of mPostSolver's two independent producer edges.
    if(lost_contacts_joined)
        throw std::logic_error(
            "Native ScPipeline completed the lost-contact continuation twice");
    lost_contacts_joined=true;
}
void Scene::RigidStepJoin::dispatch_solver_tasks(std::size_t count) {
    // Source producer branches are independent: the numerical Dy tasks
    // may run concurrently with third-pass lost-contact cleanup. The
    // dependency is on mPostSolver, never on starting Dy itself.
    if(solver_dispatched)
        throw std::logic_error(
            "Native ScPipeline dispatched the same Dy task chains twice");
    // DynamicsContext::updatePostKinematic schedules one
    // PxsSolverEndTask per active-island solver task chain. An empty
    // island graph legitimately schedules zero chains.
    solver_dispatched=true;
    outstanding_solver_tasks=count;
}
void Scene::RigidStepJoin::finish_solver_task() {
    if(!solver_dispatched || !outstanding_solver_tasks)
        throw std::logic_error(
            "Native Dy solver end task completed without a live continuation");
    --outstanding_solver_tasks;
}
void Scene::RigidStepJoin::require_post_solver() const {
    // ScPipeline::updateDynamics wires mProcessLostContactsTask3 and
    // every PxsSolverEndTask into mPostSolver. A manager handle cannot
    // be reclaimed, and no completed result can be published, before
    // both producers have joined.
    if(!lost_contacts_joined || !solver_dispatched ||
       outstanding_solver_tasks)
        throw std::logic_error(
            "Native ScPipeline postSolver reached before producer completion");
}

void Scene::step(float seconds) {
    if (!std::isfinite(seconds) || seconds <= 0.0f)
        throw std::invalid_argument("Native CPU timestep must be finite and positive");

    // Source: ScBodyCore.cpp::BodyCore dynamic defaults, and
    // DyBodyCoreIntegrator.h::bodyCoreComputeUnconstrainedVelocity:
    // gravity, linear/angular damping, then PxReal speed clamps.
    constexpr float gravity = -9.81f;
    // A failed frame must not consume impulses or publish partial
    // contacts, island nodes, body motion or result streams. Snapshot only
    // authoritative state; all per-step Dy work buffers remain reusable.
    step_checkpoint_.capture(*this);
    RigidStepJoin join;
    try {
    // ScPipeline: the Gu second pass and shared IslandSim wake stage
    // must publish the rigid solver inputs before Dy starts. The lost-
    // contact third pass may overlap Dy in the original parallel CPU
    // pipeline; our serial lowering runs it first so afterIntegration
    // does not need concurrent-deactivation pose repair. The two task
    // producer branches still join independently at postSolver.
    // IslandSim::wakeIslands snapshots the existing active prefix before
    // activation. Preserve the same boundary when source contact
    // publication subsequently rebuilds island handles.
    accurate_islands_.initial_active_node_count=
        accurate_islands_.active_nodes.size();
    speculative_islands_.initial_active_node_count=
        speculative_islands_.active_nodes.size();
    advance_collision_internal(false,true);
    third_pass_island_gen();
    activate_published_islands();
    // Source wakeIslands publishes active island membership BEFORE
    // processLostEdges tests sleep readiness. Reconcile once even when
    // no island is ready, so the deactivation pass touches active roots
    // rather than the entire island-handle inventory.
    synchronize_active_island_nodes(true);
    // Source third pass tests readiness from the prior Dy sleepCheck,
    // before this frame's unconstrained velocity/PGS processing.
    deactivate_ready_islands();
    consume_contact_interaction_activity();
    // Deactivation removes nodes and islands from the active lists;
    // ordinary awake frames do not need this second traversal.
    if(!accurate_islands_.nodes_to_deactivate.empty())
        synchronize_active_island_nodes();
    // mProcessLostContactsTask3's admitted rigid dependencies are now
    // complete: Gu lost-touch publication, both island passes, wake,
    // third-pass sleep, and Sc interaction-state consumption.
    // PhysX can overlap this branch with Dy; serial CPU execution
    // resolves it first without changing the dependency edge.
    join.finish_lost_contacts();
    // Dy::DynamicsContext takes the accurate *active* island ranges,
    // not every actor or every dormant island in the full graph.
    prepare_active_accurate_island_batches();
    for(std::size_t island:active_accurate_island_batches_)
    for(std::size_t index=accurate_island_offsets_[island];
        index<accurate_island_offsets_[island+1];++index) {
        auto& actor=actors_[accurate_island_nodes_[index]];
        if(actor.sleeping)continue;
        const auto& d=actor.dynamics;
        // ScBodySim::updateForces runs at ScPipeline::beforeSolver:
        // per-step velocity + per-second acceleration*dt are applied
        // before Dy::bodyCoreComputeUnconstrainedVelocity damping.
        const Vec3 pre_linear{
            actor.velocity.x+actor.pending_linear_velocity_delta.x+
                actor.pending_linear_acceleration.x*seconds,
            actor.velocity.y+actor.pending_linear_velocity_delta.y+
                actor.pending_linear_acceleration.y*seconds,
            actor.velocity.z+actor.pending_linear_velocity_delta.z+
                actor.pending_linear_acceleration.z*seconds};
        float vx=pre_linear.x;
        float vy=pre_linear.y+
            (d.gravity_enabled ? gravity*seconds : 0.0f);
        float vz=pre_linear.z;
        const float lin_mult = std::max(
            1.0f - d.linear_damping*seconds,0.0f);
        vx *= lin_mult; vy *= lin_mult; vz *= lin_mult;
        // Per-step angular velocity modification is accumulated on the
        // actor and consumed exactly once, as in ScBodySim::VelocityMod.
        Vec3 angular{
            actor.angular_velocity.x+actor.pending_angular_velocity_delta.x+
                actor.pending_angular_acceleration.x*seconds,
            actor.angular_velocity.y+actor.pending_angular_velocity_delta.y+
                actor.pending_angular_acceleration.y*seconds,
            actor.angular_velocity.z+actor.pending_angular_velocity_delta.z+
                actor.pending_angular_acceleration.z*seconds};
        if (!finite(angular))
            throw std::overflow_error("Native angular impulse accumulation overflow");
        const float ang_mult = std::max(
            1.0f - d.angular_damping*seconds,0.0f);
        angular.x *= ang_mult;
        angular.y *= ang_mult;
        angular.z *= ang_mult;
        const float speed_squared = vx*vx + vy*vy + vz*vz;
        const float max_linear_velocity_squared=
            d.max_linear_velocity*d.max_linear_velocity;
        const float max_angular_velocity_squared=
            d.max_angular_velocity*d.max_angular_velocity;
        if (speed_squared > max_linear_velocity_squared) {
            const float factor =
                std::sqrt(max_linear_velocity_squared / speed_squared);
            vx *= factor; vy *= factor; vz *= factor;
        }
        const float ang_speed_squared = angular.x*angular.x +
            angular.y*angular.y + angular.z*angular.z;
        if (ang_speed_squared > max_angular_velocity_squared) {
            const float factor =
                std::sqrt(max_angular_velocity_squared / ang_speed_squared);
            angular.x *= factor;
            angular.y *= factor;
            angular.z *= factor;
        }
        Vec3 velocity{vx, vy, vz};
        // DyRigidBodyToSolverBody::copyToSolverBodyData applies the
        // optional gyroscopic update after speed clamping and before
        // angular locking. All awake bodies, including no-contact
        // islands, take this source integration path exactly once.
        if(d.gyroscopic_forces_enabled) {
            if(!actor.angular_inertia_available)
                throw std::logic_error(
                    "Native gyroscopic motion requires representable inertia");
            angular=gyroscopic_velocity(
                angular,actor.inverse_inertia_local,
                actor.orientation,seconds);
        }
        // DyRigidBodyToSolverBody leaves the solver's pre-integration
        // linear baseline available to contact preparation. Original
        // integrateCore locks the motion and final linear writeback,
        // rather than suppressing contact-relative velocity beforehand.
        lock_angular_axes(angular,d.lock_flags);
        if (!finite(velocity) || !finite(angular))
            throw std::overflow_error("Native CPU unconstrained velocity overflow");
        step_checkpoint_.save_motion(*this,accurate_island_nodes_[index]);
        actor.velocity = velocity;
        actor.angular_velocity = angular;
        actor.pending_angular_velocity_delta = {};
        actor.pending_linear_velocity_delta = {};
        // ScBodySim::updateForces calls setForcesToDefaults after
        // applying per-step velocity and per-second acceleration mods.
        // PxRigidBodyFlag::eRETAIN_ACCELERATIONS preserves ONLY the
        // per-second stream for the next active frame; per-step velocity
        // deltas were consumed and cleared unconditionally above.
        // Island deactivation separately clears both streams via
        // BodySim::putToSleep::simStateClearVelMod.
        if(!d.retain_accelerations) {
            actor.pending_linear_acceleration = {};
            actor.pending_angular_acceleration = {};
        }
        // Source ScPipeline/DySolverControl: do not predict and publish
        // the position before Gu contacts and the biased PGS passes.
        // Its integration requires the saved position-solver motion
        // velocity, not the final unbiased next-frame velocity.
        dirty_ = true;
    }
    prepare_solver_bodies();
    partition_constraints();
    // Dispatch is over the ACTUAL Dy::updatePostKinematic task-group
    // descriptors, including groups without contact rows. No task
    // allocations or second island registry are introduced.
    join.dispatch_solver_tasks(solver_tasks_.size());
    prepare_constraints(seconds);
    solve_constraints();
    write_back_contacts();
    update_threshold_stream(seconds);
    publish_solver_bodies();
    // PxsBodyCore::numCountedInteractions is maintained by original Sc
    // interaction edges. In this admitted contact-only subset each Gu
    // contact manager contributes one interaction to each dynamic body.
    // Derive counted interactions from the SAME source-indexed
    // contact-manager descriptor stream used by the current Dy task
    // chains, not every Gu manager retained by dormant accurate islands.
    // Zero-response contacts still count as interactions. A sleeping
    // graph may retain its managers, but cannot add energy to any body
    // processed by this frame's Dy::sleepCheck.
    sleep_interaction_counts_.resize(actors_.size(),0);
    for(BodyId id:accurate_islands_.active_nodes)
        sleep_interaction_counts_[id]=0;
    for(std::size_t index:indexed_managers_) {
        const auto& manifold=manifolds_[index];
        for(BodyId id:{manifold.pair.a,manifold.pair.b}) {
            if(actors_[id].kind!=Kind::DynamicBox ||
               actors_[id].sleeping)continue;
            if(sleep_interaction_counts_[id]==
               std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error(
                    "Native source counted interactions overflow");
            ++sleep_interaction_counts_[id];
        }
    }
    // DyBodyCoreIntegrator.h::integrateCore uses the saved *position*
    // iteration motion velocities to move/rotate bodies. The velocity
    // iterations above publish their different, unbiased final velocities
    // for subsequent frames; they do not undo this frame's correction.
    for(const auto& task:solver_tasks_) {
        // PxsSolverEndTask owns integration and Dy::sleepCheck for its
        // contiguous range of accurate active islands, not just PGS.
        // The task contributes to postSolver only after every rigid
        // node in that same range has finished its integration.
        for(std::size_t batch=task.first_batch;
            batch<task.batch_end;++batch) {
        const auto island=active_accurate_island_batches_[batch];
        for(std::size_t index=accurate_island_offsets_[island];
            index<accurate_island_offsets_[island+1];++index) {
            const BodyId id=accurate_island_nodes_[index];
            auto& actor=actors_[id];
            if(actor.sleeping)continue;
            // integrateCore applies world linear locks *after* the PGS
            // solve. For no-contact bodies this is also their final
            // velocity publication; active solver bodies were already
            // masked by publish_solver_bodies().
            lock_linear_axes(actor.velocity,actor.dynamics.lock_flags);
            Vec3 motion_linear = actor.velocity;
            Vec3 motion_angular = actor.angular_velocity;
            // Scene clock increments in advance_collision_internal; only
            // this frame's actual contact participants own motion snapshots.
            const bool solved_contact = id < solver_rows_.size() &&
                solver_rows_[id].prepared_generation == solver_preparation_generation_;
            if (solved_contact) {
                const auto& row = solver_rows_[id];
                const Vec3& delta = row.motion_linear_delta;
                motion_linear = {
                    row.data.linear_velocity.x + delta.x,
                    row.data.linear_velocity.y + delta.y,
                    row.data.linear_velocity.z + delta.z};
                const auto& i = row.data.sqrt_inverse_inertia;
                Vec3 angular_delta=row.motion_angular_state;
                lock_angular_axes(angular_delta,actor.dynamics.lock_flags);
                motion_angular = {
                    row.data.angular_velocity.x + i[0].x*angular_delta.x +
                        i[1].x*angular_delta.y + i[2].x*angular_delta.z,
                    row.data.angular_velocity.y + i[0].y*angular_delta.x +
                        i[1].y*angular_delta.y + i[2].y*angular_delta.z,
                    row.data.angular_velocity.z + i[0].z*angular_delta.x +
                        i[1].z*angular_delta.y + i[2].z*angular_delta.z};
            }
            // integrateCore also locks position-phase motion separately
            // from the final unbiased velocity publication.
            lock_linear_axes(motion_linear,actor.dynamics.lock_flags);
            const Vec3 p{
                static_cast<float>(static_cast<double>(actor.position.x) +
                                   static_cast<double>(motion_linear.x)*seconds),
                static_cast<float>(static_cast<double>(actor.position.y) +
                                   static_cast<double>(motion_linear.y)*seconds),
                static_cast<float>(static_cast<double>(actor.position.z) +
                                   static_cast<double>(motion_linear.z)*seconds)};
            if (!finite(p) || !finite(motion_linear) || !finite(motion_angular))
                throw std::overflow_error("Native position-iteration motion overflow");
            const Quat q = integrate_quaternion(
                actor.orientation, motion_angular, seconds);
            if (!finite(q))
                throw std::overflow_error("Native CPU orientation overflow");
            step_checkpoint_.save_motion(*this,id);
            actor.position = p;
            actor.orientation = q;
            // DySleep.cpp::updateWakeCounter (default scene, without
            // stabilization). ScBodyCore::BodyCore and PxsBodyCore::init
            // initialize sleepThreshold to 5e-5 * scale.speed^2.
            // PxTolerancesScale's DEFAULT speed is 10 m/s, not 1 m/s.
            // This only produces readiness; IslandSim owns deactivation.
            constexpr float wake_reset_time=20.0f*0.02f;
            constexpr float default_scale_speed=10.0f;
            constexpr float energy_threshold=
                5.0e-5f*default_scale_speed*default_scale_speed;
            float wc=actor.wake_counter;
            if(wc<wake_reset_time*0.5f || wc<seconds) {
                const Vec3 local_angular=rotate_inverse(q,motion_angular);
                const Vec3 linear_acc{
                    actor.sleep_linear_acc.x+motion_linear.x,
                    actor.sleep_linear_acc.y+motion_linear.y,
                    actor.sleep_linear_acc.z+motion_linear.z};
                const Vec3 angular_acc{
                    actor.sleep_angular_acc.x+local_angular.x,
                    actor.sleep_angular_acc.y+local_angular.y,
                    actor.sleep_angular_acc.z+local_angular.z};
                if(!finite(linear_acc) || !finite(angular_acc))
                    throw std::overflow_error(
                        "Native Dy sleep-energy accumulator overflow");
                const Vec3 inverse_i=actor.inverse_inertia_local;
                const Vec3 inertia{
                    inverse_i.x>0.0f?1.0f/inverse_i.x:1.0f,
                    inverse_i.y>0.0f?1.0f/inverse_i.y:1.0f,
                    inverse_i.z>0.0f?1.0f/inverse_i.z:1.0f};
                const float angular_energy=(
                    angular_acc.x*angular_acc.x*inertia.x+
                    angular_acc.y*angular_acc.y*inertia.y+
                    angular_acc.z*angular_acc.z*inertia.z)*actor.inverse_mass;
                // Source uses accumulated (not instantaneous) linear
                // energy and a cluster threshold for the actual comparison.
                const float accumulated_linear=
                    linear_acc.x*linear_acc.x+
                    linear_acc.y*linear_acc.y+
                    linear_acc.z*linear_acc.z;
                const float normalized_energy=
                    0.5f*(angular_energy+accumulated_linear);
                const float cluster=1.0f+
                    static_cast<float>(sleep_interaction_counts_[id]);
                const float threshold=cluster*energy_threshold;
                if(normalized_energy>=threshold) {
                    actor.sleep_linear_acc={};
                    actor.sleep_angular_acc={};
                    const float factor=threshold==0.0f
                        ? 2.0f:std::min(normalized_energy/threshold,2.0f);
                    wc=factor*0.5f*wake_reset_time+
                        seconds*(cluster-1.0f);
                } else {
                    actor.sleep_linear_acc=linear_acc;
                    actor.sleep_angular_acc=angular_acc;
                    wc=std::max(wc-seconds,0.0f);
                }
            } else wc=std::max(wc-seconds,0.0f);
            actor.wake_counter=wc;
            actor.ready_for_sleep=(wc==0.0f);
            if(actor.ready_for_sleep) {
                actor.sleep_linear_acc={};
                actor.sleep_angular_acc={};
            }
            dirty_ = true;
        }
        }
        join.finish_solver_task();
    }
    // Dy::sleepCheck has only published readiness. The next source
    // third pass decides whether entire islands may deactivate; applying
    // that decision here would incorrectly consume this frame's flags.
    // The source mPostSolver continuation joins both actual producers:
    // processLostContactsTask3 / PostThirdPassTask and every
    // PxsSolverEndTask (including its integration and sleepCheck).
    // This check gates resource reclamation and completed publication;
    // failed frames roll back the existing single-scene checkpoint.
    join.require_post_solver();
    // The current Dy motion and island ownership are committed only
    // after the original Sc afterIntegration shape-cache publication.
    // Its bounds updates participate in the same motion undo journal.
    after_integration_cached_bounds();
    retire_destroyed_contact_edges();
    // ScNPhaseCore reports are derived only after final Gu/Dy force
    // writeback, island reconciliation and contact-manager retirement.
    publish_contact_events();
    // This is the last publication edge. The frame is now observed as
    // a completed Dy result rather than merely a Gu collision step.
    // A failed stage above leaves it unpublished; rollback restores the
    // previously completed result generation and all its buffers.
    solved_result_frame_=frame_index_;
    // Solver and lost-edge producers have both completed before this
    // successful step returns its published body and contact results.
    activity_checkpoint_active_=false;
    } catch (...) {
        step_checkpoint_.rollback(*this);
        throw;
    }
}

void Scene::advance_collision(bool force_update) {
    // No geometry, topology or handle lifecycle changes can throw on
    // a quiet collision publication. Preserve the original constant-
    // work path instead of copying persistent edge/island state.
    // A successful quiet advance still retires last frame's transient
    // event/solver reports and advances the publication clock.
    if (!dirty_ && !force_update && !island_topology_dirty_ &&
        retired_edge_handles_.empty()) {
        // No new edges or pending island changes: never scan live
        // islands on the existing constant-work collision query path.
        advance_collision_internal(false,false);
        // PxsSimpleIslandManager::clearDeactivations runs at each island
        // publication. A quiet geometry query has no new third-pass
        // work, but it must retire last publication's one-frame
        // mNodesToPutToSleep output rather than replaying the event.
        // This is constant-work and cannot throw.
        accurate_islands_.nodes_to_deactivate.clear();
        speculative_islands_.deactivating_edges.clear();
        accurate_islands_.deactivating_edges.clear();
        accurate_islands_.initial_active_node_count=
            accurate_islands_.active_nodes.size();
        speculative_islands_.initial_active_node_count=
            speculative_islands_.active_nodes.size();
        // A velocity/torque edit can queue IslandSim activation without
        // touching geometry. Complete the no-throw source wake handoff.
        // Keep the graph-owned active-node map in step with such wakes,
        // without traversing it on an unchanged, event-free query.
        const bool pending_wake=authored_activation_pending_;
        activate_published_islands();
        consume_contact_interaction_activity();
        if(pending_wake)synchronize_active_island_nodes();
        return;
    }
    // Geometry changes, forced refreshes and pending graph retirement
    // retain the full failure-atomic publication boundary. step() calls
    // advance_collision_internal directly, without nested snapshots.
    step_checkpoint_.capture(*this,false);
    try {
        accurate_islands_.initial_active_node_count=
            accurate_islands_.active_nodes.size();
        speculative_islands_.initial_active_node_count=
            speculative_islands_.active_nodes.size();
        advance_collision_internal(force_update,true);
        third_pass_island_gen();
        // A collision-only publication has no Dy consumers. The two
        // island repair passes have already joined, so handle retirement
        // may complete inside the failure-atomic collision transaction.
        retire_destroyed_contact_edges();
    } catch (...) {
        step_checkpoint_.rollback(*this);
        throw;
    }
    // Activation and third-pass deactivation only mutate existing
    // actors after the collision transaction has successfully committed.
    // Geometry-only checkpoints intentionally do not copy actor motion.
    activity_checkpoint_active_=false;
    activate_published_islands();
    synchronize_active_island_nodes(true);
    deactivate_ready_islands();
    consume_contact_interaction_activity();
    if(!accurate_islands_.nodes_to_deactivate.empty())
        synchronize_active_island_nodes();
}
void Scene::advance_collision_internal(
    bool force_update, bool checkpointed) {
    // Begin one complete source publication. The first pass can rebuild
    // island membership before third-pass lost-edge repair, so preserve
    // any resulting sleep candidates across both passes of this frame.
    speculative_split_sleep_candidates_.clear();
    // Previous mActivatedEdges flags are consumed at this wake boundary.
    clear_activated_edges(checkpointed);
    // This also serves geometry-only queries. Never publish last frame's
    // solved force-buffer values against an unsolved contact stream.
    normal_impulse_writeback_.clear();
    friction_impulse_writeback_.clear();
    // A collision-only advance is NOT equivalent to a completed
    // simulate/fetch. The next successful Scene::step can publish this
    // generation again, after its dependent solver stages finish.
    solved_result_frame_=0;
    // This clears only the exposed reports; prior exceeded-force state
    // must survive collision queries until the next simulation step.
    threshold_changes_.clear();
    // ScPipeline.cpp CPU ordering:
    // BP first pass -> BP update -> post-BP pair publication ->
    // Gu narrowphase -> physical touch publication.
    // No parallel timestep or dynamics scheduler is introduced.
    if (dirty_ || force_update) {
        update_bounds();
        update_pairs();
        publish_pairs();
        update_contacts();
    } else {
        // A quiet query/step still publishes empty per-frame event
        // streams. Rotate only their reusable buffers so a later
        // exception can undo the publication without copying events.
        // Do NOT advance pair_epoch_ or contact_epoch_: those mark
        // actual pair/manifold arena swaps in StepCheckpoint::rollback.
        next_found_.clear();
        next_lost_.clear();
        next_found_contacts_.clear();
        next_lost_contacts_.clear();
        found_.swap(next_found_);
        lost_.swap(next_lost_);
        found_contacts_.swap(next_found_contacts_);
        lost_contacts_.swap(next_lost_contacts_);
        ++quiet_event_epoch_;
    }
    // Source first speculative and second accurate pass: edge insertion
    // and active body admission, but no lost-edge island splitting yet.
    update_island_connectivity(false);
    dirty_ = false;
    ++frame_index_;
}
} // namespace fyzz::native

namespace fyzz {
namespace {
class NativeCpuWorld final : public WorldBackend {
public:
    explicit NativeCpuWorld(bool contact_notifications,
                            bool enhanced_determinism,
                            bool pcm_plane_box) {
        scene_.set_contact_notifications(contact_notifications);
        scene_.set_enhanced_determinism(enhanced_determinism);
        scene_.set_pcm_plane_box(pcm_plane_box);
    }
    std::span<const ContactEvent> contact_events() const override {
        return scene_.contact_events();
    }
    std::span<const ContactEventPoint> contact_event_points() const override {
        return scene_.contact_event_points();
    }
    BodyId add_static_box(Vec3 h, Vec3 p) override {
        return scene_.add_static_box(h, p);
    }
    BodyId add_static_floor(float height) override {
        return scene_.add_static_floor(height);
    }
    BodyId add_dynamic_box(Vec3 h, Vec3 p, float density) override {
        return scene_.add_dynamic_box(h, p, density);
    }
    void step(float seconds) override { scene_.step(seconds); }
    BodyState body_state(BodyId id) const override { return scene_.body_state(id); }
    void set_body_position(BodyId id, Vec3 p) override {
        scene_.set_position(id, p);
    }
    void set_body_pose(BodyId id, Vec3 p, Quat q) override {
        scene_.set_pose(id, p, q);
    }
    void remove_body(BodyId id) override { scene_.remove(id); }
    void set_body_velocity(BodyId id, Vec3 velocity) override {
        scene_.set_velocity(id, velocity);
    }
    void set_body_angular_velocity(BodyId id, Vec3 velocity) override {
        scene_.set_angular_velocity(id, velocity);
    }
    void apply_body_torque_impulse(BodyId id, Vec3 impulse) override {
        scene_.apply_torque_impulse(id, impulse);
    }
    void add_body_force(BodyId id, Vec3 value, ForceMode mode) override {
        scene_.add_spatial_velocity_mod(id,value,mode,false);
    }
    void add_body_torque(BodyId id, Vec3 value, ForceMode mode) override {
        scene_.add_spatial_velocity_mod(id,value,mode,true);
    }
    void clear_body_force(BodyId id, ForceMode mode) override {
        scene_.clear_spatial_velocity_mod(id,mode,false);
    }
    void clear_body_torque(BodyId id, ForceMode mode) override {
        scene_.clear_spatial_velocity_mod(id,mode,true);
    }
    void set_body_material(BodyId id, Material material) override {
        scene_.set_material(id,material);
    }
    void set_body_shape_offsets(BodyId id, float contact_offset,
                                float rest_offset) override {
        scene_.set_shape_offsets(id,contact_offset,rest_offset);
    }
    void set_body_dynamics(BodyId id, RigidBodyDynamics dynamics) override {
        scene_.set_dynamics(id,dynamics);
    }
private:
    native::Scene scene_;
};
} // namespace
std::unique_ptr<WorldBackend> create_native_cpu_backend(
    bool notify, bool enhanced_determinism, bool pcm_plane_box) {
    return std::make_unique<NativeCpuWorld>(
        notify,enhanced_determinism,pcm_plane_box);
}
} // namespace fyzz
