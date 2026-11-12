// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "native/scene.h"
#include "hardware_profile.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <bit>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <stop_token>
#include <stdexcept>
#include <thread>
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
    if(actors_.size()>=std::numeric_limits<BodyId>::max())
        throw std::overflow_error("Native scene body identifier exhausted");
    const BodyId id=static_cast<BodyId>(actors_.size());
    for(auto* graph:{&speculative_islands_,&accurate_islands_}) {
        graph->active_kinematic_nodes.reserve(actors_.size()+1);
        graph->active_kinematic_index.resize(actors_.size()+1,invalid_instance);
    }
    if(actors_.size()==actors_.capacity())
        actors_.reserve(actors_.capacity()+std::max<std::size_t>(actors_.capacity(),8));
    if(gpu_broadphase_)gpu_broadphase_->retain_rigid_actor(id);
    actor.alive=true;
    actors_.push_back(actor);
    if(actor.motion==Motion::Dynamic)++live_dynamic_count_;
    island_topology_dirty_|=actor.motion==Motion::Dynamic;
    dirty_=true;
    return id;
}
ShapeId Scene::attach(BodyId body,Shape shape,std::optional<AggregateId> aggregate) {
    auto& actor=require(body);
    if(shapes_.size()>=invalid_shape)
        throw std::overflow_error("Native scene shape identifier exhausted");
    const ShapeId id=static_cast<ShapeId>(shapes_.size());
    if(shapes_.size()==shapes_.capacity())
        shapes_.reserve(shapes_.capacity()+std::max<std::size_t>(shapes_.capacity(),8));
    if(live_shape_ids_.size()==live_shape_ids_.capacity())
        live_shape_ids_.reserve(live_shape_ids_.capacity()+std::max<std::size_t>(live_shape_ids_.capacity(),8));
    shape.local_pose=validated_pose(shape.local_pose);
    shape.aggregate=aggregate;
    shape.body=body;
    shape.motion=actor.motion;
    shape.alive=true;
    shape.device_cache_supported=!aggregate.has_value();
    refresh_shape_pose(shape);
    if(aggregate && !gpu_broadphase_)
        throw std::logic_error("Aggregate requires native GPU AABB owner");
    if(gpu_broadphase_) {
        const auto motion=actor.motion==Motion::Static
            ? GpuBroadphaseBox::Motion::Static
            : actor.motion==Motion::Kinematic ? GpuBroadphaseBox::Motion::Kinematic
                                             : GpuBroadphaseBox::Motion::Dynamic;
        if(shape.geometry==Geometry::Plane)
            gpu_broadphase_->admit_plane_element(id,body);
        else if(aggregate)
            gpu_broadphase_->admit_aggregate_shape(*aggregate,id,body,motion,shape.contact_offset);
        else gpu_broadphase_->admit_shape_element(id,body,motion,shape.contact_offset);
    }
    shape.next=actor.first_shape;
    if(actor.first_shape!=invalid_shape)shapes_[actor.first_shape].previous=id;
    actor.first_shape=id;
    ++actor.shape_count;
    if(rigid_node(actor.motion))++rigid_shape_count_;
    shapes_.push_back(shape);
    live_shape_ids_.push_back(id);
    refresh_shape_links(body);
    if(shape.geometry==Geometry::Plane)++active_plane_count_;
    gpu_broadphase_snapshot_current_=false;
    plane_pair_membership_dirty_=true;
    broadphase_endpoint_membership_dirty_=true;
    dirty_=true;
    return id;
}
BodyId Scene::create_body(Transform pose,bool dynamic) {
    pose=validated_pose(pose);
    Actor actor;
    actor.motion=dynamic?Motion::Dynamic:Motion::Static;
    actor.position=pose.position;
    actor.orientation=pose.orientation;
    if(dynamic) {
        actor.inverse_mass=1.0f;
        actor.inverse_inertia_local={1.0f,1.0f,1.0f};
        actor.angular_inertia_available=true;
    }
    return add(actor);
}
ShapeId Scene::attach_box(BodyId body,Vec3 h,Transform pose) {
    validate_box(h,pose.position);
    Shape shape;
    shape.half_extents=h;
    shape.local_pose=pose;
    return attach(body,shape);
}
ShapeId Scene::attach_sphere(BodyId body,float radius,Transform pose) {
    if(!std::isfinite(radius) || radius<=0.0f || !finite(pose.position))
        throw std::invalid_argument("Sphere geometry invalid");
    Shape shape;
    shape.geometry=Geometry::Sphere;
    shape.radius=radius;
    shape.local_pose=pose;
    return attach(body,shape);
}
ShapeId Scene::attach_convex(
    BodyId body,std::shared_ptr<const ConvexMesh> mesh,Transform pose) {
    if(!mesh || mesh->vertices().empty())
        throw std::invalid_argument("Cooked convex geometry is missing");
    // A cooked mesh is immutable. Attached instances of the same mesh
    // share its one stable resident vertex span across their lifetimes.
    const auto reused=std::find_if(shapes_.begin(),shapes_.end(),
        [&](const Shape& prior) {
            return prior.geometry==Geometry::Convex &&
                   prior.convex.get()==mesh.get();
        });
    const bool existing=reused!=shapes_.end();
    if(!existing && convex_vertices_.size()>
       std::numeric_limits<std::uint32_t>::max()-mesh->vertices().size())
        throw std::overflow_error("Convex vertex arena exceeds 32-bit device indexing");
    const auto lower=mesh->bounds_min(),upper=mesh->bounds_max();
    Shape shape;
    shape.geometry=Geometry::Convex;
    shape.convex=std::move(mesh);
    shape.half_extents={
        std::max(std::abs(lower.x),std::abs(upper.x)),
        std::max(std::abs(lower.y),std::abs(upper.y)),
        std::max(std::abs(lower.z),std::abs(upper.z))};
    shape.convex_vertex_offset=existing
        ?reused->convex_vertex_offset
        :static_cast<std::uint32_t>(convex_vertices_.size());
    shape.convex_vertex_count=static_cast<std::uint32_t>(
        shape.convex->vertices().size());
    shape.local_pose=pose;
    const auto previous=convex_vertices_.size();
    const auto previous_faces=convex_faces_.size();
    const auto previous_indices=convex_face_indices_.size();
    if(!existing &&
       (shape.convex->faces().size()>
            std::numeric_limits<std::uint32_t>::max()-previous_faces ||
        shape.convex->face_indices().size()>
            std::numeric_limits<std::uint32_t>::max()-previous_indices))
        throw std::overflow_error("Convex face resource exceeds device indexing");
    try {
        if(!existing) {
            convex_vertices_.insert(convex_vertices_.end(),
                shape.convex->vertices().begin(),shape.convex->vertices().end());
            convex_face_spans_.resize(convex_vertices_.size());
            convex_face_spans_[previous]={
                static_cast<std::uint32_t>(previous_faces),
                static_cast<std::uint32_t>(shape.convex->faces().size())};
            convex_face_indices_.insert(convex_face_indices_.end(),
                shape.convex->face_indices().begin(),
                shape.convex->face_indices().end());
            for(auto face:shape.convex->faces()) {
                face.first_index+=static_cast<std::uint32_t>(previous_indices);
                convex_faces_.push_back(face);
            }
        }
        return attach(body,std::move(shape));
    }catch(...) {
        convex_vertices_.resize(previous);
        convex_face_spans_.resize(previous);
        convex_faces_.resize(previous_faces);
        convex_face_indices_.resize(previous_indices);
        throw;
    }
}
Scene::Shape& Scene::require_shape(ShapeId id) {
    if(id>=shapes_.size() || !shapes_[id].alive)
        throw std::out_of_range("Unknown or retired native shape ID");
    return shapes_[id];
}
const Scene::Shape& Scene::require_shape(ShapeId id) const {
    if(id>=shapes_.size() || !shapes_[id].alive)
        throw std::out_of_range("Unknown or retired native shape ID");
    return shapes_[id];
}
Pair Scene::body_pair(Pair pair) const noexcept {
    return {shapes_[pair.a].body,shapes_[pair.b].body};
}
Vec3 Scene::mass_position(const Actor& actor) const noexcept {
    const Vec3 p=rotate(actor.orientation,actor.mass_local_pose.position);
    return {actor.position.x+p.x,actor.position.y+p.y,actor.position.z+p.z};
}
Quat Scene::mass_orientation(const Actor& actor) const noexcept {
    return multiply(actor.orientation,actor.mass_local_pose.orientation);
}
void Scene::refresh_shape_pose(Shape& shape) noexcept {
    const auto& actor=actors_[shape.body];
    const Vec3 p=rotate(actor.orientation,shape.local_pose.position);
    shape.position={actor.position.x+p.x,actor.position.y+p.y,actor.position.z+p.z};
    shape.orientation=multiply(actor.orientation,shape.local_pose.orientation);
}
void Scene::invalidate_body_shapes(BodyId body) {
    for(ShapeId id=actors_[body].first_shape;id!=invalid_shape;id=shapes_[id].next) {
        auto& shape=shapes_[id];
        if(gpu_broadphase_ && shape.geometry!=Geometry::Plane && shape.simulation_shape_enabled)
            gpu_broadphase_->mark_shape_bounds_changed(id,shape.contact_offset);
        refresh_shape_pose(shape);
        shape.cached_bounds_current=false;
    }
    gpu_broadphase_snapshot_current_=false;
    dirty_=true;
}
BodyId Scene::add_static_box(
    Vec3 h,Vec3 p,std::optional<AggregateId> aggregate) {
    validate_box(h, p);
    Actor a;
    Shape shape;
    a.motion = Motion::Static;
    shape.geometry = Geometry::Box;
    shape.half_extents = h;
    a.position = p;
    const BodyId id=add(a);
    try {attach(id,shape,aggregate);} catch(...) {remove(id); throw;}
    return id;
}
BodyId Scene::add_static_floor(float height) {
    if (!std::isfinite(height))
        throw std::invalid_argument("Native scene floor must be finite");
    Actor a;
    Shape shape;
    a.motion = Motion::Static;
    shape.geometry = Geometry::Plane;
    a.position = {0.0f, height, 0.0f};
    const BodyId id=add(a);
    try {attach(id,shape);} catch(...) {remove(id); throw;}
    return id;
}
BodyId Scene::add_dynamic_box(
    Vec3 h,Vec3 p,float density,std::optional<AggregateId> aggregate) {
    validate_box(h, p);
    if (!std::isfinite(density) || density <= 0.0f)
        throw std::invalid_argument("Native scene density must be finite and positive");
    Actor a;
    Shape shape;
    a.motion = Motion::Dynamic;
    shape.geometry = Geometry::Box;
    shape.half_extents = h;
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
    const BodyId id=add(a);
    try {attach(id,shape,aggregate);} catch(...) {remove(id); throw;}
    return id;
}

BodyId Scene::add_static_sphere(
    float radius,Vec3 position,std::optional<AggregateId> aggregate) {
    if(!std::isfinite(radius) || !(radius>0.0f) || !finite(position))
        throw std::invalid_argument("Sphere radius and position must be finite and valid");
    Actor actor;
    Shape shape;
    actor.motion=Motion::Static;
    shape.geometry=Geometry::Sphere;
    shape.radius=radius;
    actor.position=position;
    const BodyId id=add(actor);
    try {attach(id,shape,aggregate);} catch(...) {remove(id); throw;}
    return id;
}
BodyId Scene::add_dynamic_sphere(
    float radius,Vec3 position,float density,
    std::optional<AggregateId> aggregate) {
    if(!std::isfinite(radius) || !(radius>0.0f) || !finite(position) ||
       !std::isfinite(density) || !(density>0.0f))
        throw std::invalid_argument("Sphere radius, density and position invalid");
    const double radius64=radius;
    constexpr double four_thirds_pi=4.1887902047863909846168578443727;
    const double mass=four_thirds_pi*radius64*radius64*radius64*density;
    const double inverse_mass=1.0/mass;
    if(!std::isfinite(inverse_mass) || inverse_mass<=0.0 ||
       inverse_mass>std::numeric_limits<float>::max() ||
       static_cast<float>(inverse_mass)==0.0f)
        throw std::invalid_argument("Sphere mass outside supported range");
    Actor actor;
    Shape shape;
    actor.motion=Motion::Dynamic;
    shape.geometry=Geometry::Sphere;
    shape.radius=radius;
    actor.position=position;
    actor.density=density;
    actor.inverse_mass=static_cast<float>(inverse_mass);
    // Uniform sphere: PxMassProperties uses I=2/5 m r^2 around its center.
    const double inertia=0.4*mass*radius64*radius64;
    const double inverse_inertia=1.0/inertia;
    if(std::isfinite(inverse_inertia) && inverse_inertia>0.0 &&
       inverse_inertia<=std::numeric_limits<float>::max() &&
       static_cast<float>(inverse_inertia)>0.0f) {
        const float inv=static_cast<float>(inverse_inertia);
        actor.inverse_inertia_local={inv,inv,inv};
        actor.angular_inertia_available=true;
    }
    const BodyId id=add(actor);
    try {attach(id,shape,aggregate);} catch(...) {remove(id); throw;}
    return id;
}

AggregateId Scene::create_aggregate(
    std::uint32_t max_shapes,bool self_collision) {
    if(!gpu_broadphase_)
        throw std::logic_error(
            "Original PhysX aggregate requires native GPU broadphase");
    if(!max_shapes)
        throw std::invalid_argument("Original PhysX aggregate capacity is zero");
    return gpu_broadphase_->create_aggregate(max_shapes,self_collision);
}
void Scene::destroy_aggregate(AggregateId id) {
    if(!gpu_broadphase_)
        throw std::logic_error(
            "Original PhysX aggregate requires native GPU broadphase");
    gpu_broadphase_->destroy_aggregate(id);
    for(BodyId body=0;body<actors_.size();++body) {
        bool changed=false;
        for(ShapeId shape=actors_[body].first_shape;shape!=invalid_shape;shape=shapes_[shape].next)
            if(shapes_[shape].aggregate==id) {
                shapes_[shape].aggregate.reset();shapes_[shape].device_cache_supported=true;
                changed=true;
            }
        if(changed)refresh_shape_links(body);
    }
    gpu_broadphase_snapshot_current_=false;
    gpu_pair_refilter_dirty_=true;
    broadphase_endpoint_membership_dirty_=true;
    dirty_=true;
}
void Scene::remove_actor_from_aggregate(AggregateId aggregate,BodyId id) {
    (void)require(id);
    if(!gpu_broadphase_)
        throw std::logic_error(
            "Original PhysX aggregate requires native GPU broadphase");
    bool removed=false;
    for(ShapeId shape=actors_[id].first_shape;shape!=invalid_shape;shape=shapes_[shape].next)
        if(shapes_[shape].aggregate==aggregate) {
            gpu_broadphase_->remove_actor_from_aggregate(aggregate,shape);
            shapes_[shape].aggregate.reset();shapes_[shape].device_cache_supported=true;
            removed=true;
        }
    if(!removed)throw std::invalid_argument("Body is not attached to aggregate");
    refresh_shape_links(id);
    gpu_broadphase_snapshot_current_=false;
    gpu_pair_refilter_dirty_=true;
    broadphase_endpoint_membership_dirty_=true;
    dirty_=true;
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
    for(ShapeId shape=actor.first_shape;shape!=invalid_shape;shape=shapes_[shape].next)
    if(shapes_[shape].geometry==Geometry::Plane) {
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
        // Validate the source-owned volume and mark its original GPU
        // changed bitmap BEFORE wake, Gu/PCM cache or actor mutation.
        // A failed source owner cannot leave half-authored Sc motion.
        for(ShapeId shape=actor.first_shape;shape!=invalid_shape;shape=shapes_[shape].next)
            if(gpu_broadphase_ && shapes_[shape].simulation_shape_enabled &&
               shapes_[shape].geometry!=Geometry::Plane)
                gpu_broadphase_->mark_shape_bounds_changed(shape,shapes_[shape].contact_offset);
    }
    // NpRigidDynamic::setGlobalPose wakes even on an idempotent edit.
    if(actor.motion==Motion::Dynamic) {
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
    if(pcm_plane_box_enabled_ || pcm_box_box_enabled_) {
        bool invalidated=false;
        for(EdgeIndex edge_id:active_edge_handles_) {
            auto& edge=island_edges_[edge_id];
            if(edge.pair.a==id || edge.pair.b==id) {
                edge.pcm_cache={};
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
    erase_body_friction_history(id);
    actor.position=position;
    actor.orientation=normalized;
    invalidate_body_shapes(id);
    // Preserve the established source contact-manager wake handoff:
    // relocating or rotating either endpoint wakes old touching peers
    // before their previous Gu managers can be retired.
    for(const auto& manifold:manifolds_) {
        const Pair bodies=body_pair(manifold.pair);
        if(bodies.a!=id && bodies.b!=id)continue;
        const BodyId other=bodies.a==id?bodies.b:bodies.a;
        auto& dynamic=actors_[other];
        if(!dynamic.alive || dynamic.motion!=Motion::Dynamic ||
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
void Scene::erase_body_friction_history(BodyId id) {erase_friction_history(id,true);}
void Scene::erase_shape_friction_history(ShapeId id) {erase_friction_history(id,false);}
void Scene::erase_friction_history(std::uint32_t id,bool body_scope) {
    // Explicit authored shape discontinuities compact only affected
    // cached anchors. Reindex remaining source manager spans after
    // this explicit edit; the normal PGS path rotates its own index.
    if(!std::erase_if(friction_patches_, [this,id,body_scope](const FrictionPatch& patch) {
        return body_scope ? (shapes_[patch.pair.a].body==id || shapes_[patch.pair.b].body==id)
                          : (patch.pair.a==id || patch.pair.b==id);
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
    if (actor.motion != Motion::Dynamic)
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
    if (actor.motion != Motion::Dynamic)
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
    if(actor.motion!=Motion::Dynamic)
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
            const Vec3 local=rotate_inverse(mass_orientation(actor),value);
            converted=rotate(mass_orientation(actor),{
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
    if(actor.motion!=Motion::Dynamic)
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
void Scene::set_kinematic_target(BodyId id, Vec3 position, Quat orientation) {
    auto& actor=require(id);
    if(actor.motion!=Motion::Kinematic)
        throw std::logic_error(
            "Source kinematic target requires eKINEMATIC rigid body");
    const float magnitude=std::sqrt(
        orientation.x*orientation.x+orientation.y*orientation.y+
        orientation.z*orientation.z+orientation.w*orientation.w);
    if(!finite(position) || !finite(orientation) ||
       !std::isfinite(magnitude) ||
       std::abs(magnitude-1.0f)>=1.0e-2f)
        throw std::invalid_argument("Native kinematic target is not sane");
    // Source active kinematic target publication is a per-actor edit.
    // Preflight allocations before changing the target/wake state.
    speculative_islands_.activating_nodes.reserve(actors_.size());
    accurate_islands_.activating_nodes.reserve(actors_.size());
    if(!actor.kinematic_tracked)
        active_kinematic_targets_.push_back(id);
    actor.kinematic_tracked=true;
    actor.kinematic_target_pending=true;
    actor.kinematic_target_position=position;
    actor.kinematic_target_orientation={
        orientation.x/magnitude,orientation.y/magnitude,
        orientation.z/magnitude,orientation.w/magnitude};
    refresh_kinematic_node(id,false);
    refresh_kinematic_node(id,true);
    actor.kinematic_settling=false;
    actor.sleeping=false;
    // IslandSim::processLostEdges must recognize target-driven
    // kinematics as active until their target has been integrated.
    actor.ready_for_sleep=false;
    actor.wake_counter=0.4f;
    // ScShapeInteraction::resetManagerCachedState has the same
    // authoritative EdgeIndex and intrusive accurate endpoint list.
    // Traverse only this body's touching neighbors (degree), rather
    // than rescanning all Gu managers for each animated kinematic.
    if(id>=accurate_node_first_.size())return;
    for(std::size_t instance=accurate_node_first_[id];
        instance!=invalid_instance;) {
        const auto& edge=island_edges_[instance/2];
        const auto& link=edge.accurate[instance&1u];
        instance=link.next;
        if(!edge.live || !edge.connected || edge.retiring)continue;
        const BodyId other=edge.pair.a==id?edge.pair.b:edge.pair.a;
        auto& partner=actors_[other];
        if(!partner.alive || partner.motion!=Motion::Dynamic ||
           (!partner.sleeping && !partner.ready_for_sleep))continue;
        if(!partner.activating)queue_node_activation(other);
        partner.activating=true;
        partner.sleeping=false;
        partner.ready_for_sleep=false;
        partner.wake_counter=std::max(partner.wake_counter,0.4f);
        partner.sleep_linear_acc={};
        partner.sleep_angular_acc={};
        authored_activation_pending_=true;
    }
}

void Scene::set_kinematic(BodyId id,bool enabled) {
    auto& actor=require(id);
    if(!rigid_node(actor.motion))
        throw std::logic_error("Kinematic flag requires a rigid box");
    if((actor.motion==Motion::Kinematic)==enabled)return;

    // Source ScBodySim::setKinematic removes the node from its rigid
    // island without destroying the PxRigidDynamic or its shape. Reserve
    // wake publications before mutating actor type or graph ownership.
    speculative_islands_.activating_nodes.reserve(actors_.size());
    accurate_islands_.activating_nodes.reserve(actors_.size());
    // Original ScBodySim switches the AABB group before refiltering its
    // interactions; do not leave the GPU volume in its previous class.
    if(gpu_broadphase_)
        gpu_broadphase_->set_actor_motion(
            id,enabled ? GpuBroadphaseBox::Motion::Kinematic
                       : GpuBroadphaseBox::Motion::Dynamic);
    // The previous source island owns the indexed contact-manager
    // entries until the rigid node's motion class changes. Detach every
    // incident indexed edge while its old dynamic endpoint still
    // identifies that island. The full first pass reindexes surviving
    // Gu managers under the new dynamic endpoints without replacing
    // their Sc contact-manager handles.
    if(gpu_broadphase_)
        for(EdgeIndex edge_id:active_edge_handles_) {
            const auto& edge=island_edges_[edge_id];
            if(edge.live && edge.indexed_accurate &&
               (edge.pair.a==id || edge.pair.b==id))
                remove_accurate_island_edge(edge_id);
        }
    // No target has been authored: the kinematic body has no external
    // motion and is not eligible for Dy solver-body integration.
    erase_body_friction_history(id);
    for(EdgeIndex edge_id:active_edge_handles_) {
        auto& edge=island_edges_[edge_id];
        if(edge.pair.a==id || edge.pair.b==id)edge.pcm_cache={};
    }
    // A mode change refilters the existing shape interactions and wakes
    // formerly touching dynamic partners as in Sc interaction dirtying.
    for(const auto& manifold:manifolds_) {
        const Pair bodies=body_pair(manifold.pair);
        if(bodies.a!=id && bodies.b!=id)continue;
        const BodyId other=bodies.a==id?bodies.b:bodies.a;
        auto& partner=actors_[other];
        if(!partner.alive || partner.motion!=Motion::Dynamic ||
           (!partner.sleeping && !partner.ready_for_sleep))continue;
        if(!partner.activating)queue_node_activation(other);
        partner.activating=true;
        partner.sleeping=false;
        partner.ready_for_sleep=false;
        partner.wake_counter=std::max(partner.wake_counter,0.4f);
        partner.sleep_linear_acc={};
        partner.sleep_angular_acc={};
        authored_activation_pending_=true;
    }
    if(!enabled && actor.kinematic_tracked) {
        std::erase(active_kinematic_targets_,id);
        actor.kinematic_tracked=false;
        actor.kinematic_target_pending=false;
        actor.kinematic_settling=false;
    }
    if(enabled) {
        actor.motion=Motion::Kinematic;
        --live_dynamic_count_;
        actor.velocity={};
        actor.angular_velocity={};
        actor.pending_linear_velocity_delta={};
        actor.pending_angular_velocity_delta={};
        actor.pending_linear_acceleration={};
        actor.pending_angular_acceleration={};
        actor.activating=false;
        actor.sleeping=true;
        actor.ready_for_sleep=true;
    } else {
        actor.motion=Motion::Dynamic;
        ++live_dynamic_count_;
        queue_node_activation(id);
        actor.activating=true;
        actor.sleeping=false;
        actor.ready_for_sleep=false;
        actor.wake_counter=0.4f;
        actor.sleep_linear_acc={};
        actor.sleep_angular_acc={};
        authored_activation_pending_=true;
    }
    for(ShapeId shape=actor.first_shape;shape!=invalid_shape;shape=shapes_[shape].next)
        shapes_[shape].motion=actor.motion;
    refresh_kinematic_node(id,false);
    refresh_kinematic_node(id,true);
    // Full node-type remapping is restricted to this explicit edit.
    // Quiet frames retain persistent source island handles.
    island_topology_dirty_=true;
    broadphase_endpoint_membership_dirty_=true;
    if(gpu_broadphase_)gpu_pair_refilter_dirty_=true;
    gpu_broadphase_snapshot_current_=false;
    dirty_=true;
}

void Scene::set_shape_simulation_enabled(BodyId body,bool enabled) {
    for(ShapeId shape=require(body).first_shape;shape!=invalid_shape;shape=shapes_[shape].next)
        set_attached_shape_enabled(shape,enabled);
}
void Scene::set_shape_offsets(BodyId body,float contact,float rest) {
    for(ShapeId shape=require(body).first_shape;shape!=invalid_shape;shape=shapes_[shape].next)
        set_attached_shape_offsets(shape,contact,rest);
}
void Scene::set_material(BodyId body,Material material) {
    for(ShapeId shape=require(body).first_shape;shape!=invalid_shape;shape=shapes_[shape].next)
        set_attached_shape_material(shape,material);
}
void Scene::set_attached_shape_enabled(ShapeId id, bool enabled) {
    auto& actor=require_shape(id);
    if(actor.simulation_shape_enabled==enabled)return;
    // The source AABB removal/addition can reject the Sc shape. Run
    // that source preflight before clearing any existing PCM/friction
    // working set or publishing the native shape flag.
    if(gpu_broadphase_ && actor.geometry!=Geometry::Plane)
        gpu_broadphase_->set_shape_simulation_enabled(id,enabled);
    // NpShape::setFlag(eSIMULATION_SHAPE) refilters this shape.
    // Retire the previous contact history on this source flag change.
    // The following SAP publication is the authority for edge loss,
    // and must preserve the actor/solver body across the refilter.
    erase_shape_friction_history(id);
    if(pcm_plane_box_enabled_ || pcm_box_box_enabled_)
        for(EdgeIndex edge_id:active_edge_handles_) {
            auto& edge=island_edges_[edge_id];
            if(edge.shapes.a==id || edge.shapes.b==id)
                edge.pcm_cache={};
        }
    actor.simulation_shape_enabled=enabled;
    refresh_shape_links(actor.body);
    if(actor.geometry==Geometry::Plane) {
        if(enabled)++active_plane_count_;
        else --active_plane_count_;
    }
    broadphase_endpoint_membership_dirty_=true;
    if(gpu_broadphase_) {
        if(actor.geometry==Geometry::Plane)
            plane_pair_membership_dirty_=true;
        else
            gpu_pair_refilter_dirty_=true;
    }
    if(actor.geometry!=Geometry::Plane)
        gpu_broadphase_snapshot_current_=false;
    dirty_=true;
}

void Scene::set_attached_shape_offsets(ShapeId id, float contact_offset,
                              float rest_offset) {
    // NpShape::setContactOffset and setRestOffset require finite values,
    // contactOffset >= 0 and restOffset < contactOffset. The two fields
    // are published together, avoiding invalid intermediate authored
    // states when lowering to the source PxShape API.
    if (!std::isfinite(contact_offset) || !std::isfinite(rest_offset) ||
        contact_offset < 0.0f || rest_offset >= contact_offset)
        throw std::invalid_argument("Native PhysX shape offsets invalid");
    auto& actor=require_shape(id);
    if(actor.contact_offset==contact_offset &&
       actor.rest_offset==rest_offset)return;
    // AABBManagerBase::setContactDistance owns the source GPU changed bit
    // and the new distance; keep that publication ahead of scene edits.
    if(gpu_broadphase_ && actor.geometry!=Geometry::Plane)
        gpu_broadphase_->mark_shape_bounds_changed(id,contact_offset);
    // An authored shape change invalidates prior friction correlation;
    // retire only this shape's anchors in the existing scene-owned cache.
    erase_shape_friction_history(id);
    // ScShapeCore shape-parameter edits invalidate the native PCM
    // witnesses for this shape's active contact managers. Refreshing
    // only relative transforms would not generate candidates newly
    // admitted by a larger contact distance. Actor edits are explicit,
    // so walk current overlap handles, never historical retired slots.
    if(pcm_plane_box_enabled_ || pcm_box_box_enabled_)
        for(EdgeIndex edge_id:active_edge_handles_) {
            auto& edge=island_edges_[edge_id];
            if(edge.shapes.a==id || edge.shapes.b==id) {
                edge.pcm_cache={};
            }
        }
    actor.contact_offset=contact_offset;
    actor.rest_offset=rest_offset;
    actor.cached_bounds_current=false;
    if(actor.geometry!=Geometry::Plane)
        gpu_broadphase_snapshot_current_=false;
    dirty_=true;
}

void Scene::set_attached_shape_material(ShapeId id, Material material) {
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
    require_shape(id).material=material;
}
void Scene::set_dynamics(BodyId id, RigidBodyDynamics d) {
    auto& actor=require(id);
    if(actor.motion!=Motion::Dynamic)
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
    // ScBodyCore::contactReportThreshold belongs to PxRigidDynamic,
    // including one carrying eKINEMATIC. Both live node indices and
    // both thresholds reach DySolverConstraints::writeBackContact.
    // PX_MAX_REAL disables reporting; reject negative/nonfinite values.
    if (!std::isfinite(force) || force < 0.0f)
        throw std::invalid_argument(
            "Native contact report threshold must be nonnegative and finite");
    auto& actor=require(id);
    if (!rigid_node(actor.motion))
        throw std::logic_error(
            "Source contact report threshold requires PxRigidDynamic");
    actor.contact_report_threshold=force;
}
void Scene::remove(BodyId id) {
    auto& actor=require(id);
    const bool dynamic_node=actor.motion==Motion::Dynamic;
    // Retirement may wake multiple sleeping Gu partners. Reserve both
    // graph-local event queues before touching partner motion flags.
    if(!manifolds_.empty()) {
        speculative_islands_.activating_nodes.reserve(actors_.size());
        accurate_islands_.activating_nodes.reserve(actors_.size());
    }
    // Before removing the source volume, reserve all native graph work
    // that can grow during contact-manager and constraint retirement.
    std::size_t incident=0;
    for(EdgeIndex edge:active_constraint_handles_) {
        const Pair pair=island_edges_[edge].pair;
        incident+=(pair.a==id || pair.b==id);
    }
    retired_edge_handles_.reserve(retired_edge_handles_.size()+incident);
    // The source AABB retains its actor identity through GPU lost fetch.
    while(actor.first_shape!=invalid_shape)detach_shape(actor.first_shape);
    // Sc actor retirement removes physical contact-manager edges. A
    // dormant or sleep-ready rigid body resting on the retired actor
    // cannot continue sleeping without its supporting contact: revoke
    // prior-frame readiness before third-pass deactivation and reenter
    // gravity/Gu contact generation on the next native step.
    // Traverse the published Gu pairs only on this explicit edit path.
    for(const auto& manifold:manifolds_) {
        const Pair bodies=body_pair(manifold.pair);
        if(bodies.a!=id && bodies.b!=id)continue;
        const BodyId other=bodies.a==id?bodies.b:bodies.a;
        auto& partner=actors_[other];
        if(!partner.alive || partner.motion!=Motion::Dynamic ||
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
    // ScConstraintInteraction retires every edge incident to an actor
    // before the island node can be invalidated. The same shared handle
    // pool and edge-instance disconnect path handles explicit deletion.
    for(std::size_t cursor=0;cursor<active_constraint_handles_.size();) {
        const EdgeIndex edge=active_constraint_handles_[cursor];
        const Pair pair=island_edges_[edge].pair;
        if(pair.a==id || pair.b==id)
            remove_constraint_edge(edge);
        else ++cursor;
    }
    if(gpu_broadphase_)gpu_broadphase_->release_rigid_actor(id);
    actor.alive = false;
    broadphase_endpoint_membership_dirty_=true;
    refresh_kinematic_node(id,false);
    refresh_kinematic_node(id,true);
    if(dynamic_node)--live_dynamic_count_;
    if(actor.kinematic_tracked) {
        std::erase(active_kinematic_targets_,id);
        actor.kinematic_tracked=false;
    }
    gpu_pair_refilter_dirty_=true;
    gpu_broadphase_snapshot_current_=false;
    dirty_ = true;
    island_topology_dirty_ |= dynamic_node;
}
BodyState Scene::body_state(BodyId id) const {
    const auto& actor = require(id);
    return {actor.position, actor.velocity,
            actor.orientation, actor.angular_velocity, actor.sleeping};
}
void Scene::update_shape_bounds(Shape& actor) {
    refresh_shape_pose(actor);
    // GuBounds::computeBounds followed by Sc cached-shape publication.
    // Construct every encoded endpoint before committing.
        const Vec3 p = actor.position, h = actor.half_extents;
        // Source: GuBounds.cpp::computeBounds(eBOX), basisExtentV.
        // The world AABB of a rotated box is the sum of absolute
        // world-space basis columns weighted by the half extents,
        // plus its shape contact offset. The identity fast path
        // avoids three quaternion rotations for common unrotated boxes.
        Vec3 extent = actor.geometry==Geometry::Sphere
            ? Vec3{actor.radius,actor.radius,actor.radius}:h;
        const Quat q = actor.orientation;
        if (actor.geometry==Geometry::Box &&
            (q.x != 0.0f || q.y != 0.0f ||
             q.z != 0.0f || q.w != 1.0f)) {
            const Vec3 x = rotate(q, {1.0f,0.0f,0.0f});
            const Vec3 y = rotate(q, {0.0f,1.0f,0.0f});
            const Vec3 z = rotate(q, {0.0f,0.0f,1.0f});
            extent = {
                std::abs(x.x)*h.x + std::abs(y.x)*h.y + std::abs(z.x)*h.z,
                std::abs(x.y)*h.x + std::abs(y.y)*h.y + std::abs(z.y)*h.z,
                std::abs(x.z)*h.x + std::abs(y.z)*h.y + std::abs(z.z)*h.z
            };
        }
        if(actor.geometry==Geometry::Convex) {
            Vec3 lower{
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity()};
            Vec3 upper{-lower.x,-lower.y,-lower.z};
            for(const Vec3 v:actor.convex->vertices()) {
                const Vec3 rotated=rotate(q,v);
                lower.x=std::min(lower.x,rotated.x);
                lower.y=std::min(lower.y,rotated.y);
                lower.z=std::min(lower.z,rotated.z);
                upper.x=std::max(upper.x,rotated.x);
                upper.y=std::max(upper.y,rotated.y);
                upper.z=std::max(upper.z,rotated.z);
            }
            const Vec3 center{
                (upper.x+lower.x)*0.5f,
                (upper.y+lower.y)*0.5f,
                (upper.z+lower.z)*0.5f};
            extent={(upper.x-lower.x)*0.5f,(upper.y-lower.y)*0.5f,
                    (upper.z-lower.z)*0.5f};
            // Asymmetric hull bounds are centered away from the
            // authoring origin; adjust the six source endpoints below.
            const Vec3 offset{center.x,center.y,center.z};
            const Vec3 added{extent.x+actor.contact_offset,
                             extent.y+actor.contact_offset,
                             extent.z+actor.contact_offset};
            const std::array<float,6> bounds{
                p.x+offset.x-added.x,p.y+offset.y-added.y,p.z+offset.z-added.z,
                p.x+offset.x+added.x,p.y+offset.y+added.y,p.z+offset.z+added.z};
            std::array<std::uint32_t,6> encoded{};
            for(const float v:bounds)
                if(!std::isfinite(v))
                    throw std::overflow_error("Convex scene AABB is not finite");
            for(unsigned axis=0;axis<3u;++axis) {
                encoded[axis]=encode_min(bounds[axis]);
                encoded[axis+3u]=encode_max(bounds[axis+3u]);
            }
            actor.encoded_bounds=encoded;
            actor.cached_bounds_current=true;
            return;
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
void Scene::publish_integrated_shape_bounds(BodyId,Actor& actor) {
    for(ShapeId id=actor.first_shape;id!=invalid_shape;id=shapes_[id].next) {
        auto& shape=shapes_[id];
        refresh_shape_pose(shape);
        if(shape.geometry==Geometry::Plane)continue;
        if(!gpu_broadphase_) {update_shape_bounds(shape); continue;}
        if(shape.simulation_shape_enabled)
            gpu_broadphase_->mark_shape_bounds_changed(id,shape.contact_offset);
        if(shape.simulation_shape_enabled && gpu_broadphase_snapshot_current_) {
            auto it=std::lower_bound(gpu_broadphase_boxes_.begin(),gpu_broadphase_boxes_.end(),id,
                [](const GpuBroadphaseBox& box,ShapeId key){return box.id<key;});
            if(it==gpu_broadphase_boxes_.end() || it->id!=id)
                throw std::logic_error("Integrated shape missing from GPU SAP input");
            it->position=shape.position;
            it->orientation=shape.orientation;
        }
        shape.cached_bounds_current=true;
    }
}
void Scene::update_bounds() {
    // BpBroadPhaseSap::batchCreate/mBoxBoundsMinMax:
    // min and max endpoints are retained on every axis and mapped
    // directly from each live SapBox1D's mMinMax endpoint positions.
    // Actor ID history never changes the width of this working set.
    const auto less=[](Endpoint a,Endpoint b) {
        if(a.value!=b.value)return a.value<b.value;
        if(a.is_max!=b.is_max)return !a.is_max;
        return a.id<b.id;
    };
    const auto reindex=[&](unsigned axis) {
        const auto& endpoints=sap_axis_endpoints_[axis];
        for(std::size_t i=0;i<endpoints.size();++i) {
            const Endpoint& ep=endpoints[i];
            sap_volumes_[ep.volume_slot].
                endpoint_positions[axis][ep.is_max?1u:0u]=
                static_cast<std::uint32_t>(i);
        }
    };
    broadphase_updated_ids_.clear();
    broadphase_updated_volume_slots_.clear();
    broadphase_updated_axis_masks_.clear();
    broadphase_changed_axes_=0;
    broadphase_crossed_pairs_.clear();
    broadphase_full_pair_update_=broadphase_endpoint_membership_dirty_;
    broadphase_batch_pair_set_unchanged_=false;
    broadphase_batch_crossings_complete_=false;
    try {
        if(broadphase_endpoint_membership_dirty_) {
            // The endpoint-index scratch is persistent across completed
            // frames; only a structural proxy change invalidates it.
            sap_list_next_.clear();
            sap_list_prev_.clear();
            sap_volumes_.clear();
            sap_volumes_.reserve(live_shape_ids_.size());
            for(auto& axis:sap_axis_endpoints_) {
                axis.clear();
                axis.reserve(live_shape_ids_.size()*2);
            }
            for(BodyId id:live_shape_ids_) {
                Shape& actor=shapes_[id];
                if(!actor.simulation_shape_enabled ||
                   actor.geometry==Geometry::Plane)continue;
                if(!actor.cached_bounds_current)
                    update_shape_bounds(actor);
                const auto& bounds=actor.encoded_bounds;
                const auto slot=static_cast<std::uint32_t>(
                    sap_volumes_.size());
                sap_volumes_.push_back({id,bounds,{}});
                for(unsigned axis=0;axis<3;++axis) {
                    sap_axis_endpoints_[axis].push_back({
                        bounds[axis],id,slot,false});
                    sap_axis_endpoints_[axis].push_back({
                        bounds[axis+3],id,slot,true});
                }
            }
            for(unsigned axis=0;axis<3;++axis) {
                auto& endpoints=sap_axis_endpoints_[axis];
                std::sort(endpoints.begin(),endpoints.end(),less);
                reindex(axis);
            }
            ++broadphase_endpoint_full_sort_count_;
            broadphase_endpoint_membership_dirty_=false;
            return;
        }

        // The source mUpdated worklist uses the stored encoded bounds,
        // not dirty authored positions whose rounded AABB is unchanged.
        for(std::size_t slot=0;slot<sap_volumes_.size();++slot) {
            SapVolume& volume=sap_volumes_[slot];
            Shape& actor=shapes_[volume.id];
            if(!actor.cached_bounds_current)
                update_shape_bounds(actor);
            if(volume.published_bounds!=actor.encoded_bounds) {
                std::uint8_t axes=0;
                for(unsigned axis=0;axis<3;++axis)
                    if(volume.published_bounds[axis]!=actor.encoded_bounds[axis] ||
                       volume.published_bounds[axis+3]!=
                           actor.encoded_bounds[axis+3])
                        axes|=static_cast<std::uint8_t>(1u<<axis);
                broadphase_updated_ids_.push_back(volume.id);
                broadphase_updated_volume_slots_.push_back(
                    static_cast<std::uint32_t>(slot));
                broadphase_updated_axis_masks_.push_back(axes);
                broadphase_changed_axes_|=axes;
                volume.published_bounds=actor.encoded_bounds;
            }
        }
        if(broadphase_updated_volume_slots_.empty())return;

        // BpBroadPhaseSap::batchUpdateFewUpdates indexes only the two
        // changed endpoints per axis via SapBox1D::mMinMax. Every
        // insertion shift immediately repairs the moved endpoint's
        // index, so no O(all six endpoints) scan is needed for local
        // updates. Broad reorderings use a bounded full-sort fallback.
        // BpBroadPhaseSap::batchUpdateFewUpdates uses direct endpoint
        // shifts for sparse mUpdated handles, even across a long
        // distance. Full re-sorting for a single teleport is both
        // slower and less source-faithful. Bound insertion work only
        // on the source's large-update branch.
        bool sparse=broadphase_updated_volume_slots_.size()*5<
                    sap_volumes_.size();
        bool localized_bulk=false;
        if(!sparse) {
            // Choose the source indexed update for bounded rank motion,
            // regardless of world-space distance. A sparsely populated
            // axis can admit a long translation across only a handful
            // of endpoint neighbors; using shape width to reject that
            // update needlessly pays for linked pocket construction.
            // The old sorted keys bound the preflight work. A move
            // outside that rank envelope uses linked activity pockets.
            // Probe old-rank neighbors without allocating a worklist.
            // Restrict indexed bulk shifts to narrow rank-local
            // changes. Wider crossing groups cost less through the
            // linked source activity-pocket path.
            constexpr std::size_t local_span=24;
            bool localized=true;
            bool stable=true;
            const auto final_key=[&](Endpoint ep,unsigned axis) {
                ep.value=shapes_[ep.id].encoded_bounds[
                    axis+(ep.is_max?3u:0u)];
                return ep;
            };
            for(unsigned axis=0;axis<3;++axis) {
                const auto bit=static_cast<std::uint8_t>(1u<<axis);
                if(!(broadphase_changed_axes_&bit))continue;
                const auto& endpoints=sap_axis_endpoints_[axis];
                for(std::size_t updated=0;
                    updated<broadphase_updated_volume_slots_.size();++updated) {
                    if(!(broadphase_updated_axis_masks_[updated]&bit))continue;
                    const auto slot=broadphase_updated_volume_slots_[updated];
                    for(unsigned side=0;side<2;++side) {
                        const auto i=static_cast<std::size_t>(
                            sap_volumes_[slot].endpoint_positions[axis][side]);
                        const Endpoint moving=final_key(endpoints[i],axis);
                        if((i && less(moving,final_key(endpoints[i-1],axis))) ||
                           (i+1<endpoints.size() &&
                            less(final_key(endpoints[i+1],axis),moving)))
                            stable=false;
                        // Both reference points are in the OLD sorted
                        // array. A candidate within this rank envelope
                        // can use direct adjacent insertion safely; a
                        // larger displacement takes the pocket route.
                        if(i>=local_span &&
                           less(moving,endpoints[i-local_span]))
                            localized=false;
                        if(i+local_span<endpoints.size() &&
                           less(endpoints[i+local_span],moving))
                            localized=false;
                    }
                }
            }
            if(stable) {
                // All final neighboring keys preserve the original
                // source endpoint order, so no overlap can transition.
                for(unsigned axis=0;axis<3;++axis) {
                    const auto bit=static_cast<std::uint8_t>(1u<<axis);
                    if(!(broadphase_changed_axes_&bit))continue;
                    auto& endpoints=sap_axis_endpoints_[axis];
                    for(std::size_t updated=0;
                        updated<broadphase_updated_volume_slots_.size();++updated) {
                        if(!(broadphase_updated_axis_masks_[updated]&bit))
                            continue;
                        const auto slot=broadphase_updated_volume_slots_[updated];
                        for(unsigned side=0;side<2;++side) {
                            const auto i=sap_volumes_[slot].
                                endpoint_positions[axis][side];
                            endpoints[i].value=shapes_[endpoints[i].id].
                                encoded_bounds[axis+(side?3u:0u)];
                        }
                    }
                }
                broadphase_batch_crossings_complete_=true;
                broadphase_batch_pair_set_unchanged_=true;
                ++broadphase_batch_stable_order_count_;
                return;
            }
            localized_bulk=localized;
            sparse=localized_bulk;
        }
        // When a bounded fraction of a moderately sized SAP changes,
        // merge the sorted updated endpoints into the unchanged ordered
        // stream. This is the same final order as batchUpdate's linked
        // insertion, without relocating untouched keys repeatedly.
        // The bounded pair check below derives only net Found/Lost
        // candidates for those updated volumes; the persistent source
        // SapPairManager and Sc edge publication are not replaced.
        const std::size_t updated_count=
            broadphase_updated_volume_slots_.size();
        const bool bounded_merge=!sparse &&
            updated_count>=8 &&
            updated_count<=sap_volumes_.size()/4 &&
            sap_volumes_.size()<=65536/updated_count;
        if(bounded_merge) {
            for(unsigned axis=0;axis<3;++axis) {
                const auto bit=static_cast<std::uint8_t>(1u<<axis);
                if(!(broadphase_changed_axes_&bit))continue;
                auto& endpoints=sap_axis_endpoints_[axis];
                const std::size_t n=endpoints.size();
                sap_endpoint_updated_.resize(n);
                std::fill(sap_endpoint_updated_.begin(),
                          sap_endpoint_updated_.end(),std::uint8_t{0});
                sap_merge_changed_.clear();
                sap_merge_changed_.reserve(updated_count*2);
                for(std::size_t j=0;j<updated_count;++j) {
                    if(!(broadphase_updated_axis_masks_[j]&bit))continue;
                    const std::uint32_t slot=
                        broadphase_updated_volume_slots_[j];
                    for(unsigned side=0;side<2;++side) {
                        const auto index=sap_volumes_[slot].
                            endpoint_positions[axis][side];
                        sap_endpoint_updated_[index]=1;
                        Endpoint ep=endpoints[index];
                        ep.value=shapes_[ep.id].encoded_bounds[
                            axis+(side?3u:0u)];
                        sap_merge_changed_.push_back(ep);
                    }
                }
                // Coherent translation preserves the order within
                // the updated set, even if those endpoints crossed many
                // unchanged keys. Avoid re-sorting that common case.
                if(!std::is_sorted(sap_merge_changed_.begin(),
                                   sap_merge_changed_.end(),less))
                    std::sort(sap_merge_changed_.begin(),
                              sap_merge_changed_.end(),less);
                sap_merge_output_.clear();
                sap_merge_output_.reserve(n);
                std::size_t changed=0;
                for(std::size_t i=0;i<n;++i) {
                    if(sap_endpoint_updated_[i])continue;
                    const Endpoint unchanged=endpoints[i];
                    while(changed<sap_merge_changed_.size() &&
                          less(sap_merge_changed_[changed],unchanged))
                        sap_merge_output_.push_back(
                            sap_merge_changed_[changed++]);
                    sap_merge_output_.push_back(unchanged);
                }
                while(changed<sap_merge_changed_.size())
                    sap_merge_output_.push_back(
                        sap_merge_changed_[changed++]);
                endpoints.swap(sap_merge_output_);
                reindex(axis);
            }
            const auto overlaps=[](const Shape& a,const Shape& b) {
                const auto& aa=a.encoded_bounds;
                const auto& bb=b.encoded_bounds;
                return !(aa[0]>bb[3] || bb[0]>aa[3] ||
                         aa[1]>bb[4] || bb[1]>aa[4] ||
                         aa[2]>bb[5] || bb[2]>aa[5]);
            };
            // Bp SAP's ordered X endpoints bound the candidate work:
            // no new pair can arise between two unchanged volumes, or
            // between volumes with disjoint X intervals. Split active
            // intervals by updated membership and visit only pairs with
            // an updated participant; the source three-axis overlap and
            // persistent Sc pair keys remain authoritative.
            const std::size_t volume_count=sap_volumes_.size();
            sap_merge_updated_flags_.resize(volume_count);
            std::fill(sap_merge_updated_flags_.begin(),
                      sap_merge_updated_flags_.end(),std::uint8_t{0});
            for(const std::uint32_t slot:
                broadphase_updated_volume_slots_)
                sap_merge_updated_flags_[slot]=1;
            sap_merge_active_updated_.clear();
            sap_merge_active_unchanged_.clear();
            sap_merge_active_positions_.resize(volume_count);
            for(const Endpoint ep:sap_axis_endpoints_[0]) {
                const std::uint32_t slot=ep.volume_slot;
                const bool updated=sap_merge_updated_flags_[slot]!=0;
                auto& own=updated?sap_merge_active_updated_:
                                  sap_merge_active_unchanged_;
                if(ep.is_max) {
                    const auto index=sap_merge_active_positions_[slot];
                    const std::uint32_t tail=own.back();
                    own[index]=tail;
                    sap_merge_active_positions_[tail]=index;
                    own.pop_back();
                    continue;
                }
                const Shape& a=shapes_[ep.id];
                const auto& aa=a.encoded_bounds;
                const auto compare=[&](
                    const std::vector<std::uint32_t>& active) {
                    for(const std::uint32_t other_slot:active) {
                        const BodyId other_id=sap_volumes_[other_slot].id;
                        const Shape& b=shapes_[other_id];
                        if(a.motion!=Motion::Dynamic &&
                           b.motion!=Motion::Dynamic)continue;
                        const auto& bb=b.encoded_bounds;
                        if(aa[1]>bb[4] || bb[1]>aa[4] ||
                           aa[2]>bb[5] || bb[2]>aa[5])continue;
                        const Pair pair=ordered_pair(ep.id,other_id);
                        if(!std::binary_search(previous_pairs_.begin(),
                                               previous_pairs_.end(),pair))
                            broadphase_crossed_pairs_.push_back(pair);
                    }
                };
                compare(sap_merge_active_updated_);
                if(updated)
                    compare(sap_merge_active_unchanged_);
                sap_merge_active_positions_[slot]=
                    static_cast<std::uint32_t>(own.size());
                own.push_back(slot);
            }
            sap_merge_updated_ids_=broadphase_updated_ids_;
            std::sort(sap_merge_updated_ids_.begin(),
                      sap_merge_updated_ids_.end());
            // A Lost event also needs to be published when the new
            // bounds no longer overlap. Previous Sc pairs provide the
            // precise old membership without replaying transient
            // crossings between moving endpoint keys.
            for(const Pair pair:previous_pairs_) {
                const Shape& a=shapes_[pair.a];
                const Shape& b=shapes_[pair.b];
                if(a.geometry==Geometry::Plane ||
                   b.geometry==Geometry::Plane ||
                   (!std::binary_search(sap_merge_updated_ids_.begin(),
                                        sap_merge_updated_ids_.end(),pair.a) &&
                    !std::binary_search(sap_merge_updated_ids_.begin(),
                                        sap_merge_updated_ids_.end(),pair.b)))
                    continue;
                if(!overlaps(a,b))
                    broadphase_crossed_pairs_.push_back(pair);
            }
            broadphase_batch_crossings_complete_=true;
            broadphase_batch_pair_set_unchanged_=
                broadphase_crossed_pairs_.empty();
            ++broadphase_batch_merge_count_;
            return;
        }
        if(!sparse) {
            bool all_axes_stable=true;
            // BpBroadPhaseSap::batchUpdate: update the encoded keys in
            // their original slots, then reorder linked endpoint indices.
            // Materialize only the affected activity pockets.
            for(unsigned axis=0;axis<3;++axis) {
                const std::uint8_t axis_bit=static_cast<std::uint8_t>(1u<<axis);
                if(!(broadphase_changed_axes_&axis_bit))continue;
                auto& endpoints=sap_axis_endpoints_[axis];
                const std::size_t n=endpoints.size();
                if(!n)continue;
                bool changed=false;
                for(std::size_t updated=0;updated<broadphase_updated_volume_slots_.size();++updated) {
                    if(!(broadphase_updated_axis_masks_[updated]&axis_bit))continue;
                    const std::uint32_t slot=broadphase_updated_volume_slots_[updated];
                    for(unsigned side=0;side<2;++side) {
                        const auto index=sap_volumes_[slot].
                            endpoint_positions[axis][side];
                        Endpoint& ep=endpoints[index];
                        const auto value=shapes_[ep.id].encoded_bounds[
                            axis+(side?3:0)];
                        changed|=ep.value!=value;
                        ep.value=value;
                    }
                }
                if(!changed)continue;
                // A source batchUpdate only needs an activity pocket
                // where endpoint order actually changes. Since the old
                // stream was sorted, any new inversion must touch a
                // changed endpoint. Test those immediate neighbors in
                // O(mUpdated), without initializing the n-element linked
                // workspace on ordinary coherent bulk movement.
                bool ordering_changed=false;
                for(std::size_t updated=0;updated<broadphase_updated_volume_slots_.size();++updated) {
                    if(!(broadphase_updated_axis_masks_[updated]&axis_bit))continue;
                    const std::uint32_t slot=broadphase_updated_volume_slots_[updated];
                    for(unsigned side=0;side<2;++side) {
                        const auto index=sap_volumes_[slot].
                            endpoint_positions[axis][side];
                        if((index && less(endpoints[index],endpoints[index-1])) ||
                           (index+1<n && less(endpoints[index+1],endpoints[index]))) {
                            ordering_changed=true;
                            break;
                        }
                    }
                    if(ordering_changed)break;
                }
                if(!ordering_changed)continue;
                all_axes_stable=false;
                sap_endpoint_updated_.resize(n);
                std::fill(sap_endpoint_updated_.begin(),
                          sap_endpoint_updated_.end(),std::uint8_t{0});
                for(std::size_t updated=0;
                    updated<broadphase_updated_volume_slots_.size();++updated) {
                    if(!(broadphase_updated_axis_masks_[updated]&axis_bit))
                        continue;
                    const auto slot=broadphase_updated_volume_slots_[updated];
                    for(unsigned side=0;side<2;++side)
                        sap_endpoint_updated_[sap_volumes_[slot].
                            endpoint_positions[axis][side]]=1;
                }
                const std::size_t sentinel=n;
                // BpBroadPhaseSap's mListPrev/mListNext persist between
                // updates. They return to identity after each rebuilt
                // pocket, avoiding a whole-axis initialization on each
                // coherent batch.
                if(sap_list_next_.size()!=n+1 ||
                   sap_list_prev_.size()!=n+1) {
                    sap_list_next_.resize(n+1);
                    sap_list_prev_.resize(n+1);
                    for(std::size_t i=0;i<n;++i) {
                        sap_list_prev_[i]=i?i-1:sentinel;
                        sap_list_next_[i]=i+1<n?i+1:sentinel;
                    }
                    sap_list_next_[sentinel]=0;
                    sap_list_prev_[sentinel]=n-1;
                }
                auto& pockets=sap_activity_pockets_;
                pockets.clear();
                pockets.reserve(broadphase_updated_volume_slots_.size()*2);
                // Insertion into the linked index order follows the
                // old endpoint sequence. Later nodes moving left
                // naturally displace earlier nodes moving right.
                bool predecessor_updated=false;
                // Evaluate orthogonal separation first, with the axis
                // order determined once for this whole batch.
                const std::array<unsigned,3> check_axes{
                    (axis+2)%3,(axis+1)%3,axis};
                for(std::size_t i=0;i<n;++i) {
                    const bool updated=sap_endpoint_updated_[i]!=0;
                    if(!updated && !predecessor_updated)continue;
                    // Original batchUpdate follows each changed endpoint
                    // and the unchanged successors it displaces. Once
                    // the chain is ordered again, skip to the next
                    // changed endpoint instead of comparing every key.
                    predecessor_updated=updated;
                    const Endpoint moving=endpoints[i];
                    const Shape& moving_actor=shapes_[moving.id];
                    const bool moving_dynamic=
                        moving_actor.motion==Motion::Dynamic;
                    const auto& moving_bounds=moving_actor.encoded_bounds;
                    std::size_t previous=sap_list_prev_[i];
                    std::size_t start_index=i;
                    while(previous!=sentinel) {
                        ++broadphase_activity_pocket_comparisons_;
                        if(!less(moving,endpoints[previous]))break;
                        // Source SAP batchUpdate pair candidates are
                        // crossings between min/max endpoints, not
                        // movements of same-kind endpoint keys.
                        const Endpoint& passed=endpoints[previous];
                        if(moving.id!=passed.id &&
                           moving.is_max!=passed.is_max &&
                           (moving_dynamic ||
                            shapes_[passed.id].motion==Motion::Dynamic)) {
                            const Shape& b=shapes_[passed.id];
                            // A crossing on this axis changes the
                            // pair state only if its orthogonal extents
                            // also overlap, or it had a published pair.
                            // Test those axes first to avoid work for
                            // proxies separated in another dimension.
                            const auto& aa=moving_bounds;
                            const auto& bb=b.encoded_bounds;
                            bool final_overlap=true;
                            for(unsigned dim:check_axes) {
                                if(aa[dim]>bb[dim+3] ||
                                   bb[dim]>aa[dim+3]) {
                                    final_overlap=false;
                                    break;
                                }
                            }
                            if(final_overlap) {
                                broadphase_crossed_pairs_.push_back(
                                    ordered_pair(moving.id,passed.id));
                                ++broadphase_endpoint_crossing_count_;
                            } else {
                                const Pair pair=ordered_pair(
                                    moving.id,passed.id);
                                if(std::binary_search(previous_pairs_.begin(),
                                                      previous_pairs_.end(),pair)) {
                                    broadphase_crossed_pairs_.push_back(pair);
                                    ++broadphase_endpoint_crossing_count_;
                                }
                            }
                        }
                        start_index=std::min(start_index,previous);
                        previous=sap_list_prev_[previous];
                    }
                    if(previous==sap_list_prev_[i])continue;
                    predecessor_updated=true;
                    const std::size_t old_prev=sap_list_prev_[i];
                    const std::size_t old_next=sap_list_next_[i];
                    sap_list_next_[old_prev]=old_next;
                    sap_list_prev_[old_next]=old_prev;
                    const std::size_t new_next=sap_list_next_[previous];
                    sap_list_next_[previous]=i;
                    sap_list_prev_[i]=previous;
                    sap_list_next_[i]=new_next;
                    sap_list_prev_[new_next]=i;
                    ++broadphase_activity_pocket_relocations_;
                    std::size_t pocket_end=i;
                    while(!pockets.empty() &&
                          start_index<=pockets.back().end+1) {
                        start_index=std::min(
                            start_index,pockets.back().start);
                        pocket_end=std::max(pocket_end,pockets.back().end);
                        pockets.pop_back();
                    }
                    pockets.push_back({start_index,pocket_end});
                }
                // The final linked order can differ from the old array
                // only within the recorded, coalesced activity pockets.
                for(const ActivityPocket pocket:pockets) {
                    const std::size_t length=pocket.end-pocket.start+1;
                    sap_pocket_workspace_.resize(length);
                    std::size_t index=pocket.start==0
                        ? sap_list_next_[sentinel]
                        : sap_list_next_[pocket.start-1];
                    for(std::size_t j=0;j<length;++j) {
                        if(index==sentinel)
                            throw std::logic_error(
                                "SAP activity pocket index chain ended early");
                        sap_pocket_workspace_[j]=endpoints[index];
                        index=sap_list_next_[index];
                    }
                    for(std::size_t j=0;j<length;++j) {
                        const auto index_in_array=pocket.start+j;
                        const Endpoint ep=sap_pocket_workspace_[j];
                        endpoints[index_in_array]=ep;
                        sap_volumes_[ep.volume_slot].
                            endpoint_positions[axis][ep.is_max?1u:0u]=
                            static_cast<std::uint32_t>(index_in_array);
                    }
                    // The source restores the affected linked segment
                    // after remapping its sorted endpoint data. Restore
                    // both neighbors and the sentinel where necessary;
                    // no untouched index needs another write.
                    for(std::size_t i=pocket.start;i<=pocket.end;++i) {
                        sap_list_prev_[i]=i?i-1:sentinel;
                        sap_list_next_[i]=i+1<n?i+1:sentinel;
                    }
                    if(pocket.start)
                        sap_list_next_[pocket.start-1]=pocket.start;
                    else sap_list_next_[sentinel]=0;
                    if(pocket.end+1<n)
                        sap_list_prev_[pocket.end+1]=pocket.end;
                    else sap_list_prev_[sentinel]=n-1;
                    ++broadphase_activity_pocket_count_;
                    broadphase_activity_pocket_endpoint_count_+=length;
                }
            }
            broadphase_batch_crossings_complete_=true;
            broadphase_batch_pair_set_unchanged_=
                broadphase_crossed_pairs_.empty();
            if(all_axes_stable)
                ++broadphase_batch_stable_order_count_;
            return;
        }
        const std::size_t budget=sparse
            ? std::numeric_limits<std::size_t>::max()
            : std::max<std::size_t>(16,sap_volumes_.size()/4);
        // BpBroadPhaseSap emits overlap start/stop candidates only
        // when a minimum crosses another volume's maximum (or the
        // reverse). Do not emit same-kind crossings: they cannot
        // change the 1D overlap state.
        const auto crossing=[&](unsigned axis,
                                 const Endpoint& moving,
                                 const Endpoint& neighbor) {
            if(!sparse || moving.id==neighbor.id ||
               moving.is_max==neighbor.is_max)return;
            const Shape& a=shapes_[moving.id];
            const Shape& other=shapes_[neighbor.id];
            if(a.motion!=Motion::Dynamic &&
               other.motion!=Motion::Dynamic)return;
            // Reject non-overlapping orthogonal extents before
            // spending work on the moving axis or pair lookup.
            const auto& aa=a.encoded_bounds;
            const auto& bb=other.encoded_bounds;
            const std::array<unsigned,3> check_axes{
                (axis+2)%3,(axis+1)%3,axis};
            bool final_overlap=true;
            for(unsigned dim:check_axes) {
                if(aa[dim]>bb[dim+3] || bb[dim]>aa[dim+3]) {
                    final_overlap=false;
                    break;
                }
            }
            const Pair pair=ordered_pair(moving.id,neighbor.id);
            if(!final_overlap &&
               !std::binary_search(previous_pairs_.begin(),
                                   previous_pairs_.end(),pair))
                return;
            broadphase_crossed_pairs_.push_back(pair);
            ++broadphase_endpoint_crossing_count_;
        };
        std::size_t total_shifts=0;
        bool full_sort=false;
        for(unsigned axis=0;axis<3;++axis) {
            auto& endpoints=sap_axis_endpoints_[axis];
            std::size_t shifts=0;
            bool fallback=false;
            for(std::uint32_t slot:broadphase_updated_volume_slots_) {
                for(unsigned side=0;side<2;++side) {
                    auto& pos=sap_volumes_[slot].
                        endpoint_positions[axis][side];
                    std::size_t i=pos;
                    Endpoint moving=endpoints[i];
                    moving.value=shapes_[moving.id].encoded_bounds[
                        axis+(side?3:0)];
                    if(i && less(moving,endpoints[i-1])) {
                        do {
                            crossing(axis,moving,endpoints[i-1]);
                            endpoints[i]=endpoints[i-1];
                            const auto& moved=endpoints[i];
                            sap_volumes_[moved.volume_slot].
                                endpoint_positions[axis][
                                    moved.is_max?1u:0u]=
                                static_cast<std::uint32_t>(i);
                            --i;
                            ++shifts;
                            if(shifts>budget) {
                                fallback=true;
                                break;
                            }
                        } while(i && less(moving,endpoints[i-1]));
                    } else {
                        while(i+1<endpoints.size() &&
                              less(endpoints[i+1],moving)) {
                            crossing(axis,moving,endpoints[i+1]);
                            endpoints[i]=endpoints[i+1];
                            const auto& moved=endpoints[i];
                            sap_volumes_[moved.volume_slot].
                                endpoint_positions[axis][
                                    moved.is_max?1u:0u]=
                                static_cast<std::uint32_t>(i);
                            ++i;
                            ++shifts;
                            if(shifts>budget) {
                                fallback=true;
                                break;
                            }
                        }
                    }
                    endpoints[i]=moving;
                    pos=static_cast<std::uint32_t>(i);
                    if(fallback)break;
                }
                if(fallback)break;
            }
            if(fallback) {
                // Some updated endpoints have not yet been processed;
                // source truth is the current actor AABB for every
                // endpoint in this axis before the complete fallback.
                for(Endpoint& ep:endpoints)
                    ep.value=shapes_[ep.id].encoded_bounds[
                        axis+(ep.is_max?3:0)];
                std::sort(endpoints.begin(),endpoints.end(),less);
                reindex(axis);
                full_sort=true;
            }
            total_shifts+=shifts;
        }
        if(full_sort)++broadphase_endpoint_full_sort_count_;
        broadphase_endpoint_local_shift_count_+=total_shifts;
        if(localized_bulk) {
            ++broadphase_batch_direct_update_count_;
            broadphase_batch_crossings_complete_=true;
            broadphase_batch_pair_set_unchanged_=
                broadphase_crossed_pairs_.empty();
        }
    } catch (...) {
        // Rollback restores authoritative actors and source pair/contact
        // publication; invalidate the derived axis positions so the
        // next dirty publication rebuilds them without a dense snapshot.
        broadphase_endpoint_membership_dirty_=true;
        sap_list_next_.clear();
        sap_list_prev_.clear();
        throw;
    }
}
void Scene::calculate_kinematic_velocities(float inverse_dt) {
    // ScKinematics.cpp::BodySim::calculateKinematicVelocity. The pose
    // stays at its old value until postSolver, but Dy must see the
    // target-derived velocity when solving a moving kinematic contact.
    for(BodyId id:active_kinematic_targets_) {
        Actor& body=actors_[id];
        if(!body.alive || body.motion!=Motion::Kinematic)continue;
        step_checkpoint_.save_motion(*this,id);
        if(!body.kinematic_target_pending) {
            body.velocity={};
            body.angular_velocity={};
            continue;
        }
        const Vec3 offset=rotate(body.kinematic_target_orientation,body.mass_local_pose.position);
        const Vec3 target{body.kinematic_target_position.x+offset.x,
                          body.kinematic_target_position.y+offset.y,
                          body.kinematic_target_position.z+offset.z};
        const Vec3 center=mass_position(body);
        body.velocity={(target.x-center.x)*inverse_dt,
                       (target.y-center.y)*inverse_dt,
                       (target.z-center.z)*inverse_dt};
        const Quat q=body.kinematic_target_orientation;
        const Quat old=body.orientation;
        Quat relative=multiply(q,Quat{-old.x,-old.y,-old.z,old.w});
        if(relative.w<0)
            relative={-relative.x,-relative.y,-relative.z,-relative.w};
        const float length=std::sqrt(relative.x*relative.x+
                                     relative.y*relative.y+
                                     relative.z*relative.z);
        if(length>0) {
            const float angle=2.0f*std::atan2(length,relative.w);
            const float scale=angle*inverse_dt/length;
            body.angular_velocity={relative.x*scale,
                                   relative.y*scale,
                                   relative.z*scale};
        } else body.angular_velocity={};
        if(!finite(body.velocity) || !finite(body.angular_velocity))
            throw std::overflow_error("Source kinematic target velocity overflow");
    }
}
void Scene::publish_kinematic_targets() {
    // ScPipeline::postSolver invokes integrateKinematicPose after Dy
    // solver completion. The new pose enters afterIntegration bounds
    // publication, while this frame's Gu contacts used the old pose.
    for(BodyId id:active_kinematic_targets_) {
        Actor& body=actors_[id];
        if(!body.alive || body.motion!=Motion::Kinematic ||
           !body.kinematic_target_pending)continue;
        step_checkpoint_.save_motion(*this,id);
        body.position=body.kinematic_target_position;
        body.orientation=body.kinematic_target_orientation;
        for(ShapeId shape=body.first_shape;shape!=invalid_shape;shape=shapes_[shape].next)
            shapes_[shape].cached_bounds_current=false;
        dirty_=true;
        body.kinematic_target_pending=false;
        body.kinematic_settling=true;
        // The pose has now reached its target. Its settling update
        // publishes zero velocity next frame; it must not keep a
        // stationary, referenced kinematic awake indefinitely.
        body.ready_for_sleep=true;
    }
}
void Scene::remember_pre_integration_pose(BodyId id) noexcept {
    // DyBodyCoreIntegrator stores mLastCCDTransform before integrateCore
    // updates position/orientation. Use the original actor state, not a
    // copied solver-world body or a dense rollback array.
    Actor& body=actors_[id];
    body.last_ccd_position=body.position;
    body.last_ccd_orientation=body.orientation;
    body.last_ccd_generation=rigid_integration_generation_;
}
void Scene::reconcile_integrated_deactivations() {
    if(!rigid_integration_generation_)return;
    // ScPipeline::afterIntegration repairs rigid bodies that Dy already
    // integrated while the accurate IslandSim independently put their
    // islands to sleep. Serial third-pass-first execution has no such
    // entries: this is proportional to deactivated nodes, never all
    // historical actors, and shares source mLastCCDTransform ownership.
    for(BodyId id:accurate_islands_.nodes_to_deactivate) {
        Actor& body=actors_[id];
        if(body.last_ccd_generation!=rigid_integration_generation_)
            continue;
        // IslandSim is authoritative over Dy's later sleepCheck and
        // final integrated velocities on this same frame.
        body.position=body.last_ccd_position;
        body.orientation=body.last_ccd_orientation;
        body.velocity={};
        body.angular_velocity={};
        body.wake_counter=0.0f;
        body.ready_for_sleep=true;
        body.sleep_linear_acc={};
        body.sleep_angular_acc={};
        for(ShapeId shape=body.first_shape;shape!=invalid_shape;shape=shapes_[shape].next)
            shapes_[shape].cached_bounds_current=false;
        publish_integrated_shape_bounds(id,body);
        dirty_=true;
    }
}
void Scene::after_integration_cached_bounds() {
    FYZZ_PROFILE_SCOPE("sc.cached_bounds");
    // ScPipeline::afterIntegration::updateScBodyAndShapeSim refreshes
    // shape bounds from integrated poses. Consume the SAME task-owned
    // accurate island ranges that Dy integrated, without a scene-wide
    // actor scan, a duplicate AABB pass, or another shape registry.
    // Sleep-deactivated islands never entered those task ranges.
    // Kinematic target poses are published by postSolver, outside Dy
    // island node batches, but their source shape bounds must update.
    for(BodyId id:active_kinematic_targets_) {
        auto& actor=actors_[id];
        if(actor.alive && actor.motion==Motion::Kinematic) {
            publish_integrated_shape_bounds(id,actor);
        }
    }
    for(const auto& task:solver_tasks_)
        for(std::size_t batch=task.first_batch;batch<task.batch_end;++batch) {
            const auto island=active_accurate_island_batches_[batch];
            for(std::size_t index=accurate_island_offsets_[island];
                index<accurate_island_offsets_[island+1];++index) {
                const BodyId id=accurate_island_nodes_[index];
                Actor& body=actors_[id];
                if(!body.sleeping) {
                    publish_integrated_shape_bounds(id,body);
                }
            }
        }
}
// BpBroadPhaseSapAux::SapPairManager. The stable pair keys are BP handles;
// Sc's sorted, generation-safe contact edges remain separately owned.
std::uint32_t Scene::SapPairManager::hash(Pair pair) noexcept {
    // Unsigned arithmetic reproduces Hash32Bits_1 without signed-shift UB.
    // Keep the original packed-key hash for low 16-bit handles.
    // Native BodyId spans 32 bits, so fold both high halves before
    // mixing to avoid systematic bucket aliases in long-lived scenes.
    std::uint32_t key=pair.a | (pair.b<<16);
    key^=(pair.a>>16)*0x9e3779b9u;
    key^=(pair.b>>16)*0x85ebca6bu;
    key+=~(key<<15);
    key^=key>>10;
    key+=key<<3;
    key^=key>>6;
    key+=~(key<<11);
    key^=key>>16;
    return key;
}
std::uint32_t Scene::SapPairManager::find(Pair pair) const noexcept {
    if(hash_table.empty())return invalid;
    std::uint32_t slot=hash_table[hash(pair)&(hash_table.size()-1)];
    while(slot!=invalid && !(active_pairs[slot]==pair))
        slot=next[slot];
    return slot;
}
void Scene::SapPairManager::rehash(std::size_t buckets) {
    std::vector<std::uint32_t> heads(buckets,invalid);
    for(std::size_t i=0;i<active_pairs.size();++i) {
        const auto h=hash(active_pairs[i])&(buckets-1);
        next[i]=heads[h];
        heads[h]=static_cast<std::uint32_t>(i);
    }
    hash_table.swap(heads);
}
void Scene::SapPairManager::add(Pair pair) {
    const auto existing=find(pair);
    if(existing!=invalid) {
        states[existing]&=static_cast<std::uint8_t>(~removed);
        return;
    }
    if(active_pairs.size()>=invalid)
        throw std::overflow_error("Native SAP pair index exhausted");
    if(active_pairs.size()>=hash_table.size())
        rehash(hash_table.empty()?64:hash_table.size()*2);
    const auto index=static_cast<std::uint32_t>(active_pairs.size());
    // Reserve every parallel array before publishing a hash bucket.
    // Geometric growth retains amortized O(1) pair insertion.
    // A failed allocation invalidates the derived registry at rollback.
    if(active_pairs.capacity()==active_pairs.size())
        active_pairs.reserve(std::max<std::size_t>(
            8,active_pairs.capacity()*2));
    if(states.capacity()==states.size())
        states.reserve(std::max<std::size_t>(8,states.capacity()*2));
    if(next.capacity()==next.size())
        next.reserve(std::max<std::size_t>(8,next.capacity()*2));
    if(touched.capacity()==touched.size())
        touched.reserve(std::max<std::size_t>(
            8,touched.capacity()*2));
    const auto bucket=hash(pair)&(hash_table.size()-1);
    active_pairs.push_back(pair);
    states.push_back(in_array|newly_created);
    next.push_back(hash_table[bucket]);
    hash_table[bucket]=index;
    touched.push_back(index);
}
void Scene::SapPairManager::remove(Pair pair) {
    const auto index=find(pair);
    if(index==invalid)return;
    auto& state=states[index];
    if(!(state&in_array)) {
        touched.push_back(index);
        state|=in_array;
    }
    state|=removed;
}
void Scene::SapPairManager::begin_full() {
    touched.clear();
    touched.reserve(active_pairs.size());
    // Full membership changes use the same source pair-state machine as
    // endpoint events: survivors clear REMOVED when encountered again.
    for(std::size_t i=0;i<active_pairs.size();++i) {
        states[i]=in_array|removed;
        touched.push_back(static_cast<std::uint32_t>(i));
    }
}
void Scene::SapPairManager::remove_index(std::uint32_t index) noexcept {
    const auto bucket=hash(active_pairs[index])&(hash_table.size()-1);
    std::uint32_t* link=&hash_table[bucket];
    while(*link!=index)link=&next[*link];
    *link=next[index];
    const auto last=static_cast<std::uint32_t>(active_pairs.size()-1);
    if(index!=last) {
        const auto last_bucket=hash(active_pairs[last])&
            (hash_table.size()-1);
        std::uint32_t* last_link=&hash_table[last_bucket];
        while(*last_link!=last)last_link=&next[*last_link];
        *last_link=index;
        active_pairs[index]=active_pairs[last];
        states[index]=states[last];
        next[index]=next[last];
    }
    active_pairs.pop_back();
    states.pop_back();
    next.pop_back();
}
void Scene::SapPairManager::finish(
    std::vector<Pair>& created,std::vector<Pair>& deleted) {
    created.clear();
    deleted.clear();
    actual_deleted.clear();
    for(const auto index:touched) {
        const auto state=states[index];
        const Pair pair=active_pairs[index];
        if(state&removed) {
            // A pair born and retired in one BP update has no Sc Lost.
            if(!(state&newly_created))deleted.push_back(pair);
            actual_deleted.push_back(pair);
        } else if(state&newly_created)
            created.push_back(pair);
        states[index]=0;
    }
    std::sort(created.begin(),created.end());
    std::sort(deleted.begin(),deleted.end());
    // Collect keys before swap-last compaction: indices change on removal.
    for(const Pair pair:actual_deleted) {
        const auto index=find(pair);
        if(index!=invalid)remove_index(index);
    }
    touched.clear();
    // Keep the hash load factor bounded without an allocation per pair.
    if(hash_table.size()>64 &&
       active_pairs.size()*4<hash_table.size())
        rehash(hash_table.size()/2);
}
void Scene::SapPairManager::reset() noexcept {
    active_pairs.clear();
    states.clear();
    next.clear();
    touched.clear();
    actual_deleted.clear();
    std::fill(hash_table.begin(),hash_table.end(),invalid);
}
void Scene::SapPairManager::seed(std::span<const Pair> pairs) {
    reset();
    for(const Pair pair:pairs)add(pair);
    for(auto& state:states)state=0;
    touched.clear();
}

void Scene::set_gpu_broadphase(
    std::unique_ptr<GpuBroadphaseProducer> producer) {
    if(!producer)
        throw std::invalid_argument("Native GPU broadphase owner is null");
    if(!actors_.empty() || gpu_broadphase_)
        throw std::logic_error(
            "Native GPU SAP owner must be installed before actor admission");
    gpu_broadphase_=std::move(producer);
}
void Scene::set_gpu_dy_device_admission_enabled(bool enabled) {
    if(enabled && (!gpu_broadphase_ || !gpu_broadphase_->has_gpu_pcm_box() ||
                   !gpu_broadphase_->has_gpu_dy_distance()))
        throw std::logic_error(
            "Resident Sc island admission requires the GPU PCM and Dy producer");
    gpu_dy_device_admission_enabled_=enabled;
}

void Scene::set_gpu_dy_enabled(bool enabled) {
    if(!gpu_broadphase_ || !gpu_broadphase_->has_gpu_dy_distance())
        throw std::logic_error(
            "GPU Dy execution selection requires an installed numerical solver");
    if(gpu_dy_enabled_==enabled)return;
    // Only GPU-integrated dynamic ElementSim poses need revocation.
    // Their Sc input records already contain the last published poses.
    dirty_=true;
    for(const ShapeId id:live_shape_ids_) {
        const auto& shape=shapes_[id];
        if(shape.simulation_shape_enabled &&
           shape.motion==Motion::Dynamic &&
           shape.geometry!=Geometry::Plane)
            gpu_broadphase_->mark_shape_bounds_changed(id,shape.contact_offset);
    }
    gpu_dy_enabled_=enabled;
}

// Source GPU fetch -> Sc pair stream. This is the same pending pair and
// contact-edge publication used by the native CPU branch, but it never
// calls BpBroadPhaseSap or reads that implementation's endpoint registry.
void Scene::update_gpu_pairs(
    bool force_update,const std::function<void()>& independent_sc_work) {
    FYZZ_PROFILE_SCOPE("sap.update_fetch");
    // Full input rebuilds are required for Sc admission/shape edits.
    // afterIntegration patches the changed source body entries directly
    // using its existing task-owned island ranges. Constraint/Gu dirty
    // publications do not imply any change to this finite-box view.
    // The source AABB validates admitted SAP frames; converged quiet
    // frames do not need another source GPU submission or fetch.
    if(!gpu_broadphase_snapshot_current_) {
        gpu_broadphase_snapshot_current_=false;
        gpu_broadphase_boxes_.clear();
        gpu_broadphase_boxes_.reserve(live_shape_ids_.size());
        for(BodyId id:live_shape_ids_) {
            const Shape& actor=shapes_[id];
            if(!actor.simulation_shape_enabled)continue;
            // Infinite planes are Sc contact managers, not finite SAP.
            if(actor.geometry==Geometry::Plane)continue;
            const auto motion=actor.motion==Motion::Dynamic
                ? GpuBroadphaseBox::Motion::Dynamic
                : actor.motion==Motion::Kinematic
                    ? GpuBroadphaseBox::Motion::Kinematic
                    : GpuBroadphaseBox::Motion::Static;
            const Vec3 extents=actor.geometry==Geometry::Sphere
                ? Vec3{actor.radius,actor.radius,actor.radius}
                : actor.half_extents;
            const Quat basis=actor.orientation;
            float convex_margin_extent=0.f;
            if(actor.geometry==Geometry::Convex) {
                const Vec3 lower=actor.convex->bounds_min();
                const Vec3 upper=actor.convex->bounds_max();
                convex_margin_extent=0.5f*std::min({
                    upper.x-lower.x,upper.y-lower.y,upper.z-lower.z});
            }
            gpu_broadphase_boxes_.push_back({
                id,actor.position,extents,basis,
                actor.contact_offset,motion,actor.local_pose,actor.next_simulation_shape,
                actor.geometry==Geometry::Sphere?1u:
                (actor.geometry==Geometry::Convex?2u:0u),
                actor.convex_vertex_offset,actor.convex_vertex_count,
                convex_margin_extent});
        }
        gpu_broadphase_snapshot_current_=true;
        ++gpu_broadphase_snapshot_rebuilds_;
    }
    if(uploaded_convex_vertex_count_!=convex_vertices_.size()) {
        gpu_broadphase_->upload_convex_geometry(
            convex_vertices_,convex_face_spans_,convex_faces_,
            convex_face_indices_);
        uploaded_convex_vertex_count_=convex_vertices_.size();
    }
    bool plane_candidates_ready=false;
    const auto prepare_plane_candidates=[&] {
        if(plane_candidates_ready)return;
        FYZZ_PROFILE_SCOPE("sc.plane_pair_prepare");
        gpu_plane_pair_candidates_.clear();
        if(active_plane_count_) {
            for(const ShapeId plane:live_shape_ids_) {
                const auto& a=shapes_[plane];
                if(!a.alive || !a.simulation_shape_enabled ||
                   a.geometry!=Geometry::Plane)continue;
                for(const ShapeId shape:live_shape_ids_) {
                    const auto& b=shapes_[shape];
                    if(b.alive && b.simulation_shape_enabled &&
                       b.motion==Motion::Dynamic)
                        gpu_plane_pair_candidates_.push_back(
                            ordered_pair(plane,shape));
                }
            }
            std::sort(gpu_plane_pair_candidates_.begin(),
                      gpu_plane_pair_candidates_.end());
            gpu_plane_pair_candidates_.erase(
                std::unique(gpu_plane_pair_candidates_.begin(),
                            gpu_plane_pair_candidates_.end()),
                gpu_plane_pair_candidates_.end());
        }
        plane_candidates_ready=true;
    };
    const bool preflight_planes=active_plane_count_ &&
        (plane_pair_membership_dirty_ || gpu_pair_refilter_dirty_ ||
         sap_pair_registry_dirty_);
    const auto reports=[&] {
        FYZZ_PROFILE_SCOPE("sap.source_advance");
        return gpu_broadphase_->advance_overlapping(
            gpu_broadphase_boxes_,force_update,dirty_,[&] {
                independent_sc_work();
                if(preflight_planes)prepare_plane_candidates();
            });
    }();
    if(reports.overflow)
        throw std::overflow_error(
            "Source GPU SAP found/lost report capacity overflow");

    // GPU reports describe original Bp membership, which includes
    // overlaps that are currently ineligible for Sc contact managers.
    // The existing source BP pair hash retains those memberships for
    // ScShapeSim/ScBodySim refiltering. This is NOT another CPU SAP:
    // only GPU found/lost reports can insert an overlap.
    if(sap_pair_registry_dirty_ && !reports.full_rebuild) {
        // Scripted GPU fixtures may resume after a rejected Sc frame.
        // Their delta-only report contract retains Sc's last published
        // contacts. The real source owner returns a full GPU rebuild.
        broadphase_candidate_deltas_.clear();
        for(const Pair pair:previous_pairs_)
            if(shapes_[pair.a].geometry!=Geometry::Plane &&
               shapes_[pair.b].geometry!=Geometry::Plane)
                broadphase_candidate_deltas_.push_back(pair);
        sap_pairs_.seed(broadphase_candidate_deltas_);
        sap_pair_registry_dirty_=false;
    }
    const auto validate=[&](std::span<const Pair> raw,bool found) {
        Pair last{};
        bool have_last=false;
        for(const Pair pair:raw) {
            if(pair.a>=pair.b || pair.b>=shapes_.size() ||
               (have_last && !(last<pair)))
                throw std::invalid_argument(
                    "Noncanonical or unsorted GPU SAP pair reports");
            last=pair;
            have_last=true;
            const auto& a=shapes_[pair.a];
            const auto& b=shapes_[pair.b];
            if(a.geometry==Geometry::Plane || b.geometry==Geometry::Plane)
                throw std::invalid_argument(
                    "GPU SAP pair references a non-finite Sc plane");
            if(found && (!a.alive || !b.alive ||
                         !a.simulation_shape_enabled ||
                         !b.simulation_shape_enabled))
                throw std::invalid_argument(
                    "GPU SAP found pair references inactive Sc shapes");
        }
    };
    validate(reports.found,true);
    validate(reports.lost,false);

    // Sc's original infinite-plane interaction path has no SAP proxy.
    // Candidate membership is the Cartesian product of admitted planes
    // and dynamic shapes, just as in the native source Bp branch. Gu
    // decides accurate touch; GPU SAP alone owns finite BP membership.
    // Only Sc admission/refiltering changes infinite-plane candidate
    // membership. Ordinary Dy motion and finite GPU crossings retain
    // the previously published plane manager identities.
    const auto append_plane_managers=[&](
        bool finite_delta,bool rebuild_planes) {
        if(active_plane_count_ && rebuild_planes) {
            prepare_plane_candidates();
            // The GPU plane candidates were sorted during SAP execution.
            // Merge them into the Sc finite-pair stream in linear time.
            if(gpu_pair_refilter_dirty_)
                std::sort(pending_pairs_.begin(),pending_pairs_.end());
            const auto finite_count=pending_pairs_.size();
            const auto plane_count=gpu_plane_pair_candidates_.size();
            pending_pairs_.resize(finite_count+plane_count);
            std::size_t finite=finite_count,plane=plane_count;
            std::size_t output=pending_pairs_.size();
            while(finite && plane) {
                if(gpu_plane_pair_candidates_[plane-1u]<
                   pending_pairs_[finite-1u])
                    pending_pairs_[--output]=pending_pairs_[--finite];
                else
                    pending_pairs_[--output]=gpu_plane_pair_candidates_[--plane];
            }
            while(plane)
                pending_pairs_[--output]=gpu_plane_pair_candidates_[--plane];
            pending_pairs_.erase(
                std::unique(pending_pairs_.begin(),pending_pairs_.end()),
                pending_pairs_.end());
        }
        broadphase_pair_membership_changed_=
            finite_delta || pending_pairs_!=previous_pairs_;
        plane_pair_membership_dirty_=false;
    };

    if(reports.full_rebuild) {
        if(!reports.lost.empty())
            throw std::invalid_argument(
                "Reconstructed source GPU SAP returned prior-frame lost pairs");
        // This is the COMPLETE initial source GPU BP membership, including
        // overlaps previously ineligible for Sc contacts. Re-seeding from
        // previous_pairs_ would permanently lose those memberships.
        sap_pairs_.seed(reports.found);
        sap_pair_registry_dirty_=false;
        gpu_pair_refilter_dirty_=false;
        pending_pairs_.clear();
        pending_pairs_.reserve(reports.found.size());
        for(const Pair pair:reports.found)
            if(shapes_[pair.a].motion==Motion::Dynamic ||
               shapes_[pair.b].motion==Motion::Dynamic)
                pending_pairs_.push_back(pair);
        append_plane_managers(false,true);
        return;
    }

    // Keep the original BP overlap hash in sync with GPU membership,
    // including static/kinematic pairs ignored by Sc's contact filter.
    // The original SapPairManager coalesces created/removed handles and
    // compacts them only after the GPU report streams are validated.
    // An Sc actor reinserted from an aggregate has a new ElementSim
    // handle. Its old aggregate reports may retire the same actor pair
    // that its new standalone SAP handle reports as found. When source
    // authoring requests refiltering, consume retirements before found
    // admissions so the final Sc overlap retains its canonical identity.
    if(gpu_pair_refilter_dirty_) {
        for(const Pair pair:reports.lost)sap_pairs_.remove(pair);
        for(const Pair pair:reports.found)sap_pairs_.add(pair);
    } else {
        for(const Pair pair:reports.found)sap_pairs_.add(pair);
        for(const Pair pair:reports.lost)sap_pairs_.remove(pair);
    }

    // GPU volume removal emits no geometric crossing/lost pair. Retire
    // the BP membership here after an explicit Sc actor/shape edit,
    // before it can be carried forward as an active contact manager.
    if(gpu_pair_refilter_dirty_)
        for(const Pair pair:sap_pairs_.active_pairs) {
            const Shape& a=shapes_[pair.a];
            const Shape& b=shapes_[pair.b];
            if(!a.alive || !b.alive ||
               !a.simulation_shape_enabled || !b.simulation_shape_enabled)
                sap_pairs_.remove(pair);
        }
    // These are already-owned source BP scratch vectors. Sc's sorted
    // delta vectors remain available for normal sparse publication.
    sap_pairs_.finish(broadphase_candidate_deltas_,
                      broadphase_crossed_pairs_);

    // PhysX ScShapeSim::updateBPGroup refilters existing overlap
    // interactions without requiring another endpoint crossing.
    // This is the only path that walks all GPU-reported BP memberships;
    // quiet frames and ordinary found/lost reports remain sparse.
    if(gpu_pair_refilter_dirty_) {
        pending_pairs_.clear();
        pending_pairs_.reserve(sap_pairs_.active_pairs.size());
        for(const Pair pair:sap_pairs_.active_pairs)
            if(shapes_[pair.a].motion==Motion::Dynamic ||
               shapes_[pair.b].motion==Motion::Dynamic)
                pending_pairs_.push_back(pair);
        // The source BP hash has unordered storage; finite-only Sc
        // refiltering still needs sorted contact-manager identities.
        if(!active_plane_count_)
            std::sort(pending_pairs_.begin(),pending_pairs_.end());
        append_plane_managers(false,true);
        gpu_pair_refilter_dirty_=false;
        return;
    }

    // On ordinary frames Sc consumes only source found/lost deltas.
    // Lost reports retire previously admitted contacts; reports for
    // non-dynamic BP overlaps have no contact managers to retire.
    const auto admit=[&](std::span<const Pair> raw,
                         std::vector<Pair>& kept,bool new_pair) {
        kept.clear();
        std::size_t previous_index=0;
        const bool bulk=raw.size()>previous_pairs_.size()/16u;
        for(const Pair pair:raw) {
            if(new_pair &&
               shapes_[pair.a].motion!=Motion::Dynamic &&
               shapes_[pair.b].motion!=Motion::Dynamic)
                continue;
            bool existing;
            if(bulk) {
                while(previous_index<previous_pairs_.size() &&
                      previous_pairs_[previous_index]<pair)
                    ++previous_index;
                existing=previous_index<previous_pairs_.size() &&
                    previous_pairs_[previous_index]==pair;
            }else {
                existing=std::binary_search(
                    previous_pairs_.begin(),previous_pairs_.end(),pair);
            }
            if(!new_pair && !existing)continue;
            if(existing==new_pair)
                throw std::invalid_argument(
                    "GPU SAP pair report contradicts Sc overlap history");
            kept.push_back(pair);
        }
    };
    admit(reports.found,sap_created_pairs_,true);
    admit(reports.lost,sap_deleted_pairs_,false);
    const auto& found=sap_created_pairs_;
    const auto& lost=sap_deleted_pairs_;

    // A quiet Sc pair stream needs no merge even when the source GPU
    // reports non-dynamic overlaps or updates existing PCM contacts.
    if(found.empty() && lost.empty()) {
        // Motion dirties Gu/PCM, not Sc plane candidate membership.
        if(!plane_pair_membership_dirty_) {
            broadphase_pair_membership_changed_=false;
            return;
        }
        pending_pairs_.clear();
        for(const Pair pair:previous_pairs_)
            if(shapes_[pair.a].geometry!=Geometry::Plane &&
               shapes_[pair.b].geometry!=Geometry::Plane)
                pending_pairs_.push_back(pair);
        append_plane_managers(false,true);
        return;
    }

    pending_pairs_.clear();
    pending_pairs_.reserve(previous_pairs_.size()+found.size());
    std::size_t new_index=0,lost_index=0;
    for(const Pair prior:previous_pairs_) {
        if(plane_pair_membership_dirty_ &&
           (shapes_[prior.a].geometry==Geometry::Plane ||
            shapes_[prior.b].geometry==Geometry::Plane))continue;
        while(new_index<found.size() && found[new_index]<prior)
            pending_pairs_.push_back(found[new_index++]);
        if(lost_index<lost.size() && lost[lost_index]==prior)
            ++lost_index;
        else pending_pairs_.push_back(prior);
    }
    while(new_index<found.size())
        pending_pairs_.push_back(found[new_index++]);
    append_plane_managers(true,plane_pair_membership_dirty_);
}

void Scene::update_pairs() {
    FYZZ_PROFILE_SCOPE("sap.cpu_pairs");
    // The BP hashed manager is the overlap authority; the sorted Sc
    // manager/edge stream is a separate consumer. Rejected transactions
    // reconstruct the BP lookup from the previously published Sc stream.
    if(sap_pair_registry_dirty_) {
        broadphase_candidate_deltas_.clear();
        for(Pair pair:previous_pairs_)
            if(shapes_[pair.a].geometry!=Geometry::Plane &&
               shapes_[pair.b].geometry!=Geometry::Plane)
                broadphase_candidate_deltas_.push_back(pair);
        sap_pairs_.seed(broadphase_candidate_deltas_);
        sap_pair_registry_dirty_=false;
    }
    pending_pairs_.clear();
    sap_created_pairs_.clear();
    sap_deleted_pairs_.clear();
    broadphase_pair_membership_changed_=false;
    // No encoded endpoint changed order in any SAP axis; therefore
    // no finite overlap could transition. Planes retain their existing
    // candidate membership. Do not revisit all active BP pairs.
    if(broadphase_batch_pair_set_unchanged_ && !broadphase_full_pair_update_)
        return;
    const bool few_updates=!broadphase_full_pair_update_ &&
        (broadphase_updated_ids_.size()*5<sap_volumes_.size() ||
         broadphase_batch_crossings_complete_);
    if(few_updates) {
        ++broadphase_incremental_pair_update_count_;
        const auto& crossed=broadphase_crossed_pairs_;
        if(crossed.empty())return;
        // Only candidate crossings enter the hashed state machine.
        // Deduplication coalesces multi-axis start/stop transitions into
        // final three-axis pair membership before Sc publication.
        broadphase_candidate_deltas_=crossed;
        std::sort(broadphase_candidate_deltas_.begin(),
                  broadphase_candidate_deltas_.end());
        auto& candidates=broadphase_candidate_deltas_;
        candidates.erase(std::unique(candidates.begin(),candidates.end()),
                         candidates.end());
        for(Pair pair:candidates) {
            const Shape& a=shapes_[pair.a],&b=shapes_[pair.b];
            const auto& aa=a.encoded_bounds;
            const auto& bb=b.encoded_bounds;
            const bool overlap=a.body!=b.body && (a.motion==Motion::Dynamic ||
                                b.motion==Motion::Dynamic) &&
                !(aa[0]>bb[3] || bb[0]>aa[3] ||
                  aa[1]>bb[4] || bb[1]>aa[4] ||
                  aa[2]>bb[5] || bb[2]>aa[5]);
            if(overlap)sap_pairs_.add(pair);
            else sap_pairs_.remove(pair);
        }
        sap_pairs_.finish(sap_created_pairs_,sap_deleted_pairs_);
        if(sap_created_pairs_.empty() && sap_deleted_pairs_.empty())
            return;
        broadphase_pair_membership_changed_=true;
        pending_pairs_.reserve(previous_pairs_.size()+
                               sap_created_pairs_.size());
        // Sc maintains canonical pair/EdgeIndex order. Translate only
        // BP Found/Lost deltas into its next list; inactive managers keep
        // their identities and are never recreated by this path.
        std::size_t add=0,del=0;
        for(Pair prior:previous_pairs_) {
            while(add<sap_created_pairs_.size() &&
                  sap_created_pairs_[add]<prior)
                pending_pairs_.push_back(sap_created_pairs_[add++]);
            while(del<sap_deleted_pairs_.size() &&
                  sap_deleted_pairs_[del]<prior)++del;
            if(del<sap_deleted_pairs_.size() &&
               sap_deleted_pairs_[del]==prior)++del;
            else pending_pairs_.push_back(prior);
        }
        while(add<sap_created_pairs_.size())
            pending_pairs_.push_back(sap_created_pairs_[add++]);
        return;
    }

    // On a membership/full-update pass, every BP pair starts REMOVED,
    // and the x-axis sweep reaffirms surviving overlaps. A single
    // finish pass determines source Created/Deleted status.
    sap_pairs_.begin_full();
    sap_active_dynamic_slots_.clear();
    sap_active_static_slots_.clear();
    sap_active_positions_.resize(sap_volumes_.size());
    // BpBroadPhaseSap x-axis sweep: only a pair with at least one
    // dynamic participant passes source group filtering. Keep active
    // static and dynamic intervals separate so dense static-only
    // scenes never enumerate static/static candidates. Each endpoint
    // already owns a compact slot for constant-time active removal.
    for(const Endpoint ep:sap_axis_endpoints_[0]) {
        const bool dynamic=shapes_[ep.id].motion==Motion::Dynamic;
        auto& own=dynamic?sap_active_dynamic_slots_:
                          sap_active_static_slots_;
        if(ep.is_max) {
            const std::uint32_t slot=ep.volume_slot;
            const std::uint32_t index=sap_active_positions_[slot];
            const std::uint32_t tail=own.back();
            own[index]=tail;
            sap_active_positions_[tail]=index;
            own.pop_back();
            continue;
        }
        const auto& a=shapes_[ep.id].encoded_bounds;
        const auto compare=[&](const std::vector<std::uint32_t>& slots) {
            for(const std::uint32_t slot:slots) {
                const BodyId other_id=sap_volumes_[slot].id;
                const auto& b=shapes_[other_id].encoded_bounds;
                if(shapes_[ep.id].body==shapes_[other_id].body)continue;
                if(a[1]>b[4] || b[1]>a[4] ||
                   a[2]>b[5] || b[2]>a[5])continue;
                pending_pairs_.push_back(ordered_pair(ep.id,other_id));
            }
        };
        compare(sap_active_dynamic_slots_);
        if(dynamic)compare(sap_active_static_slots_);
        sap_active_positions_[ep.volume_slot]=
            static_cast<std::uint32_t>(own.size());
        own.push_back(ep.volume_slot);
    }
    // Infinite source planes have no finite endpoint and therefore
    // remain separate candidate broadphase shapes, as before.
    for(BodyId plane:live_shape_ids_) {
        const Shape& actor=shapes_[plane];
        if(!actor.simulation_shape_enabled ||
           actor.geometry!=Geometry::Plane)continue;
        for(const SapVolume& volume:sap_volumes_)
            if(shapes_[volume.id].motion==Motion::Dynamic)
                pending_pairs_.push_back(ordered_pair(plane,volume.id));
    }
    std::sort(pending_pairs_.begin(),pending_pairs_.end());
    pending_pairs_.erase(
        std::unique(pending_pairs_.begin(),pending_pairs_.end()),
        pending_pairs_.end());
    for(Pair pair:pending_pairs_)
        if(shapes_[pair.a].geometry!=Geometry::Plane &&
           shapes_[pair.b].geometry!=Geometry::Plane)
            sap_pairs_.add(pair);
    sap_pairs_.finish(sap_created_pairs_,sap_deleted_pairs_);
    broadphase_pair_membership_changed_=
        !std::equal(pending_pairs_.begin(),pending_pairs_.end(),
                    previous_pairs_.begin(),previous_pairs_.end());

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
    return handle!=invalid_edge && island_edges_[handle].connected;
}
bool Scene::speculative_contact_edge_active(Pair pair) const noexcept {
    const auto id=contact_edge_index(pair);
    return id!=invalid_edge && (island_edges_[id].speculative_state&1u);
}
bool Scene::accurate_contact_edge_active(Pair pair) const noexcept {
    const auto id=contact_edge_index(pair);
    return id!=invalid_edge && (island_edges_[id].accurate_state&1u);
}
bool Scene::contact_interaction_active(Pair pair) const noexcept {
    const auto id=contact_edge_index(pair);
    return id!=invalid_edge && island_edges_[id].interaction_active;
}
void Scene::change_counted_interactions(Pair pair, bool attach) {
    // ShapeInteraction construction and constraint attachment update
    // PxsBodyCore::numCountedInteractions independently of Dy activity.
    // Validate both endpoints before any change; each body is visited
    // once, with rollback sharing the graph's sparse endpoint journal.
    for(BodyId id:{pair.a,pair.b}) {
        const Actor& actor=actors_[id];
        if(!rigid_node(actor.motion))continue;
        const auto count=actor.num_counted_interactions;
        if(attach && count==std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Source body interaction count overflow");
        if(!attach && count==0)
            throw std::logic_error("Source body interaction count underflow");
    }
    for(BodyId id:{pair.a,pair.b}) {
        Actor& actor=actors_[id];
        if(!rigid_node(actor.motion))continue;
        step_checkpoint_.save_reference(*this,id);
        if(attach)++actor.num_counted_interactions;
        else --actor.num_counted_interactions;
    }
}
std::uint32_t Scene::allocate_contact_edge(Pair pair,Pair shapes) {
    // Overflow must reject before allocating a shared EdgeIndex.
    for(BodyId id:{pair.a,pair.b})
        if(rigid_node(actors_[id].motion) &&
           actors_[id].num_counted_interactions==
               std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Source body interaction count overflow");
    // SimpleIslandManager::addContactManager -> addEdge obtains the
    // edge handle once; accurate touch may later toggle independently.
    EdgeIndex id;
    if(!free_edge_handles_.empty()) {
        id=free_edge_handles_.back();
        step_checkpoint_.save_edge(*this,id);
        free_edge_handles_.pop_back();
        island_edges_[id]={IslandEdgeType::ContactManager,pair,true,false,false};
    } else {
        if(island_edges_.size()>=invalid_edge)
            throw std::overflow_error("Native island edge handles exhausted");
        id=static_cast<EdgeIndex>(island_edges_.size());
        // deactivateNodeInternal may publish a contact edge after the
        // geometry transaction commits. Reserve its bounded source
        // output before enlarging the authoritative handle arena.
        // As with the source's reusable event arrays, grow capacity
        // geometrically. Reserving the exact new watermark for every
        // edge would reallocate on every manager admission.
        const std::size_t needed=island_edges_.size()+1;
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
        const auto reserve_constraint_events=[&](IslandGraph& graph) {
            auto& events=graph.deactivating_constraint_edges;
            if(events.capacity()<needed)
                events.reserve(std::max(needed,events.capacity()+
                    std::max<std::size_t>(events.capacity(),8)));
        };
        reserve_constraint_events(speculative_islands_);
        reserve_constraint_events(accurate_islands_);
        const auto reserve_activation=[&](IslandGraph& graph) {
            auto& events=graph.activated_edges;
            if(events.capacity()>=needed)return;
            const std::size_t capacity=events.capacity();
            events.reserve(std::max(needed,capacity+
                std::max<std::size_t>(capacity,8)));
        };
        reserve_activation(speculative_islands_);
        reserve_activation(accurate_islands_);
        const auto reserve_constraint_activation=[&](IslandGraph& graph) {
            auto& events=graph.activated_constraint_edges;
            if(events.capacity()<needed)
                events.reserve(std::max(needed,events.capacity()+
                    std::max<std::size_t>(events.capacity(),8)));
        };
        reserve_constraint_activation(speculative_islands_);
        reserve_constraint_activation(accurate_islands_);
        step_checkpoint_.activity_undo.reserve(needed);
        activity_undo_generations_.resize(needed,0);
        island_edges_.push_back({IslandEdgeType::ContactManager,pair,true,false,false});
    }
    island_edges_[id].shapes=shapes;
    change_counted_interactions(pair,true);
    return id;
}
// PxsIslandSim.cpp::IslandSim::addConnectionToGraph and
// removeConnectionInternal: two intrusive endpoint instances per
// EdgeIndex. The speculative and accurate IslandSim graphs retain
// separate lists, but share the authoritative contact-manager handle.
void Scene::link_contact_edge(EdgeIndex id, bool accurate) {
    auto& edge=island_edges_[id];
    auto& heads=accurate?accurate_node_first_:speculative_node_first_;
    auto instance_at=[&](std::size_t index)->EdgeInstance& {
        auto& owner=island_edges_[index/2];
        return (accurate?owner.accurate:owner.speculative)[index&1];
    };
    for(std::size_t side=0;side<2;++side) {
        const BodyId node=side==0?edge.pair.a:edge.pair.b;
        // The PhysX static world endpoint has no dynamic island node.
        if(!rigid_node(actors_[node].motion))continue;
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
    const bool constraint=
        island_edges_[id].type==IslandEdgeType::Constraint;
    auto& first=constraint?island.first_constraint:island.first_contact;
    auto& last=constraint?island.last_constraint:island.last_contact;
    auto& count=constraint?island.constraint_count:island.contact_count;
    if(!island.live || !island_edges_[id].connected ||
       island_edges_[id].indexed_accurate || first==id ||
       island_edges_[id].island_prev!=invalid_edge ||
       island_edges_[id].island_next!=invalid_edge)
        throw std::logic_error("Invalid PhysX accurate island edge insertion");
    step_checkpoint_.save_graph_ownership(*this);
    step_checkpoint_.save_edge(*this,id);
    if(last!=invalid_edge) {
        step_checkpoint_.save_edge(*this,last);
        island_edges_[last].island_next=id;
    } else first=id;
    island_edges_[id].island_prev=last;
    island_edges_[id].island_next=invalid_edge;
    last=id;
    island_edges_[id].indexed_accurate=true;
    ++count;
}
// PxsIslandSim.cpp::removeEdgeFromIsland with
// IG_STORE_ISLAND_EDGES_IN_ARRAYS=1 replaces the removed element with the
// final array element. Rewire the final EdgeIndex into that slot in O(1).
// Retaining stable order here would be *different* from PhysX.
void Scene::remove_accurate_island_edge(EdgeIndex id) {
    // setEdgeConnected does not imply addEdgeToIsland has executed.
    // A constraint authored and retired before the next island pass
    // has EdgeInstances but no island-list slot to remove.
    if(!island_edges_[id].indexed_accurate)return;
    const Pair pair=island_edges_[id].pair;
    const BodyId node=actors_[pair.a].motion==Motion::Dynamic
        ? pair.a:pair.b;
    const auto owner=accurate_islands_.node_island[node];
    if(owner==IslandGraph::invalid) return;
    auto& island=accurate_islands_.islands[owner];
    const bool constraint=
        island_edges_[id].type==IslandEdgeType::Constraint;
    auto& first=constraint?island.first_constraint:island.first_contact;
    auto& last_ref=constraint?island.last_constraint:island.last_contact;
    auto& count=constraint?island.constraint_count:island.contact_count;
    if(first==invalid_edge || !count)
        throw std::logic_error("PhysX accurate island edge list underflow");
    step_checkpoint_.save_graph_ownership(*this);
    const EdgeIndex last=last_ref;
    const EdgeIndex before=island_edges_[id].island_prev;
    const EdgeIndex after=island_edges_[id].island_next;
    step_checkpoint_.save_edge(*this,id);
    if(last!=id) {
        const EdgeIndex last_before=island_edges_[last].island_prev;
        step_checkpoint_.save_edge(*this,last);
        if(last_before!=invalid_edge) {
            step_checkpoint_.save_edge(*this,last_before);
            island_edges_[last_before].island_next=invalid_edge;
        }
        last_ref=last_before;
        const EdgeIndex successor=after==last?invalid_edge:after;
        if(before!=invalid_edge) {
            step_checkpoint_.save_edge(*this,before);
            island_edges_[before].island_next=last;
        } else first=last;
        if(successor!=invalid_edge) {
            step_checkpoint_.save_edge(*this,successor);
            island_edges_[successor].island_prev=last;
        } else last_ref=last;
        island_edges_[last].island_prev=before;
        island_edges_[last].island_next=successor;
    } else {
        if(before!=invalid_edge) {
            step_checkpoint_.save_edge(*this,before);
            island_edges_[before].island_next=invalid_edge;
        } else first=invalid_edge;
        last_ref=before;
    }
    island_edges_[id].island_prev=invalid_edge;
    island_edges_[id].island_next=invalid_edge;
    island_edges_[id].indexed_accurate=false;
    --count;
}
void Scene::unlink_contact_edge(EdgeIndex id, bool accurate) {
    // A removed speculative bridge can isolate an already sleeping node
    // when its awake neighbor is being retired. The full graph rebuild
    // runs in the first pass, so record the affected surviving node
    // directly from this source edge before it is disconnected.
    // Third-pass deactivation validates the rebuilt island; no island
    // inventory reconstruction or unrelated island scan is needed.
    const auto& retiring=island_edges_[id];
    if(!accurate && (retiring.speculative_state&1u)) {
        for(BodyId candidate:{retiring.pair.a,retiring.pair.b}) {
            const BodyId other=(candidate==retiring.pair.a)
                ? retiring.pair.b:retiring.pair.a;
            if(actors_[candidate].alive &&
               actors_[candidate].motion==Motion::Dynamic &&
               actors_[candidate].sleeping &&
               actors_[other].motion==Motion::Dynamic &&
               !actors_[other].alive)
                speculative_split_sleep_candidates_.push_back(candidate);
        }
    }
    // Gu touch loss and manager retirement cancel activity without
    // fabricating an IslandSim sleep-deactivation event. Source Sc's
    // overlap removal destroys the interaction rather than publishing
    // an IslandSim sleep event. Retire that flag on the same EdgeIndex.
    if(!accurate && island_edges_[id].interaction_active) {
        step_checkpoint_.save_activity(*this,id);
        island_edges_[id].interaction_active=false;
        --active_contact_interaction_count_;
    }
    deactivate_contact_edge(id,accurate,false);
    // Contact manager retirement and touch loss both remove the same
    // accurate source edge before disconnecting its intrusive endpoints.
    if(accurate && island_edges_[id].connected)
        remove_accurate_island_edge(id);
    auto& edge=island_edges_[id];
    auto& heads=accurate?accurate_node_first_:speculative_node_first_;
    auto instance_at=[&](std::size_t index)->EdgeInstance& {
        auto& owner=island_edges_[index/2];
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
           actors_[node].motion==Motion::Dynamic &&
           actors_[other].motion==Motion::Dynamic) {
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
    FYZZ_PROFILE_SCOPE("sc.retire_edges");
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
        auto& edge=island_edges_[id];
        edge={};
        free_edge_handles_.push_back(id);
    }
    retired_edge_handles_.clear();
}
ConstraintId Scene::joint_token(EdgeIndex edge) const noexcept {
    const auto id=island_edges_[edge].constraint_id;
    return (static_cast<ConstraintId>(constraint_records_[id].generation)<<32)|id;
}
ConstraintId Scene::create_anchored_distance_constraint(
    BodyId a,BodyId b,Vec3 local_a,Vec3 local_b,float distance) {
    return joint_token(add_distance_constraint(a,b,distance,local_a,local_b));
}
ConstraintId Scene::create_fixed_constraint(
    BodyId a,BodyId b,Transform local_a,Transform local_b) {
    ConstraintRecord record{};
    record.kind=JointKind::Fixed;
    record.local_a=validated_pose(local_a);
    record.local_b=validated_pose(local_b);
    // Locked axes are defined in the authored A frame. Preserve endpoint
    // order independently of the canonical island pair.
    record.swapped=a>b;
    return joint_token(add_joint(a,b,record));
}
ConstraintId Scene::create_spherical_constraint(
    BodyId a,BodyId b,Transform local_a,Transform local_b) {
    ConstraintRecord record{};
    record.kind=JointKind::Spherical;
    record.local_a=validated_pose(local_a);
    record.local_b=validated_pose(local_b);
    record.swapped=a>b;
    return joint_token(add_joint(a,b,record));
}
ConstraintId Scene::create_revolute_constraint(
    BodyId a,BodyId b,Transform local_a,Transform local_b) {
    ConstraintRecord record{};
    record.kind=JointKind::Revolute;
    record.local_a=validated_pose(local_a);
    record.local_b=validated_pose(local_b);
    record.swapped=a>b;
    return joint_token(add_joint(a,b,record));
}
ConstraintId Scene::create_prismatic_constraint(
    BodyId a,BodyId b,Transform local_a,Transform local_b) {
    ConstraintRecord record{};
    record.kind=JointKind::Prismatic;
    record.local_a=validated_pose(local_a);
    record.local_b=validated_pose(local_b);
    record.swapped=a>b;
    return joint_token(add_joint(a,b,record));
}
void Scene::retire_constraint(ConstraintId id) {
    const auto index=static_cast<EdgeIndex>(id);
    const auto generation=static_cast<std::uint32_t>(id>>32);
    if(index>=constraint_records_.size() || !generation)
        throw std::out_of_range("Unknown or expired rigid constraint");
    const auto& record=constraint_records_[index];
    if(!record.live || record.generation!=generation)
        throw std::out_of_range("Unknown or expired rigid constraint");
    remove_constraint_edge(record.edge);
}
Scene::EdgeIndex Scene::add_distance_constraint(
    BodyId a,BodyId b,float length,Vec3 local_a,Vec3 local_b) {
    if(!finite(local_a) || !finite(local_b))
        throw std::invalid_argument("Source joint anchors must be finite");
    if(!std::isfinite(length) || length<0.0f)
        throw std::invalid_argument("Rigid 1D distance must be finite and nonnegative");
    ConstraintRecord record{};
    record.distance_length=length;
    record.local_a.position=local_a;
    record.local_b.position=local_b;
    return add_joint(a,b,record);
}
Scene::EdgeIndex Scene::add_joint(BodyId a,BodyId b,ConstraintRecord record) {
    const auto& left=require(a);
    const auto& right=require(b);
    if(left.motion!=Motion::Dynamic && right.motion!=Motion::Dynamic)
        throw std::logic_error("Rigid constraint requires a dynamic solver node");
    if(constraint_records_.size()>=invalid_edge &&
       free_constraint_ids_.empty())
        throw std::overflow_error("Sc constraint identifier exhausted");
    if(!free_constraint_ids_.empty() &&
       constraint_records_[free_constraint_ids_.back()].generation==
           std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("Sc constraint slot generations exhausted");
    if(free_constraint_ids_.empty())
        constraint_records_.reserve(constraint_records_.size()+1);
    free_constraint_ids_.reserve(constraint_records_.size()+1);
    const EdgeIndex edge=add_constraint_edge(a,b);
    record.edge=edge;
    record.live=true;
    EdgeIndex id;
    if(free_constraint_ids_.empty()) {
        id=static_cast<EdgeIndex>(constraint_records_.size());
        record.generation=1;
        constraint_records_.push_back(record);
    } else {
        id=free_constraint_ids_.back();
        free_constraint_ids_.pop_back();
        record.generation=constraint_records_[id].generation+1;
        constraint_records_[id]=record;
    }
    island_edges_[edge].constraint_id=id;
    return edge;
}
Scene::EdgeIndex Scene::add_constraint_edge(BodyId a, BodyId b) {
    require(a);
    require(b);
    if(a==b || (!rigid_node(actors_[a].motion) &&
                !rigid_node(actors_[b].motion)))
        throw std::invalid_argument(
            "Constraint requires two distinct actors and a rigid node");
    // EdgeIndex comes from the SAME HandleManager pool as Gu managers;
    // both IslandSim graphs use the same paired endpoint slots.
    const Pair pair=ordered_pair(a,b);
    // PhysX admits multiple independent PxConstraint objects between
    // the same two nodes. They share endpoints, not EdgeIndex identity.
    active_constraint_handles_.reserve(active_constraint_handles_.size()+1);
    speculative_node_first_.resize(actors_.size(),invalid_instance);
    accurate_node_first_.resize(actors_.size(),invalid_instance);
    // addConnectionToGraph may mark this edge active immediately. Prepare
    // the existing IslandSim reference slots before any edge mutation.
    speculative_islands_.active_reference_counts.resize(actors_.size(),0);
    accurate_islands_.active_reference_counts.resize(actors_.size(),0);
    // The source wake handoff can enqueue a previously dormant partner
    // on the next island publication after the new connection merges.
    speculative_islands_.activating_nodes.reserve(actors_.size());
    accurate_islands_.activating_nodes.reserve(actors_.size());
    const EdgeIndex id=allocate_contact_edge(pair);
    auto& edge=island_edges_[id];
    edge.type=IslandEdgeType::Constraint;
    edge.connected=true;
    link_contact_edge(id,false);
    link_contact_edge(id,true);
    active_constraint_handles_.push_back(id);
    const bool awake_a=actors_[a].motion==Motion::Dynamic &&
                       !actors_[a].sleeping;
    const bool awake_b=actors_[b].motion==Motion::Dynamic &&
                       !actors_[b].sleeping;
    if(awake_a || awake_b) {
        // Source addConnectionToGraph tests endpoint activity before
        // later island-route repair, with no scan of all constraints.
        activate_contact_edge(id,false);
        activate_contact_edge(id,true);
        for(BodyId endpoint:{a,b}) {
            auto& body=actors_[endpoint];
            if(body.motion==Motion::Dynamic && body.sleeping &&
               !body.activating) {
                queue_node_activation(endpoint);
                body.activating=true;
                authored_activation_pending_=true;
            }
        }
    }
    island_topology_dirty_=true;
    dirty_=true;
    return id;
}
void Scene::remove_constraint_edge(EdgeIndex id) {
    if(id>=island_edges_.size() ||
       island_edges_[id].type!=IslandEdgeType::Constraint ||
       !island_edges_[id].live || island_edges_[id].retiring)
        throw std::out_of_range("Unknown or retired constraint edge");
    auto it=std::find(active_constraint_handles_.begin(),
                      active_constraint_handles_.end(),id);
    if(it==active_constraint_handles_.end())
        throw std::logic_error("Constraint handle missing from source pool");
    retired_edge_handles_.reserve(retired_edge_handles_.size()+1);
    const EdgeIndex constraint_id=island_edges_[id].constraint_id;
    if(constraint_id!=invalid_edge)
        free_constraint_ids_.reserve(free_constraint_ids_.size()+1);
    unlink_contact_edge(id,true);
    unlink_contact_edge(id,false);
    change_counted_interactions(island_edges_[id].pair,false);
    island_edges_[id].retiring=true;
    island_edges_[id].connected=false;
    if(constraint_id!=invalid_edge) {
        auto& record=constraint_records_[constraint_id];
        if(!record.live || record.edge!=id)
            throw std::logic_error("Source constraint owner/index mismatch");
        // Retain the slot generation across IDTracker recycling.
        // The next admitted World token will have a different identity.
        record.edge=invalid_edge;
        record.distance_length=0.0f;
        record.live=false;
        free_constraint_ids_.push_back(constraint_id);
        island_edges_[id].constraint_id=invalid_edge;
    }
    active_constraint_handles_.erase(it);
    retired_edge_handles_.push_back(id);
    island_topology_dirty_=true;
    dirty_=true;
}
void Scene::publish_pairs() {
    FYZZ_PROFILE_SCOPE("sc.publish_pairs");
    // Graph node heads must grow on actor admission even with no BP delta.
    speculative_node_first_.resize(actors_.size(),invalid_instance);
    accurate_node_first_.resize(actors_.size(),invalid_instance);
    if(!broadphase_pair_membership_changed_) {
        next_found_.clear();
        next_lost_.clear();
        found_.swap(next_found_);
        lost_.swap(next_lost_);
        ++quiet_event_epoch_;
        return;
    }
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
            change_counted_interactions(island_edges_[id].pair,false);
            step_checkpoint_.save_edge(*this,id);
            island_edges_[id].retiring=true;
            island_edges_[id].connected=false;
            retired_edge_handles_.push_back(id);
            next_lost_.push_back(previous_pairs_[old_index++]);
        } else if(old_index==previous_pairs_.size() ||
                  pending_pairs_[new_index]<previous_pairs_[old_index]) {
            step_checkpoint_.save_handle_lists(*this);
            const Pair pair=pending_pairs_[new_index++];
            const EdgeIndex id=allocate_contact_edge(body_pair(pair),pair);
            // ScNPhaseCore::shouldSwapBodies puts static or kinematic
            // participants second. Two ordinary dynamics are ordered by
            // descending ActorID, but a later-created kinematic is not
            // swapped ahead of an earlier ordinary dynamic. BodyIds
            // preserve admission order for the supported rigid actors.
            // The manager retains this orientation for its lifetime.
            const Motion a=shapes_[pair.a].motion;
            const Motion b=shapes_[pair.b].motion;
            const bool a_static=!rigid_node(a);
            const bool both_rigid=rigid_node(a) && rigid_node(b);
            island_edges_[id].gu_order_reversed =
                a_static ||
                (both_rigid && (a==Motion::Kinematic ||
                    (shapes_[pair.a].body<shapes_[pair.b].body && b==Motion::Dynamic)));
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
void Scene::reduce_pcm_box_batch(
    PcmManifoldCache& cache,
    std::span<const PcmContactCandidate> contacts,
    float tolerance_length) {
    const auto selected=source_pcm_box_batch_selection(contacts,tolerance_length);
    for(std::size_t i=0;i<4;++i) {
        const auto& c=contacts[selected[i]];
        cache.points[i]={
            c.local_a,c.local_b,c.normal_b,c.separation};
    }
    cache.count=4;
}

void Scene::reduce_pcm_box_five(
    PcmManifoldCache& cache,const PcmPersistentContact& new_contact) {
    // GuPersistentContactManifold.cpp::reduceContactsForPCM. The fifth
    // witness enters a four-contact cache without discarding its spatial
    // coverage or ignoring the previous points' depth ordering.
    std::array<PcmPersistentContact,5> candidates{};
    for(int i=0;i<4;++i)candidates[i]=cache.points[i];
    candidates[4]=new_contact;
    std::array<bool,5> chosen{};
    std::array<int,4> selected{};
    const auto diff=[](Vec3 a,Vec3 b) {
        return Vec3{a.x-b.x,a.y-b.y,a.z-b.z};
    };
    const auto dot=[](Vec3 a,Vec3 b) {
        return a.x*b.x+a.y*b.y+a.z*b.z;
    };
    const auto sq=[&](Vec3 a) {return dot(a,a);};
    const auto segment_sq=[&](Vec3 point,Vec3 a,Vec3 b) {
        const Vec3 ab=diff(b,a),ap=diff(point,a);
        const float denominator=sq(ab);
        const float t=denominator==0.0f
            ?0.0f:std::clamp(dot(ap,ab)/denominator,0.0f,1.0f);
        const Vec3 delta{ap.x-t*ab.x,ap.y-t*ab.y,ap.z-t*ab.z};
        return sq(delta);
    };
    const auto triangle_sq=[&](Vec3 p,Vec3 a,Vec3 b,Vec3 c) {
        // Source distancePointTriangleSquaredLocal: Voronoi regions
        // of A/B/C, AB/BC/AC and the projected face interior.
        const Vec3 ab=diff(b,a),ac=diff(c,a);
        const Vec3 ap=diff(p,a),bp=diff(p,b),cp=diff(p,c);
        const float d1=dot(ab,ap),d2=dot(ac,ap);
        const float d3=dot(ab,bp),d4=dot(ac,bp);
        const float d5=dot(ab,cp),d6=dot(ac,cp);
        if(d1<0.0f && d2<0.0f)return sq(ap);
        if(d3>=0.0f && d3>=d4)return sq(bp);
        if(d6>=0.0f && d6>=d5)return sq(cp);
        const float vc=d1*d4-d3*d2;
        if(vc<0.0f && d1>=0.0f && d3<0.0f)
            return segment_sq(p,a,b);
        const float va=d3*d6-d5*d4;
        if(va<0.0f && d4>=d3 && d5>=d6)
            return segment_sq(p,b,c);
        const float vb=d5*d2-d1*d6;
        if(vb<0.0f && d2>=0.0f && d6<0.0f)
            return segment_sq(p,a,c);
        const Vec3 n{
            ab.y*ac.z-ab.z*ac.y,
            ab.z*ac.x-ab.x*ac.z,
            ab.x*ac.y-ab.y*ac.x};
        const float denominator=sq(n);
        if(denominator==0.0f)return 0.0f;
        const float projection=dot(n,ap);
        return projection*projection/denominator;
    };
    int deepest=4;
    for(int i=0;i<4;++i)
        if(candidates[i].separation<candidates[deepest].separation)
            deepest=i;
    selected[0]=deepest;
    chosen[deepest]=true;
    int farthest=0;
    float largest=sq(diff(candidates[0].local_b,
                          candidates[deepest].local_b));
    for(int i=1;i<5;++i)if(!chosen[i]) {
        const float value=sq(diff(candidates[i].local_b,
                                  candidates[deepest].local_b));
        if(value>largest){largest=value;farthest=i;}
    }
    selected[1]=farthest;
    chosen[farthest]=true;
    largest=-std::numeric_limits<float>::max();
    int third=0;
    for(int i=0;i<5;++i)if(!chosen[i]) {
        const float value=segment_sq(
            candidates[i].local_b,candidates[deepest].local_b,
            candidates[farthest].local_b);
        if(value>largest){largest=value;third=i;}
    }
    selected[2]=third;
    chosen[third]=true;
    largest=-std::numeric_limits<float>::max();
    int fourth=third;
    for(int i=0;i<5;++i)if(!chosen[i]) {
        const float value=triangle_sq(
            candidates[i].local_b,candidates[selected[0]].local_b,
            candidates[selected[1]].local_b,
            candidates[selected[2]].local_b);
        if(value>largest){largest=value;fourth=i;}
    }
    if(chosen[fourth]) {
        cache.count=3;
        for(int i=0;i<3;++i)cache.points[i]=candidates[selected[i]];
        return;
    }
    selected[3]=fourth;
    chosen[fourth]=true;
    for(int i=0;i<4;++i)cache.points[i]=candidates[selected[i]];
    cache.count=4;
    int unselected=-1;
    for(int i=0;i<5;++i)
        if(!chosen[i]){unselected=i;break;}
    if(unselected<0)return;
    float closest=std::numeric_limits<float>::max();
    int replace=0;
    for(int i=0;i<4;++i) {
        const float distance=sq(diff(cache.points[i].local_a,
                                     candidates[unselected].local_a));
        if(distance<closest){closest=distance;replace=i;}
    }
    if(candidates[unselected].separation<
       cache.points[replace].separation)
        cache.points[replace]=candidates[unselected];
}
void Scene::refresh_pcm_contact_points(
    PcmManifoldCache& cache, Vec3 position_a_in_b,
    Quat rotation_a_in_b, float projection_break) {
    // Refresh every retained witness in the second shape's local frame.
    // Removing a witness swaps in the last point, preserving the source
    // cache's update order and its local-normal distance convention.
    const float breaking_sq=projection_break*projection_break;
    for(std::size_t i=cache.count;i>0;--i) {
        auto& point=cache.points[i-1];
        const Vec3 rotated=rotate(rotation_a_in_b,point.local_a);
        const Vec3 local_a_in_b{
            position_a_in_b.x+rotated.x,
            position_a_in_b.y+rotated.y,
            position_a_in_b.z+rotated.z};
        const Vec3 v{
            local_a_in_b.x-point.local_b.x,
            local_a_in_b.y-point.local_b.y,
            local_a_in_b.z-point.local_b.z};
        const float distance=v.x*point.normal_b.x+
            v.y*point.normal_b.y+v.z*point.normal_b.z;
        const Vec3 projected_difference{
            v.x-point.normal_b.x*distance,
            v.y-point.normal_b.y*distance,
            v.z-point.normal_b.z*distance};
        const float tangential_sq=
            projected_difference.x*projected_difference.x+
            projected_difference.y*projected_difference.y+
            projected_difference.z*projected_difference.z;
        if(tangential_sq>breaking_sq)
            cache.points[i-1]=cache.points[--cache.count];
        else
            point.separation=distance;
    }
}
std::uint8_t Scene::generate_pcm_box_box(
    const Shape& first, const Shape& second, float contact_distance,
    PcmManifoldCache& cache, std::span<ContactPoint> output) {
    // GuPCMContactBoxBox.cpp::pcmContactBoxBox, operating on the same
    // Sc contact-manager edge used for islands, rollback and PGS.
    // A-to-B relative transform and each contact normal are B-local.
    const Vec3 delta{
        first.position.x-second.position.x,
        first.position.y-second.position.y,
        first.position.z-second.position.z};
    const Vec3 relative=rotate_inverse(second.orientation,delta);
    const Quat inv_b{
        -second.orientation.x,-second.orientation.y,
        -second.orientation.z,second.orientation.w};
    const Quat rotation=multiply(inv_b,first.orientation);
    const auto margin=[](const Shape& a) {
        return std::min(0.15f*std::min({
            a.half_extents.x,a.half_extents.y,a.half_extents.z}),0.15f);
    };
    const float min_margin=std::min(margin(first),margin(second));
    const std::uint8_t initial=cache.count;
    refresh_pcm_contact_points(cache,relative,rotation,min_margin*0.8f);
    const bool lost=cache.count!=initial;
    constexpr std::array<float,5> thresholds{
        0.5f,0.125f,0.25f,0.375f,0.375f};
    constexpr std::array<float,5> quaternion_thresholds{
        0.9998f,0.9999f,0.9999f,0.9999f,0.9999f};
    const float translation_delta=cache.initialized
        ? std::max({
            std::abs(relative.x-cache.relative_position.x),
            std::abs(relative.y-cache.relative_position.y),
            std::abs(relative.z-cache.relative_position.z)})
        : std::numeric_limits<float>::infinity();
    const auto quaternion_dot=[](Quat a,Quat b) {
        return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
    };
    const float qa=quaternion_dot(first.orientation,cache.orientation_a);
    const float qb=quaternion_dot(second.orientation,cache.orientation_b);
    const float threshold_p=min_margin*thresholds[cache.count];
    bool regenerate=!cache.initialized || lost ||
        translation_delta>threshold_p ||
        qa<quaternion_thresholds[cache.count] ||
        qb<quaternion_thresholds[cache.count];
    if(!regenerate) {
        // GuPersistentContactManifold::invalidate_BoxConvex also
        // measures rotational travel at the body's bounding radius.
        // Either excessive arc distance invalidates the manifold.
        const auto angular_travel=[](float d,Vec3 extents) {
            const float angle=d<1.0f?std::acos(std::max(-1.0f,d)):0.0f;
            return angle*std::sqrt(extents.x*extents.x+
                extents.y*extents.y+extents.z*extents.z);
        };
        regenerate=
            angular_travel(qa,first.half_extents)>threshold_p ||
            angular_travel(qb,second.half_extents)>threshold_p;
    }
    bool direct_generated=false;
    bool gjk_generated=false;
    Vec3 gjk_normal_b{};
    if(regenerate) {
        cache.relative_position=relative;
        cache.relative_rotation=rotation;
        cache.orientation_a=first.orientation;
        cache.orientation_b=second.orientation;
        cache.initialized=true;
        if(pcm_candidate_scratch_.size()!=64)
            pcm_candidate_scratch_.resize(64);
        auto& candidates=pcm_candidate_scratch_;
        std::size_t count=0;
        const bool overlapping=source_pcm_box_box_candidates(
            first.half_extents,first.position,first.orientation,
            second.half_extents,second.position,second.orientation,
            contact_distance,candidates,count);
        if(!overlapping) {
            // A failed regeneration cannot retain refreshed witnesses:
            // the new transform has no source PCM contact to publish.
            cache.count=0;
            cache.warmstart={};
            return 0;
        }
        if(count==0) {
            PcmContactCandidate candidate{};
            if(!source_pcm_box_box_gjk_epa(
                    first.half_extents,first.position,first.orientation,
                    second.half_extents,second.position,second.orientation,
                    contact_distance,candidate,&cache.warmstart)) {
                // Gu did not publish a contact at this transform. Retaining
                // refreshed old points would make the next quiet pass
                // replay stale witnesses without running GJK/EPA.
                // Retain the empty manifold's relative pose so an
                // unchanged separation can use the ordinary PCM cache.
                cache.count=0;
                cache.warmstart={};
                return 0;
            }
            // GJK output normal applies to the entire manifold in
            // GuPCMContactBoxBox::pcmContactBoxBox, even when an older
            // retained point occupies the first slot.
            gjk_generated=true;
            gjk_normal_b=candidate.normal_b;
            const PcmPersistentContact witness{
                candidate.local_a,candidate.local_b,
                candidate.normal_b,candidate.separation};
            const float replace_break=min_margin*0.05f;
            const float replace_sq=replace_break*replace_break;
            const auto sq=[](Vec3 v) {
                return v.x*v.x+v.y*v.y+v.z*v.z;
            };
            const auto diff=[](Vec3 a,Vec3 b) {
                return Vec3{a.x-b.x,a.y-b.y,a.z-b.z};
            };
            bool replaced=false;
            for(std::size_t i=0;i<cache.count;++i) {
                const auto& old=cache.points[i];
                if(std::min(sq(diff(old.local_a,witness.local_a)),
                            sq(diff(old.local_b,witness.local_b)))<
                   replace_sq) {
                    cache.points[i]=witness;
                    replaced=true;
                    break;
                }
            }
            if(!replaced) {
                if(cache.count<4)
                    cache.points[cache.count++]=witness;
                else
                    reduce_pcm_box_five(cache,witness);
            }
        } else if(count<=4) {
            cache.count=0;
            for(std::size_t i=0;i<count;++i) {
                const auto& c=candidates[i];
                cache.points[i]={c.local_a,c.local_b,c.normal_b,c.separation};
            }
            cache.count=static_cast<std::uint8_t>(count);
        } else {
            reduce_pcm_box_batch(cache,
                std::span<const PcmContactCandidate>{
                    candidates.data(),count},
                1.0f); // source PxTolerancesScale.length
        }
        direct_generated=count>0;
    }
    if(cache.count==0)return 0;
    // Direct face generation uses the first manifold normal. A newly
    // generated GJK witness uses GjkOutput.normal. Quiet refresh takes
    // getWorldNormal: the normalized sum of every retained local B
    // normal, not an arbitrary point's normal.
    Vec3 normal_b{};
    if(gjk_generated)
        normal_b=gjk_normal_b;
    else if(direct_generated)
        normal_b=cache.points[0].normal_b;
    else
        for(std::size_t i=0;i<cache.count;++i) {
            const auto& n=cache.points[i].normal_b;
            normal_b.x+=n.x;
            normal_b.y+=n.y;
            normal_b.z+=n.z;
        }
    const float normal_sq=normal_b.x*normal_b.x+
        normal_b.y*normal_b.y+normal_b.z*normal_b.z;
    if(!(normal_sq>0.0f) || !std::isfinite(normal_sq))
        throw std::overflow_error("PCM box-box manifold normal invalid");
    const float inverse_length=1.0f/std::sqrt(normal_sq);
    const Vec3 normalized{
        normal_b.x*inverse_length,
        normal_b.y*inverse_length,
        normal_b.z*inverse_length};
    const Vec3 normal=rotate(second.orientation,normalized);
    std::uint8_t emitted=0;
    for(std::size_t i=0;i<cache.count;++i) {
        const auto& p=cache.points[i];
        if(!direct_generated && p.separation>contact_distance)continue;
        if(emitted>=output.size())
            throw std::logic_error("PCM box-box output arena exhausted");
        const Vec3 rotated=rotate(second.orientation,p.local_b);
        const ContactPoint point{{
            rotated.x+second.position.x,
            rotated.y+second.position.y,
            rotated.z+second.position.z},
            normal,p.separation};
        if(!finite(point.position) || !finite(point.normal) ||
           !std::isfinite(point.separation))
            throw std::overflow_error("PCM box-box contact overflow");
        output[emitted++]=point;
    }
    return emitted;
}
std::uint8_t Scene::generate_pcm_convex_pair(
    const Shape& first,const Shape& second,float contact_distance,
    PcmManifoldCache& cache,std::span<ContactPoint> output) {
    // Cooked hulls participate in the same Sc EdgeIndex PCM lifetime,
    // manifold refresh and Dy contact stream as other finite shapes.
    const Vec3 delta{first.position.x-second.position.x,
                     first.position.y-second.position.y,
                     first.position.z-second.position.z};
    const Vec3 relative=rotate_inverse(second.orientation,delta);
    const Quat inverse_b{-second.orientation.x,-second.orientation.y,
                         -second.orientation.z,second.orientation.w};
    const Quat rotation=multiply(inverse_b,first.orientation);
    const auto margin=[](const Shape& shape) {
        const float extent=shape.geometry==Geometry::Sphere
            ? shape.radius
            : shape.geometry==Geometry::Convex
                ? [&] {
                    const Vec3 lower=shape.convex->bounds_min();
                    const Vec3 upper=shape.convex->bounds_max();
                    return 0.5f*std::min({upper.x-lower.x,
                                          upper.y-lower.y,upper.z-lower.z});
                  }()
                :std::min({shape.half_extents.x,
                            shape.half_extents.y,shape.half_extents.z});
        return std::min(0.15f*extent,0.15f);
    };
    const float min_margin=std::min(margin(first),margin(second));
    const auto previous_count=cache.count;
    refresh_pcm_contact_points(cache,relative,rotation,min_margin*0.8f);
    const float translation=cache.initialized
        ?std::max({std::abs(relative.x-cache.relative_position.x),
                   std::abs(relative.y-cache.relative_position.y),
                   std::abs(relative.z-cache.relative_position.z)})
        :std::numeric_limits<float>::infinity();
    const auto quaternion_dot=[](Quat a,Quat b) {
        return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
    };
    const float qa=quaternion_dot(first.orientation,cache.orientation_a);
    const float qb=quaternion_dot(second.orientation,cache.orientation_b);
    const float threshold=min_margin*(cache.count==0u?0.5f:0.125f);
    bool regenerate=!cache.initialized || cache.count!=previous_count ||
        translation>threshold || qa<0.9999f || qb<0.9999f;
    if(!regenerate) {
        const auto angular_travel=[](float q,Vec3 extents) {
            return (q<1.f?std::acos(std::max(-1.f,q)):0.f)*
                std::sqrt(extents.x*extents.x+
                          extents.y*extents.y+extents.z*extents.z);
        };
        regenerate=angular_travel(qa,first.half_extents)>threshold ||
                   angular_travel(qb,second.half_extents)>threshold;
    }
    if(regenerate) {
        cache.relative_position=relative;
        cache.relative_rotation=rotation;
        cache.orientation_a=first.orientation;
        cache.orientation_b=second.orientation;
        cache.initialized=true;
        const auto descriptor=[](const Shape& shape) {
            PcmFiniteShape descriptor{};
            descriptor.half_extents=shape.half_extents;
            descriptor.position=shape.position;
            descriptor.orientation=shape.orientation;
            if(shape.geometry==Geometry::Sphere) {
                descriptor.sphere_radius=shape.radius;
                descriptor.half_extents={shape.radius,shape.radius,shape.radius};
            }
            if(shape.geometry==Geometry::Convex) {
                descriptor.hull_vertices=shape.convex->vertices();
                descriptor.hull_faces=shape.convex->faces();
                descriptor.hull_face_indices=shape.convex->face_indices();
                const Vec3 lower=shape.convex->bounds_min();
                const Vec3 upper=shape.convex->bounds_max();
                descriptor.half_extents={
                    0.5f*(upper.x-lower.x),
                    0.5f*(upper.y-lower.y),
                    0.5f*(upper.z-lower.z)};
            }
            return descriptor;
        };
        const auto a=descriptor(first),b=descriptor(second);
        PcmContactCandidate point{};
        cache.count=0u;
        if(source_pcm_convex_gjk_epa(a,b,contact_distance,
                                    point,&cache.warmstart)) {
            std::array<PcmContactCandidate,4> clipped{};
            const auto count=source_pcm_convex_face_contacts(
                a,b,contact_distance,point,clipped);
            if(count) {
                for(unsigned i=0u;i<count;++i)
                    cache.points[i]={clipped[i].local_a,clipped[i].local_b,
                                     clipped[i].normal_b,clipped[i].separation};
                cache.count=count;
            }else {
                cache.points[0]={point.local_a,point.local_b,
                                 point.normal_b,point.separation};
                cache.count=1u;
            }
        }
    }
    if(!cache.count)return 0u;
    std::uint8_t emitted=0u;
    for(std::uint8_t i=0u;i<cache.count;++i) {
        const auto& point=cache.points[i];
        if(point.separation>contact_distance)continue;
        const Vec3 normal=rotate(second.orientation,point.normal_b);
        const Vec3 local=rotate(second.orientation,point.local_b);
        const ContactPoint contact{{
            second.position.x+local.x,
            second.position.y+local.y,
            second.position.z+local.z},normal,point.separation};
        if(!finite(contact.position) || !finite(contact.normal) ||
           !std::isfinite(contact.separation))
            throw std::overflow_error("Cooked convex PCM contact invalid");
        if(emitted>=output.size())
            throw std::logic_error("Cooked convex Gu contact arena exhausted");
        output[emitted++]=contact;
    }
    return emitted;
}
std::uint8_t Scene::generate_pcm_plane_box(
    const Shape& plane, const Shape& box, float contact_distance,
    PcmManifoldCache& cache, std::span<ContactPoint> output) {
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
    const auto sq=[](Vec3 v) {
        return v.x*v.x+v.y*v.y+v.z*v.z;
    };
    const auto diff=[](Vec3 a,Vec3 b) {
        return Vec3{a.x-b.x,a.y-b.y,a.z-b.z};
    };
    const auto dot=[](Vec3 a,Vec3 b) {
        return a.x*b.x+a.y*b.y+a.z*b.z;
    };
    const auto previous_count=cache.count;
    refresh_pcm_contact_points(cache,relative,box.orientation,project_break);
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
        // Convex plane contacts are actual cooked vertices, not corners
        // of the shape's conservative local AABB. Reuse the scene-owned
        // candidate arena across all Sc managers without per-pair malloc.
        auto& candidates=pcm_plane_candidates_;
        candidates.clear();
        if(box.geometry==Geometry::Convex)
            candidates.reserve(box.convex->vertices().size());
        else candidates.reserve(8u);
        const auto admit=[&](Vec3 local) {
            const Vec3 r=rotate(q,local);
            const Vec3 in_plane{relative.x+r.x,relative.y+r.y,
                                relative.z+r.z};
            if(in_plane.y>=contact_distance)return;
            candidates.push_back({local,
                {in_plane.x,0.0f,in_plane.z}, {0.0f,1.0f,0.0f},
                in_plane.y});
        };
        if(box.geometry==Geometry::Convex) {
            for(const auto vertex:box.convex->vertices())admit(vertex);
        }else {
            for(unsigned corner=0;corner<8u;++corner)
                admit({
                    (corner&4u)?-box.half_extents.x:box.half_extents.x,
                    (corner&2u)?-box.half_extents.y:box.half_extents.y,
                    (corner&1u)?-box.half_extents.z:box.half_extents.z});
        }
        const std::size_t count=candidates.size();
        if(count<=4) {
            for(std::size_t i=0;i<count;++i)
                cache.points[i]=candidates[i];
            cache.count=static_cast<std::uint8_t>(count);
        } else {
            // GuPersistentContactManifold.cpp::
            // reduceBatchContactsCluster (four-contact manifold). Keep
            // source deepest/farthest/segment-extrema ordering and then
            // cluster residual corners around those four witnesses.
            pcm_plane_chosen_.assign(count,0u);
            auto& chosen=pcm_plane_chosen_;
            std::array<std::size_t,4> indices{};
            std::size_t first=0;
            for(std::size_t i=1;i<count;++i)
                if(candidates[i].separation<candidates[first].separation)
                    first=i;
            indices[0]=first;
            chosen[first]=true;
            std::size_t farthest=0;
            float max_distance=sq(diff(candidates[0].local_b,
                                       candidates[first].local_b));
            for(std::size_t i=1;i<count;++i) {
                const float distance=sq(diff(candidates[i].local_b,
                                               candidates[first].local_b));
                if(distance>max_distance) {
                    max_distance=distance;
                    farthest=i;
                }
            }
            indices[1]=farthest;
            chosen[farthest]=true;
            const Vec3 segment=diff(candidates[farthest].local_b,
                                    candidates[first].local_b);
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
                const float d=dot(diff(candidates[i].local_b,
                                         candidates[first].local_b),sideways);
                if(d>largest){largest=d;third=i;}
                if(d<smallest){smallest=d;fourth=i;}
            }
            indices[2]=third;
            chosen[third]=true;
            if(smallest*largest>0.0f) {
                largest=-std::numeric_limits<float>::max();
                for(std::size_t i=0;i<count;++i) {
                    if(chosen[i])continue;
                    const float d=dot(diff(candidates[i].local_b,
                                             candidates[first].local_b),
                                      sideways);
                    if(d>largest){largest=d;fourth=i;}
                }
            }
            indices[3]=fourth;
            chosen[fourth]=true;
            // Source clusters every remaining contact around the
            // ORIGINAL four geometric witnesses. Replacement changes
            // only the chosen contact index, not the cluster center.
            std::array<PcmPersistentContact,4> cluster_centers{};
            for(std::size_t k=0;k<4;++k)
                cluster_centers[k]=candidates[indices[k]];
            for(std::size_t i=0;i<count;++i) {
                if(chosen[i])continue;
                float nearest=std::numeric_limits<float>::max();
                std::size_t slot=0;
                for(std::size_t j=0;j<4;++j) {
                    const float d=sq(diff(candidates[i].local_b,
                                           cluster_centers[j].local_b));
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
            {plane.position.x+pt.local_b.x,plane.position.y,
             plane.position.z+pt.local_b.z},
            {0.0f,-1.0f,0.0f},pt.separation};
        if(!finite(contact.position) || !std::isfinite(contact.separation))
            throw std::overflow_error("PhysX PCM plane-box contact overflow");
        output[emitted++]=contact;
    }
    return emitted;
}
std::uint8_t Scene::publish_gpu_pcm_box_box(
    const GpuPcmBoxResult& gpu,PcmManifoldCache& cache,
    std::span<ContactPoint> output) {
    if(gpu.status()!=0u || gpu.count()>4u)
        throw std::logic_error("Source GPU PCM returned an invalid Sc manifold");
    // The original EdgeIndex cache retains B-local witnesses on device.
    // This frame's device stream already contains Gu world contacts.
    cache.initialized=true;
    if(!gpu.count())return 0u;
    const float sq=gpu.normal.x*gpu.normal.x+
                   gpu.normal.y*gpu.normal.y+
                   gpu.normal.z*gpu.normal.z;
    if(!(sq>0.f) || !std::isfinite(sq) || !finite(gpu.normal))
        throw std::overflow_error("Source GPU PCM normal is invalid");
    if(gpu.count()>output.size())
        throw std::logic_error("Source GPU PCM contact arena exhausted");
    for(std::uint8_t i=0;i<gpu.count();++i) {
        const auto& p=gpu.points[i];
        const ContactPoint contact{p.position,gpu.normal,p.separation};
        if(!finite(contact.position) || !std::isfinite(contact.separation))
            throw std::overflow_error("Source GPU PCM witness invalid");
        output[i]=contact;
    }
    return static_cast<std::uint8_t>(gpu.count());
}
bool Scene::cached_friction_patch_valid(
    const FrictionPatch& old,const Actor& actor0,
    const Actor& actor1) const {
    if(old.broken || old.disable_strong_friction || old.anchor_count==0u)
        return false;
    constexpr float same_normal=0.999f;
    constexpr float correlation_distance=0.025f;
    const Quat q0=mass_orientation(actor0),q1=mass_orientation(actor1);
    const Vec3 old_normal_1_in_0=
        rotate_inverse(q0,rotate(q1,old.body1_normal));
    const auto dot=[](Vec3 a,Vec3 b) {
        return a.x*b.x+a.y*b.y+a.z*b.z;
    };
    if(dot(old.body0_normal,old_normal_1_in_0)<=same_normal)
        return false;
    const Vec3 p0=mass_position(actor0),p1=mass_position(actor1);
    for(std::size_t k=0;k<old.anchor_count;++k) {
        const Vec3 local=rotate(q1,old.body1_anchors[k]);
        const Vec3 world{local.x+p1.x,local.y+p1.y,local.z+p1.z};
        const Vec3 in0=rotate_inverse(q0,{
            world.x-p0.x,world.y-p0.y,world.z-p0.z});
        const Vec3 delta{
            old.body0_anchors[k].x-in0.x,
            old.body0_anchors[k].y-in0.y,
            old.body0_anchors[k].z-in0.z};
        if(std::abs(dot(delta,old.body0_normal))>=correlation_distance)
            return false;
    }
    return true;
}

void Scene::update_contacts() {
    FYZZ_PROFILE_SCOPE("gu.contacts_pcm");
    // Gu narrowphase consumes the sum of both PxShapeCore contact offsets.
    next_manifolds_.clear();
    next_contact_points_.clear();
    next_found_contacts_.clear();
    next_lost_contacts_.clear();
    // Original PxcNpCacheStreamPair retains source-generated geometry,
    // distinct from this frame's reconstructed midpoint contact output.
    next_gu_cache_records_.clear();
    next_gu_cached_points_.clear();
    // Build the Gu-offset lookup alongside this real contact-manager
    // publication; no second O(all Gu manifolds) pass during PGS.
    // Generation stamps invalidate untouched/retired EdgeIndex slots
    // without clearing the historical handle watermark each frame.
    next_manifold_by_edge_.resize(island_edges_.size());
    if(++indexed_manifold_generation_==0) {
        for(auto& item:next_manifold_by_edge_)item.generation=0;
        for(auto& item:manifold_by_edge_)item.generation=0;
        indexed_manifold_generation_=1;
    }
    const auto new_lookup_generation=indexed_manifold_generation_;
    gpu_pcm_pairs_.clear();
    gpu_pcm_results_.clear();
    gpu_pcm_friction_history_masks_.clear();
    gpu_dy_gu_touches_.clear();
    const bool source_gpu_pcm =
        gpu_broadphase_ && gpu_broadphase_->has_gpu_pcm_box() &&
        (pcm_box_box_enabled_ || pcm_plane_box_enabled_);
    // Sc's source EdgeIndex and Gu cache already distinguish an untouched
    // speculative pair from a touching manifold. Once both rigid bodies
    // are inactive, neither case needs another PCM refresh: retain the
    // published points for touching pairs and the authoritative absence
    // of points for zero-touch pairs. An authored shape edit resets the
    // same PCM cache and a rigid wake reenters the GPU narrowphase.
    const auto dormant_pcm=[&](Pair pair,EdgeIndex edge) noexcept {
        if(edge>=island_edges_.size() ||
           !island_edges_[edge].pcm_cache.initialized)return false;
        const auto& a=shapes_[pair.a];
        const auto& b=shapes_[pair.b];
        if(a.motion==Motion::Kinematic || b.motion==Motion::Kinematic)
            return false;
        const bool dynamic_a=a.motion==Motion::Dynamic;
        const bool dynamic_b=b.motion==Motion::Dynamic;
        return (dynamic_a || dynamic_b) &&
            (!dynamic_a || actors_[a.body].sleeping) && (!dynamic_b || actors_[b.body].sleeping);
    };
    if(source_gpu_pcm) {
        gpu_pcm_pairs_.reserve(previous_pairs_.size());
        std::size_t prior_touch_cursor=0u;
        for(std::size_t i=0;i<previous_pairs_.size();++i) {
            const Pair pair=previous_pairs_[i];
            while(prior_touch_cursor<manifolds_.size() &&
                  manifolds_[prior_touch_cursor].pair<pair)
                ++prior_touch_cursor;
            const bool previously_touching=
                prior_touch_cursor<manifolds_.size() &&
                manifolds_[prior_touch_cursor].pair==pair;
            const Shape& a=shapes_[pair.a];
            const Shape& b=shapes_[pair.b];
            const bool plane=a.geometry==Geometry::Plane ||
                b.geometry==Geometry::Plane;
            const bool sphere=a.geometry==Geometry::Sphere ||
                b.geometry==Geometry::Sphere;
            const bool convex=a.geometry==Geometry::Convex ||
                b.geometry==Geometry::Convex;
            // Finite cooked convex pairs enter Gu GJK/EPA on the GPU.
            // The infinite plane has no GPU broadphase volume. Its
            // paired box already has a resident Sc ElementSim transform.
            if((plane && !pcm_plane_box_enabled_) ||
               (!plane && !pcm_box_box_enabled_))continue;
            const EdgeIndex edge=active_edge_handles_[i];
            if(dormant_pcm(pair,edge))continue;
            if(plane) {
                const Shape& floor=a.geometry==Geometry::Plane?a:b;
                const Shape& box=a.geometry==Geometry::Plane?b:a;
                if(box.motion!=Motion::Dynamic)
                    throw std::logic_error(
                        "PCM plane-box requires the admitted dynamic box");
                gpu_pcm_pairs_.push_back({
                    a.geometry==Geometry::Plane?pair.a:pair.b,
                    a.geometry==Geometry::Plane?pair.b:pair.a,
                    edge,(!island_edges_[edge].pcm_cache.initialized ||
                          !island_edges_[edge].pcm_cache.device_history),
                    sphere?1u:16u,true,floor.position.y,
                    a.contact_offset+b.contact_offset,previously_touching,true});
            }else {
                const bool swapped=island_edges_[edge].gu_order_reversed;
                gpu_pcm_pairs_.push_back({
                    swapped?pair.b:pair.a,swapped?pair.a:pair.b,edge,
                    (!island_edges_[edge].pcm_cache.initialized ||
                     !island_edges_[edge].pcm_cache.device_history),
                    convex?4u:(sphere?1u:16u),false,0.f,0.f,
                    previously_touching,sphere && !convex});
            }
        }
        gpu_pcm_results_.resize(gpu_pcm_pairs_.size());
    }
    const bool source_device_gu_history=source_gpu_pcm &&
        gpu_broadphase_->has_gpu_dy_distance() && !gpu_pcm_pairs_.empty();
    // Admit bounded CPU Gu classifications per manager, not only for
    // primitive-only batches. Select at most four contiguous runs in
    // source pair order to bound GPU upload and readback spans.
    struct HostRun { std::size_t first=0u,end=0u; };
    std::array<HostRun,4> host_runs{};
    std::size_t host_run_count=0u;
    std::size_t host_pair_count=0u;
    if(source_gpu_pcm && gpu_broadphase_->supports_host_pcm_contacts()) {
        const auto eligible=[this](const GpuPcmBoxPair& request) {
            return request.candidate_complete &&
                (request.candidate_capacity==1u ||
                 (request.plane_pair &&
                  request.candidate_capacity==16u &&
                  shapes_[request.second].geometry==Geometry::Box));
        };
        for(std::size_t i=0u;i<gpu_pcm_pairs_.size();) {
            if(!eligible(gpu_pcm_pairs_[i])) {
                ++i;
                continue;
            }
            const std::size_t first=i;
            do { ++i; }while(i<gpu_pcm_pairs_.size() &&
                           eligible(gpu_pcm_pairs_[i]));
            if(host_run_count<host_runs.size())
                host_runs[host_run_count++]={first,i};
            else {
                std::size_t shortest=0u;
                for(std::size_t j=1u;j<host_run_count;++j)
                    if(host_runs[j].end-host_runs[j].first<
                       host_runs[shortest].end-host_runs[shortest].first)
                        shortest=j;
                if(i-first>host_runs[shortest].end-
                           host_runs[shortest].first)
                    host_runs[shortest]={first,i};
            }
        }
        for(std::size_t j=0u;j<host_run_count;++j)
            for(std::size_t i=host_runs[j].first;i<host_runs[j].end;++i) {
                gpu_pcm_pairs_[i].host_generated=true;
                ++host_pair_count;
            }
    }
    const bool gpu_results_pending=
        source_gpu_pcm && host_pair_count<gpu_pcm_pairs_.size();
    if(source_gpu_pcm && gpu_broadphase_->has_gpu_dy_distance() &&
       island_edges_.size()>std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("Sc EdgeIndex watermark exceeds GPU history capacity");
    // General GPU PCM starts before Sc produces any host primitive Gu.
    // Both numerical producers occupy disjoint ordinals in the same
    // contact allocation. The host spans enter the source queue before
    // the GPU witness join, preserving the later device dependency.
    if(gpu_results_pending || (source_gpu_pcm && gpu_pcm_pairs_.empty()))
        gpu_broadphase_->enqueue_gpu_pcm_box_contacts(
            gpu_pcm_pairs_,gpu_pcm_results_,
            gpu_broadphase_->has_gpu_dy_distance()
                ?static_cast<std::uint32_t>(island_edges_.size()):0u);
    // The speculative IslandSim consumes SAP membership, never Gu
    // witness geometry. Admit its first/second-pass topology on the CPU
    // while the same scene stream executes general GPU PCM. Accurate
    // IslandSim admission remains dependent on the real Gu touch result.
    if(gpu_results_pending &&
       (island_topology_dirty_ || !found_.empty() || !lost_.empty()))
        update_island_connectivity(true);
    // The previous source friction patch's relative normal and local
    // anchor separation depend on authored poses, not this frame's Gu.
    // Evaluate those unchanged-input tests while GPU PCM runs. After
    // Gu joins, only the new contact-normal correlation remains.
    if(gpu_results_pending && !friction_patches_.empty()) {
        gpu_pcm_friction_history_masks_.assign(gpu_pcm_pairs_.size(),0u);
        for(std::size_t i=0u;i<gpu_pcm_pairs_.size();++i) {
            const EdgeIndex edge=gpu_pcm_pairs_[i].edge;
            if(edge>=friction_by_edge_.size())continue;
            const Pair pair=island_edges_[edge].shapes;
            const auto& history=friction_by_edge_[edge];
            if(history.epoch!=friction_cache_epoch_ ||
               !history.count || history.pair!=pair ||
               history.offset>friction_patches_.size() ||
               history.count>friction_patches_.size()-history.offset)
                continue;
            if(history.count>32u)
                throw std::logic_error(
                    "Source friction history exceeds patch correlation limit");
            const bool reversed=island_edges_[edge].gu_order_reversed;
            const Shape& shape0=shapes_[reversed?pair.b:pair.a];
            const Shape& shape1=shapes_[reversed?pair.a:pair.b];
            const auto& m0=shape0.material;
            const auto& m1=shape1.material;
            if(m0.disable_friction || m1.disable_friction ||
               m0.disable_strong_friction || m1.disable_strong_friction)
                continue;
            const Actor& actor0=actors_[shape0.body];
            const Actor& actor1=actors_[shape1.body];
            std::uint32_t valid=0u;
            for(std::size_t j=0u;j<history.count;++j)
                if(cached_friction_patch_valid(
                    friction_patches_[history.offset+j],actor0,actor1))
                    valid|=1u<<j;
            gpu_pcm_friction_history_masks_[i]=valid;
        }
    }
    if(host_pair_count) {
        // The same Sc cache owns primitive witnesses even when this
        // frame also schedules general device PCM contacts.
        for(std::size_t i=0u;i<gpu_pcm_pairs_.size();++i) {
            const auto& request=gpu_pcm_pairs_[i];
            if(!request.host_generated)continue;
            const Shape& first=shapes_[request.first];
            const Shape& second=shapes_[request.second];
            const Shape& sphere=first.geometry==Geometry::Sphere
                ?first:second;
            std::array<ContactPoint,4> points{};
            std::uint8_t count=0u;
            const float distance=first.contact_offset+second.contact_offset;
            if(request.plane_pair &&
               second.geometry==Geometry::Box) {
                // Eight bounded corner tests and the original Sc
                // persistent PCM cache determine the CPU contact run.
                step_checkpoint_.save_edge(*this,request.edge);
                auto& cache=island_edges_[request.edge].pcm_cache;
                // The last GPU-only frame retained its witnesses in
                // device Gu history, not this host cache. Regenerate
                // host primitive witnesses from authoritative geometry.
                if(cache.device_history)cache={};
                count=generate_pcm_plane_box(
                    first,second,distance,cache,points);
                for(std::size_t j=0u;j<count;++j)
                    points[j].normal={
                        -points[j].normal.x,-points[j].normal.y,
                        -points[j].normal.z};
            }else if(request.plane_pair) {
                // GPU plane contacts are already in Sc's final +Y
                // orientation. The CPU-only Gu path reverses twice.
                count=source_pcm_sphere_plane(
                    sphere.position,sphere.radius,first.position.y,
                    distance,points);
            }else if(first.geometry==Geometry::Sphere &&
                     second.geometry==Geometry::Sphere) {
                count=source_pcm_sphere_sphere(
                    first.position,first.radius,second.position,second.radius,
                    distance,points);
            }else {
                const Shape& box=first.geometry==Geometry::Box?first:second;
                count=source_pcm_sphere_box(
                    sphere.position,sphere.radius,box.position,
                    box.orientation,box.half_extents,distance,points);
                if(count && first.geometry==Geometry::Box)
                    points[0].normal={
                        -points[0].normal.x,-points[0].normal.y,
                        -points[0].normal.z};
            }
            auto& result=gpu_pcm_results_[i];
            result={};
            result.edge=request.edge;
            result.set_primitive(count,request.previously_touching);
            if(count) {
                result.normal=points[0].normal;
                for(std::size_t j=0u;j<count;++j)
                    result.points[j]={
                        points[j].position,points[j].separation};
            }
        }
    }
    if(source_gpu_pcm && host_pair_count) {
        if(gpu_results_pending)
            gpu_broadphase_->enqueue_host_pcm_contacts(
                gpu_pcm_pairs_,gpu_pcm_results_);
        else
            gpu_broadphase_->upload_host_pcm_contacts(
                gpu_pcm_pairs_,gpu_pcm_results_,
                gpu_broadphase_->has_gpu_dy_distance()
                    ?static_cast<std::uint32_t>(island_edges_.size()):0u);
    }
    bool gpu_results_joined=!gpu_results_pending;
    std::size_t gpu_pcm_index=0u;
    std::size_t old_cursor = 0;
    std::size_t old_cache_cursor=0;
    std::size_t pair_index = 0;
    for (const Pair pair : previous_pairs_) {
        const EdgeIndex edge=active_edge_handles_[pair_index++];
        // The sorted source contact-manager stream retains its collision
        // axis across successive physical-touch frames.
        while (old_cursor < manifolds_.size() &&
               manifolds_[old_cursor].pair < pair)
            next_lost_contacts_.push_back(manifolds_[old_cursor++].pair);
        const bool previously_touching=
            old_cursor<manifolds_.size() &&
            manifolds_[old_cursor].pair==pair;
        const Shape& a = shapes_[pair.a];
        const Shape& b = shapes_[pair.b];
        const float contact_distance=a.contact_offset+b.contact_offset;
        if(!std::isfinite(contact_distance))
            throw std::overflow_error("Native Gu contact distance overflow");
        ContactManifold manifold{};
        manifold.pair = pair;
        manifold.edge_handle=edge;
        manifold.contact_offset = next_contact_points_.size();
        std::optional<GpuPcmTouchTransition> device_touch;
        if(previously_touching)
            manifold.collision_axis=manifolds_[old_cursor].collision_axis;

        // PxcNpBatch::discreteNarrowPhase sorts geometry types (plane
        // before box) independently of ScShapeInteraction actor order.
        // Gu box/box uses the source-oriented Sc contact manager.
        const bool plane_pair=a.geometry==Geometry::Plane ||
                              b.geometry==Geometry::Plane;
        const bool reversed=!plane_pair &&
            island_edges_[edge].gu_order_reversed;
        const Shape& first=plane_pair
            ? (a.geometry==Geometry::Plane?a:b)
            : (reversed?b:a);
        const Shape& second=plane_pair
            ? (a.geometry==Geometry::Plane?b:a)
            : (reversed?a:b);
        const bool convex_pair=a.geometry==Geometry::Convex ||
                               b.geometry==Geometry::Convex;
        const bool pcm_sphere=!convex_pair &&
            (a.geometry==Geometry::Sphere || b.geometry==Geometry::Sphere);
        if(convex_pair &&
           (plane_pair?!pcm_plane_box_enabled_:!pcm_box_box_enabled_))
            throw std::logic_error(
                "Cooked convex contacts require the source PCM generator");
        const bool pcm_plane=pcm_plane_box_enabled_ && plane_pair && !pcm_sphere;
        const bool pcm_box=pcm_box_box_enabled_ && !plane_pair &&
            (!pcm_sphere || convex_pair);
        const bool gpu_pcm_plane=source_gpu_pcm && plane_pair &&
            pcm_plane_box_enabled_;
        // Gu's PxcCacheLocalContacts pose and collision-axis record
        // belongs only to the non-PCM path. PCM owns persistent
        // witnesses through the same Sc EdgeIndex; do not reconstruct
        // an unused legacy cache for every GPU PCM manager.
        GuCacheRecord cache{};
        const GuCacheRecord* old=nullptr;
        if(!pcm_plane && !pcm_box && !pcm_sphere) {
            while(old_cache_cursor<gu_cache_records_.size() &&
                  gu_cache_records_[old_cache_cursor].pair<pair)
                ++old_cache_cursor;
            if(old_cache_cursor<gu_cache_records_.size() &&
               gu_cache_records_[old_cache_cursor].pair==pair)
                old=&gu_cache_records_[old_cache_cursor];
            cache.pair=pair;
            cache.offset=next_gu_cached_points_.size();
            cache.pose0={first.position,first.orientation};
            cache.pose1={second.position,second.orientation};
            cache.contact_distance=contact_distance;
            cache.collision_axis=old?old->collision_axis:manifold.collision_axis;
            manifold.collision_axis=cache.collision_axis;
        }
        bool cache_hit=false;
        if(pcm_plane && !source_gpu_pcm) {
            step_checkpoint_.save_edge(*this,edge);
            const Shape& plane=a.geometry==Geometry::Plane?a:b;
            const Shape& box=a.geometry==Geometry::Plane?b:a;
            if(box.motion!=Motion::Dynamic)
                throw std::logic_error(
                    "PCM plane-box requires the admitted dynamic box");
            manifold.count=generate_pcm_plane_box(
                plane,box,contact_distance,island_edges_[edge].pcm_cache,
                std::span<ContactPoint>(box_contact_scratch_));
            next_contact_points_.insert(next_contact_points_.end(),
                box_contact_scratch_.begin(),
                box_contact_scratch_.begin()+manifold.count);
        }
        if(pcm_box || (source_gpu_pcm &&
                       (pcm_plane ||
                        (pcm_sphere && (plane_pair
                            ?pcm_plane_box_enabled_:pcm_box_box_enabled_))))) {
            if(source_gpu_pcm && dormant_pcm(pair,edge)) {
                // An already-touching manager retains its original Gu
                // witnesses. A previously resolved non-touching pair
                // remains absent, with no synthetic manifold or event.
                if(previously_touching) {
                    const auto& prior=manifolds_[old_cursor];
                    manifold.count=prior.count;
                    next_contact_points_.insert(next_contact_points_.end(),
                        prior.points.begin(),prior.points.end());
                }
            }else {
                step_checkpoint_.save_edge(*this,edge);
                if(source_gpu_pcm) {
                    if(gpu_pcm_index>=gpu_pcm_results_.size())
                        throw std::logic_error(
                            "Source GPU PCM result lost Sc EdgeIndex order");
                    // CPU Gu managers preceding the first general pair
                    // publish while GPU PCM runs on the same source queue.
                    // Only a GPU-owned numerical witness requires the join.
                    if(!gpu_results_joined &&
                       !gpu_pcm_pairs_[gpu_pcm_index].host_generated) {
                        gpu_broadphase_->finish_gpu_pcm_box_contacts();
                        gpu_results_joined=true;
                    }
                    if(gpu_pcm_results_[gpu_pcm_index].edge!=edge)
                        throw std::logic_error(
                            "Source GPU PCM result lost Sc EdgeIndex order");
                    const auto& result=gpu_pcm_results_[gpu_pcm_index];
                    if(result.status()!=0u || result.count()>4u)
                        throw std::logic_error(
                            "Source GPU PCM returned an invalid Sc manifold");
                    // The source Gu arena is the only published contact
                    // storage. Decode device witnesses into that arena,
                    // not into a temporary four-contact buffer first.
                    const auto offset=next_contact_points_.size();
                    next_contact_points_.resize(offset+result.count());
                    manifold.pcm_result_index=
                        static_cast<std::uint32_t>(gpu_pcm_index++);
                    device_touch=result.touch_transition();
                    // Empty Gu results have no backing point array.
                    // Never do pointer arithmetic on a null arena data().
                    std::span<ContactPoint> contacts{};
                    if(result.count())
                        contacts={next_contact_points_.data()+offset,
                                  result.count()};
                    manifold.count=publish_gpu_pcm_box_box(
                        result,island_edges_[edge].pcm_cache,contacts);
                    island_edges_[edge].pcm_cache.device_history=
                        !gpu_pcm_pairs_[manifold.pcm_result_index].host_generated;
                }else {
                    manifold.count=convex_pair
                        ?generate_pcm_convex_pair(
                            first,second,contact_distance,
                            island_edges_[edge].pcm_cache,
                            std::span<ContactPoint>(box_contact_scratch_))
                        :generate_pcm_box_box(
                            first,second,contact_distance,
                            island_edges_[edge].pcm_cache,
                            std::span<ContactPoint>(box_contact_scratch_));
                    next_contact_points_.insert(next_contact_points_.end(),
                        box_contact_scratch_.begin(),
                        box_contact_scratch_.begin()+manifold.count);
                }
            }
        }
        if(!pcm_plane && !pcm_box && !pcm_sphere && old &&
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
        if(pcm_sphere &&
           (!source_gpu_pcm || (plane_pair && !gpu_pcm_plane) ||
            (!plane_pair && !pcm_box_box_enabled_))) {
            const bool use_pcm=plane_pair
                ? pcm_plane_box_enabled_:pcm_box_box_enabled_;
            std::array<ContactPoint,1> output{};
            const Shape& sphere=a.geometry==Geometry::Sphere?a:b;
            if(plane_pair) {
                const Shape& plane=a.geometry==Geometry::Plane?a:b;
                manifold.count=source_pcm_sphere_plane(
                    sphere.position,sphere.radius,plane.position.y,
                    contact_distance,output);
                // Sc's plane stream flips the raw Gu orientation below.
                if(manifold.count) {
                    output[0].normal={
                        -output[0].normal.x,-output[0].normal.y,
                        -output[0].normal.z};
                }
            }else if(a.geometry==Geometry::Sphere &&
                     b.geometry==Geometry::Sphere) {
                manifold.count=use_pcm
                    ? source_pcm_sphere_sphere(
                        first.position,first.radius,
                        second.position,second.radius,
                        contact_distance,output)
                    : source_gu_sphere_sphere(
                        first.position,first.radius,
                        second.position,second.radius,
                        contact_distance,output);
            }else {
                const Shape& box=a.geometry==Geometry::Box?a:b;
                manifold.count=use_pcm
                    ? source_pcm_sphere_box(
                        sphere.position,sphere.radius,
                        box.position,box.orientation,box.half_extents,
                        contact_distance,output)
                    : source_gu_sphere_box(
                        sphere.position,sphere.radius,
                        box.position,box.orientation,box.half_extents,
                        contact_distance,output);
                if(manifold.count && first.geometry==Geometry::Box) {
                    output[0].normal={
                        -output[0].normal.x,-output[0].normal.y,
                        -output[0].normal.z};
                }
            }
            if(manifold.count)
                next_contact_points_.push_back(output[0]);
        }
        if(!pcm_plane && !pcm_box && !pcm_sphere && !cache_hit && (a.geometry == Geometry::Plane || b.geometry == Geometry::Plane)) {
            // Direct source order/geometry from
            // GuContactPlaneBox.cpp::contactPlaneBox. The SDK plane's
            // logical normal is +Y, corresponding to PhysX plane-local
            // +X. This computes boxToPlane.m.column[k].x times each
            // half-extent, constructs the eight source depth tests, and
            // publishes world-space rotated vertices. Do not replace
            // this with identity-only AABB corners.
            const Shape& plane = a.geometry == Geometry::Plane ? a : b;
            const Shape& box = a.geometry == Geometry::Plane ? b : a;
            if (box.motion != Motion::Dynamic)
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
        } else if(!pcm_plane && !pcm_box && !pcm_sphere && !cache_hit) {
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
        if(!pcm_plane && !pcm_box && !pcm_sphere && !cache_hit) {
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
        if(!pcm_plane && !pcm_box && !pcm_sphere)
            next_gu_cache_records_.push_back(cache);
        // PxcNpBatch::flipContacts runs AFTER PxcCacheLocalContacts.
        // The cached points above must retain the raw plane-first Gu
        // normal, so a subsequent cache hit's motion.dot(normal) keeps
        // the source separation sign. ScShapeInteraction orders the
        // dynamic body before the plane regardless of creation order;
        // only its published contact stream receives the flip.
        // The device PCM stream is already in Sc/Dy normal orientation;
        // CPU Gu still requires its original plane-first reversal.
        if(plane_pair && !gpu_pcm_plane) {
            for(std::size_t k=0;k<manifold.count;++k) {
                Vec3& n=next_contact_points_[manifold.contact_offset+k].normal;
                n={-n.x,-n.y,-n.z};
            }
        }
        // Sc's sorted Gu contact-manager traversal already compares
        // this frame against the preceding physical-touch state. Emit
        // each transition here, without a second manifold merge pass.
        if(device_touch) {
            const auto transition=*device_touch;
            // Source Sc's prior publication is an independent guard
            // against a stale or recycled GPU EdgeIndex history.
            const bool expected_found=manifold.count>0u && !previously_touching;
            const bool expected_persists=manifold.count>0u && previously_touching;
            const bool expected_lost=manifold.count==0u && previously_touching;
            if((transition==GpuPcmTouchTransition::Found)!=expected_found ||
               (transition==GpuPcmTouchTransition::Persists)!=expected_persists ||
               (transition==GpuPcmTouchTransition::Lost)!=expected_lost)
                throw std::logic_error(
                    "Source GPU Sc touch transition disagrees with edge lifetime");
            if(transition==GpuPcmTouchTransition::Found)
                next_found_contacts_.push_back(pair);
            else if(transition==GpuPcmTouchTransition::Lost)
                next_lost_contacts_.push_back(pair);
        } else if(manifold.count>0u && !previously_touching) {
            next_found_contacts_.push_back(pair);
        } else if(manifold.count==0u && previously_touching) {
            next_lost_contacts_.push_back(pair);
        }
        if(manifold.count>0u) {
            if(source_device_gu_history &&
               manifold.pcm_result_index==
                   std::numeric_limits<std::uint32_t>::max())
                gpu_dy_gu_touches_.push_back({
                    edge,previously_touching?3u:1u});
            next_manifold_by_edge_[edge]={
                next_manifolds_.size(),new_lookup_generation};
            next_manifolds_.push_back(manifold);
        }
        if(previously_touching)++old_cursor;
    }
    if(gpu_results_pending && !gpu_results_joined)
        throw std::logic_error(
            "Source GPU PCM stream had no matching Sc join");
    if(source_gpu_pcm && gpu_pcm_index!=gpu_pcm_results_.size())
        throw std::logic_error(
            "Source GPU PCM stream left an active Sc contact manager unconsumed");
    // Broadphase-retired pairs and zero-contact pairs still lose their
    // accurate Sc touch, independently of speculative SAP membership.
    while(old_cursor<manifolds_.size())
        next_lost_contacts_.push_back(manifolds_[old_cursor++].pair);
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
            island_edges_[id].connected=true;
        }
    }
    for(Pair pair:next_lost_contacts_) {
        const auto id=contact_edge_index(pair);
        if(id!=invalid_edge) {
            unlink_contact_edge(id,true);
            step_checkpoint_.save_edge(*this,id);
            island_edges_[id].connected=false;
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
    auto& route=accurate?accurate_route_scratch_:speculative_route_scratch_;
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
    route.seen.resize(actors_.size(),0);
    route.predecessor.resize(actors_.size(),invalid);
    route.predecessor_edge.resize(actors_.size(),invalid_edge);
    // PhysX mVisitedState is shared by every dirty-node search in this
    // third-pass IslandSim, so nodes assigned to an earlier split are
    // never searched again against their newly promoted root. Preserve
    // the worklist witness state without clearing O(all actors) per pass.
    route.confirmed.resize(actors_.size(),0);
    if(++route.confirmed_stamp==0) {
        std::fill(route.confirmed.begin(),
                  route.confirmed.end(),0);
        route.confirmed_stamp=1;
    }
    const auto confirmed_stamp=route.confirmed_stamp;
    bool split_occurred=false;
    std::size_t dirty_route_checks=0, fast_route_hits=0;

    // A cached source EdgeIndex is stable until retirement, and never
    // aliases an unrelated route just because its slot is later reused:
    // both endpoints and this graph's intrusive instances must still
    // match. One edge-slot check per hop replaces linear neighbor scans.
    const auto live_hop=[&](BodyId from,BodyId to,EdgeIndex id) {
        if(id==invalid_edge || id>=island_edges_.size())return false;
        const auto& edge=island_edges_[id];
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
        if(!actors_[node].alive || actors_[node].motion!=Motion::Dynamic ||
           route.confirmed[node]==confirmed_stamp)
            continue;
        ++dirty_route_checks;
        const BodyId root=root_for(node);
        if(root==node) {
            route.confirmed[node]=confirmed_stamp;
            continue;
        }

        // Source tryFastPath: only accept a cached path when every
        // individual link is still present and hop estimates decrease.
        // Never trust the cache merely because two nodes retain the
        // same pre-split island identifier.
        BodyId probe=node;
        bool fast_found=false;
        route.split_members.clear();
        for(std::size_t depth=0;depth<actors_.size();++depth) {
            if(probe==root ||
               (route.confirmed[probe]==confirmed_stamp &&
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
            route.split_members.push_back(probe);
            probe=next;
        }
        if(fast_found) {
            ++fast_route_hits;
            // Source tryFastPath retains valid witnesses along every
            // root-directed step, not just at the starting dirty node.
            for(BodyId visited:route.split_members)
                route.confirmed[visited]=confirmed_stamp;
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
        if(++route.visit_stamp==0) {
            std::fill(route.seen.begin(),route.seen.end(),0);
            route.visit_stamp=1;
        }
        const auto visit_stamp=route.visit_stamp;
        route.queue.clear();
        route.split_members.clear();
        route.split_members.push_back(node);
        source_hop_push(route.queue,node,higher);
        route.seen[node]=visit_stamp;
        route.predecessor[node]=invalid;
        route.predecessor_edge[node]=invalid_edge;
        bool found=false;
        BodyId witness=root;
        // The reusable queue is a min-heap by estimated hop count.
        // The source's Cm::PriorityQueue has no secondary tie-break:
        // equal-hop promotion follows its exact binary-heap child rules.
        while(!route.queue.empty() && !found) {
            const BodyId from=source_hop_pop(route.queue,higher);
            for(std::size_t i=first[from];i!=invalid_instance;) {
                const IslandEdge& edge=island_edges_[i/2];
                const auto& link=(accurate?edge.accurate:
                                  edge.speculative)[i&1];
                const BodyId to=(i&1)?edge.pair.a:edge.pair.b;
                const std::size_t i_edge=i/2;
                i=link.next;
                if(!actors_[to].alive ||
                   actors_[to].motion!=Motion::Dynamic ||
                   route.seen[to]==visit_stamp)
                    continue;
                route.seen[to]=visit_stamp;
                route.predecessor[to]=from;
                route.predecessor_edge[to]=
                    static_cast<EdgeIndex>(i_edge);
                // Source findRoute can terminate on an earlier pass's
                // visited, confirmed island member as well as the root.
                // This reuses mVisitedState's witness instead of
                // rediscovering its complete fast route in this search.
                if(to==root ||
                   (route.confirmed[to]==confirmed_stamp &&
                    graph.node_island[to]==graph.node_island[node])) {
                    found=true;
                    witness=to;
                    break;
                }
                // mVisitedNodes is ordered by queue insertion, not pop.
                // A failed search has already collected the entire
                // disconnected component in its exact source visit order.
                route.split_members.push_back(to);
                source_hop_push(route.queue,to,higher);
            }
        }
        if(!found) {
            split_lost_island(accurate,node);
            for(BodyId member:route.split_members)
                route.confirmed[member]=confirmed_stamp;
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
            const BodyId predecessor=route.predecessor[current];
            step_checkpoint_.save_hint(*this,predecessor,accurate);
            fast[predecessor]=current;
            fast_edge[predecessor]=route.predecessor_edge[current];
            hops[predecessor]=hops[current]+1;
            stamps[predecessor]=epoch;
            route.confirmed[predecessor]=confirmed_stamp;
            current=predecessor;
        }
        for(BodyId visited:route.split_members) {
            if(route.confirmed[visited]==confirmed_stamp)
                continue;
            const BodyId predecessor=route.predecessor[visited];
            if(predecessor==invalid)
                throw std::logic_error(
                    "Source successful route lost its traversal parent");
            step_checkpoint_.save_hint(*this,visited,accurate);
            fast[visited]=predecessor;
            fast_edge[visited]=route.predecessor_edge[visited];
            hops[visited]=hops[predecessor]+1;
            stamps[visited]=epoch;
            route.confirmed[visited]=confirmed_stamp;
        }
    }
    // Aggregate after this task's route search, not per visited node.
    std::atomic_ref(island_dirty_route_checks_)
        .fetch_add(dirty_route_checks,std::memory_order_relaxed);
    std::atomic_ref(island_fast_route_hits_)
        .fetch_add(fast_route_hits,std::memory_order_relaxed);
    return !split_occurred;
}

// PxsIslandSim.cpp::processLostEdges: after failed findRoute, migrate
// precisely the visited component out of its previous island. The
// retained root's handle and all unrelated island chains remain intact.
// The source's heap insertion order determines the new node chain and
// indexed contact-manager splitEdges order. This is the admitted rigid
// graph; kinematic witnesses and bounded dirty budgets remain separate.
void Scene::split_lost_island(bool accurate,BodyId seed) {
    auto& route=accurate?accurate_route_scratch_:speculative_route_scratch_;
    auto& graph=accurate?accurate_islands_:speculative_islands_;
    auto& hops=accurate?accurate_hops_:speculative_hops_;
    auto& routes=accurate?accurate_fast_route_:speculative_fast_route_;
    auto& route_edges=accurate?accurate_fast_edge_:speculative_fast_edge_;
    auto& stamps=accurate?accurate_hint_stamps_:speculative_hint_stamps_;
    const auto epoch=accurate?accurate_hint_epoch_:speculative_hint_epoch_;
    const BodyId invalid=std::numeric_limits<BodyId>::max();
    if(!actors_[seed].alive ||
       actors_[seed].motion!=Motion::Dynamic)
        throw std::logic_error("Source split seed is not dynamic");
    const auto owner=graph.node_island[seed];
    if(owner==IslandGraph::invalid || !graph.islands[owner].live)
        throw std::logic_error("Source split seed lacks an island");
    const BodyId root=graph.islands[owner].root;
    // findRoute has ALREADY visited every reachable node and found no
    // route to root. Reuse its mVisitedNodes and parent/depth records:
    // repeating the heap traversal wastes work and can change ordering.
    if(root==seed || route.split_members.empty() ||
       route.split_members.front()!=seed ||
       route.seen[root]==route.visit_stamp ||
       route.split_members.size()>=graph.islands[owner].count)
        throw std::logic_error("Source failed-route split invariant lost");
    step_checkpoint_.save_graph_ownership(*this,accurate);

        // Source removeEdgeFromIsland uses swap-last on the old indexed
        // contact-manager array. Remove exactly once, in visited-node
        // adjacency order, before changing each node's island identity.
        route.split_edges.clear();
        if(accurate) {
            for(BodyId node:route.split_members)
                for(std::size_t i=accurate_node_first_[node];
                    i!=invalid_instance;) {
                    const EdgeIndex edge_index=
                        static_cast<EdgeIndex>(i/2);
                    const auto& edge=island_edges_[edge_index];
                    const auto& link=edge.accurate[i&1];
                    i=link.next;
                    if(!edge.connected ||
                       (node!=edge.pair.a &&
                        actors_[edge.pair.a].motion==Motion::Dynamic))
                        continue;
                    remove_accurate_island_edge(edge_index);
                    route.split_edges.push_back(edge_index);
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
        for(BodyId node:route.split_members) {
            const BodyId before=graph.node_prev[node];
            const BodyId after=graph.node_next[node];
            if(before!=invalid)graph.node_next[before]=after;
            else graph.islands[owner].root=after;
            if(after!=invalid)graph.node_prev[after]=before;
            else graph.islands[owner].tail=before;
            --graph.islands[owner].count;
            --graph.islands[owner].node_type_counts[
                static_cast<std::size_t>(graph.node_type[node])];
            const auto count=graph.node_static_touches[node];
            static_touches+=count;
            graph.islands[owner].static_touches-=count;
        }

        auto& island=graph.islands[created];
        island={};
        island.live=true;
        island.root=seed;
        island.tail=route.split_members.back();
        island.count=route.split_members.size();
        for(BodyId member:route.split_members)
            ++island.node_type_counts[
                static_cast<std::size_t>(graph.node_type[member])];
        island.static_touches=static_touches;
        // PhysX processLostEdges creates an active detached island when
        // the original island was active. Preserve this event order now,
        // before another split can consume a recycled island handle.
        if(graph.islands[owner].active_index!=invalid_instance) {
            island.active_index=graph.active_islands.size();
            graph.active_islands.push_back(created);
        }
        for(std::size_t i=0;i<route.split_members.size();++i) {
            const BodyId node=route.split_members[i];
            const BodyId previous=i?route.split_members[i-1]:invalid;
            const BodyId next=i+1<route.split_members.size()
                ?route.split_members[i+1]:invalid;
            graph.node_prev[node]=previous;
            graph.node_next[node]=next;
            graph.node_island[node]=created;
            step_checkpoint_.save_hint(*this,node,accurate);
            routes[node]=previous==invalid?invalid:
                route.predecessor[node];
            route_edges[node]=previous==invalid?invalid_edge:
                route.predecessor_edge[node];
            hops[node]=previous==invalid?0:
                hops[route.predecessor[node]]+1;
            stamps[node]=epoch;
        }
        // Source splitEdges appends the collected edge sequence,
        // not an order reconstructed from canonical Gu pairs.
        if(accurate)
            for(EdgeIndex id:route.split_edges)
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
        std::atomic_ref(island_local_split_count_)
            .fetch_add(1,std::memory_order_relaxed);
}

void Scene::update_island_connectivity(bool speculative_only) {
    const bool speculative_pass=
        speculative_only || !speculative_connectivity_prepared_;
    const bool accurate_pass=!speculative_only;
    if(speculative_only)speculative_connectivity_prepared_=true;
    // PxsIslandSim::IslandSim's node-to-island IDs and persistent island
    // handle slots are now the authoritative connectivity owner. The
    // speculative and accurate graphs share source contact-manager edge
    // slots, not a second actor registry or a provisional union/find map.
    const BodyId invalid=std::numeric_limits<BodyId>::max();
    using Graph=IslandGraph;
    const auto dynamic_node=[&](BodyId node) {
        return actors_[node].alive && actors_[node].motion==Motion::Dynamic;
    };
    const auto dynamic_edge=[&](Pair pair) {
        return dynamic_node(pair.a) && dynamic_node(pair.b);
    };

    const bool rebuild=island_topology_dirty_;
    if(!rebuild) {
        // IslandSim::addConnectionToGraph/removeConnectionInternal update
        // per-node and per-island static-touch counts without merging
        // dynamic islands. SAP candidates and Gu touches are independent.
        // Rebuilds instead count the final intrusive static links once.
        const auto change_static=[&](Graph& graph,Pair pair,bool added) {
            const bool da=dynamic_node(pair.a);
            const bool db=dynamic_node(pair.b);
            if(da==db)return;
            // Source PxNodeIndex::isStaticBody is an INVALID endpoint,
            // never a kinematic rigid node with its own EdgeInstance.
            if(rigid_node(actors_[da?pair.b:pair.a].motion))return;
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
        if(speculative_pass) {
            for(Pair pair:found_)
                change_static(speculative_islands_,body_pair(pair),true);
            for(Pair pair:lost_)
                change_static(speculative_islands_,body_pair(pair),false);
        }
        if(accurate_pass) {
            for(Pair pair:found_contacts_)
                change_static(accurate_islands_,body_pair(pair),true);
            for(Pair pair:lost_contacts_)
                change_static(accurate_islands_,body_pair(pair),false);
        }
    }
    if(!rebuild) {
        // No newly connected dynamic edges means no island membership,
        // handle or solver batch change. Preserve the source's quiet
        // publication contract without touching island revision/epochs.
        bool dynamic_found=false;
        if(speculative_pass)
            for(Pair pair:found_)
                if(dynamic_edge(body_pair(pair))) {
                    dynamic_found=true;
                    break;
                }
        if(accurate_pass && !dynamic_found)
            for(Pair pair:found_contacts_)
                if(dynamic_edge(body_pair(pair))) {
                    dynamic_found=true;
                    break;
                }
        // The accurate completion also publishes the earlier speculative
        // merge's single topology revision even if Gu touches are quiet.
        if(accurate_pass && speculative_connectivity_prepared_ &&
           !dynamic_found)
            for(Pair pair:found_)
                if(dynamic_edge(body_pair(pair))) {
                    dynamic_found=true;
                    break;
                }
        if(!dynamic_found) {
            // A static-only Gu touch still joins its existing accurate
            // island's contact list, but not the speculative graph.
            if(accurate_pass)
                for(Pair shapes:found_contacts_) {
                    const Pair pair=body_pair(shapes);
                    const auto id=contact_edge_index(shapes);
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
    const auto merge=[&](Graph& graph,Pair pair,EdgeIndex connecting_edge) {
        if(!dynamic_edge(pair))return false;
        const bool accurate=&graph==&accurate_islands_;
        auto& hops=accurate?accurate_hops_:speculative_hops_;
        auto& fast=accurate?accurate_fast_route_:speculative_fast_route_;
        auto& fast_edge=accurate?accurate_fast_edge_:speculative_fast_edge_;
        auto& stamps=accurate?accurate_hint_stamps_:speculative_hint_stamps_;
        const auto epoch=accurate?accurate_hint_epoch_:speculative_hint_epoch_;
        const auto a=graph.node_island[pair.a];
        const auto b=graph.node_island[pair.b];
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
        for(std::size_t type=0;type<island_node_type_count;++type)
            winner.node_type_counts[type]+=loser.node_type_counts[type];
        winner.static_touches+=loser.static_touches;
        if(&graph==&accurate_islands_) {
            // IslandSim::mergeEdges concatenates each source edge-type
            // list independently; constraint edges never become Gu rows.
            for(const bool constraint:{false,true}) {
                auto& wf=constraint?winner.first_constraint:winner.first_contact;
                auto& wl=constraint?winner.last_constraint:winner.last_contact;
                auto& wc=constraint?winner.constraint_count:winner.contact_count;
                const EdgeIndex lf=constraint?loser.first_constraint:
                                              loser.first_contact;
                const EdgeIndex ll=constraint?loser.last_constraint:
                                              loser.last_contact;
                const std::size_t lc=constraint?loser.constraint_count:
                                               loser.contact_count;
                if(lf==invalid_edge)continue;
                if(wl!=invalid_edge) {
                    step_checkpoint_.save_edge(*this,wl);
                    step_checkpoint_.save_edge(*this,lf);
                    island_edges_[wl].island_next=lf;
                    island_edges_[lf].island_prev=wl;
                } else wf=lf;
                wl=ll;
                wc+=lc;
            }
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
        graph.node_type.resize(actors_.size(),IslandNodeType::RigidBody);
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
                graph.islands[handle].node_type_counts[
                    static_cast<std::size_t>(graph.node_type[id])]=1;
            } else {
                if(graph.islands.size()>=Graph::invalid)
                    throw std::overflow_error("Source island handles exhausted");
                handle=static_cast<Graph::IslandId>(graph.islands.size());
                graph.islands.push_back({id,id,1,true});
                graph.islands.back().node_type_counts[
                    static_cast<std::size_t>(graph.node_type[id])]=1;
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
        next.node_type=graph.node_type;
        next.node_type.resize(actors_.size(),IslandNodeType::RigidBody);
        next.node_next.assign(actors_.size(),invalid);
        next.node_prev.assign(actors_.size(),invalid);
        next.node_static_touches.assign(actors_.size(),0);
        // Source-owned active-index slots and third-pass output survive
        // on the authoritative graph. Rebuild prepares enough capacity
        // for no-throw deactivation after a geometry-only publication.
        next.active_node_index.assign(actors_.size(),invalid_instance);
        next.active_nodes.reserve(actors_.size());
        next.active_kinematic_index.assign(actors_.size(),invalid_instance);
        next.active_kinematic_nodes.reserve(actors_.size());
        next.active_islands.reserve(actors_.size());
        next.activating_nodes=graph.activating_nodes;
        next.activating_nodes.reserve(actors_.size());
        next.activating_seen.assign(actors_.size(),0);
        next.active_island_seen.resize(next.islands.size(),0);
        next.nodes_to_deactivate.reserve(actors_.size());
        // Full admission/retirement rebuilds replace graph handles but
        // do not recreate contact managers or reset their active flags.
        next.active_edge_count=graph.active_edge_count;
        next.active_constraint_edge_count=graph.active_constraint_edge_count;
        // The original EdgeIndex slots, not reconstructed island
        // membership, determine each node's active references.
        next.active_reference_counts=graph.active_reference_counts;
        next.active_reference_counts.resize(actors_.size(),0);
        // Only structural rebuilds scan actor IDs. Preserve prior
        // mActiveKinematicNodes order, admitting new referenced targets.
        const auto add_active_kinematic=[&](BodyId id) {
            if(id>=actors_.size() || !actors_[id].alive ||
               actors_[id].motion!=Motion::Kinematic ||
               next.active_kinematic_index[id]!=invalid_instance)return;
            const auto& actor=actors_[id];
            if(!next.active_reference_counts[id] &&
               !actor.kinematic_target_pending &&
               !actor.kinematic_settling)return;
            next.active_kinematic_index[id]=next.active_kinematic_nodes.size();
            next.active_kinematic_nodes.push_back(id);
        };
        for(BodyId id:graph.active_kinematic_nodes)
            add_active_kinematic(id);
        for(std::size_t i=0;i<actors_.size();++i)
            add_active_kinematic(static_cast<BodyId>(i));
        next.activated_edges.reserve(island_edges_.size());
        next.deactivating_edges.reserve(island_edges_.size());
        next.activated_constraint_edges.reserve(island_edges_.size());
        next.deactivating_constraint_edges.reserve(island_edges_.size());
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
                    const auto& edge=island_edges_[i/2];
                    const auto& link=(accurate?edge.accurate:
                                      edge.speculative)[i&1];
                    const BodyId to=(i&1)?edge.pair.a:edge.pair.b;
                    i=link.next;
                    if(!dynamic_node(to)) {
                        // In each graph a static contact manager has one
                        // intrusive instance on its dynamic endpoint.
                        if(actors_[to].alive &&
                           !rigid_node(actors_[to].motion))
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
            for(BodyId member:members)
                ++island.node_type_counts[
                    static_cast<std::size_t>(next.node_type[member])];
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
                        const IslandEdge& edge=island_edges_[i/2];
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
            migrated.reserve(manifolds_.size()+active_constraint_handles_.size());
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
                    const auto& edge=island_edges_[edge_index];
                    const auto& link=edge.accurate[i&1];
                    i=link.next;
                    if(!edge.connected)continue;
                    // The first endpoint owns dynamic/dynamic entries.
                    // A world/static first endpoint cannot be visited,
                    // so admit that entry from its only dynamic endpoint.
                    const BodyId first_endpoint=edge.pair.a;
                    if((node!=first_endpoint && dynamic_node(first_endpoint)) ||
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
            retained.reserve(manifolds_.size()+active_constraint_handles_.size());
            for(const auto& island:graph.islands)
            for(EdgeIndex head:{island.first_contact,island.first_constraint})
                for(EdgeIndex id=head;id!=invalid_edge;
                    id=island_edges_[id].island_next) {
                    const Pair pair=island_edges_[id].pair;
                    const BodyId node=dynamic_node(pair.a)?pair.a:pair.b;
                    if(!dynamic_node(node) || !island_edges_[id].connected)
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
                island_edges_[item.edge].island_prev=invalid_edge;
                island_edges_[item.edge].island_next=invalid_edge;
                // The graph-owned island list was moved; the same
                // source slot must be reinserted into its new owner.
                island_edges_[item.edge].indexed_accurate=false;
            }
            for(const auto& item:retained)
                append_accurate_island_edge(item.edge,item.owner);
            for(const auto& item:migrated)
                append_accurate_island_edge(item.edge,item.owner);
        } else graph=std::move(next);
        return islands;
    };


    bool order_changed=rebuild && accurate_pass;
    if(rebuild) {
        step_checkpoint_.save_graph_ownership(*this);
        if(speculative_pass)++island_full_rebuild_count_;
        if(island_topology_dirty_) {
            // Existing source islands still own pre-loss membership.
            // Admit newly added nodes, then merge *all* found edges
            // against accumulated island sizes, prior to any split.
            if(speculative_pass) {
                admit_nodes(speculative_islands_);
                for(Pair pair:found_)
                    merge(speculative_islands_,body_pair(pair),contact_edge_index(pair));
                for(EdgeIndex id:active_constraint_handles_)
                    if(island_edges_[id].live && !island_edges_[id].retiring)
                        merge(speculative_islands_,island_edges_[id].pair,id);
            }
            if(accurate_pass) {
                admit_nodes(accurate_islands_);
                for(Pair shapes:found_contacts_) {
                    const Pair pair=body_pair(shapes);
                    merge(accurate_islands_,pair,contact_edge_index(shapes));
                    const auto id=contact_edge_index(shapes);
                    const BodyId node=dynamic_node(pair.a)?pair.a:pair.b;
                    if(id!=invalid_edge && dynamic_node(node))
                        append_accurate_island_edge(
                            id,accurate_islands_.node_island[node]);
                }
                for(EdgeIndex id:active_constraint_handles_)
                    if(island_edges_[id].live && !island_edges_[id].retiring) {
                        const Pair pair=island_edges_[id].pair;
                        merge(accurate_islands_,pair,id);
                        const BodyId node=dynamic_node(pair.a)?pair.a:pair.b;
                        if(dynamic_node(node)) {
                            if(!island_edges_[id].indexed_accurate)
                                append_accurate_island_edge(
                                    id,accurate_islands_.node_island[node]);
                        }
                    }
            }
        }
        // Reconcile lost edges and actor retirement with the authoritative
        // intrusive graph. Already-selected source handles survive on the
        // connected component that retains their original root.
        if(speculative_pass)
            speculative_island_count_=rebuild_graph(speculative_islands_,false);
        if(accurate_pass)
            accurate_island_count_=rebuild_graph(accurate_islands_,true);
        // Reclassified rigid bodies may reconnect an existing Gu manager
        // without emitting a new touch-found report. Index those live
        // managers in their newly admitted dynamic island after the
        // original full node/edge reconciliation.
        if(accurate_pass && gpu_broadphase_)
            for(EdgeIndex edge_id:active_edge_handles_) {
                const auto& edge=island_edges_[edge_id];
                if(!edge.live || !edge.connected || edge.indexed_accurate)continue;
                const BodyId node=dynamic_node(edge.pair.a)
                    ? edge.pair.a:edge.pair.b;
                if(!dynamic_node(node))continue;
                const auto owner=accurate_islands_.node_island[node];
                if(owner!=IslandGraph::invalid)
                    append_accurate_island_edge(edge_id,owner);
            }
        if(accurate_pass) {
            accurate_island_nodes_.clear();
            for(std::size_t i=0;i<actors_.size();++i)
                if(dynamic_node(static_cast<BodyId>(i)))
                    accurate_island_nodes_.push_back(static_cast<BodyId>(i));
        }
    } else {
        if(speculative_pass)
            for(Pair pair:found_)
                if(merge(speculative_islands_,body_pair(pair),contact_edge_index(pair)))
                    --speculative_island_count_;
        if(accurate_pass)
            for(Pair shapes:found_contacts_) {
                const Pair pair=body_pair(shapes);
                if(merge(accurate_islands_,pair,contact_edge_index(shapes))) {
                    --accurate_island_count_;
                    order_changed=true;
                }
                const auto id=contact_edge_index(shapes);
                const BodyId node=dynamic_node(pair.a)?pair.a:pair.b;
                if(id!=invalid_edge && dynamic_node(node))
                    append_accurate_island_edge(
                        id,accurate_islands_.node_island[node]);
            }
    }
    if(order_changed)rebuild_accurate_island_offsets();
    // Reserve each graph's wake/deactivation storage when its own
    // first/second-pass source admission finishes.
    if(speculative_pass) {
        speculative_islands_.active_island_seen.resize(
            speculative_islands_.islands.size(),0);
        speculative_pre_retirement_index_.resize(
            speculative_islands_.islands.size());
        speculative_islands_.active_islands.reserve(actors_.size());
        speculative_islands_.activating_nodes.reserve(actors_.size());
        speculative_islands_.activating_seen.resize(actors_.size(),0);
    }
    if(accurate_pass) {
        accurate_islands_.active_island_seen.resize(
            accurate_islands_.islands.size(),0);
        accurate_islands_.active_islands.reserve(actors_.size());
        accurate_islands_.activating_nodes.reserve(actors_.size());
        accurate_islands_.activating_seen.resize(actors_.size(),0);
        // Actor/topology journal flags stay open until both IslandSim
        // producers have finished admitting this scene transaction.
        island_topology_dirty_=false;
        ++island_topology_revision_;
    }
}
void Scene::rebuild_accurate_island_offsets() {
    // Only accurate islands own Dy's indexed solver-body batches.
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
bool Scene::repair_lost_island_graph(bool accurate) {
    auto& graph=accurate?accurate_islands_:speculative_islands_;
    auto& dirty=accurate?accurate_dirty_nodes_:speculative_dirty_nodes_;
    if(dirty.empty())return false;
    const bool split=!lost_island_edges_retain_routes(accurate);
    if(split) {
        if(accurate)rebuild_accurate_island_offsets();
        graph.active_island_seen.resize(graph.islands.size(),0);
        if(!accurate)
            speculative_pre_retirement_index_.resize(graph.islands.size());
    }
    auto& dirty_map=accurate?accurate_dirty_map_:speculative_dirty_map_;
    for(BodyId node:dirty)dirty_map[node]=0;
    dirty.clear();
    return split;
}
void Scene::refresh_kinematic_node(BodyId id,bool accurate) noexcept {
    auto& graph=accurate?accurate_islands_:speculative_islands_;
    if(id>=graph.active_kinematic_index.size())return;
    const auto& body=actors_[id];
    const bool active=body.alive && body.motion==Motion::Kinematic &&
        ((id<graph.active_reference_counts.size() &&
          graph.active_reference_counts[id]!=0) ||
         body.kinematic_target_pending || body.kinematic_settling);
    const auto old=graph.active_kinematic_index[id];
    if(active && old==invalid_instance) {
        graph.active_kinematic_index[id]=graph.active_kinematic_nodes.size();
        graph.active_kinematic_nodes.push_back(id);
    } else if(!active && old!=invalid_instance) {
        const BodyId last=graph.active_kinematic_nodes.back();
        graph.active_kinematic_nodes[old]=last;
        graph.active_kinematic_index[last]=old;
        graph.active_kinematic_nodes.pop_back();
        graph.active_kinematic_index[id]=invalid_instance;
    }
}
// PxsIslandSim.cpp::markEdgeActive/removeEdgeFromActivatingList:
// the shared EdgeIndex has independent activity in the two IslandSims.
void Scene::activate_contact_edge(EdgeIndex id, bool accurate) noexcept {
    auto& edge=island_edges_[id];
    if(!edge.live || edge.retiring ||
       (accurate && !edge.connected))return;
    auto& state=accurate?edge.accurate_state:edge.speculative_state;
    if(state&1u)return; // Edge::eACTIVE
    step_checkpoint_.save_activity(*this,id);
    if(rigid_node(actors_[edge.pair.a].motion) &&
       rigid_node(actors_[edge.pair.b].motion)) {
        step_checkpoint_.save_reference(*this,edge.pair.a);
        step_checkpoint_.save_reference(*this,edge.pair.b);
    }
    auto& graph=accurate?accurate_islands_:speculative_islands_;
    // PxsIslandSim indexes activation outputs and live counts by
    // EdgeType. Constraint edges never appear in contact-manager events.
    if(edge.type==IslandEdgeType::Constraint &&
       actors_[edge.pair.a].motion==Motion::Kinematic &&
       actors_[edge.pair.b].motion==Motion::Kinematic)return;
    state|=3u; // Edge::eACTIVE | Edge::eACTIVATING
    if(edge.type==IslandEdgeType::Constraint) {
        graph.activated_constraint_edges.push_back(id);
        ++graph.active_constraint_edge_count;
    } else {
        graph.activated_edges.push_back(id);
        ++graph.active_edge_count;
    }
    // Source IslandSim::markEdgeActive counts references only for
    // edges with two valid node endpoints. Static world edges do not
    // contribute to either rigid node's active reference count.
    if(rigid_node(actors_[edge.pair.a].motion) &&
       rigid_node(actors_[edge.pair.b].motion)) {
        ++graph.active_reference_counts[edge.pair.a];
        ++graph.active_reference_counts[edge.pair.b];
        refresh_kinematic_node(edge.pair.a,accurate);
        refresh_kinematic_node(edge.pair.b,accurate);
    }
}
void Scene::deactivate_contact_edge(EdgeIndex id, bool accurate,
                                    bool publish) noexcept {
    // Both IslandSims can finish their third passes concurrently.
    // The checkpoint captures BOTH graph-state fields and both
    // per-node reference counts; protect its snapshot and the
    // corresponding per-edge transition as one indivisible operation.
    // This lock is only acquired on a real edge deactivation, never
    // during quiet island scans or ordinary Dy contact solving.
    const std::scoped_lock lock(third_pass_edge_mutex_);
    auto& edge=island_edges_[id];
    auto& state=accurate?edge.accurate_state:edge.speculative_state;
    if(!(state&3u))return;
    step_checkpoint_.save_activity(*this,id);
    if(rigid_node(actors_[edge.pair.a].motion) &&
       rigid_node(actors_[edge.pair.b].motion)) {
        step_checkpoint_.save_reference(*this,edge.pair.a);
        step_checkpoint_.save_reference(*this,edge.pair.b);
    }
    auto& graph=accurate?accurate_islands_:speculative_islands_;
    if(state&2u) {
        auto& pending=edge.type==IslandEdgeType::Constraint
            ? graph.activated_constraint_edges : graph.activated_edges;
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
        if(edge.type==IslandEdgeType::Constraint)
            --graph.active_constraint_edge_count;
        else --graph.active_edge_count;
        // Source removeEdgeFromActivatingList releases both valid
        // endpoint references on every real active-to-inactive edge
        // transition, including retirement of an actor no longer alive.
        if(rigid_node(actors_[edge.pair.a].motion) &&
           rigid_node(actors_[edge.pair.b].motion)) {
            --graph.active_reference_counts[edge.pair.a];
            --graph.active_reference_counts[edge.pair.b];
            refresh_kinematic_node(edge.pair.a,accurate);
            refresh_kinematic_node(edge.pair.b,accurate);
        }
        if(publish) {
            if(edge.type==IslandEdgeType::Constraint)
                graph.deactivating_constraint_edges.push_back(id);
            else graph.deactivating_edges.push_back(id);
        }
    }
}
void Scene::clear_activated_edges(bool checkpointed) noexcept {
    // Consume the previous generation's eACTIVATING bits by edge type,
    // never merging constraint and contact-manager event streams.
    for(const bool accurate:{false,true}) {
        auto& graph=accurate?accurate_islands_:speculative_islands_;
        auto& saved=accurate?step_checkpoint_.accurate_islands:
                             step_checkpoint_.speculative_islands;
        for(const bool constraint:{false,true}) {
            auto& prior=checkpointed
                ? (constraint?saved.activated_constraint_edges:
                              saved.activated_edges)
                : (constraint?graph.activated_constraint_edges:
                              graph.activated_edges);
            for(EdgeIndex id:prior) {
                auto& state=accurate?island_edges_[id].accurate_state:
                                     island_edges_[id].speculative_state;
                if(state&2u) {
                    step_checkpoint_.save_activity(*this,id);
                    state&=static_cast<std::uint8_t>(~2u);
                }
            }
            if(!checkpointed)prior.clear();
        }
    }
}
void Scene::activate_node_contact_edges(BodyId node) noexcept {
    // activateNodeInternal walks existing source intrusive edges.
    for(const bool accurate:{false,true}) {
        const auto& heads=accurate?accurate_node_first_:speculative_node_first_;
        for(std::size_t instance=heads[node];
            instance!=invalid_instance;) {
            const EdgeIndex id=static_cast<EdgeIndex>(instance/2);
            const auto& edge=island_edges_[id];
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
    FYZZ_PROFILE_SCOPE("sc.activate_islands");
    // Quiet publications without source wake/connection events never
    // visit graph islands or allocate per-node activation records.
    if(!authored_activation_pending_ && found_contacts_.empty() &&
       found_.empty())return;
    const auto dynamic_node=[&](BodyId id) {
        return actors_[id].alive && actors_[id].motion==Motion::Dynamic;
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
    for(Pair shapes:found_) {
        const Pair pair=body_pair(shapes);
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
    for(Pair shapes:found_contacts_) {
        const Pair pair=body_pair(shapes);
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
    for(Pair shapes:found_) {
        const Pair pair=body_pair(shapes);
        const EdgeIndex id=contact_edge_index(shapes);
        if(id!=invalid_edge &&
           ((actors_[pair.a].motion==Motion::Dynamic &&
             !actors_[pair.a].sleeping) ||
            (actors_[pair.b].motion==Motion::Dynamic &&
             !actors_[pair.b].sleeping)))
            activate_contact_edge(id,false);
    }
    for(Pair shapes:found_contacts_) {
        const Pair pair=body_pair(shapes);
        const EdgeIndex id=contact_edge_index(shapes);
        if(id!=invalid_edge &&
           ((actors_[pair.a].motion==Motion::Dynamic &&
             !actors_[pair.a].sleeping) ||
            (actors_[pair.b].motion==Motion::Dynamic &&
             !actors_[pair.b].sleeping)))
            activate_contact_edge(id,true);
    }
    authored_activation_pending_=false;
}
Scene::~Scene() noexcept = default;
Scene::ThirdPassWorkerState::~ThirdPassWorkerState() noexcept {
    if(!worker.joinable())return;
    {
        const std::lock_guard lock(mutex);
        stopping=true;
    }
    cv.notify_all();
    worker.join();
}
void Scene::run_third_pass_worker() noexcept {
    auto& work=*third_pass_worker_state_;
    for(;;) {
        std::unique_lock lock(work.mutex);
        work.cv.wait(lock,[&work] {return work.pending || work.stopping;});
        if(work.stopping)return;
        const bool repair_only=work.repair_only;
        const auto solver_first=work.solver_first;
        const auto solver_end=work.solver_end;
        const bool integrate_solver=work.integrate_solver;
        const float seconds=work.seconds;
        RigidStepJoin* const solver_join=work.solver_join;
#if defined(FYZZXYZZY_PROFILE_GPU)
        const auto trace_id=work.trace_id;
        const auto trace_origin=work.trace_origin;
        const auto trace_frame=work.trace_frame;
#endif
        work.pending=false;
        lock.unlock();
#if defined(FYZZXYZZY_PROFILE_GPU)
        hardware_profile::begin_worker(trace_id,trace_origin);
#endif
        bool split=false;
        std::exception_ptr error;
        {
        FYZZ_PROFILE_SCOPE(solver_first!=invalid_instance
            ?"dy.worker_solver":"sc.worker_third_pass");
        try {
            if(solver_first!=invalid_instance) {
                // Each independent source Dy task chain can progress
                // directly from PGS to PxsSolverEndTask: no other island's
                // solve must finish before this group's writeback,
                // integration and sleepCheck.
                for(std::size_t i=solver_first;i<solver_end;++i) {
                    solve_constraint_task(i);
                    if(integrate_solver) {
                        complete_solver_end_task(
                            i,seconds,solver_ready_worker_);
                        solver_join->finish_solver_task(i);
                    }
                }
            } else {
                split=repair_only
                    ? repair_lost_island_graph(false)
                    : process_third_pass_island_task(false);
            }
        } catch(...) {error=std::current_exception();}
        }
#if defined(FYZZXYZZY_PROFILE_GPU)
        hardware_profile::end_worker(trace_frame);
#endif
        lock.lock();
        work.split=split;
        work.error=error;
        work.done=true;
        lock.unlock();
        work.cv.notify_one();
    }
}
void Scene::dispatch_third_pass_worker(
    bool repair_only,std::size_t solver_first,
    bool integrate_solver,float seconds,RigidStepJoin* solver_join) {
    if(integrate_solver && !solver_join)
        throw std::logic_error("Source Dy task missing completion join");
    if(!third_pass_worker_state_)
        third_pass_worker_state_=std::make_unique<ThirdPassWorkerState>();
    auto& work=*third_pass_worker_state_;
    if(work.outstanding)
        throw std::logic_error("Native island worker dispatched twice");
    if(!work.worker.joinable()) {
        work.worker=std::jthread([this] {run_third_pass_worker();});
        ++third_pass_worker_start_count_;
    }
    {
        const std::lock_guard lock(work.mutex);
        work.repair_only=repair_only;
        work.integrate_solver=integrate_solver;
        work.seconds=seconds;
        work.solver_join=solver_join;
        work.solver_first=solver_first;
        work.solver_end=solver_first==invalid_instance
            ? 0:solver_tasks_.size();
#if defined(FYZZXYZZY_PROFILE_GPU)
        work.trace_id=hardware_profile::current_trace_id();
        work.trace_origin=hardware_profile::current_frame_begin();
        work.trace_frame=frame_index_;
#endif
        work.done=false;
        work.split=false;
        work.error={};
        work.pending=true;
    }
    work.outstanding=true;
    if(solver_first!=invalid_instance)
        ++concurrent_solver_worker_jobs_;
    work.cv.notify_one();
}
void Scene::join_third_pass_worker() noexcept {
    if(!third_pass_worker_state_)return;
    auto& work=*third_pass_worker_state_;
    if(!work.outstanding)return;
    std::unique_lock lock(work.mutex);
    work.cv.wait(lock,[&work] {return work.done;});
    work.outstanding=false;
}
bool Scene::complete_third_pass_worker() {
    join_third_pass_worker();
    const auto& work=*third_pass_worker_state_;
    if(work.error)std::rethrow_exception(work.error);
    return work.split;
}
bool Scene::worthwhile_parallel_route_work() const noexcept {
    return std::min(speculative_dirty_nodes_.size(),
                    accurate_dirty_nodes_.size()) >=
           parallel_third_pass_min_dirty_nodes_;
}
bool Scene::repair_lost_island_graphs_concurrently() {
    // Process both source ThirdPassTask lost-route producers in parallel.
    // Edge topology is stable; accurate-only indexed-manager mutations
    // do not modify the speculative intrusive edge instances.
    if(!parallel_third_pass_enabled_ ||
       speculative_dirty_nodes_.empty() ||
       accurate_dirty_nodes_.empty() ||
       !worthwhile_parallel_route_work()) {
        const bool speculative=repair_lost_island_graph(false);
        const bool accurate=repair_lost_island_graph(true);
        return speculative || accurate;
    }
    dispatch_third_pass_worker(true);
    bool accurate_split=false;
    try {
        accurate_split=repair_lost_island_graph(true);
    } catch(...) {
        join_third_pass_worker();
        throw;
    }
    const bool speculative_split=complete_third_pass_worker();
    ++concurrent_lost_route_dispatches_;
    return speculative_split || accurate_split;
}
bool Scene::process_third_pass_island_task(bool accurate) {
    FYZZ_PROFILE_SCOPE("sc.third_pass");
    // The single source ThirdPassTask for this IslandSim owns its
    // route repair, active-node reconciliation, and sleep decision in
    // that order. Neither task waits for the OTHER graph's result.
    const bool split=repair_lost_island_graph(accurate);
    synchronize_active_island_nodes_for_graph(accurate,!accurate);
    if(accurate)deactivate_ready_accurate_islands();
    else deactivate_ready_speculative_islands();
    return split;
}
void Scene::third_pass_island_gen() {
    // Collision-only entry point repairs routes before its no-throw
    // activity publication. Reuse the same independent repair producers
    // without scheduling redundant workers in the normal Scene::step.
    if(repair_lost_island_graphs_concurrently())
        ++island_topology_revision_;
}
// PxsIslandSim.cpp::IslandSim::processLostEdges(allowDeactivation) tests
// *previously published* Node::isReadyForSleeping flags in its third pass.
// Dy::sleepCheck updates those flags only after the current solver step,
// so a newly ready island cannot sleep until the next island pass.
// Sc::Scene::afterIntegration can restore a body integrated during the
// source's third-pass work. Native third-pass tasks now run concurrently
// with one another; their later overlap with Dy is a separate dependency.
// Both accurate and speculative contact graphs retain their existing
// persistent handles and node membership while bodies are inactive.
void Scene::deactivate_ready_islands() noexcept {
    // The collision-only path and source conformance probes consume the
    // same two owned stages, without a synthetic solver/task dispatcher.
    deactivate_ready_accurate_islands();
    deactivate_ready_speculative_islands();
}
std::uint32_t Scene::mark_kinematic_awake(
    IslandGraph& graph, bool accurate) noexcept {
        if(++graph.active_island_visit_stamp==0) {
            std::fill(graph.active_island_seen.begin(),
                      graph.active_island_seen.end(),0);
            graph.active_island_visit_stamp=1;
        }
        const auto stamp=graph.active_island_visit_stamp;
        const auto& first=accurate?accurate_node_first_:
                                   speculative_node_first_;
        for(BodyId kinematic:graph.active_kinematic_nodes) {
            const Actor& body=actors_[kinematic];
            if(!body.alive || body.motion!=Motion::Kinematic ||
               body.ready_for_sleep)continue;
            for(std::size_t index=first[kinematic];
                index!=invalid_instance;) {
                const auto& edge=island_edges_[index/2];
                const auto& instance=(accurate?edge.accurate:
                                              edge.speculative)[index&1u];
                index=instance.next;
                // PxsIslandSim::processLostEdges visits all live linked
                // kinematic EdgeInstances, independently of eACTIVE.
                // An inactive edge can still veto rigid deactivation.
                if(!edge.live || edge.retiring)continue;
                const BodyId other=edge.pair.a==kinematic
                    ? edge.pair.b:edge.pair.a;
                if(other>=graph.node_island.size() ||
                   actors_[other].motion!=Motion::Dynamic)continue;
                const auto owner=graph.node_island[other];
                if(owner!=IslandGraph::invalid)
                    graph.active_island_seen[owner]=stamp;
            }
        }
        return stamp;
 }
void Scene::deactivate_ready_accurate_islands() noexcept {
    constexpr BodyId invalid=std::numeric_limits<BodyId>::max();
    // Accurate processLostEdges owns rigid sleep/deactivation. This
    // producer may finish before the speculative graph's interaction
    // retirement pass, even in the serial CPU lowering.
    accurate_islands_.nodes_to_deactivate.clear();
    accurate_islands_.deactivating_edges.clear();
    accurate_islands_.deactivating_constraint_edges.clear();
    const auto accurate_kinematic_awake_stamp=
        mark_kinematic_awake(accurate_islands_,true);
    for(std::size_t index=accurate_islands_.active_islands.size();
        index>0;--index) {
        const auto handle=accurate_islands_.active_islands[index-1];
        if(handle>=accurate_islands_.islands.size())continue;
        const auto& island=accurate_islands_.islands[handle];
        if(!island.live ||
           accurate_islands_.active_island_seen[handle]==
               accurate_kinematic_awake_stamp)
            continue;
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
                const auto& edge=island_edges_[i/2];
                const auto& link=edge.accurate[i&1];
                const BodyId other=(i&1)?edge.pair.a:edge.pair.b;
                const EdgeIndex edge_id=static_cast<EdgeIndex>(i/2);
                i=link.next;
                if(edge.live && !edge.retiring && edge.connected &&
                   (actors_[other].motion!=Motion::Dynamic ||
                    actors_[other].sleeping))
                    deactivate_contact_edge(edge_id,true,true);
            }
        }
    }
}
void Scene::deactivate_ready_speculative_islands() noexcept {
    constexpr BodyId invalid=std::numeric_limits<BodyId>::max();
    // Speculative processLostEdges owns its separate contact/constraint
    // deactivation output. The serial lowering uses the prior accurate
    // sleep result as its candidate stream; no additional graph scan.
    speculative_islands_.nodes_to_deactivate.clear();
    speculative_islands_.deactivating_edges.clear();
    speculative_islands_.deactivating_constraint_edges.clear();
    const auto speculative_kinematic_awake_stamp=
        mark_kinematic_awake(speculative_islands_,false);
    // A target that just settled may release a speculative SAP-only
    // island even when accurate IslandSim had already slept its rigid
    // members on an earlier frame. Its still-live source EdgeInstance
    // must be reconsidered without waiting for another rigid sleep event.
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
    // Dy readiness transitions and speculative split/retirement
    // events are independent of accurate IslandSim's sleep output.
    // Source processLostEdges considers this graph's active islands
    // in reverse; sort only the touched candidates in that order.
    for(BodyId candidate:speculative_ready_nodes_)
        collect_candidate(candidate);
    speculative_ready_nodes_.clear();
    for(BodyId candidate:speculative_split_sleep_candidates_)
        collect_candidate(candidate);
    for(BodyId kinematic:speculative_islands_.active_kinematic_nodes) {
        const Actor& body=actors_[kinematic];
        if(!body.kinematic_settling || !body.ready_for_sleep)continue;
        for(std::size_t index=speculative_node_first_[kinematic];
            index!=invalid_instance;) {
            const auto& edge=island_edges_[index/2];
            const auto& instance=edge.speculative[index&1u];
            index=instance.next;
            // Source kinematic settling traverses linked EdgeInstances,
            // including those already inactive after a sleep pass.
            if(!edge.live || edge.retiring)continue;
            const BodyId other=edge.pair.a==kinematic
                ? edge.pair.b:edge.pair.a;
            if(other<actors_.size() &&
               actors_[other].motion==Motion::Dynamic)
                collect_candidate(other);
        }
    }
    if(speculative_sleep_order_.empty())return;
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
        if(speculative_islands_.active_island_seen[candidate.handle]==
           speculative_kinematic_awake_stamp)
            continue;
        const auto& island=speculative_islands_.islands[candidate.handle];
        const BodyId root=island.root;
        // PhysX speculative IslandSim tests Node::isReadyForSleeping,
        // NOT Sc's rigid sleeping bit set by the accurate island task.
        // This producer therefore works in either completion order.
        bool all_ready=true;
        for(BodyId node=root;node!=invalid;
            node=speculative_islands_.node_next[node]) {
            if(!actors_[node].ready_for_sleep) {
                all_ready=false;
                break;
            }
        }
        if(!all_ready)continue;
        // Source deactivateIsland walks the original intrusive node
        // chain; deactivateNodeInternal publishes each active edge on
        // visiting its last inactive endpoint. Reuse the same stamp as
        // the candidate deduplication for the visited node prefix.
        for(BodyId node=root;node!=invalid;
            node=speculative_islands_.node_next[node]) {
            speculative_sleep_seen_[node]=stamp;
            for(std::size_t i=speculative_node_first_[node];
                i!=invalid_instance;) {
                const auto& edge=island_edges_[i/2];
                const auto& link=edge.speculative[i&1];
                const BodyId other=(i&1)?edge.pair.a:edge.pair.b;
                const EdgeIndex edge_id=static_cast<EdgeIndex>(i/2);
                i=link.next;
                if(edge.live && !edge.retiring &&
                   (actors_[other].motion!=Motion::Dynamic ||
                    speculative_sleep_seen_[other]==stamp))
                    deactivate_contact_edge(edge_id,false,true);
            }
            // PxsIslandSim::deactivateIsland retires each graph's own
            // active node, not the shared ScBodySim sleep flag. A body
            // already asleep accurately can remain speculative-active
            // while its SAP island still has an awake neighbor.
            auto& graph=speculative_islands_;
            if(graph.active_node_index[node]==invalid_instance)continue;
            graph.nodes_to_deactivate.push_back(node);
            std::size_t at=graph.active_node_index[node];
            if(at<graph.initial_active_node_count) {
                const std::size_t tail=graph.initial_active_node_count-1;
                const BodyId previous=graph.active_nodes[tail];
                graph.active_nodes[at]=previous;
                graph.active_node_index[previous]=at;
                graph.active_nodes[tail]=node;
                graph.active_node_index[node]=tail;
                --graph.initial_active_node_count;
                at=tail;
            }
            const BodyId replacement=graph.active_nodes.back();
            graph.active_nodes[at]=replacement;
            graph.active_node_index[replacement]=at;
            graph.active_nodes.pop_back();
            graph.active_node_index[node]=invalid_instance;
        }
        // markIslandInactive uses replace-with-last, not a full active
        // island inventory rebuild or a Sc actor sleep scan.
        auto& graph=speculative_islands_;
        const std::size_t at=graph.islands[candidate.handle].active_index;
        if(at!=invalid_instance) {
            const auto moved=graph.active_islands.back();
            graph.active_islands[at]=moved;
            graph.islands[moved].active_index=at;
            graph.active_islands.pop_back();
            graph.islands[candidate.handle].active_index=invalid_instance;
        }
    }}
// ScPipeline.cpp::activateEdgesInternal(eCONTACT_MANAGER) consumes the
// speculative graph's mActivatedEdges, then postThirdPassIslandGen
// consumes mDeactivatingEdges after both lost-edge tasks have joined.
// The admitted rigid contact manager is already the single EdgeIndex
// owner, so retain Sc::InteractionFlag::eIS_ACTIVE in that same slot.
// This serial execution preserves activation-before-deactivation order
// without inventing a second interaction registry or task dispatcher.
void Scene::consume_contact_interaction_activity() noexcept {
    // Collision-only and source-fixture entry points preserve both
    // transitions without allocating a separate Sc interaction table.
    consume_activated_contact_interactions();
    consume_deactivated_contact_interactions();
}
void Scene::consume_activated_contact_interactions() noexcept {
    // ScPipeline::activateEdgesInternal consumes wake/second-pass edge
    // activity before the independent third-pass retirement producers.
    for(EdgeIndex id:speculative_islands_.activated_edges) {
        auto& edge=island_edges_[id];
        if(!edge.live || edge.retiring || edge.interaction_active ||
           !(edge.speculative_state&1u))continue;
        step_checkpoint_.save_activity(*this,id);
        edge.interaction_active=true;
        ++active_contact_interaction_count_;
    }
}
void Scene::consume_deactivated_contact_interactions() noexcept {
    // ScPipeline::postThirdPassIslandGen consumes deactivation only after
    // both IslandSim producers have completed their lost-edge processing.
    // No accurate-island deactivation is consumed here. Sc explicitly
    // uses the speculative graph so an accurate component can sleep
    // while its speculative-only bridge remains interaction-active.
    for(EdgeIndex id:speculative_islands_.deactivating_edges) {
        auto& edge=island_edges_[id];
        if(!edge.live || edge.retiring || !edge.interaction_active ||
           (edge.speculative_state&1u))continue;
        step_checkpoint_.save_activity(*this,id);
        edge.interaction_active=false;
        --active_contact_interaction_count_;
    }}
// Source PxsIslandSim::mActiveIslands is distinct from the complete
// island inventory. Build active batch indices from the already-owned
// contiguous accurate island-node sequence after source wake/deactivation.
// An island is active while any member is awake; the later per-node
// sleeping checks still protect a transitional mixed-state island.
// Unlike copying a second body-ID array, only one index per active
// island is retained, with reusable allocation across frames.
void Scene::synchronize_active_island_nodes(
    bool capture_speculative_order) noexcept {
    synchronize_active_island_nodes_for_graph(false,capture_speculative_order);
    synchronize_active_island_nodes_for_graph(true,false);
}
void Scene::synchronize_active_island_nodes_for_graph(
    bool accurate,bool capture_speculative_order) noexcept {
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
    // The speculative IslandSim's eACTIVE state is independent of
    // ScBodySim::sleeping. Only the accurate graph's active nodes are
    // removed when Sc rigid sleep is published; speculative nodes are
    // retired by their own third pass instead.
    auto& graph=accurate?accurate_islands_:speculative_islands_;
    for(std::size_t index=0;index<graph.active_nodes.size();) {
        const BodyId id=graph.active_nodes[index];
        if(actors_[id].alive &&
           (!accurate || !actors_[id].sleeping) &&
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
            if(!actors_[id].alive ||
               (accurate && actors_[id].sleeping) ||
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
void Scene::prepare_active_accurate_island_batches() {
    FYZZ_PROFILE_SCOPE("sc.accurate_batches");
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
Scene::SolverBodyRow& Scene::admit_solver_row(BodyId id) {
    auto& actor=actors_[id];
    if(actor.solver_row_generation!=solver_preparation_generation_) {
        const auto index=solver_rows_.size();
        solver_rows_.emplace_back();
        actor.solver_row_index=index;
        actor.solver_row_generation=solver_preparation_generation_;
    }
    return solver_rows_[actor.solver_row_index];
}

Scene::SolverBodyRow& Scene::prepared_solver_row(BodyId id) {
    // The hot PGS path uses pre-resolved row indices, so verify source
    // admission here rather than risk aliasing an unrelated active body.
    const auto& actor=actors_[id];
    if(actor.solver_row_generation!=solver_preparation_generation_ ||
       actor.solver_row_index>=solver_rows_.size())
        throw std::logic_error("Dy solver body has no current source row");
    return solver_rows_[actor.solver_row_index];
}

void Scene::prepare_solver_bodies() {
    FYZZ_PROFILE_SCOPE("dy.prepare_bodies");
    // DyDynamics::preIntegrationParallel -> Dy::copyToSolverBodyData.
    // Retain allocation capacity, but hold rows only for admitted bodies.
    // Retired actors and free-flight dynamics do not consume GPU-bound rows.
    solver_rows_.clear();
    // StepCheckpoint may restore frame_index_ after rejected work. The
    // independent generation invalidates actor-to-row mappings on retry.
    if(++solver_preparation_generation_==0) {
        for(auto& actor:actors_)actor.solver_row_generation=0;
        solver_preparation_generation_=1;
    }
    solver_active_ids_.clear();
    world_solver_row_ = {};
    // Kinematic rigid nodes are not accurate-island solver bodies,
    // but all active contacts/1D constraints referencing them need a
    // current-frame, infinite-mass SolverBodyData row. A settled target
    // may have left active_kinematic_targets_ while its edge remains
    // active. Stamp each participant once using existing indexed
    // island lists; never scan every historical actor.
    const auto prepare_kinematic=[&](BodyId id) {
        const Actor& body=actors_[id];
        if(body.motion!=Motion::Kinematic || !body.alive)return;
        if(body.solver_row_generation==solver_preparation_generation_)return;
        auto& row=admit_solver_row(id);
        row.prepared_generation=solver_preparation_generation_;
        row.data.position=mass_position(body);
        row.data.orientation=mass_orientation(body);
        row.data.linear_velocity=body.velocity;
        row.data.angular_velocity=body.angular_velocity;
        // A kinematic participates in Jacobian velocity without any
        // inverse mass, angular response or Dy writeback ownership.
    };
    for(BodyId id:active_kinematic_targets_)
        prepare_kinematic(id);
    // DyDynamicsContext::iterateIslandsIslandEdges gives pre-integration
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
            edge=island_edges_[edge].island_next) {
            if(++visited>island.contact_count)
                throw std::logic_error(
                    "PhysX active contact-manager chain contains a cycle");
            const auto& manager=island_edges_[edge];
            if(!manager.live || !manager.connected || manager.retiring)
                throw std::logic_error(
                    "PhysX active island contains an inactive contact manager");
            const Pair pair=manager.pair;
            prepare_kinematic(pair.a);
            prepare_kinematic(pair.b);
            const Actor& first=actors_[pair.a];
            const Actor& second=actors_[pair.b];
            const bool responds=std::min(
                first.dynamics.max_contact_impulse,
                second.dynamics.max_contact_impulse)!=0.0f;
            for(BodyId id:{pair.a,pair.b})
                if(actors_[id].motion==Motion::Dynamic &&
                   !actors_[id].sleeping) {
                    auto& row=admit_solver_row(id);
                    if(responds)
                        row.prepared_generation=solver_preparation_generation_;
                }
        }
        if(visited!=island.contact_count)
            throw std::logic_error(
                "PhysX active contact-manager chain is truncated");
        // Source iterateIslandsConstraintEdges admits solver-body rows
        // even when Gu has no contact manager for the same two actors.
        visited=0;
        for(EdgeIndex edge=island.first_constraint;edge!=invalid_edge;
            edge=island_edges_[edge].island_next) {
            if(++visited>island.constraint_count)
                throw std::logic_error("Source 1D constraint island list cycle");
            const auto& joint=island_edges_[edge];
            if(!joint.live || !joint.connected ||
               joint.constraint_id==invalid_edge)
                throw std::logic_error("Source 1D constraint is not admitted");
            for(BodyId id:{joint.pair.a,joint.pair.b}) {
                prepare_kinematic(id);
                if(actors_[id].motion==Motion::Dynamic &&
                   !actors_[id].sleeping)
                    admit_solver_row(id).prepared_generation=
                        solver_preparation_generation_;
            }
        }
        if(visited!=island.constraint_count)
            throw std::logic_error("Source 1D constraint island list truncated");
    }
    // Dy::DynamicsContext::iterateIslandsNodes hands contiguous accurate
    // island nodes to preIntegrationParallel. Free-flight bodies retain the
    // source no-constraint integration path without preparing unused rows.
    const bool gpu_world_inertia=gpu_dy_active();
    for(std::size_t island:active_accurate_island_batches_) {
        // Contact and joint geometry consume the same resident Dy inertia.
        // CPU execution alone requires a host world-response matrix.
        const bool device_inertia=gpu_world_inertia;
        for(std::size_t index=accurate_island_offsets_[island];
            index<accurate_island_offsets_[island+1];++index) {
            const BodyId id=accurate_island_nodes_[index];
            if(actors_[id].solver_row_generation!=
               solver_preparation_generation_)
                continue;
            auto& row=prepared_solver_row(id);
            // A newly admitted row is already zero-initialized, including
            // DyConstraintPartition's progress words for zero-capped
            // managers. Do not clear the same large row a second time.
            if(row.prepared_generation!=solver_preparation_generation_)
                continue;
            solver_active_ids_.push_back(id);
            const Actor& actor=actors_[id];
            row.data.position=mass_position(actor);
            row.data.orientation=mass_orientation(actor);
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
            row.device_world_inertia=device_inertia;
            if(device_inertia)continue;

            // CmUtils.h::transformInertiaTensor(sqrt(invInertia), pose.q).
            // Reuse the six unique terms of the symmetric world tensor.
            const Vec3 local{
                std::sqrt(actor.inverse_inertia_local.x),
                std::sqrt(actor.inverse_inertia_local.y),
                std::sqrt(actor.inverse_inertia_local.z)};
            const Quat q=mass_orientation(actor);
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
    FYZZ_PROFILE_SCOPE("dy.partition");
    // DyConstraintPartition.cpp rigid-body classification and source
    // static-contact reservation across both 1D and contact descriptors.
    // Contact patches remain indivisible; no new solver or island owner.
    //
    // The source assigns conflict-free dynamic/dynamic constraints from
    // successive 32-bit partition masks; overflow starts a new wave.
    // Each dynamic/static or dynamic/kinematic descriptor follows its
    // dynamic body's final dynamic/dynamic partition.
    constexpr std::size_t partition_bits = 32;
    // Only active source descriptors are partitioned. Quiet dormant
    // Gu history is neither rescanned nor cleared.
    constraint_descriptors_.clear();
    constraint_partitions_.clear();
    ordered_constraints_.clear();
    partition_ends_.clear();
    partition_cursors_.clear();
    // DyConstraintPartition.cpp::classifyConstraintDesc queues only the
    // unassigned dynamic descriptors for each overflow wave; compact the
    // same reusable descriptor buffer in place to avoid an O(waves*all
    // manifolds) rescan for high-degree constraint graphs.
    // DyDynamicsBase::iterateIslandsIslandEdges enumerates the
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
    indexed_joints_.clear();
    solver_tasks_.clear();
    gpu_dy_free_ids_.clear();
    gpu_dy_responding_island_count_=0u;
    const bool gpu_integration=gpu_dy_active();
    // GPU task boundaries follow Sc's existing accurate islands.
    // CPU groups retain their source batching policy.
    std::size_t joint_task_begin=0;
    const auto sort_constraints=[&] {
        // DyDynamicsBase::ConstraintLess compares Dy::Constraint::index
        // descending, not graph EdgeIndex or body-pair identity.
        auto first=indexed_joints_.begin()+joint_task_begin;
        std::sort(first,indexed_joints_.end(),
            [&](EdgeIndex a,EdgeIndex b) {
                return island_edges_[a].constraint_id >
                       island_edges_[b].constraint_id;
            });
        joint_task_begin=indexed_joints_.size();
    };
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
        // The current Dy rigid PGS lowering consumes only registered
        // rigid body nodes. A non-rigid node may belong to this island
        // once its owning solver path is installed, but must never be
        // accidentally lowered as a PxSolverBody by this path.
        if(island.node_type_counts[rigid_node_type_index]!=island.count)
            throw std::logic_error(
                "Non-rigid island node requires its own solver admission");
        std::size_t count=0;
        for(EdgeIndex edge=island.first_contact;edge!=invalid_edge;
            edge=island_edges_[edge].island_next) {
            if(edge>=manifold_by_edge_.size() ||
               manifold_by_edge_[edge].generation!=
                   manifold_lookup_generation_ ||
               manifold_by_edge_[edge].index>=manifolds_.size() ||
               !(manifolds_[manifold_by_edge_[edge].index].pair==
                 island_edges_[edge].shapes))
                throw std::logic_error(
                    "Active PhysX island has an unindexed contact manager");
            indexed_managers_.push_back(manifold_by_edge_[edge].index);
            if(++count>island.contact_count)
                throw std::logic_error("PhysX island contact list cycle");
        }
        if(count!=island.contact_count)
            throw std::logic_error("PhysX island contact list truncated");
        count=0;
        for(EdgeIndex edge=island.first_constraint;edge!=invalid_edge;
            edge=island_edges_[edge].island_next) {
            if(++count>island.constraint_count ||
               island_edges_[edge].constraint_id==invalid_edge)
                throw std::logic_error("Source 1D edge descriptor invalid");
            indexed_joints_.push_back(edge);
        }
        if(count!=island.constraint_count)
            throw std::logic_error("Source 1D edge list truncated");
        task_body_count+=island.count;
        task_constraint_count+=island.contact_count+island.constraint_count;
        // Only islands with prepared numerical bodies can gain a Dy task
        // from device island admission; free-flight islands are separate.
        const auto prepared_before_island=prepared_solver_bodies;
        // DyDynamics.cpp::preIntegrationParallel accumulates the
        // maximum PxsBodyCore solverIterationCounts word over precisely
        // this source task's node array, including unconstrained bodies.
        for(std::size_t node=accurate_island_offsets_[batch];
            node<accurate_island_offsets_[batch+1];++node) {
            const BodyId id=accurate_island_nodes_[node];
            const auto& d=actors_[id].dynamics;
            task_pos_iters=std::max(task_pos_iters,d.position_iterations);
            task_vel_iters=std::max(task_vel_iters,d.velocity_iterations);
            if(actors_[id].solver_row_generation==
                   solver_preparation_generation_ &&
               prepared_solver_row(id).prepared_generation==
                   solver_preparation_generation_)
                ++prepared_solver_bodies;
            else if(gpu_integration && !actors_[id].sleeping)
                gpu_dy_free_ids_.push_back(id);
        }
        if(prepared_solver_bodies!=prepared_before_island)
            ++gpu_dy_responding_island_count_;
        // GPU tasks retain independent accurate-island boundaries;
        // CPU tasks retain the source 128-body batching.
        if(enhanced_determinism_ ||
           (gpu_integration && island.contact_count+island.constraint_count>0u) ||
           (task_body_count>=source_solver_batch_size &&
            task_constraint_count>=1)) {
            sort_constraints();
            solver_tasks_.push_back({task_first_batch,batch_index+1,
                indexed_managers_.size(),0,0,prepared_solver_bodies,
                task_pos_iters,task_vel_iters,
                indexed_joints_.size(),gpu_dy_free_ids_.size()});
            task_first_batch=batch_index+1;
            task_body_count=0;
            task_constraint_count=0;
            task_pos_iters=task_vel_iters=0;
        }
    }
    if(task_body_count) {
        sort_constraints();
        solver_tasks_.push_back({task_first_batch,
            active_accurate_island_batches_.size(),
            indexed_managers_.size(),0,0,prepared_solver_bodies,
            task_pos_iters,task_vel_iters,
            indexed_joints_.size(),gpu_dy_free_ids_.size()});
    }
    // The admitted non-compound rigid PGS descriptor path in
    // DyDynamics.cpp::setupDescTask consumes this indexed-manager
    // sequence directly. The compound contact reducer's separate
    // two-key solver-body sort remains an explicit ABI admission gate.
    // Each source task chain partitions only its own island batch.
    // Use disjoint global partition ranges when concatenating the serial
    // CPU lowering; the original source can execute these chains in
    // parallel, but none may share a solver body across task groups.
    std::size_t manager_begin=0;
    std::size_t joint_begin=0, joint_row_begin=0;
    std::size_t partition_base=0;
    std::size_t prepared_body_begin=0u;
    pending_dynamic_constraints_.clear();
    for(auto& task:solver_tasks_) {
        const std::size_t manager_end=task.manager_end;
        const std::size_t descriptor_begin=constraint_descriptors_.size();
        const auto gpu_first=ordered_constraints_.size();
        // GPU coloring does not need host contact endpoint ordinals
        // unless this task could use wide numerical tiling. The upper
        // bound includes sleeping descriptors so this decision cannot
        // incorrectly classify a wide task as compact.
        constexpr std::size_t wide_min=gpu_dy_wide_task_threshold;
        const auto potential_descriptors=
            task.joint_end-joint_begin+manager_end-manager_begin;
        const bool compact_gpu_task=gpu_integration &&
            (potential_descriptors<wide_min ||
             task.solver_body_end-prepared_body_begin<wide_min);
        // Source setupDescTask admits 1D edges before contact managers.
        // Native Dy receives their solver-row descriptors directly; CPU
        // PGS retains the ActorID descriptor stream.
        const auto narrow_index=[](std::size_t index) {
            if(index>std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error(
                    "Source Dy descriptor index exceeds GPU address range");
            return static_cast<std::uint32_t>(index);
        };
        const auto dense_row=[&](BodyId id) {
            const auto& actor=actors_[id];
            if(!rigid_node(actor.motion))return invalid_solver_row;
            if(actor.solver_row_generation!=solver_preparation_generation_ ||
               actor.solver_row_index>=solver_rows_.size() ||
               actor.solver_row_index>=invalid_solver_row)
                throw std::logic_error(
                    "Native device partition references an unprepared solver body");
            return static_cast<std::uint32_t>(actor.solver_row_index);
        };
        const auto admit=[&](std::uint32_t index,BodyId a,BodyId b,
                             std::uint32_t row_count) {
            if(!gpu_integration) {
                constraint_descriptors_.push_back({index,a,b,row_count});
                return;
            }
            const auto& actor_a=actors_[a];
            const auto& actor_b=actors_[b];
            const bool dynamic_a=actor_a.motion==Motion::Dynamic;
            const bool dynamic_b=actor_b.motion==Motion::Dynamic;
            if((dynamic_a && actor_a.sleeping) ||
               (dynamic_b && actor_b.sleeping) ||
               (!dynamic_a && !dynamic_b))
                return;
            const auto flags=row_count |
                (dynamic_a?0x100u:0u)|(dynamic_b?0x200u:0u);
            if(compact_gpu_task && row_count==0u) {
                // The device expands contacts from the already
                // prepared manager table. Only zero-patch contacts
                // need their endpoint ordinals for GPU coloring.
                ordered_constraints_.push_back({
                    index,invalid_solver_row,invalid_solver_row,flags});
            }else {
                ordered_constraints_.push_back({
                    index,dense_row(a),dense_row(b),flags});
            }
        };
        for(std::size_t j=joint_begin;j<task.joint_end;++j) {
            const auto& edge=island_edges_[indexed_joints_[j]];
            const auto& joint=constraint_records_[edge.constraint_id];
            const std::uint32_t count=joint.kind==JointKind::Distance?1u:
                (joint.kind==JointKind::Spherical?3u:
                 joint.kind==JointKind::Fixed?6u:5u);
            const auto a=joint.swapped?edge.pair.b:edge.pair.a;
            const auto b=joint.swapped?edge.pair.a:edge.pair.b;
            admit(narrow_index(joint_row_begin),a,b,count);
            joint_row_begin+=count;
        }
        for(std::size_t j=manager_begin;j<manager_end;++j) {
            const std::size_t index=indexed_managers_[j];
            const auto& manifold=manifolds_[index];
            const bool reversed=island_edges_[
                manifold.edge_handle].gu_order_reversed;
            admit(narrow_index(index),
                  shapes_[reversed?manifold.pair.b:manifold.pair.a].body,
                  shapes_[reversed?manifold.pair.a:manifold.pair.b].body,0u);
        }
        if(gpu_integration) {
            const auto count=ordered_constraints_.size()-gpu_first;
            if(count>std::numeric_limits<std::size_t>::max()-partition_base)
                throw std::overflow_error("Device Dy task partition capacity overflow");
            partition_base+=count;
            task.partition_end=partition_base;
            task.constraint_end=ordered_constraints_.size();
            manager_begin=manager_end;
            joint_begin=task.joint_end;
            prepared_body_begin=task.solver_body_end;
            continue;
        }
        const auto pair_of=[&](std::size_t index) {
            const auto& descriptor=constraint_descriptors_[index];
            return Pair{descriptor.body0,descriptor.body1};
        };
        constraint_partitions_.resize(constraint_descriptors_.size(),
                                      invalid_instance);
        pending_dynamic_constraints_.clear();
        for(std::size_t i=descriptor_begin;
            i<constraint_descriptors_.size();++i) {
            const Pair pair=pair_of(i);
            if(actors_[pair.a].motion==Motion::Dynamic &&
               actors_[pair.b].motion==Motion::Dynamic &&
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
                    const Pair pair=pair_of(i);
                    prepared_solver_row(pair.a).solver.partition_mask=0;
                    prepared_solver_row(pair.b).solver.partition_mask=0;
                }
            std::size_t overflow_count=0;
            for(std::size_t k=0;k<pending_dynamic_constraints_.size();++k) {
                const std::size_t i=pending_dynamic_constraints_[k];
                const Pair pair=pair_of(i);
                auto& a=prepared_solver_row(pair.a).solver;
                auto& b=prepared_solver_row(pair.b).solver;
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

        for(std::size_t i=descriptor_begin;
            i<constraint_descriptors_.size();++i) {
            if(constraint_partitions_[i]!=invalid_instance)continue;
            const Pair pair=pair_of(i);
            const bool dynamic_a=actors_[pair.a].motion==Motion::Dynamic;
            const bool dynamic_b=actors_[pair.b].motion==Motion::Dynamic;
            if((dynamic_a && actors_[pair.a].sleeping) ||
               (dynamic_b && actors_[pair.b].sleeping))continue;
            if(!dynamic_a && !dynamic_b)continue;
            auto& active=prepared_solver_row(dynamic_a?pair.a:pair.b).solver;
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
            next_partition_base=std::max(next_partition_base,
                                         global_partition+1);
        }
        partition_base=next_partition_base;
        task.partition_end=partition_base;
        manager_begin=manager_end;
        joint_begin=task.joint_end;
    }

    joint_rows_.reserve(joint_row_begin);
    if(gpu_integration)return;

    // Prefix-count and stable-scatter the unified descriptor sequence.
    // Source 1D rows precede contacts within each conflict-free batch;
    // source Gu contact offsets remain unaffected by this permutation.
    std::size_t count=0;
    for(std::size_t i=0;i<constraint_descriptors_.size();++i) {
        const std::size_t p=constraint_partitions_[i];
        if(p==invalid_instance)continue;
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
    const auto dense_row=[&](BodyId id) {
        const auto& actor=actors_[id];
        if(!rigid_node(actor.motion))return invalid_solver_row;
        if(actor.solver_row_generation!=solver_preparation_generation_ ||
           actor.solver_row_index>=solver_rows_.size() ||
           actor.solver_row_index>=invalid_solver_row)
            throw std::logic_error(
                "Native Dy partition references an unprepared solver body");
        return static_cast<std::uint32_t>(actor.solver_row_index);
    };
    for(std::size_t i=0;i<constraint_descriptors_.size();++i) {
        const std::size_t partition=constraint_partitions_[i];
        if(partition==invalid_instance)continue;
        const auto& source=constraint_descriptors_[i];
        ordered_constraints_[partition_cursors_[partition]++]={
            source.index,dense_row(source.body0),
            dense_row(source.body1),source.row_count};
    }
}

void Scene::prepare_constraints(float seconds) {
    FYZZ_PROFILE_SCOPE("dy.prepare_constraints");
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
    prepared_contact_rows_=0u;
    gpu_dy_gu_inputs_.clear();
    solver_contact_order_.clear();
    gpu_dy_contact_order_.clear();
    joint_rows_.clear();
    gpu_dy_rows_.clear();
    gpu_dy_joint_inputs_.clear();
    const bool gpu_normal_prep=gpu_dy_active();
    gpu_dy_contact_prep_pairs_.clear();
    if(gpu_normal_prep)
        gpu_dy_contact_prep_indices_.assign(manifolds_.size(),0xffffffffu);
    solver_kinematic_participant_=false;
    for(EdgeIndex edge_id:indexed_joints_) {
        const auto& edge=island_edges_[edge_id];
        const auto& joint=constraint_records_[edge.constraint_id];
        if(!joint.live || joint.edge!=edge_id)
            throw std::logic_error("Source 1D constraint descriptor owner missing");
        const auto id0=joint.swapped?edge.pair.b:edge.pair.a;
        const auto id1=joint.swapped?edge.pair.a:edge.pair.b;
        const auto& body0=actors_[id0];
        const auto& body1=actors_[id1];
        solver_kinematic_participant_ |=
            body0.motion==Motion::Kinematic || body1.motion==Motion::Kinematic;
        if(gpu_normal_prep) {
            // Only authored local frames and non-dynamic endpoint poses
            // cross into Dy. World Jacobians use resident actor motion.
            GpuDyJointInput input{};
            const auto pack3=[](float(&out)[3],Vec3 v) {
                out[0]=v.x;out[1]=v.y;out[2]=v.z;
            };
            const auto pack4=[](float(&out)[4],Quat q) {
                out[0]=q.x;out[1]=q.y;out[2]=q.z;out[3]=q.w;
            };
            pack3(input.local_a_position,joint.local_a.position);
            pack4(input.local_a_orientation,joint.local_a.orientation);
            pack3(input.local_b_position,joint.local_b.position);
            pack4(input.local_b_orientation,joint.local_b.orientation);
            if(body0.motion!=Motion::Dynamic) {
                pack3(input.actor_a_position,body0.position);
                pack4(input.actor_a_orientation,body0.orientation);
                pack3(input.center_a,mass_position(body0));
            }
            if(body1.motion!=Motion::Dynamic) {
                pack3(input.actor_b_position,body1.position);
                pack4(input.actor_b_orientation,body1.orientation);
                pack3(input.center_b,mass_position(body1));
            }
            input.distance_length=joint.distance_length;
            input.kind=static_cast<std::uint32_t>(joint.kind);
            input.source_slot=edge.constraint_id;
            input.source_generation=joint.generation;
            if(gpu_dy_joint_inputs_.size()>=std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Native GPU joint admission count overflow");
            const auto source_index=static_cast<std::uint32_t>(gpu_dy_joint_inputs_.size());
            gpu_dy_joint_inputs_.push_back(input);
            const auto count=joint.kind==JointKind::Distance?1u:
                joint.kind==JointKind::Spherical?3u:
                joint.kind==JointKind::Fixed?6u:5u;
            const float mass0=body0.motion==Motion::Dynamic?body0.inverse_mass:0.0f;
            const float mass1=body1.motion==Motion::Dynamic?body1.inverse_mass:0.0f;
            for(std::uint32_t i=0u;i<count;++i) {
                joint_rows_.emplace_back();
                auto& gpu=gpu_dy_rows_.emplace_back();
                gpu.inverse_mass_a=mass0;
                gpu.inverse_mass_b=mass1;
                gpu.reserved=2u;
                if(i==0u) {
                    gpu.applied_force=std::bit_cast<float>(source_index);
                    // The device verifies both source slot and generation
                    // before replacing these transient admission words.
                    gpu.constant=std::bit_cast<float>(input.source_slot);
                    gpu.unbiased_constant=std::bit_cast<float>(input.source_generation);
                }
            }
            continue;
        }
        const auto anchor0=rotate(body0.orientation,joint.local_a.position);
        const auto anchor1=rotate(body1.orientation,joint.local_b.position);
        const Vec3 world0{body0.position.x+anchor0.x,
                          body0.position.y+anchor0.y,
                          body0.position.z+anchor0.z};
        const Vec3 world1{body1.position.x+anchor1.x,
                          body1.position.y+anchor1.y,
                          body1.position.z+anchor1.z};
        const auto center0=mass_position(body0),center1=mass_position(body1);
        const auto ra=sub(world0,center0),rb=sub(world1,center1);
        const float mass0=body0.motion==Motion::Dynamic?body0.inverse_mass:0.0f;
        const float mass1=body1.motion==Motion::Dynamic?body1.inverse_mass:0.0f;
        const auto response_angular=[&](BodyId id,Vec3 angular) {
            if(actors_[id].motion!=Motion::Dynamic)return Vec3{};
            return transform(prepared_solver_row(id).data,angular);
        };
        const auto relative=sub(body0.velocity,body1.velocity);
        const auto emit=[&](Vec3 axis,Vec3 angular0,Vec3 angular1,
                            Vec3 ja,Vec3 jb,float error,float response) {
            if(!(response>0.0f) || !std::isfinite(response))
                throw std::logic_error("Source 1D joint requires finite response");
            const float reciprocal=1.0f/response;
            // Hard PGS rows use unit ERP, then remove bias for velocity passes.
            JointRow row{};
            row.axis=axis;
            row.angular_a=ja;
            row.angular_b=jb;
            row.inverse_mass_a=mass0;
            row.inverse_mass_b=mass1;
            row.velocity_multiplier=-reciprocal;
            const float speed=
                dot(axis,relative)+dot(angular0,body0.angular_velocity)-
                dot(angular1,body1.angular_velocity);
            row.constant=-reciprocal*(error*inv_dt+speed);
            row.unbiased_constant=-reciprocal*speed;
            if(!std::isfinite(row.constant) || !std::isfinite(row.unbiased_constant))
                throw std::overflow_error("Source 1D solver constants overflow");
            joint_rows_.push_back(row);
        };
        if(joint.kind==JointKind::Distance) {
            // Equal min/max distance rows use +X for coincident anchors.
            const Vec3 offset{center0.x+ra.x-center1.x-rb.x,
                              center0.y+ra.y-center1.y-rb.y,
                              center0.z+ra.z-center1.z-rb.z};
            const float distance=std::sqrt(dot(offset,offset));
            if(!std::isfinite(distance))
                throw std::overflow_error("Source 1D distance overflow");
            const Vec3 axis=distance<1.192092896e-07f?Vec3{1,0,0}:
                Vec3{offset.x/distance,offset.y/distance,offset.z/distance};
            const auto angular0=cross(ra,axis),angular1=cross(rb,axis);
            const auto ja=response_angular(id0,angular0),jb=response_angular(id1,angular1);
            emit(axis,angular0,angular1,ja,jb,distance-joint.distance_length,
                mass0+mass1+dot(ja,ja)+dot(jb,jb));
            continue;
        }
        // Prismatic joints lock Y/Z translation and all rotations.
        // Revolute joints lock XYZ translation and Y/Z rotation; spherical
        // joints lock only translation and fixed joints lock every axis.
        const auto qa=multiply(body0.orientation,joint.local_a.orientation);
        const auto error_p=rotate_inverse(qa,sub(world0,world1));
        const bool lock_rotation=joint.kind!=JointKind::Spherical;
        Vec3 error_angle{},va{},vb{},c{};
        float d=0.0f,angular_overlap=0.0f;
        if(lock_rotation) {
            auto qb=multiply(body1.orientation,joint.local_b.orientation);
            if(qa.x*qb.x+qa.y*qb.y+qa.z*qb.z+qa.w*qb.w<0.0f)
                qb={-qb.x,-qb.y,-qb.z,-qb.w};
            const auto error_q=multiply({-qa.x,-qa.y,-qa.z,qa.w},qb);
            error_angle={error_q.x,error_q.y,error_q.z};
            va={qa.x,qa.y,qa.z};
            vb={qb.x,qb.y,qb.z};
            d=qa.w*qb.w-dot(va,vb);
            angular_overlap=qa.w*qb.w+dot(va,vb);
            c={qa.w*vb.x+qb.w*va.x,qa.w*vb.y+qb.w*va.y,qa.w*vb.z+qb.w*va.z};
        }
        const std::array<float,6> errors{
            error_p.x,error_p.y,error_p.z,
            -error_angle.x,-error_angle.y,-error_angle.z};
        const std::array<Vec3,3> basis{Vec3{1,0,0},Vec3{0,1,0},Vec3{0,0,1}};
        const std::array<float,3> av{va.x,va.y,va.z},bv{vb.x,vb.y,vb.z};
        const std::array<Vec3,3> diagonal{
            Vec3{d,c.z,-c.y},Vec3{-c.z,d,c.x},Vec3{c.y,-c.x,d}};
        struct EqualityRow {
            Vec3 linear{},angular0{},angular1{},ja{},jb{};
            float error=0.0f,inverse_response=0.0f;
        };
        std::array<EqualityRow,6> rows{};
        const auto weighted_dot=[&](const EqualityRow& a,const EqualityRow& b) {
            return (mass0+mass1)*dot(a.linear,b.linear)+dot(a.ja,b.ja)+dot(a.jb,b.jb);
        };
        const auto remove_projection=[](Vec3& a,Vec3 b,float t) {
            a={a.x-b.x*t,a.y-b.y*t,a.z-b.z*t};
        };
        const std::size_t row_count=joint.kind==JointKind::Spherical?3u:
            joint.kind==JointKind::Fixed?6u:5u;
        for(std::size_t i=0;i<row_count;++i) {
            auto& row=rows[i];
            // Source locked-axis order excludes the slider's free X
            // translation or the hinge's free X rotation.
            const auto component=i+
                (joint.kind==JointKind::Prismatic?1u:0u)+
                (joint.kind==JointKind::Revolute && i>=3u?1u:0u);
            row.error=errors[component];
            if(component<3u) {
                row.linear=rotate(qa,basis[component]);
                // The slider's transverse axes rotate with frame A.
                // At nonzero extension, d(axis)/dt contributes a lever
                // arm from A's mass center to B's joint origin.
                const Vec3 arm=joint.kind==JointKind::Prismatic
                    ? sub(world1,center0):ra;
                row.angular0=cross(arm,row.linear);
                row.angular1=cross(rb,row.linear);
            } else {
                const auto k=component-3u;
                const auto v=diagonal[k];
                row.angular0={
                    0.5f*(va.x*bv[k]+vb.x*av[k]+v.x),
                    0.5f*(va.y*bv[k]+vb.y*av[k]+v.y),
                    0.5f*(va.z*bv[k]+vb.z*av[k]+v.z)};
                // The relative quaternion's 180-degree Jacobian is singular.
                if(angular_overlap==0.0f)
                    remove_projection(row.angular0,basis[k],
                                      -std::numeric_limits<float>::epsilon());
                row.angular1=row.angular0;
            }
            row.ja=response_angular(id0,row.angular0);
            row.jb=response_angular(id1,row.angular1);
            // DyConstraintSetup orthogonalizes bilateral PGS rows in the
            // inverse-mass metric, including the geometric error terms.
            for(std::size_t j=0;j<i;++j) {
                const auto& prev=rows[j];
                const float t=weighted_dot(row,prev)*prev.inverse_response;
                remove_projection(row.linear,prev.linear,t);
                remove_projection(row.angular0,prev.angular0,t);
                remove_projection(row.angular1,prev.angular1,t);
                remove_projection(row.ja,prev.ja,t);
                remove_projection(row.jb,prev.jb,t);
                row.error-=prev.error*t;
            }
            const float response=weighted_dot(row,row);
            row.inverse_response=response>0.0f?1.0f/response:0.0f;
            emit(row.linear,row.angular0,row.angular1,row.ja,row.jb,row.error,response);
        }
    }
    solver_friction_rows_.clear();
    prepared_friction_rows_=0u;
    gpu_dy_friction_anchors_.clear();
    next_friction_patches_.clear();
    // PXC_SAME_NORMAL and DyCorrelationBuffer::MAX_FRICTION_PATCHES.
    constexpr float same_normal = 0.999f;
    constexpr float friction_offset_threshold = 0.04f;
    const auto local_point = [this](const Actor& actor, Vec3 world) {
        const Quat q = mass_orientation(actor);
        const Vec3 p=mass_position(actor);
        return rotate(Quat{-q.x,-q.y,-q.z,q.w},
            {world.x-p.x,world.y-p.y,world.z-p.z});
    };
    const auto local_normal = [this](const Actor& actor, Vec3 world_normal) {
        const Quat q = mass_orientation(actor);
        return rotate(Quat{-q.x,-q.y,-q.z,q.w}, world_normal);
    };
    const auto world_point = [this](const Actor& actor, Vec3 local) {
        const Vec3 r=rotate(mass_orientation(actor),local);
        const Vec3 p=mass_position(actor);
        return Vec3{r.x+p.x,r.y+p.y,r.z+p.z};
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
    next_friction_by_edge_.resize(island_edges_.size());
    for(std::size_t manager_index:indexed_managers_) {
        auto& manifold=manifolds_[manager_index];
        const EdgeIndex edge=manifold.edge_handle;
        if(edge>=island_edges_.size() ||
           !(island_edges_[edge].shapes==manifold.pair))
            throw std::logic_error(
                "Dy source manager preparation has a mismatched island edge");
        step_checkpoint_.save_manifold_solver_metadata(*this,manager_index);
        manifold.prepared_friction_epoch=next_cache_epoch;
        // An inactive accurate island retains its Gu contact manager
        // but the source does not prepare or solve constraints for it.
        // Keep the Gu-indexed zero-impulse result slots without paying
        // for contact Jacobians, friction correlation or solver rows.
        const Actor& actor_a=actors_[shapes_[manifold.pair.a].body];
        const Actor& actor_b=actors_[shapes_[manifold.pair.b].body];
        const bool active_dynamic=
            (actor_a.motion==Motion::Dynamic && !actor_a.sleeping) ||
            (actor_b.motion==Motion::Dynamic && !actor_b.sleeping);
        if(!active_dynamic) {
            manifold.solver_offset=prepared_contact_rows_;
            manifold.friction_patch_offset=next_friction_patches_.size();
            manifold.friction_patch_count=0;
            // No Dy contact rows are admitted for dormant Gu managers.
            // The independently Gu-indexed writeback provides zeros.
            continue;
        }
        // Only an admitted active manager can share writable kinematic
        // solver state across otherwise independent Dy task groups.
        // Dormant Gu history must not suppress concurrent solving.
        solver_kinematic_participant_ |=
            actor_a.motion==Motion::Kinematic ||
            actor_b.motion==Motion::Kinematic;
        // Preserve ScShapeInteraction's actual ordered actor pair
        // through DyContactPrep. A canonical pair key is only used
        // for Gu arena identity, and must not reorder the two
        // momentum/Jacobian inputs after the source Gu generator.
        // PxcNpBatch sorts plane-box Gu geometry plane-first and
        // flips its returned normal back to ScShapeInteraction order.
        // ScNPhaseCore::shouldSwapBodies puts static shapes SECOND,
        // even when the floor was created before its dynamic partner.
        // The persistent source-order flag applies to every shape pair.
        const bool reversed=island_edges_[edge].gu_order_reversed;
        const Shape& shape0=shapes_[reversed?manifold.pair.b:manifold.pair.a];
        const Shape& shape1=shapes_[reversed?manifold.pair.a:manifold.pair.b];
        const BodyId id0=shape0.body,id1=shape1.body;
        const auto& a = rigid_node(actors_[id0].motion)
            ? prepared_solver_row(id0).data : world_solver_row_.data;
        const auto& b = rigid_node(actors_[id1].motion)
            ? prepared_solver_row(id1).data : world_solver_row_.data;
        manifold.solver_offset = prepared_contact_rows_;
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
        const float rest_distance=shape0.rest_offset+shape1.rest_offset;
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
        const Material& m0=shape0.material;
        const Material& m1=shape1.material;
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
        // GuPCMContactBoxBox publishes one common world normal for all
        // selected witnesses of a source box pair. Its contiguous run
        // is already known: build the original patch bounds once without
        // repeating general Gu normal/material compatibility checks.
        if(manifold.pcm_result_index!=
           std::numeric_limits<std::uint32_t>::max()) {
            ContactPatchData cp{};
            cp.start=0u;
            cp.count=manifold.count;
            cp.normal=manifold.points.front().normal;
            if(!gpu_normal_prep) {
                // CPU Dy owns the exact contact bounds for correlation.
                // Device Dy derives the extent from its resident PCM
                // witnesses during friction-anchor preparation instead.
                cp.bounds_min=cp.bounds_max=manifold.points.front().position;
                for(std::size_t j=1;j<manifold.count;++j) {
                    const auto& p=manifold.points[j].position;
                    cp.bounds_min={
                        std::min(cp.bounds_min.x,p.x),
                        std::min(cp.bounds_min.y,p.y),
                        std::min(cp.bounds_min.z,p.z)};
                    cp.bounds_max={
                        std::max(cp.bounds_max.x,p.x),
                        std::max(cp.bounds_max.y,p.y),
                        std::max(cp.bounds_max.z,p.z)};
                }
            }
            contact_patch_scratch_.push_back(cp);
        }else {
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
                const bool checked_before_gu=
                    manifold.pcm_result_index<
                        gpu_pcm_friction_history_masks_.size();
                const bool valid=checked_before_gu
                    ?(gpu_pcm_friction_history_masks_[manifold.pcm_result_index] &
                       (1u<<(previous-history.offset)))!=0u
                    :cached_friction_patch_valid(old,actor0,actor1);
                if(!valid)continue;
                if(manifold.friction_patch_count==max_source_patches)
                    throw std::overflow_error(
                        "Native source cached friction exceeds 32 patches");
                FrictionPatch copied=old;
                copied.normal=rotate(mass_orientation(actor0),old.body0_normal);
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
        if(gpu_normal_prep) {
            if(gpu_dy_contact_prep_pairs_.size()>=
                std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Source GPU contact preparation limit");
            const auto pack_body=[](native::GpuDyContactPrepBody& out,
                                    const SolverBodyData& body) {
                const auto pack=[](float(&to)[3],Vec3 v) {
                    to[0]=v.x;to[1]=v.y;to[2]=v.z;
                };
                pack(out.position,body.position);
                pack(out.linear_velocity,body.linear_velocity);
                pack(out.angular_velocity,body.angular_velocity);
                for(std::size_t i=0;i<3u;++i) {
                    const Vec3 c=body.sqrt_inverse_inertia[i];
                    out.sqrt_inverse_inertia[i*3u+0u]=c.x;
                    out.sqrt_inverse_inertia[i*3u+1u]=c.y;
                    out.sqrt_inverse_inertia[i*3u+2u]=c.z;
                }
                out.inverse_mass=body.inverse_mass;
            };
            native::GpuDyContactPrepPair prep{};
            pack_body(prep.body0,a);
            pack_body(prep.body1,b);
            // The prepared contact endpoints already own their solver
            // rows. Compact Dy descriptors omit duplicate row indices.
            const auto pack_ordinal=[&](native::GpuDyContactPrepBody& out,
                                        const Actor& actor) {
                if(!rigid_node(actor.motion))return;
                if(actor.solver_row_generation!=solver_preparation_generation_ ||
                   actor.solver_row_index>=solver_rows_.size() ||
                   actor.solver_row_index>=invalid_solver_row)
                    throw std::logic_error(
                        "GPU Dy contact preparation lacks a solver row");
                out.solver_ordinal=
                    static_cast<std::uint32_t>(actor.solver_row_index);
            };
            pack_ordinal(prep.body0,actor0);
            pack_ordinal(prep.body1,actor1);
            // Gu anchors are actor-local, including the immovable shape.
            // World solver-body data uses a zero-mass static sentinel.
            const auto pack_pose=[this](native::GpuDyContactPrepBody& out,
                                    const Actor& actor) {
                const Vec3 p=mass_position(actor);
                const Quat q=mass_orientation(actor);
                out.position[0]=p.x;out.position[1]=p.y;out.position[2]=p.z;
                out.orientation[0]=q.x;out.orientation[1]=q.y;
                out.orientation[2]=q.z;out.orientation[3]=q.w;
            };
            pack_pose(prep.body0,actor0);
            pack_pose(prep.body1,actor1);
            prep.rest_distance=rest_distance;
            prep.pair_slop=pair_slop;
            prep.restitution=restitution;
            prep.damping=damping;
            prep.max_penetration_bias=max_penetration_bias;
            prep.inv_dt=inv_dt;
            prep.seconds=seconds;
            prep.rigid_contact_bias=rigid_contact_bias;
            prep.bounce_threshold=bounce_threshold;
            prep.acceleration_spring=acceleration_spring?1u:0u;
            prep.max_contact_impulse=pair_max_impulse;
            // Sc records PCM provenance when its original Gu manager
            // is published. Dy consumes that exact source ordinal; no
            // second host-side search or independent pair registry.
            if(manifold.pcm_result_index!=
               std::numeric_limits<std::uint32_t>::max()) {
                if(manifold.pcm_result_index>=gpu_pcm_pairs_.size() ||
                   gpu_pcm_pairs_[manifold.pcm_result_index].edge!=edge)
                    throw std::logic_error(
                        "Source Sc PCM manager lost GPU result provenance");
                prep.pcm_pair_index=manifold.pcm_result_index;
                prep.pcm_edge=edge;
            }
            if(manifold.pcm_result_index==
               std::numeric_limits<std::uint32_t>::max()) {
                if(gpu_dy_gu_inputs_.size()>=
                   std::numeric_limits<std::uint32_t>::max())
                    throw std::overflow_error(
                        "Source CPU Gu geometry exceeds 32-bit view");
                prep.cpu_gu_offset=static_cast<std::uint32_t>(
                    gpu_dy_gu_inputs_.size());
            }
            gpu_dy_contact_prep_indices_[manager_index]=
                static_cast<std::uint32_t>(gpu_dy_contact_prep_pairs_.size());
            gpu_dy_contact_prep_pairs_.push_back(prep);
        }
        // CPU PGS retains the complete correlated contact order.
        // Device Dy needs explicit indices only for non-PCM Gu:
        // PCM's source-ordered contact run is already resident.
        const bool resident_order=gpu_normal_prep &&
            manifold.pcm_result_index!=
                std::numeric_limits<std::uint32_t>::max();
        const auto order_begin=gpu_normal_prep
            ?gpu_dy_contact_order_.size():solver_contact_order_.size();
        if(!resident_order)
            for(std::size_t j=0;j<kept;++j)
                for(std::size_t cp=used_heads[j];cp!=invalid_instance;
                    cp=contact_patch_scratch_[cp].next)
                    for(std::size_t c=contact_patch_scratch_[cp].start;
                        c<contact_patch_scratch_[cp].start+
                          contact_patch_scratch_[cp].count;++c) {
                        const auto index=manifold.solver_offset+c;
                        if(gpu_normal_prep) {
                            if(index>=std::numeric_limits<std::uint32_t>::max())
                                throw std::overflow_error(
                                    "GPU Dy contact order exceeds 32-bit rows");
                            gpu_dy_contact_order_.push_back(
                                static_cast<std::uint32_t>(index));
                        } else solver_contact_order_.push_back(index);
                    }
        const auto order_end=gpu_normal_prep
            ?gpu_dy_contact_order_.size():solver_contact_order_.size();
        if(order_end-order_begin!=(resident_order?0u:manifold.count))
            throw std::logic_error(
                "Source correlation omitted a Gu contact descriptor");
        for(std::size_t j=0;j<manifold.count;++j) {
            const ContactPoint& contact=manifold.points[j];
            const std::size_t patch_id=contact_patch_ids[j];
            const Vec3 normal=header_normals[
                patch_id-manifold.friction_patch_offset];
            if(gpu_normal_prep) {
                // Sc retains the Gu ordinal, not a shadow Jacobian.
                // CPU Gu packs the correlated header normal while
                // PCM geometry remains in the resident contact stream.
                if(manifold.pcm_result_index==
                   std::numeric_limits<std::uint32_t>::max()) {
                    GpuDyGuContactInput gu{};
                    gu.normal[0]=normal.x;
                    gu.normal[1]=normal.y;
                    gu.normal[2]=normal.z;
                    gu.position[0]=contact.position.x;
                    gu.position[1]=contact.position.y;
                    gu.position[2]=contact.position.z;
                    gu.separation=contact.separation;
                    gpu_dy_gu_inputs_.push_back(gu);
                }
                ++prepared_contact_rows_;
                continue;
            }
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
            const float response = (a.inverse_mass+dot(ra,ra)) +
                (b.inverse_mass+dot(rb,rb));
            const float vel_multiplier = response > 0.0f
                ? 1.0f / response : 0.0f;
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
            const float target=(target_velocity-vrel);
            if(restitution<0.0f) {
                // DyContactPrepShared::computeCompliantContactCoefficients
                // disables damping for separated nonclosing contacts.
                // Force springs use unitResponse; acceleration springs
                // use the reciprocal response as a mass factor.
                const float enabled_damping=
                    separated && !closing ? 0.0f : damping;
                const float nrdt=seconds*restitution;
                const float a_coeff=seconds*
                    (enabled_damping-nrdt);
                const float mass_factor=acceleration_spring
                    ? vel_multiplier : 1.0f;
                const float x=1.0f/(1.0f+a_coeff*
                    (acceleration_spring ? 1.0f : response));
                const float bias=x*(-nrdt*penetration*mass_factor);
                row.velocity_multiplier=x*a_coeff*mass_factor;
                row.impulse_multiplier=1.0f-x;
                row.biased_error=target*row.velocity_multiplier-bias;
                row.unbiased_error=row.biased_error;
            } else {
                // Normal restitution keeps a distinct unbiased
                // velocity-pass error after the biased position passes.
                const float penetration_inv_dt_scaled=separated
                    ? penetration_inv_dt
                    : penetration*(inv_dt*rigid_contact_bias);
                const float scaled_bias=bouncing ? 0.0f :
                    vel_multiplier*std::max(
                        max_penetration_bias,penetration_inv_dt_scaled);
                row.velocity_multiplier=vel_multiplier;
                row.biased_error=target*vel_multiplier-scaled_bias;
                row.unbiased_error=target*vel_multiplier-
                    (bouncing?0.0f:std::max(scaled_bias,0.0f));
            }
            if (!std::isfinite(response) ||
                !std::isfinite(row.biased_error) ||
                !std::isfinite(row.unbiased_error) ||
                !std::isfinite(row.impulse_multiplier) ||
                !std::isfinite(row.velocity_multiplier))
                throw std::overflow_error("Native Dy contact preparation overflow");
            solver_contact_points_.push_back(row);
            ++prepared_contact_rows_;
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
                patch.row_offset=prepared_friction_rows_;
                continue;
            }
            if(gpu_normal_prep && manifold.pcm_result_index!=
               std::numeric_limits<std::uint32_t>::max()) {
                // A PCM box manager publishes one common Gu normal and
                // therefore one correlated patch. Retain the source
                // cached anchors as input; GPU Dy grows them directly
                // from device Gu contacts before building tangent rows.
                if(manifold.friction_patch_count!=1u ||
                   patch.contact_count!=manifold.count)
                    throw std::logic_error(
                        "Source GPU PCM friction is not one correlated patch");
                patch.row_offset=prepared_friction_rows_;
                // Device PCM grows up to two anchors in its own numerical
                // workspace; only their compact source slots are needed.
                gpu_dy_friction_anchors_.resize(
                    gpu_dy_friction_anchors_.size()+2u);
                prepared_friction_rows_+=4u;
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
            patch.row_offset=prepared_friction_rows_;
            if (count==0) continue;
            // DyContactPrep.cpp constructs tangents from relative linear
            // velocity, with the same 0.7071f normal-axis fallback and
            // 1e-4 squared-tangent threshold as the source.
            if(gpu_normal_prep) {
                // Sc retains the body-local anchors; the device authors
                // both tangent Jacobians without CPU row placeholders.
                for(std::size_t k=0u;k<count;++k) {
                    GpuDyFrictionAnchorInput anchor{};
                    const Vec3 x=patch.body0_anchors[k];
                    const Vec3 y=patch.body1_anchors[k];
                    anchor.local_a[0]=x.x;anchor.local_a[1]=x.y;
                    anchor.local_a[2]=x.z;
                    anchor.local_b[0]=y.x;anchor.local_b[1]=y.y;
                    anchor.local_b[2]=y.z;
                    gpu_dy_friction_anchors_.push_back(anchor);
                }
                prepared_friction_rows_+=count*2u;
                continue;
            }
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
                const Vec3 ra=rotate(mass_orientation(actor0),patch.body0_anchors[k]);
                const Vec3 rb=rotate(mass_orientation(actor1),patch.body1_anchors[k]);
                const Vec3 center0=mass_position(actor0),center1=mass_position(actor1);
                const Vec3 error=suppress_slop(sub(
                    {ra.x+center0.x,ra.y+center0.y,ra.z+center0.z},
                    {rb.x+center1.x,rb.y+center1.y,rb.z+center1.z}),pair_slop);
                for (const Vec3 tangent : {t0,t1}) {
                    const Vec3 ax0=suppress_slop(cross(ra,tangent),pair_slop);
                    const Vec3 ax1=suppress_slop(cross(rb,tangent),pair_slop);
                    const Vec3 j0=transform(a,ax0);
                    const Vec3 j1=transform(b,ax1);
                    const float response=(a.inverse_mass+dot(j0,j0))
                        +(b.inverse_mass+dot(j1,j1));
                    SolverContactFriction fr{};
                    fr.tangent=tangent;
                    fr.ra_x_t_sqrt_inertia=j0;
                    fr.rb_x_t_sqrt_inertia=j1;
                    fr.velocity_multiplier=response>0.0f
                        ? rigid_contact_bias/response : 0.0f;
                    fr.bias=dot(tangent,error)*inv_dt;
                    fr.target_velocity=-(dot(tangent,rel)
                        +dot(ax0,a.angular_velocity)
                        -dot(ax1,b.angular_velocity));
                    if (!std::isfinite(response) ||
                        !std::isfinite(fr.velocity_multiplier) ||
                        !std::isfinite(fr.bias) ||
                        !std::isfinite(fr.target_velocity))
                        throw std::overflow_error(
                            "Native source friction row preparation overflow");
                    solver_friction_rows_.push_back(fr);
                }
            }
            prepared_friction_rows_+=count*2u;
        }
    }
    // Swap only after the new rows and patch anchors are fully prepared.
    // Old shape-local anchors remain available throughout correlation.
    friction_patches_.swap(next_friction_patches_);
    friction_by_edge_.swap(next_friction_by_edge_);
    friction_cache_epoch_=next_cache_epoch;
}
void Scene::publish_cpu_gu_touch_history() {
    FYZZ_PROFILE_SCOPE("gu.touch_history");
    // CPU Gu touch transitions update the shared EdgeIndex history even
    // when the already prepared Sc task groups need no device admission.
    if(gpu_broadphase_ && gpu_broadphase_->has_gpu_dy_distance() &&
       gpu_broadphase_->has_gpu_pcm_box() &&
       (pcm_box_box_enabled_ || pcm_plane_box_enabled_) &&
       !gpu_pcm_pairs_.empty()) {
        // Sc publishes live Gu touch bits while building the contact
        // manifold. Only retired touches still require edge lookup here.
        for(const Pair pair:lost_contacts_) {
            const EdgeIndex edge=contact_edge_index(pair);
            if(edge==invalid_edge)continue;
            const auto it=std::lower_bound(
                gpu_pcm_pairs_.begin(),gpu_pcm_pairs_.end(),pair,
                [this](const GpuPcmBoxPair& record,Pair key) {
                    return island_edges_[record.edge].shapes<key;
                });
            if(it!=gpu_pcm_pairs_.end() &&
               island_edges_[it->edge].shapes==pair)continue;
            gpu_dy_gu_touches_.push_back({edge,2u});
        }
        if(!gpu_dy_gu_touches_.empty())
            gpu_broadphase_->enqueue_cpu_gu_touch_history(
                gpu_dy_gu_touches_);
    }
}
void Scene::prepare_gpu_dy_device_islands() {
    if(!gpu_dy_device_admission_enabled_ || gpu_dy_tasks_.empty())return;
    if(gpu_dy_tasks_.size()>gpu_broadphase_->gpu_dy_direct_task_limit())
        throw std::out_of_range(
            "Resident Sc island admission exceeds device task capacity");
    std::size_t prior_descriptors=0u,prior_bodies=0u;
    for(const auto& task:solver_tasks_) {
        if(task.constraint_end<prior_descriptors ||
           task.solver_body_end<prior_bodies)
            throw std::logic_error(
                "Source Sc island task ranges are not ordered");
        if(task.constraint_end-prior_descriptors>=gpu_dy_wide_task_threshold &&
           task.solver_body_end-prior_bodies>=gpu_dy_wide_task_threshold)
            throw std::logic_error(
                "Resident Sc compact task admission cannot execute a wide island");
        prior_descriptors=task.constraint_end;
        prior_bodies=task.solver_body_end;
    }
    if(gpu_dy_tasks_.size()<gpu_dy_responding_island_count_)
        throw std::logic_error(
            "Resident Sc task capacity cannot represent accurate islands");
    const auto narrow=[](std::size_t n) {
        if(n>=std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error(
                "Sc device island source index exceeds 32 bits");
        return static_cast<std::uint32_t>(n);
    };
    const auto position_limit=std::numeric_limits<std::uint8_t>::max();
    constexpr std::uint32_t invalid=0xffffffffu;
    constexpr std::uint32_t active=1u<<16u;
    constexpr std::uint32_t nonresponding=1u<<17u;
    constexpr std::uint32_t joint_bit=1u;
    constexpr std::uint32_t no_response=2u;
    constexpr std::uint32_t device_joint=16u;
    gpu_dy_island_node_flags_.assign(solver_rows_.size(),nonresponding);
    for(const BodyId id:solver_active_ids_) {
        const auto& actor=actors_[id];
        if(actor.motion!=Motion::Dynamic || actor.sleeping ||
           actor.solver_row_generation!=solver_preparation_generation_ ||
           actor.solver_row_index>=gpu_dy_island_node_flags_.size() ||
           !actor.dynamics.position_iterations)
            throw std::logic_error(
                "Resident Sc responding node lacks an accurate solver row");
        const auto pos=actor.dynamics.position_iterations;
        const auto vel=actor.dynamics.velocity_iterations;
        if(pos>position_limit || vel>position_limit)
            throw std::out_of_range(
                "Resident Sc island iteration word exceeds device encoding");
        auto& word=gpu_dy_island_node_flags_[actor.solver_row_index];
        if(word!=nonresponding)
            throw std::logic_error(
                "Resident Sc responding solver row assigned twice");
        word=pos|(std::uint32_t(vel)<<8u)|active;
    }
    const auto node=[&](BodyId id) {
        const auto& actor=actors_[id];
        if(actor.motion!=Motion::Dynamic ||
           actor.solver_row_generation!=solver_preparation_generation_ ||
           actor.solver_row_index>=gpu_dy_island_node_flags_.size() ||
           (gpu_dy_island_node_flags_[actor.solver_row_index]&nonresponding))
            return invalid;
        return narrow(actor.solver_row_index);
    };
    gpu_dy_island_edges_.clear();
    gpu_dy_island_edges_.reserve(
        indexed_joints_.size()+indexed_managers_.size());
    std::size_t joint_cursor=0u,manager_cursor=0u,descriptor_cursor=0u;
    std::size_t row_cursor=0u,full_descriptor_cursor=0u;
    for(const auto& task:solver_tasks_) {
        if(task.joint_end>indexed_joints_.size() ||
           task.manager_end>indexed_managers_.size() ||
           task.constraint_end>ordered_constraints_.size())
            throw std::logic_error(
                "Source Sc task exceeds its admitted edge tables");
        for(;joint_cursor<task.joint_end;++joint_cursor) {
            const auto edge=indexed_joints_[joint_cursor];
            const auto& graph=island_edges_[edge];
            if(graph.type!=IslandEdgeType::Constraint ||
               graph.constraint_id>=constraint_records_.size())
                throw std::logic_error(
                    "Source Sc device joint has no stable constraint");
            const auto& joint=constraint_records_[graph.constraint_id];
            const std::size_t rows=joint.kind==JointKind::Distance?1u:
                joint.kind==JointKind::Spherical?3u:
                joint.kind==JointKind::Fixed?6u:5u;
            std::uint32_t source=0u,kind=joint_bit|no_response;
            if(descriptor_cursor<task.constraint_end) {
                const auto& desc=ordered_constraints_[descriptor_cursor];
                if((desc.row_count&255u)!=0u && desc.index==row_cursor) {
                    if(desc.index>=gpu_dy_rows_.size())
                        throw std::logic_error(
                            "Source Sc joint row exceeds Dy input table");
                    const bool direct=gpu_dy_rows_[desc.index].reserved==2u;
                    source=direct?narrow(desc.index):
                        narrow(full_descriptor_cursor++);
                    kind=joint_bit|(direct?device_joint:0u);
                    ++descriptor_cursor;
                }
            }
            gpu_dy_island_edges_.push_back({
                edge,node(graph.pair.a),node(graph.pair.b),source,kind});
            row_cursor+=rows;
        }
        for(;manager_cursor<task.manager_end;++manager_cursor) {
            const auto index=indexed_managers_[manager_cursor];
            if(index>=manifolds_.size())
                throw std::logic_error(
                    "Source Sc island manager exceeds Gu arena");
            const auto& m=manifolds_[index];
            const auto pair=m.pair;
            std::uint32_t kind=no_response;
            if(descriptor_cursor<task.constraint_end) {
                const auto& desc=ordered_constraints_[descriptor_cursor];
                if((desc.row_count&255u)==0u && desc.index==index) {
                    if(m.friction_patch_count)kind=0u;
                    ++descriptor_cursor;
                }
            }
            gpu_dy_island_edges_.push_back({
                m.edge_handle,node(shapes_[pair.a].body),
                node(shapes_[pair.b].body),narrow(index),kind});
        }
        if(descriptor_cursor!=task.constraint_end)
            throw std::logic_error(
                "Source Sc device island descriptor order differs from Dy");
    }
    if(descriptor_cursor!=ordered_constraints_.size() ||
       joint_cursor!=indexed_joints_.size() ||
       manager_cursor!=indexed_managers_.size() ||
       row_cursor!=gpu_dy_rows_.size())
        throw std::logic_error(
            "Source Sc device admission did not cover its numerical tables");
    gpu_broadphase_->enqueue_gpu_dy_island_admission(
        gpu_dy_island_edges_,gpu_dy_island_node_flags_,
        {narrow(manifolds_.size()),narrow(gpu_dy_rows_.size()),
         narrow(full_descriptor_cursor)});
}

void Scene::solve_gpu_distance_constraints(float seconds) {
    FYZZ_PROFILE_SCOPE("dy.gpu_enqueue_join");
    // Source Dy owns task, body and 1D row order; the device receives
    // transient packed views of those exact prepared source inputs.
    if(ordered_constraints_.empty() && solver_active_ids_.empty() &&
       gpu_dy_free_ids_.empty()) {
        begin_contact_writeback();
        return;
    }
    const auto narrow=[](std::size_t value) {
        if(value>=std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Native GPU Dy row index exceeds 32 bits");
        return static_cast<std::uint32_t>(value);
    };
    const auto pack=[](float(&out)[3],Vec3 v) {
        out[0]=v.x;out[1]=v.y;out[2]=v.z;
    };
    gpu_dy_bodies_.resize(solver_rows_.size());
    for(std::size_t i=0;i<solver_rows_.size();++i) {
        const auto& row=solver_rows_[i];
        auto& body=gpu_dy_bodies_[i];
        // Dy initializes fresh transient deltas from the stable per-actor
        // inverse masses retained on the device.
        body.inverse_mass=row.data.inverse_mass;
    }
    const auto integration_count=solver_rows_.size()+gpu_dy_free_ids_.size();
    gpu_dy_integration_inputs_.assign(integration_count,{});
    gpu_dy_integrated_poses_.resize(integration_count);
    gpu_dy_sleep_states_.resize(integration_count);
    const auto pack_integration=[&](BodyId id,const Actor& actor,
                                   GpuDyBodyIntegrationInput& input) {
        input.source_actor=id;
        input.source_inverse_mass=actor.inverse_mass;
        const auto& d=actor.dynamics;
        input.linear_damping=d.linear_damping;
        input.angular_damping=d.angular_damping;
        input.max_linear_speed_squared=d.max_linear_velocity*d.max_linear_velocity;
        input.max_angular_speed_squared=d.max_angular_velocity*d.max_angular_velocity;
        pack(input.inverse_inertia_local,actor.inverse_inertia_local);
        input.dynamics_flags=(d.gravity_enabled?1u:0u)|
            (d.gyroscopic_forces_enabled?2u:0u);
        input.wake_counter=actor.wake_counter;
        pack(input.sleep_linear_acc,actor.sleep_linear_acc);
        pack(input.sleep_angular_acc,actor.sleep_angular_acc);
        input.counted_interactions=actor.num_counted_interactions;
        pack(input.pending_linear_velocity_delta,
             actor.pending_linear_velocity_delta);
        pack(input.pending_linear_acceleration,
             actor.pending_linear_acceleration);
        pack(input.pending_angular_velocity_delta,
             actor.pending_angular_velocity_delta);
        pack(input.pending_angular_acceleration,
             actor.pending_angular_acceleration);
        // Native Dy reconstructs its mass frame from the resident actor
        // pose and mass-local transform; Sc supplies only actor edits.
        pack(input.actor_position,actor.position);
        input.actor_orientation[0]=actor.orientation.x;
        input.actor_orientation[1]=actor.orientation.y;
        input.actor_orientation[2]=actor.orientation.z;
        input.actor_orientation[3]=actor.orientation.w;
        pack(input.mass_local_position,actor.mass_local_pose.position);
        const auto local_q=actor.mass_local_pose.orientation;
        input.mass_local_orientation[0]=local_q.x;input.mass_local_orientation[1]=local_q.y;
        input.mass_local_orientation[2]=local_q.z;input.mass_local_orientation[3]=local_q.w;
        input.source_shape_count=actor.simulation_shape_count;
        input.source_element=actor.first_simulation_shape;
        input.lock_flags=actor.dynamics.lock_flags;
        input.seconds=seconds;
        input.active=1u;
        // Every finite shape now retains the authored quaternion.
        // The sphere AABB remains rotation-invariant on the device.
        if(actor.first_simulation_shape!=invalid_shape)
            input.active|=4u;
    };
    for(const BodyId id:solver_active_ids_) {
        const auto& actor=actors_[id];
        const auto& row=solver_rows_[actor.solver_row_index];
        auto& input=gpu_dy_integration_inputs_[actor.solver_row_index];
        pack_integration(id,actor,input);
        pack(input.linear_velocity,row.data.linear_velocity);
        if(row.device_world_inertia)input.active|=32u;
        // GPU preintegration clamps and applies gyroscopic correction
        // before masking angular axes, just like CPU source preintegration.
        pack(input.angular_velocity,actor.angular_velocity);
        for(std::size_t axis=0;axis<3u;++axis) {
            const auto& inertia=row.data.sqrt_inverse_inertia[axis];
            input.sqrt_inverse_inertia[axis*3u]=inertia.x;
            input.sqrt_inverse_inertia[axis*3u+1u]=inertia.y;
            input.sqrt_inverse_inertia[axis*3u+2u]=inertia.z;
        }
    }
    // Kinematic 1D endpoints contribute authored velocities but are
    // not integrated or treated as dynamic solver-body owners.
    for(const EdgeIndex edge:indexed_joints_) {
        const auto pair=island_edges_[edge].pair;
        for(const BodyId id:{pair.a,pair.b}) {
            const auto& actor=actors_[id];
            if(actor.motion!=Motion::Kinematic)continue;
            if(actor.solver_row_generation!=solver_preparation_generation_ ||
               actor.solver_row_index>=solver_rows_.size())
                throw std::logic_error(
                    "Source GPU 1D kinematic lacks its prepared row");
            auto& input=gpu_dy_integration_inputs_[actor.solver_row_index];
            input.active=16u;
            pack(input.linear_velocity,actor.velocity);
            pack(input.angular_velocity,actor.angular_velocity);
        }
    }
    for(std::size_t i=0;i<gpu_dy_free_ids_.size();++i) {
        const auto& actor=actors_[gpu_dy_free_ids_[i]];
        auto& input=gpu_dy_integration_inputs_[solver_rows_.size()+i];
        pack_integration(gpu_dy_free_ids_[i],actor,input);
        pack(input.linear_velocity,actor.velocity);
        pack(input.angular_velocity,actor.angular_velocity);
    }
    if(gpu_dy_rows_.size()!=joint_rows_.size())
        throw std::logic_error(
            "Source GPU 1D row preparation did not cover source rows");
    gpu_dy_body_ordinals_.resize(solver_active_ids_.size());
    gpu_dy_tasks_.clear();
    if(gpu_dy_device_admission_enabled_ && !solver_tasks_.empty()) {
        // The device owns actual task ranges and responding-body order.
        // These spans provide capacity only; no host task/ordinal
        // records are constructed or uploaded for numerical execution.
        gpu_dy_tasks_.resize(solver_tasks_.size());
    }else {
        for(std::size_t i=0;i<solver_active_ids_.size();++i)
            gpu_dy_body_ordinals_[i]=narrow(
                actors_[solver_active_ids_[i]].solver_row_index);
        for(const auto& task:solver_tasks_) {
            const auto prev=gpu_dy_tasks_.empty()
                ?GpuDyDistanceTask{}:gpu_dy_tasks_.back();
            if(task.constraint_end==prev.descriptor_end &&
               task.partition_end==prev.partition_end &&
               task.solver_body_end==prev.body_end)
                continue;
            gpu_dy_tasks_.push_back({
                prev.descriptor_end,narrow(task.constraint_end),
                prev.partition_end,narrow(task.partition_end),
                prev.body_end,narrow(task.solver_body_end),
                task.position_iterations,task.velocity_iterations,0u});
        }
    }
    // Sc owns Gu point identity and correlated patch normals. The GPU
    // constructs every writable Dy contact row; only CPU-origin Gu
    // geometry needs a compact source upload.
    if(!solver_contact_points_.empty())
        throw std::logic_error("GPU Dy materialized CPU numerical contact rows");
    gpu_dy_contact_forces_.resize(prepared_contact_rows_);
    // Contact correlation already packed the Gu header normals in
    // source order. Validate that every CPU-origin manager owns its
    // exact range without traversing shadow solver contacts again.
    std::size_t cpu_gu_cursor=0u;
    for(const auto manager_index:indexed_managers_) {
        const auto& m=manifolds_[manager_index];
        if(!m.friction_patch_count)continue;
        if(m.solver_offset>gpu_dy_contact_forces_.size() ||
           m.count>gpu_dy_contact_forces_.size()-m.solver_offset)
            throw std::logic_error("Source GPU contact preparation range invalid");
        const auto prep_index=gpu_dy_contact_prep_indices_[manager_index];
        if(prep_index>=gpu_dy_contact_prep_pairs_.size())
            throw std::logic_error("Source GPU Dy contact pair index invalid");
        const auto& prep=gpu_dy_contact_prep_pairs_[prep_index];
        if(prep.pcm_pair_index!=0xffffffffu) {
            if(prep.cpu_gu_offset!=0xffffffffu)
                throw std::logic_error("Resident PCM acquired CPU Gu geometry");
            continue;
        }
        if(prep.cpu_gu_offset!=cpu_gu_cursor ||
           cpu_gu_cursor>gpu_dy_gu_inputs_.size() ||
           m.count>gpu_dy_gu_inputs_.size()-cpu_gu_cursor)
            throw std::logic_error("Source CPU Gu geometry lost numerical contact order");
        cpu_gu_cursor+=m.count;
    }
    if(cpu_gu_cursor!=gpu_dy_gu_inputs_.size())
        throw std::logic_error("Source CPU Gu geometry has trailing records");
    // One source friction anchor generates the two correlated tangents.
    // The device constructs both 56-byte tangent rows and returns one
    // combined world impulse per source friction anchor.
    if(!solver_friction_rows_.empty() ||
       (prepared_friction_rows_&1u)!=0u ||
       gpu_dy_friction_anchors_.size()!=prepared_friction_rows_/2u)
        throw std::logic_error(
            "GPU Dy must consume only source-correlated friction anchors");
    gpu_dy_friction_impulses_.resize(gpu_dy_friction_anchors_.size());
    gpu_dy_patches_.resize(friction_patches_.size());
    gpu_dy_manifolds_.assign(manifolds_.size(),{});
    std::size_t explicit_order_cursor=0u;
    constexpr std::uint32_t device_contact_order=0x80000000u;
    for(const auto manager_index:indexed_managers_) {
        const auto& m=manifolds_[manager_index];
        if(m.prepared_friction_epoch!=friction_cache_epoch_)
            throw std::logic_error("GPU Dy contact manager is not prepared");
        if(m.friction_patch_offset>friction_patches_.size() ||
           m.friction_patch_count>
               friction_patches_.size()-m.friction_patch_offset)
            throw std::logic_error("GPU Dy contact patch range invalid");
        // A dormant Gu manager retains geometry but has no prepared
        // normal rows; source CPU PGS skips its zero-patch descriptor.
        if(!m.friction_patch_count)continue;
        if(m.solver_offset>prepared_contact_rows_ ||
           m.count>prepared_contact_rows_-m.solver_offset)
            throw std::logic_error("GPU Dy contact row window invalid");
        if(manager_index>=gpu_dy_contact_prep_indices_.size() ||
           gpu_dy_contact_prep_indices_[manager_index]==0xffffffffu)
            throw std::logic_error("Source GPU contact pair was not prepared");
        gpu_dy_manifolds_[manager_index]={
            narrow(m.friction_patch_offset),m.friction_patch_count,
            narrow(m.solver_offset),m.count,
            gpu_dy_contact_prep_indices_[manager_index]};
        std::size_t cursor=m.solver_offset;
        for(std::size_t i=m.friction_patch_offset;
            i<m.friction_patch_offset+m.friction_patch_count;++i) {
            const auto& patch=friction_patches_[i];
            if(patch.row_offset>prepared_friction_rows_ ||
               patch.anchor_count>2u ||
               patch.anchor_count*2u>
                   prepared_friction_rows_-patch.row_offset ||
               patch.contact_count>m.count-(cursor-m.solver_offset))
                throw std::logic_error("GPU Dy friction patch rows invalid");
            const bool pcm_growth=m.pcm_result_index!=
                std::numeric_limits<std::uint32_t>::max() &&
                !patch.disable_friction;
            if(pcm_growth) {
                if(patch.row_offset%2u || patch.row_offset/2u+2u>
                   gpu_dy_friction_anchors_.size())
                    throw std::logic_error(
                        "Source GPU PCM has no room for two friction anchors");
                for(std::size_t k=0u;k<patch.anchor_count;++k) {
                    auto& anchor=gpu_dy_friction_anchors_[patch.row_offset/2u+k];
                    pack(anchor.local_a,patch.body0_anchors[k]);
                    pack(anchor.local_b,patch.body1_anchors[k]);
                }
            }
            std::uint32_t order_offset;
            if(m.pcm_result_index!=
               std::numeric_limits<std::uint32_t>::max()) {
                if(cursor>=device_contact_order ||
                   patch.contact_count>device_contact_order-cursor)
                    throw std::overflow_error(
                        "GPU PCM direct contact order exceeds address space");
                // PCM retains source order in the resident contact arena.
                // Do not upload a second list of identical indices.
                order_offset=device_contact_order|narrow(cursor);
            } else {
                order_offset=narrow(explicit_order_cursor);
                if(explicit_order_cursor>gpu_dy_contact_order_.size() ||
                   patch.contact_count>
                       gpu_dy_contact_order_.size()-explicit_order_cursor)
                    throw std::logic_error(
                        "CPU Gu contact correlation order is incomplete");
                explicit_order_cursor+=patch.contact_count;
            }
            gpu_dy_patches_[i]={
                narrow(patch.row_offset),patch.contact_count,
                patch.anchor_count,(patch.broken?1u:0u)|
                    (pcm_growth?0x80000000u:0u),
                patch.static_friction,patch.dynamic_friction,
                order_offset,0u};
            cursor+=patch.contact_count;
        }
        if(cursor!=m.solver_offset+m.count)
            throw std::logic_error("GPU Dy source patch contact stream truncated");
    }
    // Zero-response managers have no DyContactPrep record. Preserve
    // their original endpoint ordinals and dynamic classification in
    // the otherwise unused manifold slots for device task admission.
    for(const auto& desc:ordered_constraints_) {
        if((desc.row_count&255u)!=0u)continue;
        if(desc.index>=gpu_dy_manifolds_.size())
            throw std::logic_error("GPU Dy zero-response manager index invalid");
        auto& manager=gpu_dy_manifolds_[desc.index];
        if(manager.patch_count!=0u)continue;
        auto row0=desc.row0,row1=desc.row1;
        if(row0==invalid_solver_row && row1==invalid_solver_row) {
            // Compact numerical contacts carry only their manager
            // token. Recover endpoints solely for zero-response
            // managers, which have no device contact-prep record.
            const auto& manifold=manifolds_[desc.index];
            const auto pair=manifold.pair;
            const bool reversed=island_edges_[
                manifold.edge_handle].gu_order_reversed;
            const auto dense_row=[&](BodyId id) {
                const auto& actor=actors_[id];
                if(!rigid_node(actor.motion))return invalid_solver_row;
                if(actor.solver_row_generation!=solver_preparation_generation_ ||
                   actor.solver_row_index>=solver_rows_.size() ||
                   actor.solver_row_index>=invalid_solver_row)
                    throw std::logic_error(
                        "Source zero-response manager lacks a prepared solver row");
                return static_cast<std::uint32_t>(actor.solver_row_index);
            };
            row0=dense_row(shapes_[
                reversed?pair.b:pair.a].body);
            row1=dense_row(shapes_[
                reversed?pair.a:pair.b].body);
        }
        manager.patch_offset=row0;
        manager.order_offset=row1;
        manager.prep_index=desc.row_count;
    }
    if(explicit_order_cursor!=gpu_dy_contact_order_.size())
        throw std::logic_error(
            "GPU Dy explicit Gu contact order was not fully consumed");
    // Sc owns persistent island identities. Device admission consumes
    // physical EdgeIndex touch history and authors only numerical ranges.
    prepare_gpu_dy_device_islands();
    gpu_broadphase_->enqueue_gpu_dy_distance(
        gpu_dy_bodies_,gpu_dy_rows_,ordered_constraints_,
        {},gpu_dy_body_ordinals_,gpu_dy_tasks_,
        gpu_dy_contact_forces_,gpu_dy_gu_inputs_,
        {},gpu_dy_friction_anchors_,
        gpu_dy_patches_,gpu_dy_manifolds_,gpu_dy_contact_order_,
        gpu_dy_contact_prep_pairs_,
        gpu_dy_integration_inputs_,gpu_dy_integrated_poses_,true,
        gpu_dy_friction_impulses_,true,true,true,gpu_dy_sleep_states_,
        gpu_dy_joint_inputs_);
    // Sc's Gu-indexed force buffer and retired-manager bookkeeping are
    // independent of the submitted numerical solve. Prepare them while
    // the ordered GPU Dy task graph executes, before its result join.
    begin_contact_writeback();
    const auto consume_preintegration=[this](BodyId id) {
        auto& actor=actors_[id];
        actor.pending_angular_velocity_delta={};
        actor.pending_linear_velocity_delta={};
        if(!actor.dynamics.retain_accelerations) {
            actor.pending_linear_acceleration={};
            actor.pending_angular_acceleration={};
        }
    };
    for(const BodyId id:solver_active_ids_)consume_preintegration(id);
    for(const BodyId id:gpu_dy_free_ids_)consume_preintegration(id);
    gpu_broadphase_->finish_gpu_dy_distance();
    gpu_dy_pose_ready_=true;
    // Dy keeps transient PGS deltas on device. Sc receives world-space
    // final and position-phase velocities needed by its task continuations.
    if(!solver_contact_points_.empty() ||
       gpu_dy_contact_forces_.size()!=prepared_contact_rows_)
        throw std::logic_error("GPU Dy returned mismatched contact force slots");
    // Solved friction rows stay on device; Sc receives only the anchor
    // world impulses needed by its existing source patch writeback.
    // Sc retains the authoritative body-local strong-friction history.
    // GPU PCM selected changed anchors using the same Gu point ordinals;
    // reconstruct their local coordinates only after Dy's source task
    // completed and its compact patch report has been fetched.
    const auto local_point=[this](const Actor& actor,Vec3 world) {
        const auto q=mass_orientation(actor);
        const auto p=mass_position(actor);
        return rotate(Quat{-q.x,-q.y,-q.z,q.w},
            {world.x-p.x,world.y-p.y,world.z-p.z});
    };
    for(const auto manager_index:indexed_managers_) {
        const auto& m=manifolds_[manager_index];
        if(m.pcm_result_index==std::numeric_limits<std::uint32_t>::max())
            continue;
        const bool reversed=island_edges_[m.edge_handle].gu_order_reversed;
        const Actor& a=actors_[shapes_[reversed?m.pair.b:m.pair.a].body];
        const Actor& b=actors_[shapes_[reversed?m.pair.a:m.pair.b].body];
        for(std::size_t i=m.friction_patch_offset;
            i<m.friction_patch_offset+m.friction_patch_count;++i) {
            auto& patch=friction_patches_[i];
            if(patch.disable_friction)continue;
            const auto& returned=gpu_dy_patches_[i];
            const auto count=returned.anchor_count;
            if(count>2u)throw std::logic_error(
                "Source GPU PCM returned more than two friction anchors");
            const std::uint32_t report=returned.reserved;
            for(std::uint32_t k=0u;k<count;++k) {
                const std::uint32_t source=(report>>(3u+3u*k))&7u;
                if(!source)continue;
                const auto j=source-1u;
                if(j>=m.count)throw std::logic_error(
                    "Source GPU friction selected an invalid Gu witness");
                patch.contact_ids[k]=static_cast<std::uint16_t>(j);
                patch.body0_anchors[k]=local_point(a,m.points[j].position);
                patch.body1_anchors[k]=local_point(b,m.points[j].position);
            }
            if(count==0u) {
                patch.body0_anchors[0]={};
                patch.body1_anchors[0]={};
            }
            patch.anchor_count=static_cast<std::uint8_t>(count);
        }
    }
    for(std::size_t i=0;i<friction_patches_.size();++i)
        friction_patches_[i].broken=gpu_dy_patches_[i].broken!=0u;
}

void Scene::solve_constraint_task(std::size_t task_index) {
    FYZZ_PROFILE_SCOPE("dy.cpu_solve");
    // DySolverConstraintsShared.h::solveDynamicContacts /
    // solveStaticContacts and DySolverConstraints.cpp friction patch solve,
    // with the PhysX default of four position
    // iterations and one final velocity/writeback iteration. The final
    // writeback is itself a solve pass, not a second velocity sweep.
    // Constraint preparation and iteration use source PxReal arithmetic.
    // Source task groups carry their own maximum of constituent
    // PxsBodyCore solverIterationCounts. In particular, an unrelated
    // high-iteration island must never increase another group's solve.
    // Static-world solver data must not be written by independent island
    // workers through the shared source zero-mass body. Each task uses a
    // stack-local copy of that same prepared zero-response state.
    SolverBodyRow local_world_solver_row=world_solver_row_;
    constexpr float max_float =
        static_cast<float>(std::numeric_limits<float>::max());
    const auto dot = [](Vec3 u, Vec3 v) {
        return u.x*v.x + u.y*v.y + u.z*v.z;
    };
    // Dy task groups own disjoint accurate islands. Derive exclusive
    // ranges from the source cumulative task descriptors; no per-task
    // contact/solver-body copies or alternate owner are allocated.
    if(task_index>=solver_tasks_.size())
        throw std::out_of_range("Unknown source Dy solver task");
    const auto& task=solver_tasks_[task_index];
    const std::size_t task_constraint_begin=task_index
        ? solver_tasks_[task_index-1].constraint_end:0;
    const std::size_t task_solver_begin=task_index
        ? solver_tasks_[task_index-1].solver_body_end:0;
    if(task.constraint_end<task_constraint_begin ||
       task.constraint_end>ordered_constraints_.size() ||
       task.solver_body_end<task_solver_begin ||
       task.solver_body_end>solver_active_ids_.size())
        throw std::logic_error("Native Dy task owns an invalid descriptor range");
    const unsigned position_iterations=task.position_iterations;
    // DySolverControl::solveV_Blocks executes max(requested, 1):
    // a zero requested velocity count skips optional velocity passes,
    // never the mandatory final solve/writeback pass. Preserve the raw
    // requested count in the source task descriptor.
    const unsigned velocity_iterations=std::max<unsigned>(
        task.velocity_iterations,1u);
    for (unsigned iter = 0;
         iter < position_iterations + velocity_iterations; ++iter) {
        // Unified conflict partitions order both source 1D and contact
        // descriptors through this same PGS position/velocity pass.
        for(std::size_t desc=task_constraint_begin;
            desc<task.constraint_end;++desc) {
            const auto descriptor=ordered_constraints_[desc];
            if(descriptor.row_count) {
                auto& a=descriptor.row0==invalid_solver_row
                    ? local_world_solver_row:solver_rows_[descriptor.row0];
                auto& b=descriptor.row1==invalid_solver_row
                    ? local_world_solver_row:solver_rows_[descriptor.row1];
                const auto end=std::size_t(descriptor.index)+descriptor.row_count;
                for(std::size_t index=descriptor.index;index<end;++index) {
                    auto& row=joint_rows_[index];
                    const float normal_delta=
                        row.axis.x*
                            (a.solver.linear_velocity_delta.x-
                             b.solver.linear_velocity_delta.x)+
                        row.axis.y*
                            (a.solver.linear_velocity_delta.y-
                             b.solver.linear_velocity_delta.y)+
                        row.axis.z*
                            (a.solver.linear_velocity_delta.z-
                             b.solver.linear_velocity_delta.z)+
                        dot(row.angular_a,a.solver.angular_state)-
                        dot(row.angular_b,b.solver.angular_state);
                    const float constant=iter<position_iterations
                        ? row.constant:row.unbiased_constant;
                    const float next=std::clamp(
                        row.applied_force+
                        normal_delta*row.velocity_multiplier+constant,
                        -max_float,max_float);
                    const float impulse=next-row.applied_force;
                    const float ia=impulse*row.inverse_mass_a;
                    const float ib=impulse*row.inverse_mass_b;
                    const Vec3 va=a.solver.linear_velocity_delta;
                    const Vec3 vb=b.solver.linear_velocity_delta;
                    const Vec3 aa=a.solver.angular_state,ab=b.solver.angular_state;
                    const std::array<float,12> candidate{
                        va.x+ia*row.axis.x,va.y+ia*row.axis.y,
                        va.z+ia*row.axis.z,vb.x-ib*row.axis.x,
                        vb.y-ib*row.axis.y,vb.z-ib*row.axis.z,
                        aa.x+impulse*row.angular_a.x,
                        aa.y+impulse*row.angular_a.y,
                        aa.z+impulse*row.angular_a.z,
                        ab.x-impulse*row.angular_b.x,
                        ab.y-impulse*row.angular_b.y,
                        ab.z-impulse*row.angular_b.z};
                    for(std::size_t k=0;k<candidate.size();++k) {
                        if(!std::isfinite(candidate[k]))
                            throw std::overflow_error("Source 1D solver velocity overflow");
                    }
                    a.solver.linear_velocity_delta={candidate[0],candidate[1],candidate[2]};
                    b.solver.linear_velocity_delta={candidate[3],candidate[4],candidate[5]};
                    a.solver.angular_state={candidate[6],candidate[7],candidate[8]};
                    b.solver.angular_state={candidate[9],candidate[10],candidate[11]};
                    row.applied_force=next;
                }
            } else {
                // Each contact descriptor retains its source friction-patch
                // order: all normal rows followed immediately by tangent rows.
                // Gu force-buffer offsets are independent of this permutation.
                const auto index=descriptor.index;
                const auto& manifold = manifolds_[index];
                // Stable source-orientation and dense solver-body rows
                // are resolved once at scatter, never per PGS iteration.
                auto& a=descriptor.row0==invalid_solver_row
                    ? local_world_solver_row:solver_rows_[descriptor.row0];
                auto& b=descriptor.row1==invalid_solver_row
                    ? local_world_solver_row:solver_rows_[descriptor.row1];
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
                    float normal_impulse=0.0f;
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
                    const float error = iter < position_iterations
                        ? row.biased_error : row.unbiased_error;
                    const float delta_candidate = std::max(
                        -normal_velocity *
                            row.velocity_multiplier + error,
                        -row.applied_force);
                    // DySolverConstraintsShared::solveDynamicContacts
                    // caps EACH normal row's accumulated impulse, not
                    // the manager total and not the body velocity delta.
                    // DySolverConstraintsShared multiplies the old force
                    // by impulseMultiplier for every compliant PGS solve.
                    // Noncompliant rows retain a multiplier of exactly 1.
                    const float next_impulse=std::min(row.max_impulse,
                        row.impulse_multiplier*row.applied_force+delta_candidate);
                    const float delta = next_impulse-row.applied_force;
                    const float da = delta*a.data.inverse_mass;
                    const float db = delta*b.data.inverse_mass;
                    const std::array<float,12> results{
                        va.x+n.x*da,
                        va.y+n.y*da,
                        va.z+n.z*da,
                        vb.x-n.x*db,
                        vb.y-n.y*db,
                        vb.z-n.z*db,
                        a.solver.angular_state.x+row.ra_x_n_sqrt_inertia.x*delta,
                        a.solver.angular_state.y+row.ra_x_n_sqrt_inertia.y*delta,
                        a.solver.angular_state.z+row.ra_x_n_sqrt_inertia.z*delta,
                        b.solver.angular_state.x-row.rb_x_n_sqrt_inertia.x*delta,
                        b.solver.angular_state.y-row.rb_x_n_sqrt_inertia.y*delta,
                        b.solver.angular_state.z-row.rb_x_n_sqrt_inertia.z*delta};
                    for (std::size_t k=0;k<results.size();++k) {
                        if (!std::isfinite(results[k]))
                            throw std::overflow_error(
                                "Native Dy contact velocity delta exceeds float range");
                    }
                    a.solver.linear_velocity_delta = {results[0],results[1],results[2]};
                    b.solver.linear_velocity_delta = {results[3],results[4],results[5]};
                    a.solver.angular_state = {results[6],results[7],results[8]};
                    b.solver.angular_state = {results[9],results[10],results[11]};
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
                        const float coefficient=patch.anchor_count==2
                            ? 0.5f : 1.0f;
                        const float max_static=
                            patch.static_friction*coefficient*normal_impulse;
                        const float max_dynamic=
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
                            const float bias=iter < position_iterations
                                ? fr.bias : 0.0f;
                            const float candidate=fr.applied_force+
                                (fr.target_velocity-bias-tangent_delta)*
                                    fr.velocity_multiplier;
                            const bool clamped=std::abs(candidate)>max_static;
                            // Preserve the source's signed FMin/FMax
                            // expression without invoking std::clamp on
                            // negative, unordered compliant patch bounds.
                            const float next=source_contact_friction_limit(
                                candidate,max_dynamic,clamped);
                            const float delta=next-fr.applied_force;
                            const float da=delta*a.data.inverse_mass;
                            const float db=delta*b.data.inverse_mass;
                            const auto& j0=fr.ra_x_t_sqrt_inertia;
                            const auto& j1=fr.rb_x_t_sqrt_inertia;
                            const std::array<float,12> results{
                                va.x+n.x*da,
                                va.y+n.y*da,
                                va.z+n.z*da,
                                vb.x-n.x*db,
                                vb.y-n.y*db,
                                vb.z-n.z*db,
                                a.solver.angular_state.x+j0.x*delta,
                                a.solver.angular_state.y+j0.y*delta,
                                a.solver.angular_state.z+j0.z*delta,
                                b.solver.angular_state.x-j1.x*delta,
                                b.solver.angular_state.y-j1.y*delta,
                                b.solver.angular_state.z-j1.z*delta};
                            for (std::size_t v=0;v<results.size();++v) {
                                if (!std::isfinite(results[v]))
                                    throw std::overflow_error(
                                        "Native Dy friction velocity delta exceeds float range");
                            }
                            a.solver.linear_velocity_delta={
                                results[0],results[1],results[2]};
                            b.solver.linear_velocity_delta={
                                results[3],results[4],results[5]};
                            a.solver.angular_state={
                                results[6],results[7],results[8]};
                            b.solver.angular_state={
                                results[9],results[10],results[11]};
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
        }
        if (iter + 1 == position_iterations) {
            // DySolverControl.cpp::saveMotionVelocities occurs immediately
            // after the last biased/conclude position iteration, BEFORE
            // unbiased velocity iterations alter the solver deltas.
            for(std::size_t body=task_solver_begin;
                body<task.solver_body_end;++body) {
                const BodyId id=solver_active_ids_[body];
                auto& row = prepared_solver_row(id);
                row.motion_linear_delta = row.solver.linear_velocity_delta;
                row.motion_angular_state = row.solver.angular_state;
            }
        }
    }
    }
void Scene::publish_contact_events() {
    FYZZ_PROFILE_SCOPE("sc.contact_events");
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
    const auto append_event=[&](Pair pair,ContactEventKind kind,std::uint8_t count,
                                std::size_t offset,double impulse) {
        const Pair bodies=body_pair(pair);
        const bool reverse=bodies.a>bodies.b;
        next_contact_events_.push_back({reverse?bodies.b:bodies.a,reverse?bodies.a:bodies.b,
            kind,count,offset,impulse,reverse?pair.b:pair.a,reverse?pair.a:pair.b});
    };
    std::size_t lost=0,found=0;
    for(std::size_t index:contact_report_managers_) {
        const auto& manifold=manifolds_[index];
        while(lost<lost_contacts_.size() &&
              lost_contacts_[lost]<manifold.pair) {
            const Pair pair=lost_contacts_[lost++];
            append_event(pair,ContactEventKind::Lost,0,next_contact_event_points_.size(),0.0);
        }
        while(found<found_contacts_.size() &&
              found_contacts_[found]<manifold.pair)++found;
        const bool is_new=found<found_contacts_.size() &&
            found_contacts_[found]==manifold.pair;
        // ScNPhaseCore::processPersistentContactEvents tests each
        // ShapeInteraction actor's Sc::ActorSim::isActive state. Kinematic
        // BodySims are active during target motion and their subsequent
        // settling update, even though they are not Dy dynamic bodies.
        // An active kinematic alone can authorize a persistent report;
        // an idle kinematic only retained by graph references cannot.
        const auto sc_actor_active=[&](BodyId id) {
            const auto& actor=actors_[id];
            return (actor.motion==Motion::Dynamic && !actor.sleeping) ||
                   (actor.motion==Motion::Kinematic &&
                    (actor.kinematic_target_pending ||
                     actor.kinematic_settling));
        };
        if(!is_new &&
           !sc_actor_active(shapes_[manifold.pair.a].body) &&
           !sc_actor_active(shapes_[manifold.pair.b].body))
            continue;
        if(manifold.contact_offset>normal_impulse_writeback_.size() ||
           manifold.count>normal_impulse_writeback_.size()-
                              manifold.contact_offset)
            throw std::logic_error(
                "Contact notification references unpublished Gu forces");
        double impulse=0.0;
        const auto offset=next_contact_event_points_.size();
        for(std::size_t i=0;i<manifold.count;++i) {
            const std::size_t point_index=manifold.contact_offset+i;
            const auto& point=contact_points_[point_index];
            const double normal_force=normal_impulse_writeback_[point_index];
            impulse+=normal_force;
            Vec3 outward=point.normal;
            if(island_edges_[manifold.edge_handle].gu_order_reversed !=
               (shapes_[manifold.pair.a].body>shapes_[manifold.pair.b].body))
                outward={-outward.x,-outward.y,-outward.z};
            next_contact_event_points_.push_back({
                point.position,outward,point.separation,normal_force});
        }
        if(!std::isfinite(impulse))
            throw std::overflow_error(
                "Contact notification impulse is not finite");
        append_event(manifold.pair,is_new?ContactEventKind::Found:ContactEventKind::Persists,
                     manifold.count,offset,impulse);
    }
    for(;lost<lost_contacts_.size();++lost) {
        const Pair pair=lost_contacts_[lost];
        append_event(pair,ContactEventKind::Lost,0,next_contact_event_points_.size(),0.0);
    }
    // Publish only after every Sc/Dy stage and this event preparation
    // succeeds. A rejected step leaves the previous returned span and
    // backing storage untouched; the scratch is reused on retry.
    contact_event_points_.swap(next_contact_event_points_);
    contact_events_.swap(next_contact_events_);
}
void Scene::begin_contact_writeback() {
    // DySolverConstraints.cpp::writeBackContact copies final forceBuffer
    // impulses after the last velocity/writeback solve. Gu contact order,
    // not the reordered Dy partition list, owns output offsets. Here the
    // source force buffer is represented by SolverContactPoint::applied_force
    // and the output retains the signed source impulses.
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
}

void Scene::write_back_contacts(std::size_t first,std::size_t last) {
    // PxsSolverEndTask writes the Gu-indexed force/patch windows owned
    // by this source island task before the task completion edge. The
    // zeroed Gu arena is prepared once, not once per task.
    if(first>last || last>indexed_managers_.size())
        throw std::logic_error("Native Dy task owns an invalid Gu manager range");
    for(std::size_t manager=first;manager<last;++manager) {
        const std::size_t manifold_index=indexed_managers_[manager];
        const auto& manifold=manifolds_[manifold_index];
        if(manifold.prepared_friction_epoch!=friction_cache_epoch_ ||
           manifold.contact_offset>contact_points_.size() ||
           manifold.count>contact_points_.size()-manifold.contact_offset)
            throw std::logic_error(
                "Active Dy writeback references invalid Gu contact owner");
        if(manifold.friction_patch_count==0)continue;
        const std::size_t row_count=gpu_dy_pose_ready_
            ?gpu_dy_contact_forces_.size():solver_contact_points_.size();
        if(manifold.solver_offset>row_count ||
           manifold.count>row_count-manifold.solver_offset)
            throw std::logic_error(
                "Native Dy contact writeback offset exceeds solver rows");
        for (std::size_t i=0;i<manifold.count;++i) {
            const double impulse=gpu_dy_pose_ready_
                ?static_cast<double>(gpu_dy_contact_forces_[
                    manifold.solver_offset+i])
                :static_cast<double>(solver_contact_points_[
                    manifold.solver_offset+i].applied_force);
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
        const std::size_t friction_rows=gpu_dy_pose_ready_
            ?prepared_friction_rows_:solver_friction_rows_.size();
        for (std::size_t p=manifold.friction_patch_offset;
             p<manifold.friction_patch_offset+
                 manifold.friction_patch_count; ++p) {
            const auto& source=friction_patches_[p];
            if (!(source.pair==manifold.pair) || source.anchor_count>2 ||
                source.row_offset>friction_rows ||
                source.anchor_count*2>friction_rows-source.row_offset)
                throw std::logic_error(
                    "Native Dy friction writeback patch ownership invalid");
            FrictionPatchWriteback& result=friction_impulse_writeback_[p];
            result={};
            result.pair=source.pair;
            result.anchor_count=source.anchor_count;
            result.broken=source.broken;
            // DyFrictionCorrelation stores body0 anchors in Sc's
            // source-ordered frame, not the canonical Gu report pair.
            const BodyId body0=shapes_[island_edges_[manifold.edge_handle]
                .gu_order_reversed ? source.pair.b : source.pair.a].body;
            const auto& actor=actors_[body0];
            for (std::size_t k=0;k<source.anchor_count;++k) {
                const Vec3 local=source.body0_anchors[k];
                const Vec3 rotated=rotate(mass_orientation(actor),local);
                const Vec3 center=mass_position(actor);
                result.anchor_positions[k]={center.x+rotated.x,center.y+rotated.y,center.z+rotated.z};
                // DyContactPrep.cpp::writeBackContactFriction reconstructs
                // world impulse by summing both tangent vectors scaled
                // by each source SolverContactFriction::appliedForce.
                if(gpu_dy_pose_ready_) {
                    if((source.row_offset&1u)!=0u ||
                       source.row_offset/2u+k>=gpu_dy_friction_impulses_.size())
                        throw std::logic_error(
                            "Source GPU friction anchor writeback range invalid");
                    const auto& impulse=
                        gpu_dy_friction_impulses_[source.row_offset/2u+k];
                    result.anchor_impulses[k]={
                        impulse.world[0],impulse.world[1],impulse.world[2]};
                }else {
                    const auto& x=solver_friction_rows_[source.row_offset+2*k];
                    const auto& y=solver_friction_rows_[source.row_offset+2*k+1];
                    result.anchor_impulses[k]={
                        static_cast<double>(x.tangent.x)*x.applied_force+
                            static_cast<double>(y.tangent.x)*y.applied_force,
                        static_cast<double>(x.tangent.y)*x.applied_force+
                            static_cast<double>(y.tangent.y)*y.applied_force,
                        static_cast<double>(x.tangent.z)*x.applied_force+
                            static_cast<double>(y.tangent.z)*y.applied_force};
                }
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
    FYZZ_PROFILE_SCOPE("sc.threshold_events");
    // DySolverConstraints.cpp::writeBackContact emits a threshold element
    // only for a supported rigid-body manager with a nonzero final
    // normal impulse and an enabled reportThreshold on either body.
    // DyThresholdTable groups genuine static contacts against a dynamic
    // body using the SAME source world solver node. A kinematic rigid
    // body has its OWN source PxNodeIndex and PxsSolverBodyData, so
    // dynamic/kinematic pairs retain both actor indices and both report
    // thresholds. Never merge two kinematic supports into one world key.
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
        const Pair bodies=body_pair(manifold.pair);
        const Pair pair=ordered_pair(bodies.a,bodies.b);
        const bool dynamic_a=actors_[pair.a].motion==Motion::Dynamic;
        const bool dynamic_b=actors_[pair.b].motion==Motion::Dynamic;
        if (!dynamic_a && !dynamic_b)
            continue; // no admitted response chain without a dynamic
        const bool static_world=
            !rigid_node(actors_[pair.a].motion) ||
            !rigid_node(actors_[pair.b].motion);
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
    // Only actual static endpoints map to the shared world identity
    // before DyThresholdTable aggregation. Kinematic contacts retain
    // their individually indexed source pair identities. Sort the
    // reusable candidate arena because the static-world mapping can
    // disrupt Gu manager order.
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
void Scene::publish_solver_bodies(std::size_t first,std::size_t last) {
    // Each Dy task chain publishes only its own contiguous solver-body
    // range before integration and sleepCheck. No other active island's
    // body state can be committed by this chain.
    if(first>last || last>solver_active_ids_.size())
        throw std::logic_error("Native Dy task solver-body range invalid");
    for(std::size_t index=first;index<last;++index) {
        const BodyId id=solver_active_ids_[index];
        auto& actor = actors_[id];
        if (!actor.alive || actor.motion != Motion::Dynamic) continue;
        Vec3 linear{},angular{};
        if(gpu_dy_pose_ready_) {
            const auto& result=gpu_dy_bodies_[actor.solver_row_index];
            linear=result.final_linear_velocity();
            angular=result.final_angular_velocity();
        }else {
            const auto& row = prepared_solver_row(id);
            const Vec3& d = row.solver.linear_velocity_delta;
            linear={row.data.linear_velocity.x+d.x,
                    row.data.linear_velocity.y+d.y,
                    row.data.linear_velocity.z+d.z};
            lock_linear_axes(linear,actor.dynamics.lock_flags);
            // Solver angular state is expressed in sqrt-inertia space.
            Vec3 angular_state=row.solver.angular_state;
            lock_angular_axes(angular_state,actor.dynamics.lock_flags);
            const auto& i = row.data.sqrt_inverse_inertia;
            angular={
                row.data.angular_velocity.x + i[0].x*angular_state.x +
                    i[1].x*angular_state.y + i[2].x*angular_state.z,
                row.data.angular_velocity.y + i[0].y*angular_state.x +
                    i[1].y*angular_state.y + i[2].y*angular_state.z,
                row.data.angular_velocity.z + i[0].z*angular_state.x +
                    i[1].z*angular_state.y + i[2].z*angular_state.z};
        }
        if (!finite(linear) || !finite(angular))
            throw std::overflow_error("Native solver publication is not finite");
        step_checkpoint_.save_motion(*this,id);
        actor.velocity = linear;
        actor.angular_velocity = angular;
    }
}
void Scene::StepCheckpoint::save_motion(Scene& scene, BodyId id) noexcept {
    if(!scene.activity_checkpoint_active_)return;
    if(scene.motion_undo_generations_[id]==scene.motion_undo_generation_)
        return;
    const Actor& body=scene.actors_[id];
    if(!body.alive || !rigid_node(body.motion))return;
    scene.motion_undo_generations_[id]=scene.motion_undo_generation_;
    // capture() already reserved one record for each live dynamic body;
    // neither a speculative wake nor source deactivation can allocate.
    motions.push_back({id,body.position,body.velocity,body.angular_velocity,
        body.orientation,body.pending_angular_velocity_delta,
        body.pending_linear_velocity_delta,body.pending_linear_acceleration,
        body.pending_angular_acceleration,
        body.wake_counter,body.sleep_linear_acc,body.sleep_angular_acc,
        body.ready_for_sleep,body.sleeping,body.activating,
        body.kinematic_target_pending,body.kinematic_settling});
    for(ShapeId shape=body.first_shape;shape!=invalid_shape;shape=scene.shapes_[shape].next) {
        const auto& value=scene.shapes_[shape];
        shape_bounds.push_back({shape,value.encoded_bounds,value.cached_bounds_current});
    }
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
    // Accurate splitEdges copies the entire indexed-manager EdgeIndex,
    // including speculative activity fields. A speculative third pass
    // can retire those fields concurrently; its sparse activity journal
    // and this full-record copy must observe an indivisible edge state.
    // Neither graph traversal nor indexed-manager rewiring is locked.
    if(id<initial_edges) {
        const std::scoped_lock lock(scene.third_pass_edge_mutex_);
        edge_undo.push_back({id,scene.island_edges_[id]});
    }
}
void Scene::StepCheckpoint::save_activity(
    Scene& scene, EdgeIndex id) noexcept {
    if(!scene.activity_checkpoint_active_ || id>=initial_edges)return;
    // Generation-deduplicated and pre-reserved before publication.
    if(scene.activity_undo_generations_[id]==
       scene.activity_undo_generation_)return;
    scene.activity_undo_generations_[id]=scene.activity_undo_generation_;
    const auto& edge=scene.island_edges_[id];
    activity_undo.push_back({id,edge.speculative_state,edge.accurate_state,
                             edge.interaction_active});
}
void Scene::StepCheckpoint::save_reference(
    Scene& scene, BodyId node) noexcept {
    if(!scene.activity_checkpoint_active_)return;
    if(scene.reference_undo_generations_[node]==
       scene.reference_undo_generation_)return;
    scene.reference_undo_generations_[node]=
        scene.reference_undo_generation_;
    const auto& speculative=scene.speculative_islands_.active_reference_counts;
    const auto& accurate=scene.accurate_islands_.active_reference_counts;
    reference_undo.push_back({node,
        node<initial_speculative_references?speculative[node]:0,
        node<initial_accurate_references?accurate[node]:0,
        scene.actors_[node].num_counted_interactions});
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
        (accurate?accurate_hint_undo:speculative_hint_undo).push_back(
            {node,routes[node],edges[node],hops[node],stamps[node],accurate});
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
    save_graph_ownership(scene,false);
    save_graph_ownership(scene,true);
}
void Scene::StepCheckpoint::save_graph_ownership(
    Scene& scene, bool accurate) {
    const std::uint8_t bit=accurate?2u:1u;
    if(graph_ownership_mask.load(std::memory_order_acquire)&bit)return;
    // Source IslandSims have disjoint structural graph owners. A
    // parallel findRoute split snapshots only its own owner, before
    // changing its node chains or handle lists. Retry uses the same
    // graph-local checkpoint; quiet frames copy neither owner.
    const auto save=[](IslandGraph& saved,const IslandGraph& live) {
        saved.islands.assign(live.islands.begin(),live.islands.end());
        saved.node_island.assign(live.node_island.begin(),live.node_island.end());
        saved.node_type.assign(live.node_type.begin(),live.node_type.end());
        saved.node_next.assign(live.node_next.begin(),live.node_next.end());
        saved.node_prev.assign(live.node_prev.begin(),live.node_prev.end());
        saved.node_static_touches.assign(live.node_static_touches.begin(),
                                          live.node_static_touches.end());
        saved.free_handles.assign(live.free_handles.begin(),
                                   live.free_handles.end());
        saved.active_island_seen.assign(live.active_island_seen.begin(),
                                        live.active_island_seen.end());
        saved.active_island_visit_stamp=live.active_island_visit_stamp;
    };
    if(accurate) {
        save(accurate_islands,scene.accurate_islands_);
        accurate_nodes.assign(scene.accurate_island_nodes_.begin(),
                               scene.accurate_island_nodes_.end());
        accurate_offsets.assign(scene.accurate_island_offsets_.begin(),
                                 scene.accurate_island_offsets_.end());
    } else save(speculative_islands,scene.speculative_islands_);
    // One checkpoint event per frame even if both producer copies
    // complete simultaneously. The mask records successful copies only.
    const auto previous=graph_ownership_mask.fetch_or(
        bit,std::memory_order_acq_rel);
    if(!previous)
        std::atomic_ref(scene.island_graph_checkpoint_count_)
            .fetch_add(1,std::memory_order_relaxed);
}
void Scene::StepCheckpoint::capture(Scene& scene, bool capture_solver_state) {
    // Reuse capacities across frames. The checkpoint contains only state
    // that persists beyond a step or is exposed by Scene's read-only APIs.
    // It intentionally excludes rebuilt solver constraints and scratch.
    motions.clear();
    shape_bounds.clear();
    shape_bounds.reserve(scene.rigid_shape_count_);
    speculative_ready_before.assign(
        scene.speculative_ready_nodes_.begin(),
        scene.speculative_ready_nodes_.end());
    // A currently awake rigid may become ready once in this Dy frame.
    // Reserve before source task-local integration, reuse on quiet frames.
    scene.speculative_ready_nodes_.reserve(scene.live_dynamic_count_);
    {
        // Source wakeIslands can change actor state on collision-only
        // publications before third-pass lost-edge repair. Save only
        // actually changed bodies, not a dense copy on every query.
        // Reserve before entering the noexcept island publication.
        if(++scene.motion_undo_generation_==0) {
            std::fill(scene.motion_undo_generations_.begin(),
                      scene.motion_undo_generations_.end(),0);
            scene.motion_undo_generation_=1;
        }
        scene.motion_undo_generations_.resize(scene.actors_.size(),0);
        motions.reserve(scene.live_dynamic_count_+
                        scene.active_kinematic_targets_.size());
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
    scene.activity_undo_generations_.resize(scene.island_edges_.size(),0);
    activity_undo.reserve(scene.island_edges_.size());
    speculative_islands.activated_edges.reserve(
        scene.island_edges_.size());
    accurate_islands.activated_edges.reserve(
        scene.island_edges_.size());
    speculative_hint_undo.clear();
    accurate_hint_undo.clear();
    head_undo.clear();
    initial_edges=scene.island_edges_.size();
    initial_active_interactions=scene.active_contact_interaction_count_;
    initial_speculative_hints=scene.speculative_fast_route_.size();
    initial_accurate_hints=scene.accurate_fast_route_.size();
    speculative_hint_epoch=scene.speculative_hint_epoch_;
    accurate_hint_epoch=scene.accurate_hint_epoch_;
    initial_speculative_heads=scene.speculative_node_first_.size();
    initial_accurate_heads=scene.accurate_node_first_.size();
    handles_saved=false;
    graph_ownership_mask.store(0,std::memory_order_release);
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
        saved.active_kinematic_nodes.assign(
            live.active_kinematic_nodes.begin(),
            live.active_kinematic_nodes.end());
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
        saved.active_constraint_edge_count=live.active_constraint_edge_count;
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
    // Keep noexcept edge/reference transitions preallocated after
    // capture; kinematic membership itself is not a second actor owner.
    scene.speculative_islands_.active_kinematic_nodes.reserve(
        scene.actors_.size());
    scene.accurate_islands_.active_kinematic_nodes.reserve(
        scene.actors_.size());
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
    speculative_islands.nodes_to_deactivate.reserve(scene.actors_.size());
    speculative_islands.deactivating_edges.reserve(
        scene.island_edges_.size());
    accurate_islands.deactivating_edges.reserve(
        scene.island_edges_.size());
    speculative_islands.deactivating_constraint_edges.reserve(
        scene.island_edges_.size());
    accurate_islands.deactivating_constraint_edges.reserve(
        scene.island_edges_.size());
    // The rollback owner's spare activation buffers become the next
    // publication's working storage after the swap. Pre-reserve before
    // noexcept wakeIslands can activate a retained sleeping connection.
    speculative_islands.activated_constraint_edges.reserve(
        scene.island_edges_.size());
    accurate_islands.activated_constraint_edges.reserve(
        scene.island_edges_.size());
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
    plane_pair_membership_dirty=scene.plane_pair_membership_dirty_;
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
    scene.speculative_islands_.nodes_to_deactivate.swap(
        speculative_islands.nodes_to_deactivate);
    scene.speculative_islands_.deactivating_edges.swap(
        speculative_islands.deactivating_edges);
    scene.accurate_islands_.deactivating_edges.swap(
        accurate_islands.deactivating_edges);
    scene.speculative_islands_.activated_edges.swap(
        speculative_islands.activated_edges);
    scene.accurate_islands_.activated_edges.swap(
        accurate_islands.activated_edges);
    scene.speculative_islands_.deactivating_constraint_edges.swap(
        speculative_islands.deactivating_constraint_edges);
    scene.accurate_islands_.deactivating_constraint_edges.swap(
        accurate_islands.deactivating_constraint_edges);
    scene.speculative_islands_.activated_constraint_edges.swap(
        speculative_islands.activated_constraint_edges);
    scene.accurate_islands_.activated_constraint_edges.swap(
        accurate_islands.activated_constraint_edges);
    // The checkpoint owns the prior published lists; its former scratch
    // buffers must start empty before the next activation publication.
    scene.speculative_islands_.activated_edges.clear();
    scene.accurate_islands_.activated_edges.clear();
    scene.speculative_islands_.activated_constraint_edges.clear();
    scene.accurate_islands_.activated_constraint_edges.clear();
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
    // Recover unread prior-Dy wake readiness even if third pass consumed
    // the queue before a later solver/bounds publication threw.
    scene.speculative_ready_nodes_.swap(speculative_ready_before);
    // Restore the exact pre-publication route cache and epochs so retries
    // reuse valid prior paths. Only written hint slots were journaled;
    // replay and shrinking newly admitted slots cannot allocate.
    const auto restore_hints=[&](const std::vector<HintUndo>& undo) {
        for(auto it=undo.rbegin();it!=undo.rend();++it) {
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
    };
    restore_hints(speculative_hint_undo);
    restore_hints(accurate_hint_undo);
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
        scene.island_edges_[it->id]=it->previous;
    // Structural edge records can be taken after an activity change.
    // The compact activity journal is authoritative for the original
    // eACTIVE/eACTIVATING state and must replay LAST for each old slot.
    for(auto it=activity_undo.rbegin();it!=activity_undo.rend();++it) {
        scene.island_edges_[it->id].speculative_state=it->speculative;
        scene.island_edges_[it->id].accurate_state=it->accurate;
        scene.island_edges_[it->id].interaction_active=
            it->interaction_active;
    }
    while(scene.island_edges_.size()>initial_edges)
        scene.island_edges_.pop_back();
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
    {
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
            a.kinematic_target_pending=motion.kinematic_target_pending;
            a.kinematic_settling=motion.kinematic_settling;
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
    const auto graph_mask=graph_ownership_mask.load(std::memory_order_acquire);
    if(graph_mask&1u) {
        scene.speculative_islands_.islands.swap(speculative_islands.islands);
        scene.speculative_islands_.node_island.swap(speculative_islands.node_island);
        scene.speculative_islands_.node_type.swap(speculative_islands.node_type);
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
    if(!(graph_mask&1u))
        for(auto handle:scene.speculative_islands_.active_islands)
            if(handle<scene.speculative_islands_.islands.size())
                scene.speculative_islands_.islands[handle].active_index=
                    invalid_instance;
    scene.speculative_islands_.active_islands.swap(
        speculative_islands.active_islands);
    scene.speculative_islands_.activating_nodes.swap(
        speculative_islands.activating_nodes);
    if(!(graph_mask&1u))
        for(std::size_t i=0;i<scene.speculative_islands_.active_islands.size();++i)
            scene.speculative_islands_.islands[
                scene.speculative_islands_.active_islands[i]].active_index=i;
    scene.speculative_islands_.nodes_to_deactivate.swap(
        speculative_islands.nodes_to_deactivate);
    scene.speculative_islands_.deactivating_edges.swap(
        speculative_islands.deactivating_edges);
    scene.speculative_islands_.activated_edges.swap(
        speculative_islands.activated_edges);
    scene.speculative_islands_.deactivating_constraint_edges.swap(
        speculative_islands.deactivating_constraint_edges);
    scene.speculative_islands_.activated_constraint_edges.swap(
        speculative_islands.activated_constraint_edges);
    scene.speculative_islands_.active_edge_count=
        speculative_islands.active_edge_count;
    scene.speculative_islands_.active_constraint_edge_count=
        speculative_islands.active_constraint_edge_count;
    scene.speculative_islands_.initial_active_node_count=
        speculative_islands.initial_active_node_count;
    if(graph_mask&2u) {
        scene.accurate_islands_.islands.swap(accurate_islands.islands);
        scene.accurate_islands_.node_island.swap(accurate_islands.node_island);
        scene.accurate_islands_.node_type.swap(accurate_islands.node_type);
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
    if(!(graph_mask&2u))
        for(auto handle:scene.accurate_islands_.active_islands)
            if(handle<scene.accurate_islands_.islands.size())
                scene.accurate_islands_.islands[handle].active_index=
                    invalid_instance;
    scene.accurate_islands_.active_islands.swap(
        accurate_islands.active_islands);
    scene.accurate_islands_.activating_nodes.swap(
        accurate_islands.activating_nodes);
    if(!(graph_mask&2u))
        for(std::size_t i=0;i<scene.accurate_islands_.active_islands.size();++i)
            scene.accurate_islands_.islands[
                scene.accurate_islands_.active_islands[i]].active_index=i;
    scene.accurate_islands_.nodes_to_deactivate.swap(
        accurate_islands.nodes_to_deactivate);
    scene.accurate_islands_.deactivating_edges.swap(
        accurate_islands.deactivating_edges);
    scene.accurate_islands_.activated_edges.swap(
        accurate_islands.activated_edges);
    scene.accurate_islands_.deactivating_constraint_edges.swap(
        accurate_islands.deactivating_constraint_edges);
    scene.accurate_islands_.activated_constraint_edges.swap(
        accurate_islands.activated_constraint_edges);
    scene.accurate_islands_.active_edge_count=
        accurate_islands.active_edge_count;
    scene.accurate_islands_.active_constraint_edge_count=
        accurate_islands.active_constraint_edge_count;
    scene.accurate_islands_.initial_active_node_count=
        accurate_islands.initial_active_node_count;
    // The reference arrays belong to their original graph owners even
    // after a full island rebuild. Rewind only changed endpoint slots
    // after structural ownership and edge-state undo have completed.
    for(auto it=reference_undo.rbegin();it!=reference_undo.rend();++it) {
        scene.actors_[it->node].num_counted_interactions=
            it->counted_interactions;
        if(it->node<scene.speculative_islands_.active_reference_counts.size())
            scene.speculative_islands_.active_reference_counts[it->node]=
                it->speculative;
        if(it->node<scene.accurate_islands_.active_reference_counts.size())
            scene.accurate_islands_.active_reference_counts[it->node]=
                it->accurate;
    }
    scene.speculative_islands_.active_reference_counts.resize(
        initial_speculative_references);
    scene.accurate_islands_.active_reference_counts.resize(
        initial_accurate_references);
    // Restore the pre-publication kinematic permutation with O(active
    // kinematic) work only on rejected transactions.
    const auto restore_kinematics=[](IslandGraph& graph,IslandGraph& saved) {
        for(BodyId id:graph.active_kinematic_nodes)
            if(id<graph.active_kinematic_index.size())
                graph.active_kinematic_index[id]=invalid_instance;
        graph.active_kinematic_nodes.swap(saved.active_kinematic_nodes);
        for(std::size_t i=0;i<graph.active_kinematic_nodes.size();++i)
            graph.active_kinematic_index[graph.active_kinematic_nodes[i]]=i;
    };
    restore_kinematics(scene.speculative_islands_,speculative_islands);
    restore_kinematics(scene.accurate_islands_,accurate_islands);
    if(graph_mask&2u) {
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
    scene.broadphase_endpoint_membership_dirty_=true;
    scene.sap_pair_registry_dirty_=true;
    scene.dirty_=dirty;
    // Rollback restores authoritative actor motion; no scratch view
    // from the rejected Sc/Dy frame is allowed to stand in for it.
    scene.gpu_broadphase_snapshot_current_=false;
    scene.plane_pair_membership_dirty_=plane_pair_membership_dirty;
    scene.authored_activation_pending_=authored_activation_pending;
    // GPU source histories already rotate at successful fetch, before
    // the Sc/Dy transaction has committed. Rejecting this transaction
    // must force their original AABB/SAP owner to rebuild next frame.
    if(scene.gpu_broadphase_)
        scene.gpu_broadphase_->invalidate_after_rollback();
    for(const auto& undo:shape_bounds) {
        auto& shape=scene.shapes_[undo.id];
        scene.refresh_shape_pose(shape);
        shape.encoded_bounds=undo.encoded_bounds;
        shape.cached_bounds_current=undo.cached_bounds_current;
    }
}

void Scene::RigidStepJoin::finish_accurate_third_pass() {
    // The accurate and speculative IslandSims run independent
    // ThirdPassTask continuations into the SAME PostThirdPassTask.
    if(accurate_third_pass_joined || third_pass_joined)
        throw std::logic_error(
            "Native ScPipeline completed accurate third pass twice/out of order");
    accurate_third_pass_joined=true;
}
void Scene::RigidStepJoin::finish_speculative_third_pass() {
    if(speculative_third_pass_joined || third_pass_joined)
        throw std::logic_error(
            "Native ScPipeline completed speculative third pass twice/out of order");
    speculative_third_pass_joined=true;
}
void Scene::RigidStepJoin::finish_third_pass() {
    // PxsSimpleIslandManager::thirdPassIslandGen wires BOTH island task
    // completions into PostThirdPassTask. Its handle retirement and
    // postThirdPassIslandGen continuation cannot run after only one.
    if(!accurate_third_pass_joined || !speculative_third_pass_joined ||
       third_pass_joined)
        throw std::logic_error(
            "Native ScPipeline postThirdPassIslandGen missing island producer");
    third_pass_joined=true;
}
void Scene::RigidStepJoin::finish_lost_contacts() {
    // Sc::Scene::processLostContacts3 completes after the lost-touch
    // and third-pass consumers have finished. The source continuation
    // contributes one of mPostSolver's two independent producer edges.
    if(!third_pass_joined || lost_contacts_joined)
        throw std::logic_error(
            "Native ScPipeline completed lost contacts before third pass or twice");
    lost_contacts_joined=true;
}
void Scene::RigidStepJoin::dispatch_solver_tasks(
    std::span<std::uint32_t> stamps,std::uint32_t generation) {
    // Source PxsSolverEndTask contributes one completion edge per
    // real Dy task chain, not merely an anonymous task-count decrement.
    // Generation stamps reuse the Scene's task storage across frames.
    if(solver_dispatched || generation==0)
        throw std::logic_error(
            "Native ScPipeline dispatched duplicate/invalid Dy task chains");
    solver_dispatched=true;
    solver_completion_stamps=stamps;
    completion_generation=generation;
    outstanding_solver_tasks=stamps.size();
}
void Scene::RigidStepJoin::finish_solver_task(std::size_t task_index) {
    const std::lock_guard lock(solver_completion_mutex);
    if(!solver_dispatched ||
       task_index>=solver_completion_stamps.size() ||
       solver_completion_stamps[task_index]==completion_generation ||
       !outstanding_solver_tasks)
        throw std::logic_error(
            "Native Dy solver end task was invalid or already completed");
    solver_completion_stamps[task_index]=completion_generation;
    --outstanding_solver_tasks;
}
void Scene::RigidStepJoin::require_post_solver() const {
    // ScPipeline::updateDynamics wires mProcessLostContactsTask3 and
    // every PxsSolverEndTask into mPostSolver. A manager handle cannot
    // be reclaimed, and no completed result can be published, before
    // both producers have joined.
    if(!third_pass_joined || !lost_contacts_joined ||
       !solver_dispatched || outstanding_solver_tasks)
        throw std::logic_error(
            "Native ScPipeline postSolver reached before producer completion");
}
void Scene::RigidStepJoin::finish_post_solver() {
    require_post_solver();
    if(post_solver_completed)
        throw std::logic_error("Native ScPipeline completed postSolver twice");
    post_solver_completed=true;
}
void Scene::RigidStepJoin::finish_after_integration() {
    // afterIntegration depends on postSolver, and publishes completed
    // transform/bounds state only after all PxsSolverEndTask joins.
    if(!post_solver_completed || after_integration_completed)
        throw std::logic_error(
            "Native afterIntegration ran before postSolver or twice");
    after_integration_completed=true;
}
void Scene::RigidStepJoin::finish_finalization() {
    // Finalization runs after afterIntegration and CPU reporting. A
    // completed solver task alone cannot publish a finished World frame.
    if(!after_integration_completed || finalization_completed)
        throw std::logic_error(
            "Native ScPipeline finalized before afterIntegration or twice");
    finalization_completed=true;
}
void Scene::RigidStepJoin::publish_results() {
    if(!finalization_completed || results_published)
        throw std::logic_error(
            "Native ScPipeline published before finalization or twice");
    results_published=true;
}

void Scene::prepare_unconstrained_velocities(float seconds) {
    FYZZ_PROFILE_SCOPE("dy.preintegration");
    // Accurate IslandSim sleep arbitration has completed, permitting
    // Dy preIntegration on its awake island ranges. Speculative
    // processLostEdges remains independent of Actor motion state.
    constexpr float gravity = -9.81f;
    calculate_kinematic_velocities(1.0f/seconds);
    // Dy::DynamicsContext takes the accurate *active* island ranges,
    // not every actor or every dormant island in the full graph.
    prepare_active_accurate_island_batches();
    for(std::size_t island:active_accurate_island_batches_)
    for(std::size_t index=accurate_island_offsets_[island];
        index<accurate_island_offsets_[island+1];++index) {
        auto& actor=actors_[accurate_island_nodes_[index]];
        if(actor.sleeping)continue;
        const auto& d=actor.dynamics;
        if(gpu_dy_active()) {
            // Dy's float gravity term cannot be admitted if its
            // timestep product already overflows the source arithmetic.
            if(d.gravity_enabled && !std::isfinite(gravity*seconds))
                throw std::overflow_error(
                    "Native GPU unconstrained gravity timestep overflow");
            if(d.gyroscopic_forces_enabled && !actor.angular_inertia_available)
                throw std::logic_error(
                    "Native gyroscopic motion requires representable inertia");
            // Velocity mods remain authoritative until the device solve
            // succeeds. The frame journal restores them on rejection.
            step_checkpoint_.save_motion(*this,accurate_island_nodes_[index]);
            dirty_=true;
            continue;
        }
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
                mass_orientation(actor),seconds);
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
}
void Scene::integrate_solver_task(
    std::size_t task_index,float seconds,std::vector<BodyId>& ready_nodes) {
    // One PxsSolverEndTask owns both integrateCore and sleepCheck for
    // its accurate-island batch range. Caller and worker only mutate
    // actors inside mutually exclusive task groups. The frame journal
    // has pre-recorded every active rigid actor before parallel entry.
    const auto& task=solver_tasks_[task_index];
    auto free_index=task_index?solver_tasks_[task_index-1].gpu_free_end:0u;
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
            // Scene clock increments in advance_collision_internal; only
            // this frame's actual contact participants own motion snapshots.
            const bool solved_contact =
                actor.solver_row_generation==solver_preparation_generation_ &&
                prepared_solver_row(id).prepared_generation==
                    solver_preparation_generation_;
            if(gpu_dy_pose_ready_ && !solved_contact) {
                if(free_index>=task.gpu_free_end ||
                   gpu_dy_free_ids_[free_index]!=id)
                    throw std::logic_error(
                        "Source GPU free-body velocity publication order invalid");
                const auto& result=gpu_dy_integration_inputs_[
                    solver_rows_.size()+free_index];
                actor.velocity={result.linear_velocity[0],
                    result.linear_velocity[1],result.linear_velocity[2]};
                actor.angular_velocity={result.angular_velocity[0],
                    result.angular_velocity[1],result.angular_velocity[2]};
            }
            lock_linear_axes(actor.velocity,actor.dynamics.lock_flags);
            Vec3 motion_linear = actor.velocity;
            Vec3 motion_angular = actor.angular_velocity;
            if(solved_contact && !gpu_dy_pose_ready_) {
                const auto& row = prepared_solver_row(id);
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
            if(!gpu_dy_pose_ready_) {
                lock_linear_axes(motion_linear,actor.dynamics.lock_flags);
                if(!finite(motion_linear) || !finite(motion_angular))
                    throw std::overflow_error(
                        "Native position-iteration motion overflow");
            }
            Vec3 p{};
            Quat q{};
            if(gpu_dy_pose_ready_) {
                if(!solved_contact && (free_index>=task.gpu_free_end ||
                   gpu_dy_free_ids_[free_index]!=id))
                    throw std::logic_error("Source GPU free-body order mismatch");
                const auto pose_index=solved_contact?actor.solver_row_index:
                    solver_rows_.size()+free_index++;
                if(pose_index>=gpu_dy_integrated_poses_.size() ||
                   gpu_dy_integrated_poses_[pose_index].valid!=1u)
                    throw std::logic_error(
                        "Source GPU Dy integrateCore pose missing");
                const auto& result=gpu_dy_integrated_poses_[pose_index];
                p={result.position[0],result.position[1],result.position[2]};
                q={result.orientation[0],result.orientation[1],
                   result.orientation[2],result.orientation[3]};
            }else {
                const Vec3 center=mass_position(actor);
                p={center.x+motion_linear.x*seconds,
                   center.y+motion_linear.y*seconds,
                   center.z+motion_linear.z*seconds};
                if(!finite(p))
                    throw std::overflow_error(
                        "Native position-iteration motion overflow");
                q=integrate_quaternion(
                    mass_orientation(actor),motion_angular,seconds);
                const auto local=actor.mass_local_pose.orientation;
                q=multiply(q,Quat{-local.x,-local.y,-local.z,local.w});
                const Vec3 offset=rotate(q,actor.mass_local_pose.position);
                p={p.x-offset.x,p.y-offset.y,p.z-offset.z};
                if(!finite(q))
                    throw std::overflow_error("Native CPU orientation overflow");
            }
            if(!finite(p) || !finite(q))
                throw std::overflow_error("Source GPU Dy integrateCore pose overflow");
            step_checkpoint_.save_motion(*this,id);
            remember_pre_integration_pose(id);
            actor.position = p;
            actor.orientation = q;
            if(gpu_dy_pose_ready_) {
                const auto result_index=solved_contact
                    ? actor.solver_row_index
                    : solver_rows_.size()+free_index-1u;
                if(result_index>=gpu_dy_sleep_states_.size())
                    throw std::logic_error(
                        "Native GPU sleep state ordinal is invalid");
                const auto& result=gpu_dy_sleep_states_[result_index];
                const Vec3 linear{result.linear_acc[0],
                    result.linear_acc[1],result.linear_acc[2]};
                const Vec3 angular{result.angular_acc[0],
                    result.angular_acc[1],result.angular_acc[2]};
                if(!finite(linear) || !finite(angular) ||
                   !std::isfinite(result.wake_counter) ||
                   result.wake_counter<0.0f)
                    throw std::overflow_error(
                        "Native GPU Dy sleep state is not finite");
                const bool was_ready=actor.ready_for_sleep;
                actor.sleep_linear_acc=linear;
                actor.sleep_angular_acc=angular;
                actor.wake_counter=result.wake_counter;
                actor.ready_for_sleep=result.wake_counter==0.0f;
                if(actor.ready_for_sleep && !was_ready)
                    ready_nodes.push_back(id);
                continue;
            }
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
                    static_cast<float>(actor.num_counted_interactions);
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
            const bool was_ready=actor.ready_for_sleep;
            actor.ready_for_sleep=(wc==0.0f);
            if(actor.ready_for_sleep && !was_ready)
                ready_nodes.push_back(id);
            if(actor.ready_for_sleep) {
                actor.sleep_linear_acc={};
                actor.sleep_angular_acc={};
            }
        }
        }
}
void Scene::complete_solver_end_task(
    std::size_t task_index,float seconds,std::vector<BodyId>& ready_nodes) {
    FYZZ_PROFILE_SCOPE("dy.integrate_sleep");
    // PxsSolverEndTask owns the contiguous source task chain through
    // writeback, final-velocity publication, integrateCore and
    // sleepCheck. Task boundaries already reflect accurate island
    // ownership; no actor or Gu manager may appear in a second chain.
    if(task_index>=solver_tasks_.size())
        throw std::out_of_range("Unknown source PxsSolverEndTask");
    const auto& task=solver_tasks_[task_index];
    const std::size_t first_manager=task_index
        ? solver_tasks_[task_index-1].manager_end:0;
    const std::size_t first_body=task_index
        ? solver_tasks_[task_index-1].solver_body_end:0;
    write_back_contacts(first_manager,task.manager_end);
    publish_solver_bodies(first_body,task.solver_body_end);
    integrate_solver_task(task_index,seconds,ready_nodes);
}
void Scene::step(float seconds) {
    for(EdgeIndex edge:active_constraint_handles_)
        if(island_edges_[edge].constraint_id==invalid_edge)
            throw std::logic_error(
                "Unconfigured source constraint edge has no Dy 1D solver row");
    if (!std::isfinite(seconds) || seconds <= 0.0f)
        throw std::invalid_argument("Native CPU timestep must be finite and positive");

    // Source: ScBodyCore.cpp::BodyCore dynamic defaults, and
    // DyBodyCoreIntegrator.h::bodyCoreComputeUnconstrainedVelocity:
    // gravity, linear/angular damping, then PxReal speed clamps.
    // A failed frame must not consume impulses or publish partial
    // contacts, island nodes, body motion or result streams. Snapshot only
    // authoritative state; all per-step Dy work buffers remain reusable.
    FYZZ_PROFILE_BEGIN_FRAME();
    try {
        FYZZ_PROFILE_SCOPE("sc.checkpoint_capture");
        step_checkpoint_.capture(*this);
    } catch (...) {
        FYZZ_PROFILE_END_FRAME(frame_index_,gpu_dy_active(),false);
        throw;
    }
    if(++rigid_integration_generation_==0)
        ++rigid_integration_generation_;
    RigidStepJoin join;
    try {
    // ScPipeline: Gu second pass and wakeIslands publish both source
    // ThirdPassTask inputs. After accurate completion, Dy preIntegration
    // may overlap speculative processLostEdges. PostThirdPass,
    // contact-manager retirement and constraint solving join both.
    // IslandSim::wakeIslands snapshots the existing active prefix before
    // activation. Preserve the same boundary when source contact
    // publication subsequently rebuilds island handles.
    accurate_islands_.initial_active_node_count=
        accurate_islands_.active_nodes.size();
    speculative_islands_.initial_active_node_count=
        speculative_islands_.active_nodes.size();
    advance_collision_internal(false,true);
    // The source first/second-pass wakeIslands handoffs precede
    // third-pass processLostEdges, preserving old-island wake reach.
    activate_published_islands();
    consume_activated_contact_interactions();
    // PxsSimpleIslandManager schedules TWO complete independent
    // ThirdPassTasks. Each producer repairs its lost routes, reconciles
    // its own active IslandSim, and makes its own sleep decision.
    // Dispatch once per frame when both graph tasks have actual work,
    // rather than launching separate workers for repair and sleep.
    bool speculative_work=!speculative_dirty_nodes_.empty() ||
                          !speculative_ready_nodes_.empty() ||
                          !speculative_split_sleep_candidates_.empty();
    if(!speculative_work)
        for(BodyId id:speculative_islands_.active_kinematic_nodes)
            if(actors_[id].kinematic_settling &&
               actors_[id].ready_for_sleep) {
                speculative_work=true;
                break;
            }
    const bool accurate_work=!accurate_dirty_nodes_.empty() ||
                             !accurate_islands_.active_islands.empty();
    const bool both_dirty=!speculative_dirty_nodes_.empty() &&
                          !accurate_dirty_nodes_.empty();
    bool speculative_split=false,accurate_split=false;
    bool dy_prepared=false;
    // Route-only work over short independent components loses time to
    // the wake/complete handoff even with a persistent worker. Keep
    // source-order serial execution below the measured work floor.
    // Prior-Dy sleep events remain eligible without a route-size gate.
    const bool independent_sleep_work=
        !speculative_ready_nodes_.empty() ||
        !speculative_split_sleep_candidates_.empty();
    if(parallel_third_pass_enabled_ && speculative_work && accurate_work &&
       (independent_sleep_work || worthwhile_parallel_route_work())) {
        dispatch_third_pass_worker(false);
        try {
            accurate_split=process_third_pass_island_task(true);
            join.finish_accurate_third_pass();
            ++concurrent_third_pass_dispatches_;
            if(!accurate_islands_.nodes_to_deactivate.empty())
                synchronize_active_island_nodes_for_graph(true);
            // Dy preIntegration is safe after accurate third-pass
            // completion while speculative IslandSim remains active.
            prepare_unconstrained_velocities(seconds);
            dy_prepared=true;
            ++overlapped_dy_preintegration_dispatches_;
        } catch(...) {
            // The worker outlives the stack frame. Rollback cannot
            // run while its speculative graph mutation is outstanding.
            join_third_pass_worker();
            throw;
        }
        speculative_split=complete_third_pass_worker();
        join.finish_speculative_third_pass();
        if(both_dirty)++concurrent_lost_route_dispatches_;
    } else {
        speculative_split=process_third_pass_island_task(false);
        accurate_split=process_third_pass_island_task(true);
        join.finish_speculative_third_pass();
        join.finish_accurate_third_pass();
        if(!accurate_islands_.nodes_to_deactivate.empty())
            synchronize_active_island_nodes();
    }
    if(speculative_split || accurate_split)++island_topology_revision_;
    // PxsSimpleIslandManager::PostThirdPassTask may consume deactivations
    // only after both producer edges have joined.
    join.finish_third_pass();
    consume_deactivated_contact_interactions();
    // PxsSimpleIslandManager::PostThirdPassTask returns destroyed
    // EdgeIndex handles to its allocator only after both IslandSims'
    // processLostEdges consumers have completed. The lifetime remains
    // checkpointed until the whole frame succeeds.
    retire_destroyed_contact_edges();
    join.finish_lost_contacts();
    if(!dy_prepared)prepare_unconstrained_velocities(seconds);
    prepare_solver_bodies();
    partition_constraints();
    // Dispatch is over the ACTUAL Dy::updatePostKinematic task-group
    // descriptors, including groups without contact rows. No task
    // allocations or second island registry are introduced.
    solver_task_completion_stamps_.resize(solver_tasks_.size(),0);
    if(++solver_task_completion_generation_==0) {
        std::fill(solver_task_completion_stamps_.begin(),
                  solver_task_completion_stamps_.end(),0);
        solver_task_completion_generation_=1;
    }
    join.dispatch_solver_tasks(
        solver_task_completion_stamps_,solver_task_completion_generation_);
    prepare_constraints(seconds);
    publish_cpu_gu_touch_history();
    const bool gpu_dy_distance=gpu_dy_active();
    gpu_dy_pose_ready_=false;
    if(gpu_dy_distance)solve_gpu_distance_constraints(seconds);
    else begin_contact_writeback();
    // Each source Dy group owns independent accurate islands and disjoint
    // contact, friction, distance and solver-body rows. Admit complete
    // PGS -> PxsSolverEndTask chains when both ranges justify the worker.
    // All chains join before Sc postSolver/afterIntegration publication.
    std::size_t parallel_solver_split=0;
    if(!gpu_dy_distance && parallel_solver_enabled_ &&
       !solver_kinematic_participant_ && solver_tasks_.size()>1 &&
       ordered_constraints_.size()>=parallel_solver_min_descriptors_) {
        // Prepared task rows are disjoint only without shared kinematic
        // solver endpoints; their presence is recorded during the
        // existing contact and 1D preparation loops.
        const std::size_t total=ordered_constraints_.size();
        const std::size_t minimum=parallel_solver_min_descriptors_
            ? std::max<std::size_t>(1,parallel_solver_min_descriptors_/4)
            : 1;
        std::size_t best_balance=total;
        for(std::size_t i=1;i<solver_tasks_.size();++i) {
            const auto left=solver_tasks_[i-1].constraint_end;
            const auto right=total-left;
            if(left<minimum || right<minimum)continue;
            const auto imbalance=left>right?left-right:right-left;
            if(!parallel_solver_split || imbalance<best_balance) {
                parallel_solver_split=i;
                best_balance=imbalance;
            }
        }
    }
    // ScBodySim counted interactions are updated on Gu overlap and Sc
    // constraint attachment/retirement. DySleep consumes the lifetime
    // count, including non-touching managers and joint-only islands,
    // without scanning dormant Gu contacts or clearing per-frame stamps.
    // DyBodyCoreIntegrator.h::integrateCore uses the saved *position*
    // iteration motion velocities to move/rotate bodies. The velocity
    // iterations above publish their different, unbiased final velocities
    // for subsequent frames; they do not undo this frame's correction.
    // Source Dy task descriptors partition the same stable Gu manager
    // and solver-body ranges. Their final cumulative ends must cover
    // the authoritative active input streams before any task can
    // publish completion, whether serial or concurrent.
    if((solver_tasks_.empty()
            ? !solver_active_ids_.empty() || !indexed_managers_.empty()
            : solver_tasks_.back().solver_body_end!=solver_active_ids_.size() ||
              solver_tasks_.back().manager_end!=indexed_managers_.size()))
        throw std::logic_error(
            "Native Dy task ranges did not cover prepared solver bodies/Gu managers");
    if(parallel_solver_split) {
        // Dy preIntegration already journaled every awake body in these
        // exact accurate-island task ranges before touching its velocity.
        // Do not duplicate that O(active rigid) scan here. Subsequent
        // per-task save_motion() calls observe the existing generation
        // mark and cannot append to the checkpoint from either worker.
        solver_ready_caller_.clear();
        solver_ready_worker_.clear();
        solver_ready_caller_.reserve(live_dynamic_count_);
        solver_ready_worker_.reserve(live_dynamic_count_);
        dispatch_third_pass_worker(
            false,parallel_solver_split,true,seconds,&join);
        try {
            for(std::size_t i=0;i<parallel_solver_split;++i) {
                solve_constraint_task(i);
                complete_solver_end_task(i,seconds,solver_ready_caller_);
                join.finish_solver_task(i);
            }
        } catch(...) {
            // No checkpoint restoration while another source chain
            // still mutates disjoint actor motion.
            join_third_pass_worker();
            throw;
        }
        complete_third_pass_worker();
        // Dy sleepCheck supplies next-frame speculative candidates in
        // original task order, regardless of which complete task chain
        // finished first.
        speculative_ready_nodes_.insert(speculative_ready_nodes_.end(),
            solver_ready_caller_.begin(),solver_ready_caller_.end());
        speculative_ready_nodes_.insert(speculative_ready_nodes_.end(),
            solver_ready_worker_.begin(),solver_ready_worker_.end());
        ++concurrent_solver_dispatches_;
        ++concurrent_integration_dispatches_;
    } else {
        for(std::size_t i=0;i<solver_tasks_.size();++i) {
            if(!gpu_dy_distance)solve_constraint_task(i);
            complete_solver_end_task(i,seconds,speculative_ready_nodes_);
            join.finish_solver_task(i);
        }
    }
    if(!solver_tasks_.empty())dirty_=true;
    // Each actual PxsSolverEndTask now contributes its own completion
    // edge immediately after integrateCore/sleepCheck. The worker and
    // caller have joined before mPostSolver tests the final dependency.
    // Dy::sleepCheck has only published readiness. The next source
    // third pass decides whether entire islands may deactivate; applying
    // that decision here would incorrectly consume this frame's flags.
    // The source mPostSolver continuation joins both actual producers:
    // processLostContactsTask3 / PostThirdPassTask and every
    // PxsSolverEndTask (including its integration and sleepCheck).
    // This check gates resource reclamation and completed publication;
    // failed frames roll back the existing single-scene checkpoint.
    join.finish_post_solver();
    publish_kinematic_targets();
    // Concurrent IslandSim/Dy schedules can integrate bodies that the
    // accurate third pass deactivated; restore their original rigid
    // poses before Sc publishes cached body/shape transforms.
    reconcile_integrated_deactivations();
    // The current Dy motion and island ownership are committed only
    // after the original Sc afterIntegration shape-cache publication.
    // Its bounds updates participate in the same motion undo journal.
    after_integration_cached_bounds();
    if(gpu_dy_pose_ready_) {
        // Only reconciled poses can authorize the next SAP to use Dy's
        // cached device bounds. A restored pose stays dirty on the host.
        const auto authorize=[&](BodyId id,std::size_t index) {
            if(!gpu_dy_integration_inputs_[index].source_shape_count)return;
            const auto& actor=actors_[id];
            const auto& result=gpu_dy_integrated_poses_[index];
            if(actor.position.x!=result.position[0] ||
               actor.position.y!=result.position[1] ||
               actor.position.z!=result.position[2] ||
               actor.orientation.x!=result.orientation[0] ||
               actor.orientation.y!=result.orientation[1] ||
               actor.orientation.z!=result.orientation[2] ||
               actor.orientation.w!=result.orientation[3])
                return;
            for(ShapeId shape=actor.first_simulation_shape;shape!=invalid_shape;
                shape=shapes_[shape].next_simulation_shape)
                gpu_broadphase_->authorize_device_integrated_pose(shape);
        };
        for(const auto id:solver_active_ids_)
            authorize(id,actors_[id].solver_row_index);
        for(std::size_t i=0;i<gpu_dy_free_ids_.size();++i)
            authorize(gpu_dy_free_ids_[i],solver_rows_.size()+i);
    }
    join.finish_after_integration();
    // Threshold checks consume current Dy forces after the
    // integration/cache phase, independently from task-local solve.
    update_threshold_stream(seconds);
    // Reports consume completed writeback and retired manager state.
    publish_contact_events();
    join.finish_finalization();
    join.publish_results();
    // Source BF_KINEMATIC_SETTLING consumes one subsequent zero-motion
    // update; prune those no longer target-driven only after commit.
    std::erase_if(active_kinematic_targets_,[&](BodyId id) {
        auto& body=actors_[id];
        if(body.kinematic_target_pending)return false;
        if(body.kinematic_settling) {
            body.kinematic_settling=false;
            return false;
        }
        body.kinematic_tracked=false;
        refresh_kinematic_node(id,false);
        refresh_kinematic_node(id,true);
        return true;
    });
    // This is the last publication edge. The frame is now observed as
    // a completed Dy result rather than merely a Gu collision step.
    // A failed stage above leaves it unpublished; rollback restores the
    // previously completed result generation and all its buffers.
    solved_result_frame_=frame_index_;
    if(gpu_broadphase_)gpu_broadphase_->commit_gpu_step();
    // Solver and lost-edge producers have both completed before this
    // successful step returns its published body and contact results.
    activity_checkpoint_active_=false;
    FYZZ_PROFILE_COUNTER("live_dynamic",live_dynamic_count_);
    FYZZ_PROFILE_COUNTER("accurate_islands",accurate_island_count_);
    FYZZ_PROFILE_COUNTER("solver_tasks",solver_tasks_.size());
    FYZZ_PROFILE_COUNTER("solver_bodies",solver_rows_.size());
    FYZZ_PROFILE_COUNTER("gu_managers",indexed_managers_.size());
    FYZZ_PROFILE_COUNTER("contact_rows",prepared_contact_rows_);
    FYZZ_PROFILE_COUNTER("friction_rows",prepared_friction_rows_);
    FYZZ_PROFILE_COUNTER("constraint_descriptors",ordered_constraints_.size());
    FYZZ_PROFILE_COUNTER("device_responding_islands",gpu_dy_responding_island_count_);
    FYZZ_PROFILE_END_FRAME(frame_index_,gpu_dy_active(),true);
    } catch (...) {
        { FYZZ_PROFILE_SCOPE("sc.rollback");
          step_checkpoint_.rollback(*this); }
        FYZZ_PROFILE_END_FRAME(frame_index_,gpu_dy_active(),false);
        throw;
    }
}

void Scene::advance_collision(bool force_update) {
    // No geometry, topology or handle lifecycle changes can throw on
    // a quiet collision publication. Preserve the original constant-
    // work path instead of copying persistent edge/island state.
    // A successful quiet advance still retires last frame's transient
    // event/solver reports and advances the publication clock.
    if (!gpu_broadphase_ && !dirty_ && !force_update &&
        !island_topology_dirty_ && retired_edge_handles_.empty()) {
        // No new edges or pending island changes: never scan live
        // islands on the existing constant-work collision query path.
        advance_collision_internal(false,false);
        // PxsSimpleIslandManager::clearDeactivations runs at each island
        // publication. A quiet geometry query has no new third-pass
        // work, but it must retire last publication's one-frame
        // mNodesToPutToSleep output rather than replaying the event.
        // This is constant-work and cannot throw.
        accurate_islands_.nodes_to_deactivate.clear();
        speculative_islands_.nodes_to_deactivate.clear();
        speculative_islands_.deactivating_edges.clear();
        accurate_islands_.deactivating_edges.clear();
        speculative_islands_.deactivating_constraint_edges.clear();
        accurate_islands_.deactivating_constraint_edges.clear();
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
        // First/second-pass wakeIslands must see the old connected
        // island before third-pass lost-edge repair can split it.
        // Sparse motion journals restore these wakes if repair fails.
        activate_published_islands();
        consume_activated_contact_interactions();
        third_pass_island_gen();
        // A collision-only publication has no Dy consumers. The two
        // island repair passes have already joined, so handle retirement
        // may complete inside the failure-atomic collision transaction.
        retire_destroyed_contact_edges();
        // Accept the PCM history only after the Sc publication commits.
        if(gpu_broadphase_)gpu_broadphase_->commit_gpu_step();
    } catch (...) {
        step_checkpoint_.rollback(*this);
        throw;
    }
    // Activation and third-pass deactivation only mutate existing
    // actors after the collision transaction has successfully committed.
    // Geometry-only checkpoints intentionally do not copy actor motion.
    activity_checkpoint_active_=false;
    synchronize_active_island_nodes(true);
    deactivate_ready_islands();
    consume_deactivated_contact_interactions();
    if(!accurate_islands_.nodes_to_deactivate.empty())
        synchronize_active_island_nodes();
}
void Scene::advance_collision_internal(
    bool force_update,bool checkpointed) {
    FYZZ_PROFILE_SCOPE("sc.collision_pipeline");
    // Sc retirement and event clearing do not consume SAP reports or
    // change source AABB inputs. On GPU frames, execute this work after
    // the SAP producer submits but before its host-facing fetch joins.
    const auto prepare_sc_publication=[&] {
        speculative_connectivity_prepared_=false;
        gpu_pcm_friction_history_masks_.clear();
        speculative_split_sleep_candidates_.clear();
        clear_activated_edges(checkpointed);
        normal_impulse_writeback_.clear();
        friction_impulse_writeback_.clear();
        solved_result_frame_=0;
        threshold_changes_.clear();
    };
    // ScPipeline.cpp CPU ordering:
    // BP first pass -> BP update -> post-BP pair publication ->
    // Gu narrowphase -> physical touch publication.
    // No parallel timestep or dynamics scheduler is introduced.
    // Source PxgAABBManager/PxgCudaBroadPhaseSap owns the update/fetch
    // when selected. CPU SAP must not even prepare its endpoint state.
    // SAP executes its required final bounds/history convergence after
    // motion, then skips unchanged quiet frames.
    if(gpu_broadphase_)
        update_gpu_pairs(force_update,prepare_sc_publication);
    else
        prepare_sc_publication();
    if (dirty_ || force_update ||
        (gpu_broadphase_ &&
         (broadphase_pair_membership_changed_ ||
          gpu_broadphase_->needs_gpu_contact_rebuild()))) {
        if(!gpu_broadphase_) {
            update_bounds();
            update_pairs();
        }
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
    update_island_connectivity();
    if(gpu_broadphase_)gpu_broadphase_->post_reports_cleanup();
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
                            bool pcm_plane_box,
                            bool pcm_box_box,
                            std::unique_ptr<native::GpuBroadphaseProducer>
                                source_producer = {}) {
        scene_.set_contact_notifications(contact_notifications);
        scene_.set_enhanced_determinism(enhanced_determinism);
        scene_.set_pcm_plane_box(pcm_plane_box);
        scene_.set_pcm_box_box(pcm_box_box);
        if(source_producer)
            scene_.set_gpu_broadphase(std::move(source_producer));
    }
    std::span<const ContactEvent> contact_events() const override {
        return scene_.contact_events();
    }
    std::span<const ContactEventPoint> contact_event_points() const override {
        return scene_.contact_event_points();
    }
    std::span<const ForceThresholdReport> force_threshold_changes() const override {
        return scene_.force_threshold_changes();
    }
    BodyId create_static_body(Transform pose) override {return scene_.create_body(pose,false);}
    BodyId create_dynamic_body(Transform pose) override {return scene_.create_body(pose,true);}
    ShapeId attach_box_shape(BodyId body,Vec3 half,Transform pose) override {
        return scene_.attach_box(body,half,pose);
    }
    ShapeId attach_sphere_shape(BodyId body,float radius,Transform pose) override {
        return scene_.attach_sphere(body,radius,pose);
    }
    ShapeId attach_convex_shape(
        BodyId body,std::shared_ptr<const ConvexMesh> mesh,
        Transform pose) override {
        return scene_.attach_convex(body,std::move(mesh),pose);
    }
    void detach_shape(ShapeId shape) override {scene_.detach_shape(shape);}
    void set_shape_local_pose(ShapeId shape,Transform pose) override {scene_.set_shape_local_pose(shape,pose);}
    void set_shape_material(ShapeId shape,Material material) override {scene_.set_attached_shape_material(shape,material);}
    void set_shape_simulation_enabled(ShapeId shape,bool enabled) override {scene_.set_attached_shape_enabled(shape,enabled);}
    void set_shape_query_enabled(ShapeId shape,bool enabled) override {scene_.set_attached_shape_query_enabled(shape,enabled);}
    void set_shape_offsets(ShapeId shape,float contact,float rest) override {scene_.set_attached_shape_offsets(shape,contact,rest);}
    void update_mass_properties(BodyId body,float density) override {scene_.update_mass_properties(body,density);}
    BodyMassProperties body_mass_properties(BodyId body) const override {return scene_.body_mass_properties(body);}
    BodyId add_static_box(Vec3 h, Vec3 p) override {
        return scene_.add_static_box(h, p);
    }
    BodyId add_static_floor(float height) override {
        return scene_.add_static_floor(height);
    }
    BodyId add_dynamic_box(Vec3 h, Vec3 p, float density) override {
        return scene_.add_dynamic_box(h, p, density);
    }
    BodyId add_static_sphere(float radius,Vec3 position) override {
        return scene_.add_static_sphere(radius,position);
    }
    BodyId add_dynamic_sphere(float radius,Vec3 position,float density) override {
        return scene_.add_dynamic_sphere(radius,position,density);
    }
    AggregateId create_aggregate(
        std::uint32_t max_shapes,bool self_collision) override {
        return scene_.create_aggregate(max_shapes,self_collision);
    }
    void destroy_aggregate(AggregateId id) override {
        scene_.destroy_aggregate(id);
    }
    void remove_actor_from_aggregate(AggregateId id,BodyId actor) override {
        scene_.remove_actor_from_aggregate(id,actor);
    }
    BodyId add_static_box_to_aggregate(
        AggregateId aggregate,Vec3 h,Vec3 p) override {
        return scene_.add_static_box(h,p,aggregate);
    }
    BodyId add_dynamic_box_to_aggregate(
        AggregateId aggregate,Vec3 h,Vec3 p,float density) override {
        return scene_.add_dynamic_box(h,p,density,aggregate);
    }
    ConstraintId add_center_distance_constraint(
        BodyId a,BodyId b,float distance) override {
        return scene_.create_center_distance_constraint(a,b,distance);
    }
    ConstraintId add_anchored_distance_constraint(
        BodyId a,BodyId b,Vec3 local_a,Vec3 local_b,float distance) override {
        return scene_.create_anchored_distance_constraint(
            a,b,local_a,local_b,distance);
    }
    ConstraintId add_fixed_constraint(
        BodyId a,BodyId b,Transform local_a,Transform local_b) override {
        return scene_.create_fixed_constraint(a,b,local_a,local_b);
    }
    ConstraintId add_spherical_constraint(
        BodyId a,BodyId b,Transform local_a,Transform local_b) override {
        return scene_.create_spherical_constraint(a,b,local_a,local_b);
    }
    ConstraintId add_revolute_constraint(
        BodyId a,BodyId b,Transform local_a,Transform local_b) override {
        return scene_.create_revolute_constraint(a,b,local_a,local_b);
    }
    ConstraintId add_prismatic_constraint(
        BodyId a,BodyId b,Transform local_a,Transform local_b) override {
        return scene_.create_prismatic_constraint(a,b,local_a,local_b);
    }
    void remove_constraint(ConstraintId id) override {
        scene_.retire_constraint(id);
    }
    void set_gpu_dy_enabled(bool enabled) override {
        scene_.set_gpu_dy_enabled(enabled);
    }
    void step(float seconds) override { scene_.step(seconds); }
    BodyState body_state(BodyId id) const override { return scene_.body_state(id); }
    std::optional<RaycastHit> raycast(Vec3 origin,Vec3 direction,float max_distance) const override {
        return scene_.raycast(origin,direction,max_distance);
    }
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
    void set_body_contact_report_threshold(BodyId id, float force) override {
        scene_.set_contact_report_threshold(id,force);
    }
    void set_body_shape_offsets(BodyId id, float contact_offset,
                                float rest_offset) override {
        scene_.set_shape_offsets(id,contact_offset,rest_offset);
    }
    void set_body_simulation_shape_enabled(BodyId id, bool enabled) override {
        scene_.set_shape_simulation_enabled(id,enabled);
    }
    void set_body_kinematic(BodyId id, bool enabled) override {
        scene_.set_kinematic(id,enabled);
    }
    void set_body_kinematic_target(BodyId id, Vec3 pos, Quat q) override {
        scene_.set_kinematic_target(id,pos,q);
    }
    void set_body_dynamics(BodyId id, RigidBodyDynamics dynamics) override {
        scene_.set_dynamics(id,dynamics);
    }
private:
    native::Scene scene_;
};
} // namespace
std::unique_ptr<WorldBackend> create_native_cpu_backend(
    bool notify, bool enhanced_determinism,
    bool pcm_plane_box, bool pcm_box_box) {
    return std::make_unique<NativeCpuWorld>(
        notify,enhanced_determinism,pcm_plane_box,pcm_box_box);
}
std::unique_ptr<WorldBackend> create_native_gpu_broadphase_backend(
    std::unique_ptr<native::GpuBroadphaseProducer> source_producer,
    bool notify, bool enhanced_determinism,
    bool pcm_plane_box, bool pcm_box_box) {
    if(!source_producer)
        throw std::invalid_argument(
            "Native GPU scene requires the original PhysX GPU broadphase owner");
    return std::make_unique<NativeCpuWorld>(
        notify,enhanced_determinism,pcm_plane_box,pcm_box_box,
        std::move(source_producer));
}
} // namespace fyzz
