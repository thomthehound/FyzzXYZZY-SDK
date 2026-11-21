// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"

#include <cmath>
#include <memory>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz {

#if FYZZXYZZY_ENABLE_PHYSX_SOURCE
// Implemented in the separately compiled reference/PhysX CPU target.
std::unique_ptr<WorldBackend> create_reference_physx_cpu_backend(
    bool legacy_gu, bool contact_notifications, bool enhanced_determinism);
#endif
// Implemented by the SDK-owned native scene, with no reference PhysX dependency.
std::unique_ptr<WorldBackend> create_native_cpu_backend(
    bool contact_notifications, bool enhanced_determinism,
    bool pcm_plane_box, bool pcm_box_box);

namespace {
void validate_half_extents(Vec3 half_extents) {
    if (!std::isfinite(half_extents.x) || !std::isfinite(half_extents.y) ||
        !std::isfinite(half_extents.z) || half_extents.x <= 0.0f ||
        half_extents.y <= 0.0f || half_extents.z <= 0.0f) {
        throw std::invalid_argument("Box half-extents must be finite and positive");
    }
}

void validate_position(Vec3 position) {
    if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
        !std::isfinite(position.z)) {
        throw std::invalid_argument("Position must be finite");
    }
}
void validate_orientation(Quat q) {
    // PxQuat::isSane tests a finite unit magnitude within 1e-2.
    const float magnitude=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
    if (!std::isfinite(q.x) || !std::isfinite(q.y) ||
        !std::isfinite(q.z) || !std::isfinite(q.w) ||
        !std::isfinite(magnitude) || std::abs(magnitude-1.0f)>=1.0e-2f)
        throw std::invalid_argument("Rigid pose orientation is not sane");
}

Transform validated_transform(Transform pose) {
    validate_position(pose.position);
    validate_orientation(pose.orientation);
    auto& q=pose.orientation;
    const float length=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
    q={q.x/length,q.y/length,q.z/length,q.w/length};
    return pose;
}
void validate_material(Material material) {
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
            "PhysX contact material coefficients/modes invalid");
 }

std::unique_ptr<WorldBackend> create_world_backend(WorldOptions options) {
    // Native PCM selectors are deliberately irrelevant to independent
    // PhysX reference execution. ReferencePhysXCpu retains eENABLE_PCM;
    // ReferencePhysXCpuLegacyGu remains the explicit non-PCM oracle.
    switch (options.execution) {
    case ExecutionMode::ReferencePhysXCpu:
    case ExecutionMode::ReferencePhysXCpuLegacyGu:
#if FYZZXYZZY_ENABLE_PHYSX_SOURCE
        return create_reference_physx_cpu_backend(
            options.execution==ExecutionMode::ReferencePhysXCpuLegacyGu,
            options.contact_notifications,options.enhanced_determinism);
#else
        throw std::logic_error(
            "Reference PhysX CPU backend was disabled at SDK build time; "
            "select NativeCpu for independent SDK execution");
#endif
    case ExecutionMode::NativeCpu:
        return create_native_cpu_backend(
            options.contact_notifications,options.enhanced_determinism,
            options.pcm_plane_box,options.pcm_box_box);
    case ExecutionMode::NativePortableGpu:
        throw std::logic_error(
            "Native FyzzXYZZY GPU scene is not yet implemented; "
            "the reference PhysX scene will not be substituted");
    }
    throw std::invalid_argument("Unknown FyzzXYZZY execution mode");
}
} // namespace

World::World() : World(WorldOptions{}) {}
World::World(WorldOptions options)
    : World(create_world_backend(options)) {}

World::World(std::unique_ptr<WorldBackend> backend) : impl_(std::move(backend)) {
    if (!impl_) {
        throw std::invalid_argument("World requires a simulation backend");
    }
}

World::~World() = default;
World::World(World&&) noexcept = default;
World& World::operator=(World&&) noexcept = default;

BodyId World::create_static_body(Transform pose) {
    return impl_->create_static_body(validated_transform(pose));
}
BodyId World::create_dynamic_body(Transform pose) {
    return impl_->create_dynamic_body(validated_transform(pose));
}
ShapeId World::attach_box_shape(BodyId body,Vec3 half,Transform local) {
    validate_half_extents(half);
    return impl_->attach_box_shape(body,half,validated_transform(local));
}
ShapeId World::attach_sphere_shape(BodyId body,float radius,Transform local) {
    if(!std::isfinite(radius) || radius<=0.0f)
        throw std::invalid_argument("Sphere radius must be finite and positive");
    return impl_->attach_sphere_shape(body,radius,validated_transform(local));
}
ShapeId World::attach_convex_shape(
    BodyId body,std::shared_ptr<const ConvexMesh> mesh,Transform local) {
    if(!mesh)throw std::invalid_argument("Cooked convex mesh is null");
    return impl_->attach_convex_shape(body,std::move(mesh),
                                     validated_transform(local));
}
void World::detach_shape(ShapeId shape) {impl_->detach_shape(shape);}
void World::set_shape_local_pose(ShapeId shape,Transform pose) {
    impl_->set_shape_local_pose(shape,validated_transform(pose));
}
void World::set_shape_material(ShapeId shape,Material material) {
    validate_material(material);
    impl_->set_shape_material(shape,material);
}
void World::set_shape_simulation_enabled(ShapeId shape,bool enabled) {
    impl_->set_shape_simulation_enabled(shape,enabled);
}
void World::set_shape_query_enabled(ShapeId shape,bool enabled) {
    impl_->set_shape_query_enabled(shape,enabled);
}
void World::set_shape_offsets(ShapeId shape,float contact,float rest) {
    if(!std::isfinite(contact) || !std::isfinite(rest) || contact<0.0f || rest>=contact)
        throw std::invalid_argument("Invalid contact/rest offsets");
    impl_->set_shape_offsets(shape,contact,rest);
}
void World::update_mass_properties(BodyId body,float density) {
    if(!std::isfinite(density) || density<=0.0f)
        throw std::invalid_argument("Density must be finite and positive");
    impl_->update_mass_properties(body,density);
}
BodyMassProperties World::body_mass_properties(BodyId body) const {
    return impl_->body_mass_properties(body);
}

BodyId World::add_static_box(Vec3 half_extents, Vec3 position) {
    validate_half_extents(half_extents);
    validate_position(position);
    return impl_->add_static_box(half_extents, position);
}

BodyId World::add_static_floor(float height) {
    if (!std::isfinite(height)) {
        throw std::invalid_argument("Floor height must be finite");
    }
    return impl_->add_static_floor(height);
}

BodyId World::add_dynamic_box(Vec3 half_extents, Vec3 position, float density) {
    validate_half_extents(half_extents);
    validate_position(position);
    if (!std::isfinite(density) || density <= 0.0f) {
        throw std::invalid_argument("Density must be finite and positive");
    }
    return impl_->add_dynamic_box(half_extents, position, density);
}

BodyId World::add_static_sphere(float radius,Vec3 position) {
    validate_position(position);
    if(!std::isfinite(radius) || radius<=0.0f)
        throw std::invalid_argument("Sphere radius must be finite and positive");
    return impl_->add_static_sphere(radius,position);
}
BodyId World::add_dynamic_sphere(float radius,Vec3 position,float density) {
    validate_position(position);
    if(!std::isfinite(radius) || radius<=0.0f ||
       !std::isfinite(density) || density<=0.0f)
        throw std::invalid_argument("Sphere radius and density must be finite and positive");
    return impl_->add_dynamic_sphere(radius,position,density);
}

AggregateId World::create_aggregate(
    std::uint32_t max_shapes,bool self_collision) {
    if(!max_shapes)
        throw std::invalid_argument("Original PhysX aggregate cannot be empty-capacity");
    return impl_->create_aggregate(max_shapes,self_collision);
}

void World::destroy_aggregate(AggregateId id) {
    impl_->destroy_aggregate(id);
}
void World::remove_actor_from_aggregate(AggregateId id,BodyId actor) {
    impl_->remove_actor_from_aggregate(id,actor);
}

BodyId World::add_static_box_to_aggregate(
    AggregateId aggregate,Vec3 half_extents,Vec3 position) {
    validate_half_extents(half_extents);
    validate_position(position);
    return impl_->add_static_box_to_aggregate(
        aggregate,half_extents,position);
}

BodyId World::add_dynamic_box_to_aggregate(
    AggregateId aggregate,Vec3 half_extents,Vec3 position,float density) {
    validate_half_extents(half_extents);
    validate_position(position);
    if(!std::isfinite(density) || density<=0.0f)
        throw std::invalid_argument("Density must be finite and positive");
    return impl_->add_dynamic_box_to_aggregate(
        aggregate,half_extents,position,density);
}

ConstraintId World::add_center_distance_constraint(
    BodyId a,BodyId b,float distance) {
    if(a==b || !std::isfinite(distance) || distance<0.0f)
        throw std::invalid_argument(
            "Rigid center-distance endpoints or distance invalid");
    return impl_->add_center_distance_constraint(a,b,distance);
}

ConstraintId World::add_anchored_distance_constraint(
    BodyId a,BodyId b,Vec3 local_a,Vec3 local_b,float distance) {
    if(a==b || !std::isfinite(distance) || distance<0.0f)
        throw std::invalid_argument("Invalid anchored rigid distance constraint");
    validate_position(local_a);
    validate_position(local_b);
    return impl_->add_anchored_distance_constraint(
        a,b,local_a,local_b,distance);
}

ConstraintId World::add_fixed_constraint(
    BodyId a,BodyId b,Transform local_a,Transform local_b) {
    if(a==b)throw std::invalid_argument("Fixed joint requires distinct actors");
    return impl_->add_fixed_constraint(
        a,b,validated_transform(local_a),validated_transform(local_b));
}

ConstraintId World::add_spherical_constraint(
    BodyId a,BodyId b,Transform local_a,Transform local_b) {
    if(a==b)throw std::invalid_argument("Spherical joint requires distinct actors");
    return impl_->add_spherical_constraint(
        a,b,validated_transform(local_a),validated_transform(local_b));
}

ConstraintId World::add_revolute_constraint(
    BodyId a,BodyId b,Transform local_a,Transform local_b) {
    if(a==b)throw std::invalid_argument("Revolute joint requires distinct actors");
    return impl_->add_revolute_constraint(
        a,b,validated_transform(local_a),validated_transform(local_b));
}

ConstraintId World::add_prismatic_constraint(
    BodyId a,BodyId b,Transform local_a,Transform local_b) {
    if(a==b)throw std::invalid_argument("Prismatic joint requires distinct actors");
    return impl_->add_prismatic_constraint(
        a,b,validated_transform(local_a),validated_transform(local_b));
}

void World::remove_constraint(ConstraintId id) {
    impl_->remove_constraint(id);
}

void World::set_gpu_dy_enabled(bool enabled) {
    impl_->set_gpu_dy_enabled(enabled);
}
void World::step(float seconds) {
    if (!std::isfinite(seconds) || seconds <= 0.0f) {
        throw std::invalid_argument("Simulation step must be finite and positive");
    }
    impl_->step(seconds);
}

BodyState World::body_state(BodyId id) const {
    return impl_->body_state(id);
}
std::optional<RaycastHit> World::raycast(
    Vec3 origin,Vec3 direction,float max_distance) const {
    return impl_->raycast(origin,direction,max_distance);
}
std::span<const ContactEvent> World::contact_events() const {
    return impl_->contact_events();
}
std::span<const ContactEventPoint> World::contact_event_points() const {
    return impl_->contact_event_points();
}
std::span<const ForceThresholdReport> World::force_threshold_changes() const {
    return impl_->force_threshold_changes();
}
void World::set_body_position(BodyId id, Vec3 position) {
    validate_position(position);
    impl_->set_body_position(id, position);
}
void World::set_body_pose(BodyId id, Vec3 position, Quat orientation) {
    validate_position(position);
    validate_orientation(orientation);
    impl_->set_body_pose(id, position, orientation);
}
void World::remove_body(BodyId id) {
    impl_->remove_body(id);
}
void World::set_body_velocity(BodyId id, Vec3 velocity) {
    validate_position(velocity); // same finite 3-vector validation
    impl_->set_body_velocity(id, velocity);
}
void World::set_body_angular_velocity(BodyId id, Vec3 velocity) {
    validate_position(velocity);
    impl_->set_body_angular_velocity(id, velocity);
}
void World::apply_body_torque_impulse(BodyId id, Vec3 impulse) {
    validate_position(impulse);
    impl_->apply_body_torque_impulse(id,impulse);
}
void World::add_body_force(BodyId id, Vec3 value, ForceMode mode) {
    validate_position(value);
    if(static_cast<unsigned>(mode)>3)
        throw std::invalid_argument("Unknown PhysX force mode");
    impl_->add_body_force(id,value,mode);
}
void World::add_body_torque(BodyId id, Vec3 value, ForceMode mode) {
    validate_position(value);
    if(static_cast<unsigned>(mode)>3)
        throw std::invalid_argument("Unknown PhysX torque mode");
    impl_->add_body_torque(id,value,mode);
}
void World::clear_body_force(BodyId id, ForceMode mode) {
    if(static_cast<unsigned>(mode)>3)
        throw std::invalid_argument("Unknown PhysX force mode");
    impl_->clear_body_force(id,mode);
}
void World::clear_body_torque(BodyId id, ForceMode mode) {
    if(static_cast<unsigned>(mode)>3)
        throw std::invalid_argument("Unknown PhysX torque mode");
    impl_->clear_body_torque(id,mode);
}
void World::set_body_material(BodyId id, Material material) {
    validate_material(material);
    impl_->set_body_material(id,material);
}

void World::set_body_contact_report_threshold(BodyId id, float force) {
    if(!std::isfinite(force) || force<0.0f)
        throw std::invalid_argument("Invalid source contact report threshold");
    impl_->set_body_contact_report_threshold(id,force);
}

void World::set_body_kinematic_target(BodyId id, Vec3 position,
                                      Quat orientation) {
    validate_position(position);
    validate_orientation(orientation);
    impl_->set_body_kinematic_target(id,position,orientation);
}

void World::set_body_kinematic(BodyId id, bool enabled) {
    impl_->set_body_kinematic(id,enabled);
}

void World::set_body_simulation_shape_enabled(BodyId id, bool enabled) {
    impl_->set_body_simulation_shape_enabled(id,enabled);
}

void World::set_body_shape_offsets(BodyId id, float contact_offset,
                                   float rest_offset) {
    if (!std::isfinite(contact_offset) || !std::isfinite(rest_offset) ||
        contact_offset < 0.0f || rest_offset >= contact_offset)
        throw std::invalid_argument("Invalid PhysX contact/rest shape offsets");
    impl_->set_body_shape_offsets(id, contact_offset, rest_offset);
}

void World::set_body_dynamics(BodyId id, RigidBodyDynamics d) {
    // PxsBodyCore holds squared velocity limits in PxReal. Reject
    // overflow before any backend can partially mutate an actor.
    if(!std::isfinite(d.linear_damping) ||
       !std::isfinite(d.angular_damping) ||
       !std::isfinite(d.max_linear_velocity) ||
       !std::isfinite(d.max_angular_velocity) ||
       !std::isfinite(d.max_depenetration_velocity) ||
       !std::isfinite(d.max_contact_impulse) ||
       !std::isfinite(d.contact_slop_coefficient) ||
       d.max_depenetration_velocity<=0.0f ||
       d.max_contact_impulse<0.0f ||
       d.contact_slop_coefficient<0.0f ||
       d.linear_damping<0.0f || d.angular_damping<0.0f ||
       d.max_linear_velocity<0.0f || d.max_angular_velocity<0.0f ||
       d.max_linear_velocity*d.max_linear_velocity >
           std::numeric_limits<float>::max() ||
       d.max_angular_velocity*d.max_angular_velocity >
           std::numeric_limits<float>::max() ||
       d.position_iterations==0 ||
       (d.lock_flags & ~rigid_body_lock_mask)!=0)
        throw std::invalid_argument(
            "Source rigid dynamics inputs/iteration counts invalid");
    impl_->set_body_dynamics(id,d);
}

} // namespace fyzz
