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
    bool legacy_gu, bool contact_notifications);
#endif
// Implemented by the SDK-owned native scene, with no reference PhysX dependency.
std::unique_ptr<WorldBackend> create_native_cpu_backend(bool contact_notifications);

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

std::unique_ptr<WorldBackend> create_world_backend(WorldOptions options) {
    switch (options.execution) {
    case ExecutionMode::ReferencePhysXCpu:
    case ExecutionMode::ReferencePhysXCpuLegacyGu:
#if FYZZXYZZY_ENABLE_PHYSX_SOURCE
        return create_reference_physx_cpu_backend(
            options.execution==ExecutionMode::ReferencePhysXCpuLegacyGu,
            options.contact_notifications);
#else
        throw std::logic_error(
            "Reference PhysX CPU backend was disabled at SDK build time; "
            "the native CPU solver is not yet integrated");
#endif
    case ExecutionMode::NativeCpu:
        return create_native_cpu_backend(options.contact_notifications);
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

void World::step(float seconds) {
    if (!std::isfinite(seconds) || seconds <= 0.0f) {
        throw std::invalid_argument("Simulation step must be finite and positive");
    }
    impl_->step(seconds);
}

BodyState World::body_state(BodyId id) const {
    return impl_->body_state(id);
}
std::span<const ContactEvent> World::contact_events() const {
    return impl_->contact_events();
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
    impl_->apply_body_torque_impulse(id, impulse);
}
void World::set_body_material(BodyId id, Material material) {
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
    impl_->set_body_material(id,material);
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
       d.position_iterations==0)
        throw std::invalid_argument(
            "Source rigid dynamics inputs/iteration counts invalid");
    impl_->set_body_dynamics(id,d);
}

} // namespace fyzz
