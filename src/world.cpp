// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace fyzz {

#if FYZZXYZZY_ENABLE_PHYSX_SOURCE
// Implemented in the separately compiled reference/PhysX CPU target.
std::unique_ptr<WorldBackend> create_reference_physx_cpu_backend(bool legacy_gu);
#endif
// Implemented by the SDK-owned native scene, with no reference PhysX dependency.
std::unique_ptr<WorldBackend> create_native_cpu_backend();

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

std::unique_ptr<WorldBackend> create_world_backend(WorldOptions options) {
    switch (options.execution) {
    case ExecutionMode::ReferencePhysXCpu:
    case ExecutionMode::ReferencePhysXCpuLegacyGu:
#if FYZZXYZZY_ENABLE_PHYSX_SOURCE
        return create_reference_physx_cpu_backend(
            options.execution==ExecutionMode::ReferencePhysXCpuLegacyGu);
#else
        throw std::logic_error(
            "Reference PhysX CPU backend was disabled at SDK build time; "
            "the native CPU solver is not yet integrated");
#endif
    case ExecutionMode::NativeCpu:
        return create_native_cpu_backend();
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
void World::set_body_position(BodyId id, Vec3 position) {
    validate_position(position);
    impl_->set_body_position(id, position);
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
        material.static_friction<0.0f ||
        material.dynamic_friction<0.0f ||
        material.restitution<0.0f || material.restitution>1.0f ||
        static_cast<unsigned>(material.friction_combine)>3 ||
        static_cast<unsigned>(material.restitution_combine)>3)
        throw std::invalid_argument(
            "Noncompliant contact material coefficients/modes invalid");
    impl_->set_body_material(id,material);
}

} // namespace fyzz
