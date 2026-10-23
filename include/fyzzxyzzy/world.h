// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>

namespace fyzz {

struct Vec3 {
    float x;
    float y;
    float z;
};

// PhysX PxQuat order (x, y, z, w). Source-owned body state uses an
// identity rotation by default; no external PhysX type leaks into SDK API.
struct Quat {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

// Reference PhysX CPU execution is transitional and isolated from the native
// FyzzXYZZY engine. Native execution must never silently select the reference.
enum class ExecutionMode : std::uint8_t {
    ReferencePhysXCpu,
    NativeCpu,
    NativePortableGpu,
    // Independent PhysX PGS reference with the source legacy Gu contact
    // generator selected explicitly, rather than default PCM.
    ReferencePhysXCpuLegacyGu,
};

struct WorldOptions {
    ExecutionMode execution = ExecutionMode::ReferencePhysXCpu;
};

// Provider-neutral noncompliant contact material, matching PhysX
// PxsMaterialCombiner's rigid-material combine modes and friction flags.
// PhysX source PxsMaterialData: regular and compliant PGS materials.
enum class MaterialCombineMode : std::uint8_t {
    Average=0, Minimum=1, Multiply=2, Maximum=3
};
struct Material {
    float static_friction=0.5f;
    float dynamic_friction=0.5f;
    // Negative restitution is compliant stiffness in PhysX (e.g. -100).
    float restitution=0.2f;
    float damping=0.0f;
    MaterialCombineMode friction_combine=MaterialCombineMode::Average;
    MaterialCombineMode restitution_combine=MaterialCombineMode::Average;
    MaterialCombineMode damping_combine=MaterialCombineMode::Average;
    bool compliant_acceleration_spring=false;
    bool disable_friction=false;
    bool disable_strong_friction=false;
};

// ScBodyCore/PxsBodyCore::solverIterationCounts and
// DyDynamics::preIntegrationParallel rigid-body inputs. These values
// belong to each dynamic body, not to a second scene/global solver.
struct RigidBodyDynamics {
    float linear_damping=0.0f;
    float angular_damping=0.05f;
    float max_linear_velocity=1.0e16f;
    float max_angular_velocity=100.0f;
    // PxRigidBody::setMaxDepenetrationVelocity -> -penBiasClamp; the
    // default 1e32 reproduces the existing PxsBodyCore unrestricted bias.
    float max_depenetration_velocity=1.0e32f;
    // PxRigidBody::setMaxContactImpulse -> the minimum body cap is
    // assigned to each Gu contact before Dy preparation.
    float max_contact_impulse=std::numeric_limits<float>::max();
    // PxRigidBody::setContactSlopCoefficient. ScShapeInteraction uses the
    // maximum of the two body coefficients when preparing Dy constraints.
    float contact_slop_coefficient=0.0f;
    bool gravity_enabled=true;
    std::uint8_t position_iterations=4;
    std::uint8_t velocity_iterations=1;
};

using BodyId = std::uint32_t;

struct BodyState {
    Vec3 position;
    Vec3 velocity;
    // Logical actor orientation, independent of provider-specific shape
    // alignment. For add_static_floor the +Y-normal floor starts at
    // identity even though reference PhysX rotates its internal +X plane.
    Quat orientation{};
    Vec3 angular_velocity{};
    // ScBodyCore::isSleeping: full-island deactivation, not merely
    // an exhausted wake counter on an otherwise active body.
    bool sleeping=false;
};

// SDK simulation backend. Each World owns exactly one implementation.
// Additional backends must reject unsupported scene operations explicitly.
class WorldBackend {
public:
    virtual ~WorldBackend() = default;
    virtual BodyId add_static_box(Vec3 half_extents, Vec3 position) = 0;
    virtual BodyId add_static_floor(float height) = 0;
    virtual BodyId add_dynamic_box(Vec3 half_extents, Vec3 position, float density) = 0;
    virtual void step(float seconds) = 0;
    virtual BodyState body_state(BodyId id) const = 0;
    // All built-in CPU backends implement scene actor lifecycle. Additional
    // backends must reject unsupported operations explicitly.
    virtual void set_body_position(BodyId, Vec3) {
        throw std::logic_error("This backend cannot relocate an actor");
    }
    virtual void set_body_pose(BodyId, Vec3, Quat) {
        throw std::logic_error("This backend cannot author rigid pose");
    }
    virtual void remove_body(BodyId) {
        throw std::logic_error("This backend cannot remove an actor");
    }
    virtual void set_body_velocity(BodyId, Vec3) {
        throw std::logic_error("This backend cannot set linear velocity");
    }
    virtual void set_body_angular_velocity(BodyId, Vec3) {
        throw std::logic_error("This backend cannot set angular velocity");
    }
    virtual void apply_body_torque_impulse(BodyId, Vec3) {
        throw std::logic_error("This backend cannot apply a torque impulse");
    }
    virtual void set_body_material(BodyId, Material) {
        throw std::logic_error("This backend cannot author rigid material");
    }
    // PxShape's contact and rest distances belong to the existing actor shape.
    virtual void set_body_shape_offsets(BodyId, float, float) {
        throw std::logic_error("This backend cannot author shape offsets");
    }
    virtual void set_body_dynamics(BodyId, RigidBodyDynamics) {
        throw std::logic_error("This backend cannot author rigid dynamics");
    }
};

class World final {
public:
    World();
    explicit World(WorldOptions options);
    explicit World(std::unique_ptr<WorldBackend> backend);
    ~World();

    World(const World&) = delete;
    World& operator=(const World&) = delete;
    World(World&&) noexcept;
    World& operator=(World&&) noexcept;

    BodyId add_static_box(Vec3 half_extents, Vec3 position);
    BodyId add_static_floor(float height);
    BodyId add_dynamic_box(Vec3 half_extents, Vec3 position, float density);
    void step(float seconds);
    BodyState body_state(BodyId id) const;
    void set_body_position(BodyId id, Vec3 position);
    // PxRigidActor::setGlobalPose, with logical identity for the SDK floor.
    void set_body_pose(BodyId id, Vec3 position, Quat orientation);
    void remove_body(BodyId id);
    void set_body_velocity(BodyId id, Vec3 velocity);
    void set_body_angular_velocity(BodyId id, Vec3 velocity);
    // Impulse is angular momentum in world coordinates (N*m*s).
    // Matches PxRigidDynamic::addTorque(..., PxForceMode::eIMPULSE).
    void apply_body_torque_impulse(BodyId id, Vec3 impulse);
    void set_body_material(BodyId id, Material material);
    // PhysX PxShape::setContactOffset / setRestOffset: both in metres.
    // Contact offset must be nonnegative and greater than rest offset.
    void set_body_shape_offsets(BodyId id, float contact_offset, float rest_offset);
    void set_body_dynamics(BodyId id, RigidBodyDynamics dynamics);

private:
    std::unique_ptr<WorldBackend> impl_;
};

} // namespace fyzz
