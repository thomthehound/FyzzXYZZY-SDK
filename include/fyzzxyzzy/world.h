// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
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
};

struct WorldOptions {
    ExecutionMode execution = ExecutionMode::ReferencePhysXCpu;
};

// Provider-neutral noncompliant contact material, matching PhysX
// PxsMaterialCombiner's rigid-material combine modes and friction flags.
// Compliant springs and per-triangle material tables are future features.
enum class MaterialCombineMode : std::uint8_t {
    Average=0, Minimum=1, Multiply=2, Maximum=3
};
struct Material {
    float static_friction=0.5f;
    float dynamic_friction=0.5f;
    float restitution=0.2f;
    MaterialCombineMode friction_combine=MaterialCombineMode::Average;
    MaterialCombineMode restitution_combine=MaterialCombineMode::Average;
    bool disable_friction=false;
    bool disable_strong_friction=false;
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
    void remove_body(BodyId id);
    void set_body_velocity(BodyId id, Vec3 velocity);
    void set_body_angular_velocity(BodyId id, Vec3 velocity);
    // Impulse is angular momentum in world coordinates (N*m*s).
    // Matches PxRigidDynamic::addTorque(..., PxForceMode::eIMPULSE).
    void apply_body_torque_impulse(BodyId id, Vec3 impulse);
    void set_body_material(BodyId id, Material material);

private:
    std::unique_ptr<WorldBackend> impl_;
};

} // namespace fyzz
