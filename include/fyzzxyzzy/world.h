// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

namespace fyzz {

struct Vec3 {
    float x;
    float y;
    float z;
};

// Source PhysX remains the scene and task-graph owner. Choosing a GPU
// broadphase must never silently admit a CPU scene as a successful GPU path.
enum class BroadphaseMode : std::uint8_t {
    PhysXCpu,
    PhysXPortableGpu,
};

struct WorldOptions {
    BroadphaseMode broadphase = BroadphaseMode::PhysXCpu;
};

using BodyId = std::uint32_t;

struct BodyState {
    Vec3 position;
    Vec3 velocity;
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

private:
    std::unique_ptr<WorldBackend> impl_;
};

} // namespace fyzz
