// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

namespace fy_ {

struct Vec3 {
    float x;
    float y;
    float z;
};

using BodyId = std::uint32_t;

struct BodyState {
    Vec3 position;
    Vec3 velocity;
};

// Minimal M0 host-facing facade. The API is provisional until the engine
// ownership, scene, and resource contracts have been exercised by consumers.
class World final {
public:
    World();
    ~World();

    World(const World&) = delete;
    World& operator=(const World&) = delete;
    World(World&&) noexcept;
    World& operator=(World&&) noexcept;

    BodyId add_static_box(Vec3 half_extents, Vec3 position);
    BodyId add_dynamic_box(Vec3 half_extents, Vec3 position, float density);
    void step(float seconds);
    BodyState body_state(BodyId id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fy_
