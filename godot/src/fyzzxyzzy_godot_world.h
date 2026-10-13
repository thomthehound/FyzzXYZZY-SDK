// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fyzzxyzzy/world.h"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <cstdint>
#include <memory>

namespace fyzz::godot_adapter {

class FyzzXYZZYGodotWorld : public godot::RefCounted {
    GDCLASS(FyzzXYZZYGodotWorld, godot::RefCounted)

    std::unique_ptr<fyzz::World> world_;

protected:
    static void _bind_methods();

public:
    FyzzXYZZYGodotWorld();
    ~FyzzXYZZYGodotWorld() override = default;

    int64_t add_static_box(godot::Vector3 half_extents, godot::Vector3 position);
    int64_t add_dynamic_box(godot::Vector3 half_extents, godot::Vector3 position, float density);
    bool step(float delta_seconds);
    godot::Vector3 get_body_position(int64_t id) const;
    godot::Vector3 get_body_velocity(int64_t id) const;
};

} // namespace fyzz::godot_adapter
