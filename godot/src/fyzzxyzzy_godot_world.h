// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fyzzxyzzy/world.h"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/quaternion.hpp>

#include <cstddef>
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
    int64_t add_static_floor(float height);
    int64_t add_dynamic_box(godot::Vector3 half_extents, godot::Vector3 position, float density);
    bool step(float delta_seconds);
    godot::Vector3 get_body_position(int64_t id) const;
    godot::Vector3 get_body_velocity(int64_t id) const;
    // Event indices and point counts describe the latest completed
    // native World::step; the Scene retains authoritative contact data.
    int64_t get_contact_event_count() const;
    int64_t get_contact_event_kind(int64_t index) const;
    int64_t get_contact_event_body0(int64_t index) const;
    int64_t get_contact_event_body1(int64_t index) const;
    int64_t get_contact_event_point_count(int64_t index) const;
    double get_contact_event_normal_impulse(int64_t index) const;
    godot::Quaternion get_body_orientation(int64_t id) const;
    godot::Vector3 get_body_angular_velocity(int64_t id) const;
    bool set_body_pose(int64_t id, godot::Vector3 position,
                       godot::Quaternion orientation);
    bool set_body_velocity(int64_t id, godot::Vector3 velocity);
    bool set_body_angular_velocity(int64_t id, godot::Vector3 velocity);
    bool remove_body(int64_t id);
};

} // namespace fyzz::godot_adapter
