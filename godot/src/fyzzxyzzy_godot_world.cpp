// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy_godot_world.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <exception>
#include <limits>
#include <stdexcept>

namespace fyzz::godot_adapter {
namespace {

fyzz::Vec3 to_fyzzxyzzy(godot::Vector3 value) {
    return {value.x, value.y, value.z};
}

godot::Vector3 to_godot(fyzz::Vec3 value) {
    return godot::Vector3(value.x, value.y, value.z);
}

fyzz::BodyId checked_body_id(int64_t id) {
    if (id < 0 || id > static_cast<int64_t>(std::numeric_limits<fyzz::BodyId>::max())) {
        throw std::out_of_range("Invalid FyzzXYZZY body ID");
    }
    return static_cast<fyzz::BodyId>(id);
}

void report_failure(const std::exception& error) {
    godot::UtilityFunctions::push_error(godot::String("FyzzXYZZY: ") + error.what());
}

} // namespace

FyzzXYZZYGodotWorld::FyzzXYZZYGodotWorld() : world_(std::make_unique<fyzz::World>()) {}

void FyzzXYZZYGodotWorld::_bind_methods() {
    godot::ClassDB::bind_method(
        godot::D_METHOD("add_static_box", "half_extents", "position"),
        &FyzzXYZZYGodotWorld::add_static_box);
    godot::ClassDB::bind_method(
        godot::D_METHOD("add_dynamic_box", "half_extents", "position", "density"),
        &FyzzXYZZYGodotWorld::add_dynamic_box);
    godot::ClassDB::bind_method(
        godot::D_METHOD("step", "delta_seconds"),
        &FyzzXYZZYGodotWorld::step);
    godot::ClassDB::bind_method(
        godot::D_METHOD("get_body_position", "id"),
        &FyzzXYZZYGodotWorld::get_body_position);
    godot::ClassDB::bind_method(
        godot::D_METHOD("get_body_velocity", "id"),
        &FyzzXYZZYGodotWorld::get_body_velocity);
}

int64_t FyzzXYZZYGodotWorld::add_static_box(godot::Vector3 half_extents,
                                        godot::Vector3 position) {
    try {
        return world_->add_static_box(to_fyzzxyzzy(half_extents), to_fyzzxyzzy(position));
    } catch (const std::exception& error) {
        report_failure(error);
        return -1;
    }
}

int64_t FyzzXYZZYGodotWorld::add_dynamic_box(godot::Vector3 half_extents,
                                         godot::Vector3 position,
                                         float density) {
    try {
        return world_->add_dynamic_box(to_fyzzxyzzy(half_extents), to_fyzzxyzzy(position), density);
    } catch (const std::exception& error) {
        report_failure(error);
        return -1;
    }
}

bool FyzzXYZZYGodotWorld::step(float delta_seconds) {
    try {
        world_->step(delta_seconds);
        return true;
    } catch (const std::exception& error) {
        report_failure(error);
        return false;
    }
}

godot::Vector3 FyzzXYZZYGodotWorld::get_body_position(int64_t id) const {
    try {
        return to_godot(world_->body_state(checked_body_id(id)).position);
    } catch (const std::exception& error) {
        report_failure(error);
        return {};
    }
}

godot::Vector3 FyzzXYZZYGodotWorld::get_body_velocity(int64_t id) const {
    try {
        return to_godot(world_->body_state(checked_body_id(id)).velocity);
    } catch (const std::exception& error) {
        report_failure(error);
        return {};
    }
}

} // namespace fyzz::godot_adapter
