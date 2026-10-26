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

fyzz::Quat to_fyzzxyzzy(godot::Quaternion value) {
    return {value.x,value.y,value.z,value.w};
}

godot::Quaternion to_godot(fyzz::Quat value) {
    return {value.x,value.y,value.z,value.w};
}

std::size_t checked_event_index(int64_t index,std::size_t count) {
    if(index<0 || static_cast<std::uint64_t>(index)>=count)
        throw std::out_of_range("Invalid completed-frame contact event index");
    return static_cast<std::size_t>(index);
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

FyzzXYZZYGodotWorld::FyzzXYZZYGodotWorld()
    : world_(std::make_unique<fyzz::World>(
          fyzz::WorldOptions{fyzz::ExecutionMode::NativeCpu,true})) {}

void FyzzXYZZYGodotWorld::_bind_methods() {
    godot::ClassDB::bind_method(
        godot::D_METHOD("add_static_box", "half_extents", "position"),
        &FyzzXYZZYGodotWorld::add_static_box);
    godot::ClassDB::bind_method(
        godot::D_METHOD("add_static_floor", "height"),
        &FyzzXYZZYGodotWorld::add_static_floor);
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
    godot::ClassDB::bind_method(
        godot::D_METHOD("get_contact_event_count"),
        &FyzzXYZZYGodotWorld::get_contact_event_count);
    godot::ClassDB::bind_method(
        godot::D_METHOD("get_contact_event_kind", "index"),
        &FyzzXYZZYGodotWorld::get_contact_event_kind);
    godot::ClassDB::bind_method(
        godot::D_METHOD("get_contact_event_body0", "index"),
        &FyzzXYZZYGodotWorld::get_contact_event_body0);
    godot::ClassDB::bind_method(
        godot::D_METHOD("get_contact_event_body1", "index"),
        &FyzzXYZZYGodotWorld::get_contact_event_body1);
    godot::ClassDB::bind_method(
        godot::D_METHOD("get_contact_event_point_count", "index"),
        &FyzzXYZZYGodotWorld::get_contact_event_point_count);
    godot::ClassDB::bind_method(
        godot::D_METHOD("get_contact_event_normal_impulse", "index"),
        &FyzzXYZZYGodotWorld::get_contact_event_normal_impulse);
    godot::ClassDB::bind_method(
        godot::D_METHOD("get_body_orientation", "id"),
        &FyzzXYZZYGodotWorld::get_body_orientation);
    godot::ClassDB::bind_method(
        godot::D_METHOD("get_body_angular_velocity", "id"),
        &FyzzXYZZYGodotWorld::get_body_angular_velocity);
    godot::ClassDB::bind_method(
        godot::D_METHOD("set_body_pose", "id", "position", "orientation"),
        &FyzzXYZZYGodotWorld::set_body_pose);
    godot::ClassDB::bind_method(
        godot::D_METHOD("set_body_velocity", "id", "velocity"),
        &FyzzXYZZYGodotWorld::set_body_velocity);
    godot::ClassDB::bind_method(
        godot::D_METHOD("set_body_angular_velocity", "id", "velocity"),
        &FyzzXYZZYGodotWorld::set_body_angular_velocity);
    godot::ClassDB::bind_method(
        godot::D_METHOD("remove_body", "id"),
        &FyzzXYZZYGodotWorld::remove_body);
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

int64_t FyzzXYZZYGodotWorld::add_static_floor(float height) {
    try {
        return world_->add_static_floor(height);
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

int64_t FyzzXYZZYGodotWorld::get_contact_event_count() const {
    return static_cast<int64_t>(world_->contact_events().size());
}
int64_t FyzzXYZZYGodotWorld::get_contact_event_kind(int64_t index) const {
    try {
        const auto events=world_->contact_events();
        return static_cast<int64_t>(
            events[checked_event_index(index,events.size())].kind);
    } catch(const std::exception& error) {
        report_failure(error);
        return -1;
    }
}
int64_t FyzzXYZZYGodotWorld::get_contact_event_body0(int64_t index) const {
    try {
        const auto events=world_->contact_events();
        return events[checked_event_index(index,events.size())].body0;
    } catch(const std::exception& error) {
        report_failure(error);
        return -1;
    }
}
int64_t FyzzXYZZYGodotWorld::get_contact_event_body1(int64_t index) const {
    try {
        const auto events=world_->contact_events();
        return events[checked_event_index(index,events.size())].body1;
    } catch(const std::exception& error) {
        report_failure(error);
        return -1;
    }
}
int64_t FyzzXYZZYGodotWorld::get_contact_event_point_count(int64_t index) const {
    try {
        const auto events=world_->contact_events();
        return events[checked_event_index(index,events.size())].contact_count;
    } catch(const std::exception& error) {
        report_failure(error);
        return -1;
    }
}
double FyzzXYZZYGodotWorld::get_contact_event_normal_impulse(int64_t index) const {
    try {
        const auto events=world_->contact_events();
        return events[checked_event_index(index,events.size())].normal_impulse;
    } catch(const std::exception& error) {
        report_failure(error);
        return 0.0;
    }
}

godot::Quaternion FyzzXYZZYGodotWorld::get_body_orientation(int64_t id) const {
    try {
        return to_godot(world_->body_state(checked_body_id(id)).orientation);
    } catch (const std::exception& error) {
        report_failure(error);
        return {};
    }
}

godot::Vector3 FyzzXYZZYGodotWorld::get_body_angular_velocity(int64_t id) const {
    try {
        return to_godot(world_->body_state(checked_body_id(id)).angular_velocity);
    } catch (const std::exception& error) {
        report_failure(error);
        return {};
    }
}

bool FyzzXYZZYGodotWorld::set_body_pose(int64_t id, godot::Vector3 position,
                                         godot::Quaternion orientation) {
    try {
        world_->set_body_pose(checked_body_id(id),
                              to_fyzzxyzzy(position),to_fyzzxyzzy(orientation));
        return true;
    } catch (const std::exception& error) {
        report_failure(error);
        return false;
    }
}

bool FyzzXYZZYGodotWorld::set_body_velocity(int64_t id,
                                             godot::Vector3 velocity) {
    try {
        world_->set_body_velocity(checked_body_id(id),to_fyzzxyzzy(velocity));
        return true;
    } catch (const std::exception& error) {
        report_failure(error);
        return false;
    }
}

bool FyzzXYZZYGodotWorld::set_body_angular_velocity(int64_t id,
                                                     godot::Vector3 velocity) {
    try {
        world_->set_body_angular_velocity(
            checked_body_id(id),to_fyzzxyzzy(velocity));
        return true;
    } catch (const std::exception& error) {
        report_failure(error);
        return false;
    }
}

bool FyzzXYZZYGodotWorld::remove_body(int64_t id) {
    try {
        world_->remove_body(checked_body_id(id));
        return true;
    } catch (const std::exception& error) {
        report_failure(error);
        return false;
    }
}

} // namespace fyzz::godot_adapter
