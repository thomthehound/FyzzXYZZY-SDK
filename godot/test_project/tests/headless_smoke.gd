# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

extends SceneTree

func _initialize() -> void:
    var extension_path := "res://addons/fyzzxyzzy/fyzzxyzzy.gdextension"
    if not ClassDB.class_exists("FyzzXYZZYGodotWorld"):
        var load_status: int = GDExtensionManager.load_extension(extension_path)
        if load_status != GDExtensionManager.LOAD_STATUS_OK and (
            load_status != GDExtensionManager.LOAD_STATUS_ALREADY_LOADED
        ):
            printerr("FyzzXYZZY GDExtension load failed: status=", load_status)
            quit(1)
            return

    if not ClassDB.class_exists("FyzzXYZZYGodotWorld"):
        printerr("FyzzXYZZYGodotWorld was not registered after loading the GDExtension")
        quit(1)
        return

    var physics_world = ClassDB.instantiate("FyzzXYZZYGodotWorld")
    if physics_world == null:
        printerr("Failed to instantiate FyzzXYZZYGodotWorld")
        quit(1)
        return

    var floor_id: int = physics_world.add_static_floor(0.0)
    var support_id: int = physics_world.add_static_box(
        Vector3(0.5, 0.5, 0.5), Vector3(8.0, -0.5, 0.0))
    var box_id: int = physics_world.add_dynamic_box(
        Vector3(0.25, 0.25, 0.25), Vector3(0.0, 2.0, 0.0), 1.0)
    if floor_id < 0 or support_id < 0 or box_id < 0:
        printerr("FyzzXYZZY body creation failed")
        quit(1)
        return

    # These public Godot calls must operate on the same native rigid
    # scene that owns the contact and island state, not a CPU reference.
    var spin_id: int = physics_world.add_dynamic_box(
        Vector3(0.25, 0.35, 0.45), Vector3(3.0, 8.0, 0.0), 1.0)
    var initial_pose := Quaternion(Vector3.UP, 0.2)
    if spin_id < 0 or not physics_world.set_body_pose(
        spin_id, Vector3(3.0, 8.0, 0.0), initial_pose
    ) or not physics_world.set_body_angular_velocity(
        spin_id, Vector3(0.0, 1.5, 0.0)
    ) or not physics_world.set_body_velocity(
        spin_id, Vector3(0.2, 0.0, 0.0)
    ):
        printerr("Native Godot rigid authoring failed")
        quit(1)
        return

    for frame in range(20):
        if not physics_world.step(1.0 / 120.0):
            printerr("Native Godot rotating rigid step failed at frame ", frame)
            quit(1)
            return
    var spin_pose: Quaternion = physics_world.get_body_orientation(spin_id)
    var spin_omega: Vector3 = physics_world.get_body_angular_velocity(spin_id)
    if not is_finite(spin_pose.y) or spin_pose.y < initial_pose.y + 0.09 or (
        not is_finite(spin_omega.y) or spin_omega.y < 1.0
    ) or physics_world.get_body_position(spin_id).x <= 3.0:
        printerr("Native Godot angular integration/pose publication failed")
        quit(1)
        return
    if not physics_world.remove_body(spin_id):
        printerr("Native Godot actor retirement failed")
        quit(1)
        return

    var initial_y: float = physics_world.get_body_position(box_id).y
    var saw_found := false
    var saw_persists := false
    for frame in range(180):
        if not physics_world.step(1.0 / 60.0):
            printerr("FyzzXYZZY step failed at frame ", frame)
            quit(1)
            return
        for index in range(physics_world.get_contact_event_count()):
            if physics_world.get_contact_event_body0(index) != floor_id or (
                physics_world.get_contact_event_body1(index) != box_id
            ):
                continue
            var kind: int = physics_world.get_contact_event_kind(index)
            if kind == 0:
                saw_found = physics_world.get_contact_event_point_count(index) > 0
                saw_found = saw_found and is_finite(
                    physics_world.get_contact_event_normal_impulse(index))
            elif kind == 1:
                saw_persists = true
    if not saw_found or not saw_persists:
        printerr("Native Godot completed-frame contact Found/Persists missing")
        quit(1)
        return

    var final_position: Vector3 = physics_world.get_body_position(box_id)
    var final_velocity: Vector3 = physics_world.get_body_velocity(box_id)
    print("FyzzXYZZY Godot: initial_y=", initial_y,
        " final_y=", final_position.y, " vy=", final_velocity.y)

    if not is_finite(final_position.y) or (
        final_position.y >= initial_y or
        final_position.y < 0.1 or final_position.y > 0.6 or
        not is_finite(final_velocity.y) or abs(final_velocity.y) > 1.0
    ):
        printerr("FyzzXYZZY Godot contact/settling acceptance failed")
        quit(1)
        return

    if not physics_world.set_body_pose(
        box_id, Vector3(0.0, 2.0, 0.0), Quaternion.IDENTITY
    ) or not physics_world.step(1.0 / 120.0):
        printerr("Native Godot contact separation failed")
        quit(1)
        return
    var saw_lost := false
    for index in range(physics_world.get_contact_event_count()):
        if physics_world.get_contact_event_body0(index) == floor_id and (
            physics_world.get_contact_event_body1(index) == box_id
        ) and physics_world.get_contact_event_kind(index) == 2:
            saw_lost = physics_world.get_contact_event_point_count(index) == 0
    if not saw_lost:
        printerr("Native Godot completed-frame Lost contact missing")
        quit(1)
        return

    print("FyzzXYZZY Godot acceptance PASS (native rigid Scene)")
    quit(0)
