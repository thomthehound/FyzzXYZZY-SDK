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

    var floor_id: int = physics_world.add_static_box(
        Vector3(5.0, 0.5, 5.0), Vector3(0.0, -0.5, 0.0))
    var box_id: int = physics_world.add_dynamic_box(
        Vector3(0.25, 0.25, 0.25), Vector3(0.0, 2.0, 0.0), 1.0)
    if floor_id < 0 or box_id < 0:
        printerr("FyzzXYZZY body creation failed")
        quit(1)
        return

    var initial_y: float = physics_world.get_body_position(box_id).y
    for frame in range(180):
        if not physics_world.step(1.0 / 60.0):
            printerr("FyzzXYZZY step failed at frame ", frame)
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

    print("FyzzXYZZY Godot acceptance PASS")
    quit(0)
