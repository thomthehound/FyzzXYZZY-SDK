# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

extends SceneTree

func _initialize() -> void:
    if not ClassDB.class_exists("FizzXGodotWorld"):
        printerr("FizzXGodotWorld is not registered")
        quit(1)
        return

    var physics_world = ClassDB.instantiate("FizzXGodotWorld")
    if physics_world == null:
        printerr("Failed to instantiate FizzXGodotWorld")
        quit(1)
        return

    var floor_id: int = physics_world.add_static_box(
        Vector3(5.0, 0.5, 5.0), Vector3(0.0, -0.5, 0.0))
    var box_id: int = physics_world.add_dynamic_box(
        Vector3(0.25, 0.25, 0.25), Vector3(0.0, 2.0, 0.0), 1.0)
    if floor_id < 0 or box_id < 0:
        printerr("FizzX body creation failed")
        quit(1)
        return

    var initial_y: float = physics_world.get_body_position(box_id).y
    for frame in range(180):
        if not physics_world.step(1.0 / 60.0):
            printerr("FizzX step failed at frame ", frame)
            quit(1)
            return

    var final_position: Vector3 = physics_world.get_body_position(box_id)
    var final_velocity: Vector3 = physics_world.get_body_velocity(box_id)
    print("FizzX Godot M0: initial_y=", initial_y,
        " final_y=", final_position.y, " vy=", final_velocity.y)

    if not is_finite(final_position.y) or (
        final_position.y >= initial_y or
        final_position.y < 0.1 or final_position.y > 0.6 or
        not is_finite(final_velocity.y) or abs(final_velocity.y) > 1.0
    ):
        printerr("FizzX Godot contact/settling acceptance failed")
        quit(1)
        return

    print("FizzX Godot M0 acceptance PASS")
    quit(0)
