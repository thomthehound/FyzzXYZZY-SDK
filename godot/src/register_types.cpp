// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy_godot_world.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

static void initialize_fyzzxyzzy(ModuleInitializationLevel level) {
    if (level == MODULE_INITIALIZATION_LEVEL_SCENE) {
        ClassDB::register_class<fyzz::godot_adapter::FyzzXYZZYGodotWorld>();
    }
}

static void uninitialize_fyzzxyzzy(ModuleInitializationLevel level) {
    (void)level;
}

extern "C" {

GDExtensionBool GDE_EXPORT fyzzxyzzy_godot_library_init(
    GDExtensionInterfaceGetProcAddress get_proc_address,
    GDExtensionClassLibraryPtr library,
    GDExtensionInitialization* initialization) {
    GDExtensionBinding::InitObject init(get_proc_address, library, initialization);
    init.register_initializer(initialize_fyzzxyzzy);
    init.register_terminator(uninitialize_fyzzxyzzy);
    init.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
    return init.init();
}

} // extern "C"
