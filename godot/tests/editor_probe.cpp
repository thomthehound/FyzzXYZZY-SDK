// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

#ifdef FIZZX_PROBE_REGISTER_CLASS
class FizzXEditorProbe final : public RefCounted {
    GDCLASS(FizzXEditorProbe, RefCounted)

protected:
    static void _bind_methods() {}
};
#endif

static void initialize_probe(ModuleInitializationLevel level) {
#ifdef FIZZX_PROBE_REGISTER_CLASS
    if (level == MODULE_INITIALIZATION_LEVEL_SCENE) {
        ClassDB::register_class<FizzXEditorProbe>();
    }
#else
    (void)level;
#endif
}

static void uninitialize_probe(ModuleInitializationLevel level) {
    (void)level;
}

extern "C" {
GDExtensionBool GDE_EXPORT fizzx_godot_library_init(
    GDExtensionInterfaceGetProcAddress get_proc_address,
    GDExtensionClassLibraryPtr library,
    GDExtensionInitialization *initialization) {
    GDExtensionBinding::InitObject init(get_proc_address, library, initialization);
    init.register_initializer(initialize_probe);
    init.register_terminator(uninitialize_probe);
    init.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
    return init.init();
}
} // extern "C"
