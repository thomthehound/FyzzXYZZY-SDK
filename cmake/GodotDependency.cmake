# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

# Dependency and binding build contract. CI hashes this module and
# godot/dependency.json, not unrelated SDK CMake targets, for the cache key.
# One dependency manifest drives the engine acceptance release, Godot's
# extension API and the exact godot-cpp revision on both CI platforms.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/godot/dependency.json" FYZZXYZZY_GODOT_DEPENDENCY_JSON)
string(JSON FYZZXYZZY_GODOT_ENGINE_VERSION GET "${FYZZXYZZY_GODOT_DEPENDENCY_JSON}" engine_version)
string(JSON FYZZXYZZY_GODOT_API_VERSION GET "${FYZZXYZZY_GODOT_DEPENDENCY_JSON}" api_version)
string(JSON FYZZXYZZY_GODOT_CPP_COMMIT GET "${FYZZXYZZY_GODOT_DEPENDENCY_JSON}" godot_cpp_commit)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" FYZZXYZZY_GODOT_ENGINE_API "${FYZZXYZZY_GODOT_ENGINE_VERSION}")
if(NOT FYZZXYZZY_GODOT_ENGINE_API STREQUAL FYZZXYZZY_GODOT_API_VERSION)
  message(FATAL_ERROR "Godot dependency manifest engine/API version mismatch")
endif()
# The checked-in extension descriptor advertises which editor version
# can load the native binary. Do not claim compatibility with an older
# editor when the bindings API is upgraded for newer Godot features.
file(STRINGS
  "${CMAKE_CURRENT_SOURCE_DIR}/godot/test_project/addons/fyzzxyzzy/fyzzxyzzy.gdextension"
  FYZZXYZZY_GODOT_MINIMUM_DECLARATION REGEX "^compatibility_minimum =")
if(NOT FYZZXYZZY_GODOT_MINIMUM_DECLARATION STREQUAL
   "compatibility_minimum = \"${FYZZXYZZY_GODOT_API_VERSION}\"")
  message(FATAL_ERROR
    "Update the .gdextension compatibility_minimum when changing godot/dependency.json api_version")
endif()
set(GODOTCPP_API_VERSION "${FYZZXYZZY_GODOT_API_VERSION}" CACHE STRING "Target Godot extension API version" FORCE)
# Use godot-cpp's complete supported API bindings until an upstream-compatible
# trimmed profile can cover its internal runtime dependencies.
set(GODOTCPP_BUILD_PROFILE "" CACHE FILEPATH "Godot binding profile" FORCE)
FetchContent_Declare(fyzzxyzzy_godot_cpp
  GIT_REPOSITORY https://github.com/godotengine/godot-cpp.git
  GIT_TAG "${FYZZXYZZY_GODOT_CPP_COMMIT}"
  GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(fyzzxyzzy_godot_cpp)

