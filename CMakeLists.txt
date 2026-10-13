# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)
project(FyzzXYZZY VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(FetchContent)

# The CPU backend uses the pinned PhysX solver; the upstream CUDA build is disabled.
set(PX_GENERATE_GPU_PROJECTS OFF CACHE BOOL "Disable upstream CUDA" FORCE)
set(PX_GENERATE_GPU_PROJECTS_ONLY OFF CACHE BOOL "" FORCE)
set(PX_GENERATE_STATIC_LIBRARIES ON CACHE BOOL "Use static PhysX for the standalone SDK" FORCE)
set(PX_GENERATE_GPU_STATIC_LIBRARIES OFF CACHE BOOL "" FORCE)
set(PX_BUILDSNIPPETS OFF CACHE BOOL "" FORCE)
set(PX_BUILDPVDRUNTIME OFF CACHE BOOL "" FORCE)
set(NV_USE_STATIC_WINCRT OFF CACHE BOOL "Use the DLL MSVC CRT for downstream consumers" FORCE)

# Point this at the root of a local PhysX checkout to iterate without fetching.
set(FYZZXYZZY_PHYSX_SOURCE_DIR "" CACHE PATH "Root of local PhysX 5.11 source checkout (contains physx/CMakeLists.txt)")
if(NOT FYZZXYZZY_PHYSX_SOURCE_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/PhysX/physx/CMakeLists.txt")
  set(FYZZXYZZY_PHYSX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/PhysX"
    CACHE PATH "Root of local PhysX 5.11 source checkout" FORCE)
endif()
if(FYZZXYZZY_PHYSX_SOURCE_DIR)
  if(NOT EXISTS "${FYZZXYZZY_PHYSX_SOURCE_DIR}/physx/CMakeLists.txt")
    message(FATAL_ERROR "FYZZXYZZY_PHYSX_SOURCE_DIR must point at a checkout root containing physx/CMakeLists.txt")
  endif()
  set(FETCHCONTENT_SOURCE_DIR_FYZZXYZZY_PHYSX "${FYZZXYZZY_PHYSX_SOURCE_DIR}")
endif()

# Pinned PhysX source provenance and upstream notices remain intact.
FetchContent_Declare(fyzzxyzzy_physx
  GIT_REPOSITORY https://github.com/thomthehound/PhysX.git
  GIT_TAG 6dc53585d7db07346f53d0598a5fd3bfdce47d8d
  SOURCE_SUBDIR physx
  GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(fyzzxyzzy_physx)

if(NOT TARGET physx_lib)
  message(FATAL_ERROR "Expected PhysX 5.11 CMake interface target 'physx_lib'")
endif()

# PhysX static consumers must not import symbols from PhysX DLLs on Windows.
# The upstream static build defines this internally but does not export it
# through the physx_lib interface target.
if(PX_GENERATE_STATIC_LIBRARIES)
  target_compile_definitions(physx_lib INTERFACE PX_PHYSX_STATIC_LIB)
endif()

add_library(fyzzxyzzy_core STATIC src/world.cpp)
add_library(FyzzXYZZY::Core ALIAS fyzzxyzzy_core)
target_compile_features(fyzzxyzzy_core PUBLIC cxx_std_17)
target_include_directories(fyzzxyzzy_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(fyzzxyzzy_core PRIVATE physx_lib)
set_target_properties(fyzzxyzzy_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_options(fyzzxyzzy_core PRIVATE /W4)
else()
  target_compile_options(fyzzxyzzy_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
  add_executable(fyzzxyzzy_headless_smoke tests/headless_smoke.cpp)
  target_link_libraries(fyzzxyzzy_headless_smoke PRIVATE FyzzXYZZY::Core)
  add_test(NAME fyzzxyzzy_headless_smoke COMMAND fyzzxyzzy_headless_smoke)
endif()

# Native compute resources and operations do not depend on Godot or PhysX.
option(FYZZXYZZY_BUILD_VULKAN "Build the standalone Vulkan compute backend" OFF)
if(FYZZXYZZY_BUILD_VULKAN)
  find_package(Vulkan REQUIRED)
  add_library(fyzzxyzzy_compute STATIC src/compute_vulkan.cpp)
  add_library(FyzzXYZZY::Compute ALIAS fyzzxyzzy_compute)
  target_compile_features(fyzzxyzzy_compute PUBLIC cxx_std_17)
  target_include_directories(fyzzxyzzy_compute PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(fyzzxyzzy_compute PRIVATE Vulkan::Vulkan)
  set_target_properties(fyzzxyzzy_compute PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # These independent physics demonstrations are not translations of PhysX
  # GPU algorithms or orchestration. Do not link them into the SDK by default.
  option(FYZZXYZZY_ENABLE_GPU_PHYSICS_PROTOTYPES
    "Build non-PhysX experimental Vulkan physics demonstrations" OFF)
  if(FYZZXYZZY_ENABLE_GPU_PHYSICS_PROTOTYPES)
    target_sources(fyzzxyzzy_compute PRIVATE src/floor_contact.cpp)
    add_library(fyzzxyzzy_vulkan_world STATIC src/world_vulkan.cpp)
    add_library(FyzzXYZZY::VulkanWorld ALIAS fyzzxyzzy_vulkan_world)
    target_compile_features(fyzzxyzzy_vulkan_world PUBLIC cxx_std_17)
    target_include_directories(fyzzxyzzy_vulkan_world PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(fyzzxyzzy_vulkan_world PUBLIC
      FyzzXYZZY::Core FyzzXYZZY::Compute
    )
  endif()
  if(MSVC)
    target_compile_options(fyzzxyzzy_compute PRIVATE /W4)
  else()
    target_compile_options(fyzzxyzzy_compute PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(BUILD_TESTING)
    add_executable(fyzzxyzzy_vulkan_memory_smoke tests/vulkan_memory_smoke.cpp)
    target_link_libraries(fyzzxyzzy_vulkan_memory_smoke PRIVATE FyzzXYZZY::Compute)
    add_test(NAME fyzzxyzzy_vulkan_memory_smoke COMMAND fyzzxyzzy_vulkan_memory_smoke)

    find_program(FYZZXYZZY_GLSLANG_VALIDATOR NAMES glslangValidator REQUIRED)
    set(FYZZXYZZY_AFFINE_SHADER "${CMAKE_CURRENT_BINARY_DIR}/kernels/affine.comp.spv")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_AFFINE_SHADER}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_GLSLANG_VALIDATOR}" -V -S comp
        -o "${FYZZXYZZY_AFFINE_SHADER}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/affine.comp"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/affine.comp"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_dispatch_shader DEPENDS "${FYZZXYZZY_AFFINE_SHADER}")
    add_executable(fyzzxyzzy_vulkan_dispatch_smoke tests/vulkan_dispatch_smoke.cpp)
    target_link_libraries(fyzzxyzzy_vulkan_dispatch_smoke PRIVATE FyzzXYZZY::Compute)
    add_dependencies(fyzzxyzzy_vulkan_dispatch_smoke fyzzxyzzy_dispatch_shader)
    add_test(NAME fyzzxyzzy_vulkan_dispatch_smoke
      COMMAND fyzzxyzzy_vulkan_dispatch_smoke "${FYZZXYZZY_AFFINE_SHADER}")

    if(FYZZXYZZY_ENABLE_GPU_PHYSICS_PROTOTYPES)
      set(FYZZXYZZY_FLOOR_SHADER "${CMAKE_CURRENT_BINARY_DIR}/kernels/rigid_floor.comp.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_FLOOR_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_GLSLANG_VALIDATOR}" -V -S comp
          -o "${FYZZXYZZY_FLOOR_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/rigid_floor.comp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/rigid_floor.comp"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_floor_shader DEPENDS "${FYZZXYZZY_FLOOR_SHADER}")
      add_executable(fyzzxyzzy_vulkan_floor_physics tests/vulkan_floor_physics.cpp)
      target_link_libraries(fyzzxyzzy_vulkan_floor_physics PRIVATE FyzzXYZZY::Compute)
      add_dependencies(fyzzxyzzy_vulkan_floor_physics fyzzxyzzy_floor_shader)
      add_test(NAME fyzzxyzzy_vulkan_floor_physics
        COMMAND fyzzxyzzy_vulkan_floor_physics "${FYZZXYZZY_FLOOR_SHADER}")
  
      add_executable(fyzzxyzzy_vulkan_world_smoke tests/vulkan_world_smoke.cpp)
      target_link_libraries(fyzzxyzzy_vulkan_world_smoke PRIVATE FyzzXYZZY::VulkanWorld)
      add_dependencies(fyzzxyzzy_vulkan_world_smoke fyzzxyzzy_floor_shader)
      add_test(NAME fyzzxyzzy_vulkan_world_smoke
        COMMAND fyzzxyzzy_vulkan_world_smoke "${FYZZXYZZY_FLOOR_SHADER}")
    endif()
  endif()
endif()

# Godot is a first SDK consumer, never a dependency of FyzzXYZZY::Core.
option(FYZZXYZZY_BUILD_GODOT "Build the optional Godot 4.7 GDExtension consumer" OFF)
if(FYZZXYZZY_BUILD_GODOT)
  set(GODOTCPP_API_VERSION "4.7" CACHE STRING "Target Godot extension API version" FORCE)
  # Use godot-cpp's complete supported API bindings until an upstream-compatible
  # trimmed profile can cover its internal runtime dependencies.
  set(GODOTCPP_BUILD_PROFILE "" CACHE FILEPATH "Godot binding profile" FORCE)
  FetchContent_Declare(fyzzxyzzy_godot_cpp
    GIT_REPOSITORY https://github.com/godotengine/godot-cpp.git
    GIT_TAG 507ed9d840c01a3c5b2a39af8bb4000bfac30bf5
    GIT_SHALLOW FALSE
  )
  FetchContent_MakeAvailable(fyzzxyzzy_godot_cpp)

  add_library(fyzzxyzzy_godot SHARED
    godot/src/fyzzxyzzy_godot_world.cpp
    godot/src/register_types.cpp
  )
  target_compile_features(fyzzxyzzy_godot PRIVATE cxx_std_17)
  target_link_libraries(fyzzxyzzy_godot PRIVATE FyzzXYZZY::Core godot::cpp)
  set_target_properties(fyzzxyzzy_godot PROPERTIES PREFIX "" OUTPUT_NAME "fyzzxyzzy_godot")

  # The sample descriptor references the same basename on Linux and Windows.
  set(FYZZXYZZY_GODOT_SAMPLE_BIN "${CMAKE_CURRENT_SOURCE_DIR}/godot/test_project/bin")
  add_custom_command(TARGET fyzzxyzzy_godot POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${FYZZXYZZY_GODOT_SAMPLE_BIN}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "$<TARGET_FILE:fyzzxyzzy_godot>"
      "${FYZZXYZZY_GODOT_SAMPLE_BIN}/$<TARGET_FILE_NAME:fyzzxyzzy_godot>"
    VERBATIM
  )
endif()

# Produce a complete, offline-configurable CPU SDK source distribution.
# PhysX source is included in the archive; optional godot-cpp bindings are not.
add_custom_target(fyzzxyzzy_source_bundle
  COMMAND "${CMAKE_COMMAND}"
    "-DFYZZXYZZY_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DFYZZXYZZY_PHYSX_ROOT=${fyzzxyzzy_physx_SOURCE_DIR}"
    "-DFYZZXYZZY_BUNDLE_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/fyzzxyzzy-source.tar.gz"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package-source.cmake"
  BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/fyzzxyzzy-source.tar.gz"
  VERBATIM
)
