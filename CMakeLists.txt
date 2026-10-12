# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)
project(FizzX VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(FetchContent)

# M0 uses the unmodified PhysX CPU solver. Portable GPU runtimes come later.
set(PX_GENERATE_GPU_PROJECTS OFF CACHE BOOL "Disable upstream CUDA during M0" FORCE)
set(PX_GENERATE_GPU_PROJECTS_ONLY OFF CACHE BOOL "" FORCE)
set(PX_GENERATE_STATIC_LIBRARIES ON CACHE BOOL "Use static PhysX for the standalone SDK" FORCE)
set(PX_GENERATE_GPU_STATIC_LIBRARIES OFF CACHE BOOL "" FORCE)
set(PX_BUILDSNIPPETS OFF CACHE BOOL "" FORCE)
set(PX_BUILDPVDRUNTIME OFF CACHE BOOL "" FORCE)
set(NV_USE_STATIC_WINCRT OFF CACHE BOOL "Use the DLL MSVC CRT for downstream consumers" FORCE)

# Point this at the root of a local PhysX checkout to iterate without fetching.
set(FIZZX_PHYSX_SOURCE_DIR "" CACHE PATH "Root of local PhysX 5.11 source checkout (contains physx/CMakeLists.txt)")
if(FIZZX_PHYSX_SOURCE_DIR)
  if(NOT EXISTS "${FIZZX_PHYSX_SOURCE_DIR}/physx/CMakeLists.txt")
    message(FATAL_ERROR "FIZZX_PHYSX_SOURCE_DIR must point at a checkout root containing physx/CMakeLists.txt")
  endif()
  set(FETCHCONTENT_SOURCE_DIR_FIZZX_PHYSX "${FIZZX_PHYSX_SOURCE_DIR}")
endif()

# Pinned upstream source is a bootstrap dependency, not FizzX's final vendoring
# strategy. PhysX source provenance and existing notices remain intact.
FetchContent_Declare(fizzx_physx
  GIT_REPOSITORY https://github.com/thomthehound/PhysX.git
  GIT_TAG 6dc53585d7db07346f53d0598a5fd3bfdce47d8d
  SOURCE_SUBDIR physx
  GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(fizzx_physx)

if(NOT TARGET physx_lib)
  message(FATAL_ERROR "Expected PhysX 5.11 CMake interface target 'physx_lib'")
endif()

# PhysX static consumers must not import symbols from PhysX DLLs on Windows.
# The upstream static build defines this internally but does not export it
# through the physx_lib interface target.
if(PX_GENERATE_STATIC_LIBRARIES)
  target_compile_definitions(physx_lib INTERFACE PX_PHYSX_STATIC_LIB)
endif()

add_library(fizzx_core STATIC src/world.cpp)
add_library(FizzX::Core ALIAS fizzx_core)
target_compile_features(fizzx_core PUBLIC cxx_std_17)
target_include_directories(fizzx_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(fizzx_core PRIVATE physx_lib)
set_target_properties(fizzx_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_options(fizzx_core PRIVATE /W4)
else()
  target_compile_options(fizzx_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
  add_executable(fizzx_headless_smoke tests/headless_smoke.cpp)
  target_link_libraries(fizzx_headless_smoke PRIVATE FizzX::Core)
  add_test(NAME fizzx_headless_smoke COMMAND fizzx_headless_smoke)
endif()

# Godot is a first SDK consumer, never a dependency of FizzX::Core.
option(FIZZX_BUILD_GODOT "Build the optional Godot 4.7 GDExtension consumer" OFF)
if(FIZZX_BUILD_GODOT)
  set(GODOTCPP_API_VERSION "4.7" CACHE STRING "Target Godot extension API version" FORCE)
  FetchContent_Declare(fizzx_godot_cpp
    GIT_REPOSITORY https://github.com/godotengine/godot-cpp.git
    GIT_TAG 507ed9d840c01a3c5b2a39af8bb4000bfac30bf5
    GIT_SHALLOW FALSE
  )
  FetchContent_MakeAvailable(fizzx_godot_cpp)

  add_library(fizzx_godot SHARED
    godot/src/fizzx_godot_world.cpp
    godot/src/register_types.cpp
  )
  target_compile_features(fizzx_godot PRIVATE cxx_std_17)
  target_link_libraries(fizzx_godot PRIVATE FizzX::Core godot::cpp)
  set_target_properties(fizzx_godot PROPERTIES PREFIX "" OUTPUT_NAME "fizzx_godot")

  # The sample descriptor references the same basename on Linux and Windows.
  set(FIZZX_GODOT_SAMPLE_BIN "${CMAKE_CURRENT_SOURCE_DIR}/godot/test_project/bin")
  add_custom_command(TARGET fizzx_godot POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${FIZZX_GODOT_SAMPLE_BIN}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "$<TARGET_FILE:fizzx_godot>"
      "${FIZZX_GODOT_SAMPLE_BIN}/$<TARGET_FILE_NAME:fizzx_godot>"
    VERBATIM
  )
endif()
