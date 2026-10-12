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
