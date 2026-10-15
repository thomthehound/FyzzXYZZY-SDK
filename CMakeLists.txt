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

  # Translated PhysX GPU stages. These operations preserve upstream data
  # contracts; the PhysX scene integration is introduced incrementally.
  add_library(fyzzxyzzy_physx_gpu_stages STATIC
    src/physx_gpu/aabb_encoding.cpp
    src/physx_gpu/mark_removed_pairs_arena.cpp
    src/physx_gpu/removed_projections_arena.cpp
    src/physx_gpu/created_pairs_arena.cpp
    src/physx_gpu/radix_count_arena.cpp
    src/physx_gpu/radix_scatter_arena.cpp
    src/physx_gpu/sap_box_indices_arena.cpp
    src/physx_gpu/sap_ranked_sort_arena.cpp
    src/physx_gpu/sap_endpoint_histogram_arena.cpp
    src/physx_gpu/sap_incremental_comparisons_arena.cpp
    src/physx_gpu/sap_incremental_pairs_arena.cpp
    src/physx_gpu/sap_clear_new_flags_arena.cpp
    src/physx_gpu/broadphase_copy_reports_arena.cpp
    src/physx_gpu/broadphase_report_classification_arena.cpp
    src/physx_gpu/sap_host_frame_roles.cpp
    src/physx_gpu/sap_region_producer_arena.cpp
    src/physx_gpu/sap_region_histograms_arena.cpp
    src/physx_gpu/sap_active_region_lists_arena.cpp
    src/physx_gpu/broadphase_completion.cpp
  )
  target_compile_features(fyzzxyzzy_physx_gpu_stages PUBLIC cxx_std_17)
  target_include_directories(fyzzxyzzy_physx_gpu_stages PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(fyzzxyzzy_physx_gpu_stages PUBLIC FyzzXYZZY::Compute physx_lib)
  set_target_properties(fyzzxyzzy_physx_gpu_stages
    PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

    # Only this source-derived PhysX acceptance test requires DXC. Normal
    # Vulkan compute/memory/dispatch tests remain usable without it.
    option(FYZZXYZZY_BUILD_PHYSX_GPU_AABB_TEST
      "Enable PhysX GPU AABB HLSL/SPIR-V acceptance test (requires DXC)" OFF)
    if(FYZZXYZZY_BUILD_PHYSX_GPU_AABB_TEST)
      # PhysX broadphase.cu::translateAABBsLaunch is compiled from one HLSL
      # source for Vulkan SPIR-V and (in Windows CI) D3D12 DXIL.
      find_program(FYZZXYZZY_DXC NAMES dxc REQUIRED)
      set(FYZZXYZZY_PHYSX_AABB_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_translate_aabbs.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_AABB_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_PHYSX_AABB_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_aabb_shader DEPENDS "${FYZZXYZZY_PHYSX_AABB_SHADER}")
      add_executable(fyzzxyzzy_physx_aabb_encoding tests/physx_gpu_aabb_encoding.cpp)
      target_include_directories(fyzzxyzzy_physx_aabb_encoding PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_aabb_encoding
        PRIVATE fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_aabb_encoding fyzzxyzzy_physx_aabb_shader)
      add_test(NAME fyzzxyzzy_physx_aabb_encoding
        COMMAND fyzzxyzzy_physx_aabb_encoding "${FYZZXYZZY_PHYSX_AABB_SHADER}")

      # PhysX PxgAABBManager::postBroadPhase result-publication boundary:
      # GPU completion precedes found/lost-pair task reference release.
      add_executable(fyzzxyzzy_physx_broadphase_completion
        tests/physx_gpu_broadphase_completion.cpp)
      target_include_directories(fyzzxyzzy_physx_broadphase_completion PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_broadphase_completion PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_test(NAME fyzzxyzzy_physx_broadphase_completion
        COMMAND fyzzxyzzy_physx_broadphase_completion)

      # Source-derived PxgBroadPhaseDesc subset and markRemovedPairsLaunch:
      # a single GPU arena, no per-pointer descriptors, exact SAP state oracle.
      set(FYZZXYZZY_PHYSX_REMOVED_ARENA_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_mark_removed_pairs_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_REMOVED_ARENA_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_PHYSX_REMOVED_ARENA_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_mark_removed_pairs_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_mark_removed_pairs_arena.hlsl"
        VERBATIM
      )
      # Same source and dataflow: five offset fields are pre-resolved into
      # dispatch constants instead of loaded from the GPU descriptor.
      set(FYZZXYZZY_PHYSX_REMOVED_RESOLVED_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_mark_removed_pairs_resolved.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_REMOVED_RESOLVED_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main -D FYZZXYZZY_RESOLVED_SUBSET=1
          -Fo "${FYZZXYZZY_PHYSX_REMOVED_RESOLVED_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_mark_removed_pairs_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_mark_removed_pairs_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_removed_resolved_shader
        DEPENDS "${FYZZXYZZY_PHYSX_REMOVED_RESOLVED_SHADER}")
      add_custom_target(fyzzxyzzy_physx_removed_arena_shader
        DEPENDS "${FYZZXYZZY_PHYSX_REMOVED_ARENA_SHADER}")
      add_executable(fyzzxyzzy_physx_mark_removed_arena
        tests/physx_gpu_mark_removed_arena.cpp)
      target_include_directories(fyzzxyzzy_physx_mark_removed_arena PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_mark_removed_arena PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_mark_removed_arena
        fyzzxyzzy_physx_removed_arena_shader
        fyzzxyzzy_physx_removed_resolved_shader)
      add_test(NAME fyzzxyzzy_physx_mark_removed_arena
        COMMAND fyzzxyzzy_physx_mark_removed_arena
          "${FYZZXYZZY_PHYSX_REMOVED_ARENA_SHADER}"
          "${FYZZXYZZY_PHYSX_REMOVED_RESOLVED_SHADER}")

      # markRemovedPairsProjectionsLaunch, after the incremental SAP pass.
      set(FYZZXYZZY_PHYSX_REMOVED_PROJECTIONS_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_removed_projections_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_REMOVED_PROJECTIONS_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_PHYSX_REMOVED_PROJECTIONS_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_removed_projections_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_removed_projections_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_removed_projections_shader
        DEPENDS "${FYZZXYZZY_PHYSX_REMOVED_PROJECTIONS_SHADER}")
      add_executable(fyzzxyzzy_physx_removed_projections
        tests/physx_gpu_removed_projections.cpp)
      target_include_directories(fyzzxyzzy_physx_removed_projections PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_removed_projections PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_removed_projections
        fyzzxyzzy_physx_removed_arena_shader
        fyzzxyzzy_physx_removed_projections_shader)
      add_test(NAME fyzzxyzzy_physx_removed_projections
        COMMAND fyzzxyzzy_physx_removed_projections
          "${FYZZXYZZY_PHYSX_REMOVED_ARENA_SHADER}"
          "${FYZZXYZZY_PHYSX_REMOVED_PROJECTIONS_SHADER}")
      # PhysX broadphase.cu::markCreatedPairsLaunch (32x256).
      set(FYZZXYZZY_PHYSX_CREATED_PAIRS_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_created_pairs_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_CREATED_PAIRS_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_PHYSX_CREATED_PAIRS_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_created_pairs_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_created_pairs_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_created_pairs_shader
        DEPENDS "${FYZZXYZZY_PHYSX_CREATED_PAIRS_SHADER}")
      add_executable(fyzzxyzzy_physx_created_pairs
        tests/physx_gpu_created_pairs.cpp)
      target_include_directories(fyzzxyzzy_physx_created_pairs PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_created_pairs PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_created_pairs
        fyzzxyzzy_physx_created_pairs_shader)
      add_test(NAME fyzzxyzzy_physx_created_pairs
        COMMAND fyzzxyzzy_physx_created_pairs
          "${FYZZXYZZY_PHYSX_CREATED_PAIRS_SHADER}")

      # PhysX radixSortImpl.cu multi-block count stage (32x3x1024).
      set(FYZZXYZZY_PHYSX_RADIX_COUNT_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_radix_count_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_RADIX_COUNT_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_PHYSX_RADIX_COUNT_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_count_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_count_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_radix_count_shader
        DEPENDS "${FYZZXYZZY_PHYSX_RADIX_COUNT_SHADER}")
      set(FYZZXYZZY_PHYSX_RADIX_COUNT_WAVE_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_radix_count_wave_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_RADIX_COUNT_WAVE_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_6 -E main
          -Fo "${FYZZXYZZY_PHYSX_RADIX_COUNT_WAVE_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_count_wave_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_count_wave_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_radix_count_wave_shader
        DEPENDS "${FYZZXYZZY_PHYSX_RADIX_COUNT_WAVE_SHADER}")
      add_executable(fyzzxyzzy_physx_radix_count
        tests/physx_gpu_radix_count.cpp)
      target_include_directories(fyzzxyzzy_physx_radix_count PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_radix_count PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_radix_count
        fyzzxyzzy_physx_radix_count_shader fyzzxyzzy_physx_radix_count_wave_shader)
      add_test(NAME fyzzxyzzy_physx_radix_count
        COMMAND fyzzxyzzy_physx_radix_count
          "${FYZZXYZZY_PHYSX_RADIX_COUNT_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_COUNT_WAVE_SHADER}")

      # PhysX radixSortImpl.cu::radixSortMultiCalculateRanksLaunchWithCount.
      set(FYZZXYZZY_PHYSX_RADIX_SCATTER_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_radix_scatter_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_RADIX_SCATTER_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_PHYSX_RADIX_SCATTER_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_scatter_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_scatter_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_radix_scatter_shader
        DEPENDS "${FYZZXYZZY_PHYSX_RADIX_SCATTER_SHADER}")

      # SM6.6 / Vulkan compute-subgroup fast path: 8 packed 2-digit scans
      # instead of 16 separate 1024-thread Hillis-Steele scans.
      set(FYZZXYZZY_PHYSX_RADIX_SCATTER_WAVE_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_radix_scatter_wave_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_RADIX_SCATTER_WAVE_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_6 -E main
          -Fo "${FYZZXYZZY_PHYSX_RADIX_SCATTER_WAVE_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_scatter_wave_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_scatter_wave_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_radix_scatter_wave_shader
        DEPENDS "${FYZZXYZZY_PHYSX_RADIX_SCATTER_WAVE_SHADER}")
      add_executable(fyzzxyzzy_physx_radix_scatter
        tests/physx_gpu_radix_scatter.cpp)
      target_include_directories(fyzzxyzzy_physx_radix_scatter PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_radix_scatter PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_radix_scatter
        fyzzxyzzy_physx_radix_count_shader fyzzxyzzy_physx_radix_count_wave_shader
        fyzzxyzzy_physx_radix_scatter_shader
        fyzzxyzzy_physx_radix_scatter_wave_shader)
      add_test(NAME fyzzxyzzy_physx_radix_scatter
        COMMAND fyzzxyzzy_physx_radix_scatter
          "${FYZZXYZZY_PHYSX_RADIX_COUNT_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_SCATTER_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_COUNT_WAVE_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_SCATTER_WAVE_SHADER}")

      # broadphase.cu::initializeSapBox1DLaunch (ceil(2 * handles / 256)).
      set(FYZZXYZZY_PHYSX_SAP_BOX_INDICES_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_box_indices_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_SAP_BOX_INDICES_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_PHYSX_SAP_BOX_INDICES_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_box_indices_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_box_indices_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_sap_box_indices_shader
        DEPENDS "${FYZZXYZZY_PHYSX_SAP_BOX_INDICES_SHADER}")
      add_executable(fyzzxyzzy_physx_sap_box_indices
        tests/physx_gpu_sap_box_indices.cpp)
      target_include_directories(fyzzxyzzy_physx_sap_box_indices PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_sap_box_indices PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_sap_box_indices
        fyzzxyzzy_physx_created_pairs_shader
        fyzzxyzzy_physx_radix_count_shader
        fyzzxyzzy_physx_radix_scatter_shader
        fyzzxyzzy_physx_sap_box_indices_shader
        fyzzxyzzy_physx_radix_count_wave_shader
        fyzzxyzzy_physx_radix_scatter_wave_shader
)
      add_test(NAME fyzzxyzzy_physx_sap_box_indices
        COMMAND fyzzxyzzy_physx_sap_box_indices
          "${FYZZXYZZY_PHYSX_CREATED_PAIRS_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_COUNT_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_SCATTER_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_BOX_INDICES_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_COUNT_WAVE_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_SCATTER_WAVE_SHADER}"
)

      # PhysX WR: initializeRadixRanks → eight radix passes → updateHandles.
      foreach(_sap_kernel IN ITEMS initialize_ranks update_handles)
        set(FYZZXYZZY_PHYSX_SAP_${_sap_kernel}_SHADER
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_${_sap_kernel}_arena.spv")
        add_custom_command(
          OUTPUT "${FYZZXYZZY_PHYSX_SAP_${_sap_kernel}_SHADER}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
            -T cs_6_0 -E main
            -Fo "${FYZZXYZZY_PHYSX_SAP_${_sap_kernel}_SHADER}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_${_sap_kernel}_arena.hlsl"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_${_sap_kernel}_arena.hlsl"
          VERBATIM
        )
        add_custom_target(fyzzxyzzy_physx_sap_${_sap_kernel}_shader
          DEPENDS "${FYZZXYZZY_PHYSX_SAP_${_sap_kernel}_SHADER}")
      endforeach()
      add_executable(fyzzxyzzy_physx_sap_ranked_sort
        tests/physx_gpu_sap_ranked_sort.cpp)
      target_include_directories(fyzzxyzzy_physx_sap_ranked_sort PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_sap_ranked_sort PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_sap_ranked_sort
        fyzzxyzzy_physx_radix_count_shader
        fyzzxyzzy_physx_radix_scatter_shader
        fyzzxyzzy_physx_sap_initialize_ranks_shader
        fyzzxyzzy_physx_sap_update_handles_shader
        fyzzxyzzy_physx_radix_count_wave_shader
        fyzzxyzzy_physx_radix_scatter_wave_shader
)
      add_test(NAME fyzzxyzzy_physx_sap_ranked_sort
        COMMAND fyzzxyzzy_physx_sap_ranked_sort
          "${FYZZXYZZY_PHYSX_RADIX_COUNT_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_SCATTER_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_initialize_ranks_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_update_handles_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_COUNT_WAVE_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_SCATTER_WAVE_SHADER}"
)

      # Original BP_COMPUTE_ENDPT_HISTOGRAM / BP_OUTPUT_ENDPT_HISTOGRAM.
      foreach(_hist_pass IN ITEMS first second)
        set(FYZZXYZZY_PHYSX_SAP_ENDPOINT_${_hist_pass}_SHADER
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_endpoint_histogram_${_hist_pass}_arena.spv")
        add_custom_command(
          OUTPUT "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_${_hist_pass}_SHADER}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
            -T cs_6_0 -E main
            -Fo "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_${_hist_pass}_SHADER}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_endpoint_histogram_${_hist_pass}_arena.hlsl"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_endpoint_histogram_${_hist_pass}_arena.hlsl"
          VERBATIM
        )
        add_custom_target(fyzzxyzzy_physx_sap_endpoint_${_hist_pass}_shader
          DEPENDS "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_${_hist_pass}_SHADER}")
      endforeach()

      # Source's packed 10-bit triple-counter first pass, native SM6.6
      # wave prefix; the original first-pass shader stays as the oracle
      # fallback and the original second-pass output kernel is unchanged.
      set(FYZZXYZZY_PHYSX_SAP_ENDPOINT_WAVE_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_endpoint_histogram_wave.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_WAVE_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_6 -E main
          -Fo "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_WAVE_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_endpoint_histogram_wave_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_endpoint_histogram_wave_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_sap_endpoint_wave_shader
        DEPENDS "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_WAVE_SHADER}")
      add_executable(fyzzxyzzy_physx_sap_endpoint_histogram
        tests/physx_gpu_sap_endpoint_histogram.cpp)
      target_include_directories(fyzzxyzzy_physx_sap_endpoint_histogram PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_sap_endpoint_histogram PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_sap_endpoint_histogram
        fyzzxyzzy_physx_sap_endpoint_first_shader
        fyzzxyzzy_physx_sap_endpoint_second_shader
        fyzzxyzzy_physx_sap_endpoint_wave_shader)
      add_test(NAME fyzzxyzzy_physx_sap_endpoint_histogram
        COMMAND fyzzxyzzy_physx_sap_endpoint_histogram
          "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_first_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_second_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_WAVE_SHADER}")

      # PhysX BP_COMPUTE_INCREMENTAL_CMP_COUNTS1 / COUNTS2 (32x256).
      foreach(_comparison_pass IN ITEMS first second)
        set(FYZZXYZZY_PHYSX_SAP_COMPARISONS_${_comparison_pass}_SHADER
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_incremental_comparisons_${_comparison_pass}_arena.spv")
        add_custom_command(
          OUTPUT "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_${_comparison_pass}_SHADER}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
            -T cs_6_0 -E main
            -Fo "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_${_comparison_pass}_SHADER}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_comparisons_${_comparison_pass}_arena.hlsl"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_comparisons_${_comparison_pass}_arena.hlsl"
          VERBATIM
        )
        add_custom_target(fyzzxyzzy_physx_sap_comparisons_${_comparison_pass}_shader
          DEPENDS "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_${_comparison_pass}_SHADER}")
      endforeach()

      # PhysX stage1 CUDA warp-first scan: 3 independent 32-bit counts,
      # one native wave prefix per axis and original 32x256 launch.
      set(FYZZXYZZY_PHYSX_SAP_COMPARISONS_WAVE_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_incremental_comparisons_wave.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_WAVE_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_6 -E main
          -Fo "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_WAVE_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_comparisons_wave_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_comparisons_wave_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_sap_comparisons_wave_shader
        DEPENDS "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_WAVE_SHADER}")
      add_executable(fyzzxyzzy_physx_sap_incremental_comparisons
        tests/physx_gpu_sap_incremental_comparisons.cpp)
      target_include_directories(fyzzxyzzy_physx_sap_incremental_comparisons PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_sap_incremental_comparisons PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_sap_incremental_comparisons
        fyzzxyzzy_physx_sap_endpoint_first_shader
        fyzzxyzzy_physx_sap_endpoint_second_shader
        fyzzxyzzy_physx_sap_comparisons_first_shader
        fyzzxyzzy_physx_sap_comparisons_second_shader
        fyzzxyzzy_physx_sap_endpoint_wave_shader
        fyzzxyzzy_physx_sap_comparisons_wave_shader)
      add_test(NAME fyzzxyzzy_physx_sap_incremental_comparisons
        COMMAND fyzzxyzzy_physx_sap_incremental_comparisons
          "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_first_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_second_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_first_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_second_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_WAVE_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_WAVE_SHADER}")

      # PhysX BP_INCREMENTAL_SAP: 256 x 3 x 256 source dispatch.
      set(FYZZXYZZY_PHYSX_SAP_INCREMENTAL_PAIRS_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_incremental_pairs_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_SAP_INCREMENTAL_PAIRS_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_PHYSX_SAP_INCREMENTAL_PAIRS_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_pairs_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_pairs_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_sap_incremental_pairs_shader
        DEPENDS "${FYZZXYZZY_PHYSX_SAP_INCREMENTAL_PAIRS_SHADER}")
            # Native packed found/lost wave scan, original 256x3x256 dispatch.
      set(FYZZXYZZY_PHYSX_SAP_INCREMENTAL_PAIRS_WAVE_SHADER "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_incremental_pairs_wave_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_SAP_INCREMENTAL_PAIRS_WAVE_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_6 -E main
          -Fo "${FYZZXYZZY_PHYSX_SAP_INCREMENTAL_PAIRS_WAVE_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_pairs_wave_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_pairs_wave_arena.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_physx_sap_incremental_pairs_wave_shader DEPENDS "${FYZZXYZZY_PHYSX_SAP_INCREMENTAL_PAIRS_WAVE_SHADER}")
      add_executable(fyzzxyzzy_physx_sap_incremental_pairs
        tests/physx_gpu_sap_incremental_pairs.cpp)
      target_include_directories(fyzzxyzzy_physx_sap_incremental_pairs PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_sap_incremental_pairs PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_sap_incremental_pairs
        fyzzxyzzy_physx_sap_endpoint_first_shader
        fyzzxyzzy_physx_sap_endpoint_second_shader
        fyzzxyzzy_physx_sap_comparisons_first_shader
        fyzzxyzzy_physx_sap_comparisons_second_shader
        fyzzxyzzy_physx_sap_incremental_pairs_shader
        fyzzxyzzy_physx_sap_endpoint_wave_shader
        fyzzxyzzy_physx_sap_comparisons_wave_shader
        fyzzxyzzy_physx_sap_incremental_pairs_wave_shader)
      add_test(NAME fyzzxyzzy_physx_sap_incremental_pairs
        COMMAND fyzzxyzzy_physx_sap_incremental_pairs
          "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_first_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_second_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_first_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_second_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_INCREMENTAL_PAIRS_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ENDPOINT_WAVE_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_COMPARISONS_WAVE_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_INCREMENTAL_PAIRS_WAVE_SHADER}")

      # PhysX clearNewFlagLaunch: 32 x 256, after new-pair generation.
      set(FYZZXYZZY_PHYSX_SAP_CLEAR_NEW_FLAGS_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_clear_new_flags_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_SAP_CLEAR_NEW_FLAGS_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_PHYSX_SAP_CLEAR_NEW_FLAGS_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_clear_new_flags_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_clear_new_flags_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_sap_clear_new_flags_shader
        DEPENDS "${FYZZXYZZY_PHYSX_SAP_CLEAR_NEW_FLAGS_SHADER}")
      add_executable(fyzzxyzzy_physx_sap_clear_new_flags
        tests/physx_gpu_sap_clear_new_flags.cpp)
      target_include_directories(fyzzxyzzy_physx_sap_clear_new_flags PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_sap_clear_new_flags PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_sap_clear_new_flags
        fyzzxyzzy_physx_created_pairs_shader
        fyzzxyzzy_physx_radix_count_shader
        fyzzxyzzy_physx_radix_scatter_shader
        fyzzxyzzy_physx_sap_box_indices_shader
        fyzzxyzzy_physx_sap_clear_new_flags_shader
        fyzzxyzzy_physx_radix_count_wave_shader
        fyzzxyzzy_physx_radix_scatter_wave_shader
)
      add_test(NAME fyzzxyzzy_physx_sap_clear_new_flags
        COMMAND fyzzxyzzy_physx_sap_clear_new_flags
          "${FYZZXYZZY_PHYSX_CREATED_PAIRS_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_COUNT_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_SCATTER_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_BOX_INDICES_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_CLEAR_NEW_FLAGS_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_COUNT_WAVE_SHADER}"
          "${FYZZXYZZY_PHYSX_RADIX_SCATTER_WAVE_SHADER}"
)

      # PhysX broadphase.cu::copyReports: 64 x 256 publication stage.
      set(FYZZXYZZY_PHYSX_BP_COPY_REPORTS_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_broadphase_copy_reports_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_BP_COPY_REPORTS_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_PHYSX_BP_COPY_REPORTS_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_copy_reports_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_copy_reports_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_bp_copy_reports_shader
        DEPENDS "${FYZZXYZZY_PHYSX_BP_COPY_REPORTS_SHADER}")
      add_executable(fyzzxyzzy_physx_bp_copy_reports
        tests/physx_gpu_broadphase_copy_reports.cpp)
      target_include_directories(fyzzxyzzy_physx_bp_copy_reports PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_bp_copy_reports PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_bp_copy_reports
        fyzzxyzzy_physx_bp_copy_reports_shader)
      add_test(NAME fyzzxyzzy_physx_bp_copy_reports
        COMMAND fyzzxyzzy_physx_bp_copy_reports
          "${FYZZXYZZY_PHYSX_BP_COPY_REPORTS_SHADER}")

      # Pinned PhysX accumulateReportsStage_1 / _2: 32 x 4 x 256 each.
      foreach(_report_pass IN ITEMS first second)
        set(FYZZXYZZY_PHYSX_BP_REPORT_CLASS_${_report_pass}_SHADER
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_broadphase_report_classification_${_report_pass}_arena.spv")
        add_custom_command(
          OUTPUT "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_${_report_pass}_SHADER}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
            -T cs_6_0 -E main
            -Fo "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_${_report_pass}_SHADER}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_report_classification_${_report_pass}_arena.hlsl"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_report_classification_${_report_pass}_arena.hlsl"
          VERBATIM
        )
        add_custom_target(fyzzxyzzy_physx_bp_report_class_${_report_pass}_shader
          DEPENDS "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_${_report_pass}_SHADER}")
      endforeach()
      # Native wave reductions and stable compaction for the original
      # 32 x 4 x 256 report-classification dispatches.
      foreach(_report_pass IN ITEMS first second)
        set(FYZZXYZZY_PHYSX_BP_REPORT_CLASS_${_report_pass}_WAVE_SHADER
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_broadphase_report_classification_${_report_pass}_wave.spv")
        add_custom_command(
          OUTPUT "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_${_report_pass}_WAVE_SHADER}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
            -T cs_6_6 -E main
            -Fo "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_${_report_pass}_WAVE_SHADER}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_report_classification_${_report_pass}_wave_arena.hlsl"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_report_classification_${_report_pass}_wave_arena.hlsl"
          VERBATIM)
        add_custom_target(fyzzxyzzy_physx_bp_report_class_${_report_pass}_wave_shader
          DEPENDS "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_${_report_pass}_WAVE_SHADER}")
      endforeach()
      add_executable(fyzzxyzzy_physx_bp_report_classification
        tests/physx_gpu_broadphase_report_classification.cpp)
      target_include_directories(fyzzxyzzy_physx_bp_report_classification PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_bp_report_classification PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_bp_report_classification
        fyzzxyzzy_physx_bp_report_class_first_shader
        fyzzxyzzy_physx_bp_report_class_second_shader
        fyzzxyzzy_physx_bp_report_class_first_wave_shader
        fyzzxyzzy_physx_bp_report_class_second_wave_shader
        fyzzxyzzy_physx_bp_copy_reports_shader)
      add_test(NAME fyzzxyzzy_physx_bp_report_classification
        COMMAND fyzzxyzzy_physx_bp_report_classification
          "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_first_SHADER}"
          "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_second_SHADER}"
          "${FYZZXYZZY_PHYSX_BP_COPY_REPORTS_SHADER}"
          "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_first_WAVE_SHADER}"
          "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_second_WAVE_SHADER}")

      # PhysX BP_CREATE_REGIONS: original 32 x 256 region-range producer.
      set(FYZZXYZZY_PHYSX_SAP_REGION_PRODUCER_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_region_producer_arena.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_PHYSX_SAP_REGION_PRODUCER_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_PHYSX_SAP_REGION_PRODUCER_SHADER}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_producer_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_producer_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_physx_sap_region_producer_shader
        DEPENDS "${FYZZXYZZY_PHYSX_SAP_REGION_PRODUCER_SHADER}")
      add_executable(fyzzxyzzy_physx_sap_region_producer
        tests/physx_gpu_sap_region_producer.cpp)
      target_include_directories(fyzzxyzzy_physx_sap_region_producer PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_sap_region_producer PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_sap_region_producer
        fyzzxyzzy_physx_sap_region_producer_shader)
      add_test(NAME fyzzxyzzy_physx_sap_region_producer
        COMMAND fyzzxyzzy_physx_sap_region_producer
          "${FYZZXYZZY_PHYSX_SAP_REGION_PRODUCER_SHADER}")

      # Four distinct PhysX region histogram kernels, in source order.
      foreach(_region_kind IN ITEMS start region)
        foreach(_region_pass IN ITEMS first second)
          if(_region_kind STREQUAL "start")
            set(_region_flag 1)
          else()
            set(_region_flag 0)
          endif()
          set(FYZZXYZZY_PHYSX_SAP_REGION_${_region_kind}_${_region_pass}_SHADER
            "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_region_${_region_kind}_histogram_${_region_pass}.spv")
          add_custom_command(
            OUTPUT "${FYZZXYZZY_PHYSX_SAP_REGION_${_region_kind}_${_region_pass}_SHADER}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
            COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
              -T cs_6_0 -E main -DFYZZXYZZY_REGION_START=${_region_flag}
              -Fo "${FYZZXYZZY_PHYSX_SAP_REGION_${_region_kind}_${_region_pass}_SHADER}"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_histogram_${_region_pass}_arena.hlsl"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_histogram_${_region_pass}_arena.hlsl"
            VERBATIM
          )
          add_custom_target(fyzzxyzzy_physx_sap_region_${_region_kind}_${_region_pass}_shader
            DEPENDS "${FYZZXYZZY_PHYSX_SAP_REGION_${_region_kind}_${_region_pass}_SHADER}")
        endforeach()
      endforeach()
      # Fast source-style warp-prefix lowering; source geometry stays
      # 32x256 and the four scan/offset dispatches remain separate.
      foreach(_region_kind IN ITEMS start region)
        if(_region_kind STREQUAL "start")
          set(_region_flag 1)
        else()
          set(_region_flag 0)
        endif()
        set(FYZZXYZZY_PHYSX_SAP_REGION_${_region_kind}_WAVE_SHADER
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_region_${_region_kind}_histogram_wave.spv")
        add_custom_command(
          OUTPUT "${FYZZXYZZY_PHYSX_SAP_REGION_${_region_kind}_WAVE_SHADER}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
            -T cs_6_6 -E main -DFYZZXYZZY_REGION_START=${_region_flag}
            -Fo "${FYZZXYZZY_PHYSX_SAP_REGION_${_region_kind}_WAVE_SHADER}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_histogram_wave_arena.hlsl"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_histogram_wave_arena.hlsl"
          VERBATIM
        )
        add_custom_target(fyzzxyzzy_physx_sap_region_${_region_kind}_wave_shader
          DEPENDS "${FYZZXYZZY_PHYSX_SAP_REGION_${_region_kind}_WAVE_SHADER}")
      endforeach()
      add_executable(fyzzxyzzy_physx_sap_region_histograms
        tests/physx_gpu_sap_region_producer.cpp)
      target_include_directories(fyzzxyzzy_physx_sap_region_histograms PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_sap_region_histograms PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_sap_region_histograms
        fyzzxyzzy_physx_sap_region_producer_shader
        fyzzxyzzy_physx_sap_region_start_first_shader
        fyzzxyzzy_physx_sap_region_start_second_shader
        fyzzxyzzy_physx_sap_region_region_first_shader
        fyzzxyzzy_physx_sap_region_region_second_shader
        fyzzxyzzy_physx_sap_region_start_wave_shader
        fyzzxyzzy_physx_sap_region_region_wave_shader)
      add_test(NAME fyzzxyzzy_physx_sap_region_histograms
        COMMAND fyzzxyzzy_physx_sap_region_histograms
          "${FYZZXYZZY_PHYSX_SAP_REGION_PRODUCER_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_REGION_start_first_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_REGION_start_second_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_REGION_region_first_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_REGION_region_second_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_REGION_start_WAVE_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_REGION_region_WAVE_SHADER}")

      # Three PhysX start/active-region kernels: 32x256, 64x512, 64x256.
      foreach(_active_pass IN ITEMS writeout compute_wave compute_portable output)
        if(_active_pass STREQUAL "compute_wave")
          set(_active_source compute)
          set(_active_flag 1)
          set(_active_profile cs_6_6)
        elseif(_active_pass STREQUAL "compute_portable")
          set(_active_source compute)
          set(_active_flag 0)
          set(_active_profile cs_6_0)
        else()
          set(_active_source ${_active_pass})
          set(_active_flag 0)
          set(_active_profile cs_6_0)
        endif()
        set(FYZZXYZZY_PHYSX_SAP_ACTIVE_${_active_pass}_SHADER
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_active_region_${_active_pass}.spv")
        add_custom_command(
          OUTPUT "${FYZZXYZZY_PHYSX_SAP_ACTIVE_${_active_pass}_SHADER}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
            -T ${_active_profile} -E main -DFYZZXYZZY_ACTIVE_REGION_WAVE=${_active_flag}
            -Fo "${FYZZXYZZY_PHYSX_SAP_ACTIVE_${_active_pass}_SHADER}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_active_region_${_active_source}_arena.hlsl"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_active_region_${_active_source}_arena.hlsl"
          VERBATIM
        )
        add_custom_target(fyzzxyzzy_physx_sap_active_${_active_pass}_shader
          DEPENDS "${FYZZXYZZY_PHYSX_SAP_ACTIVE_${_active_pass}_SHADER}")
      endforeach()
      add_executable(fyzzxyzzy_physx_sap_active_region_lists
        tests/physx_gpu_sap_active_region_lists.cpp)
      target_include_directories(fyzzxyzzy_physx_sap_active_region_lists PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_sap_active_region_lists PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_physx_sap_active_region_lists
        fyzzxyzzy_physx_sap_active_writeout_shader
        fyzzxyzzy_physx_sap_active_compute_wave_shader
        fyzzxyzzy_physx_sap_active_compute_portable_shader
        fyzzxyzzy_physx_sap_active_output_shader)
      add_test(NAME fyzzxyzzy_physx_sap_active_region_lists
        COMMAND fyzzxyzzy_physx_sap_active_region_lists
          "${FYZZXYZZY_PHYSX_SAP_ACTIVE_writeout_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ACTIVE_compute_wave_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ACTIVE_compute_portable_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ACTIVE_output_SHADER}")
    endif()

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

# Native D3D12 execution of PhysX's AABB stage, using the exact same
# AabbEncodingStage host dispatch and reference fixtures as Vulkan.
# The initial executor supports this kernel's 3 SRV + 1 UAV/8-constant ABI.
# It remains a separate static target pending backend-neutral runtime dispatch.
option(FYZZXYZZY_BUILD_D3D12 "Build the native Windows D3D12 compute backend" OFF)
if(FYZZXYZZY_BUILD_D3D12)
  if(NOT WIN32)
    message(FATAL_ERROR "FYZZXYZZY_BUILD_D3D12 requires Windows")
  endif()
  add_library(fyzzxyzzy_compute_d3d12 STATIC src/compute_d3d12.cpp)
  add_library(FyzzXYZZY::ComputeD3D12 ALIAS fyzzxyzzy_compute_d3d12)
  target_compile_features(fyzzxyzzy_compute_d3d12 PUBLIC cxx_std_17)
  target_include_directories(fyzzxyzzy_compute_d3d12 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_link_libraries(fyzzxyzzy_compute_d3d12 PUBLIC d3d12 dxgi d3dcompiler)
  set_target_properties(fyzzxyzzy_compute_d3d12 PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  if(MSVC)
    target_compile_options(fyzzxyzzy_compute_d3d12 PRIVATE /W4)
  endif()

  add_library(fyzzxyzzy_physx_gpu_stages_d3d12 STATIC
    src/physx_gpu/aabb_encoding.cpp
    src/physx_gpu/mark_removed_pairs_arena.cpp
    src/physx_gpu/removed_projections_arena.cpp
    src/physx_gpu/created_pairs_arena.cpp
    src/physx_gpu/radix_count_arena.cpp
    src/physx_gpu/radix_scatter_arena.cpp
    src/physx_gpu/sap_box_indices_arena.cpp
    src/physx_gpu/sap_ranked_sort_arena.cpp
    src/physx_gpu/sap_endpoint_histogram_arena.cpp
    src/physx_gpu/sap_incremental_comparisons_arena.cpp
    src/physx_gpu/sap_incremental_pairs_arena.cpp
    src/physx_gpu/sap_clear_new_flags_arena.cpp
    src/physx_gpu/broadphase_copy_reports_arena.cpp
    src/physx_gpu/broadphase_report_classification_arena.cpp
    src/physx_gpu/sap_host_frame_roles.cpp
    src/physx_gpu/sap_region_producer_arena.cpp
    src/physx_gpu/sap_region_histograms_arena.cpp
    src/physx_gpu/sap_active_region_lists_arena.cpp
    src/physx_gpu/broadphase_completion.cpp)
  target_compile_features(fyzzxyzzy_physx_gpu_stages_d3d12 PUBLIC cxx_std_17)
  target_include_directories(fyzzxyzzy_physx_gpu_stages_d3d12 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(fyzzxyzzy_physx_gpu_stages_d3d12 PUBLIC
    FyzzXYZZY::ComputeD3D12 physx_lib)

  option(FYZZXYZZY_BUILD_D3D12_AABB_TEST
    "Enable native D3D12 PhysX AABB conformance test (requires DXC)" OFF)
  if(BUILD_TESTING AND FYZZXYZZY_BUILD_D3D12_AABB_TEST)
    find_program(FYZZXYZZY_DXC NAMES dxc REQUIRED)
    set(FYZZXYZZY_D3D12_AABB_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_translate_aabbs.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_AABB_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_D3D12_AABB_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_aabb_shader
      DEPENDS "${FYZZXYZZY_D3D12_AABB_DXIL}")
    add_executable(fyzzxyzzy_physx_aabb_encoding_d3d12
      tests/physx_gpu_aabb_encoding.cpp)
    target_compile_definitions(fyzzxyzzy_physx_aabb_encoding_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_aabb_encoding_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_aabb_encoding_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_aabb_encoding_d3d12
      fyzzxyzzy_d3d12_aabb_shader)
    add_test(NAME fyzzxyzzy_physx_aabb_encoding_d3d12
      COMMAND fyzzxyzzy_physx_aabb_encoding_d3d12
        "${FYZZXYZZY_D3D12_AABB_DXIL}")

    add_executable(fyzzxyzzy_physx_broadphase_completion_d3d12
      tests/physx_gpu_broadphase_completion.cpp)
    target_compile_definitions(fyzzxyzzy_physx_broadphase_completion_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_broadphase_completion_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_broadphase_completion_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_test(NAME fyzzxyzzy_physx_broadphase_completion_d3d12
      COMMAND fyzzxyzzy_physx_broadphase_completion_d3d12)

    set(FYZZXYZZY_D3D12_REMOVED_ARENA_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_mark_removed_pairs_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_REMOVED_ARENA_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_D3D12_REMOVED_ARENA_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_mark_removed_pairs_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_mark_removed_pairs_arena.hlsl"
      VERBATIM
    )
    set(FYZZXYZZY_D3D12_REMOVED_RESOLVED_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_mark_removed_pairs_resolved.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_REMOVED_RESOLVED_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -D FYZZXYZZY_RESOLVED_SUBSET=1
        -Fo "${FYZZXYZZY_D3D12_REMOVED_RESOLVED_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_mark_removed_pairs_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_mark_removed_pairs_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_removed_resolved_shader
      DEPENDS "${FYZZXYZZY_D3D12_REMOVED_RESOLVED_DXIL}")
    add_custom_target(fyzzxyzzy_d3d12_removed_arena_shader
      DEPENDS "${FYZZXYZZY_D3D12_REMOVED_ARENA_DXIL}")
    add_executable(fyzzxyzzy_physx_mark_removed_arena_d3d12
      tests/physx_gpu_mark_removed_arena.cpp)
    target_compile_definitions(fyzzxyzzy_physx_mark_removed_arena_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_mark_removed_arena_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_mark_removed_arena_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_mark_removed_arena_d3d12
      fyzzxyzzy_d3d12_removed_arena_shader
      fyzzxyzzy_d3d12_removed_resolved_shader)
    add_test(NAME fyzzxyzzy_physx_mark_removed_arena_d3d12
      COMMAND fyzzxyzzy_physx_mark_removed_arena_d3d12
        "${FYZZXYZZY_D3D12_REMOVED_ARENA_DXIL}"
        "${FYZZXYZZY_D3D12_REMOVED_RESOLVED_DXIL}")

    set(FYZZXYZZY_D3D12_REMOVED_PROJECTIONS_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_removed_projections_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_REMOVED_PROJECTIONS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_D3D12_REMOVED_PROJECTIONS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_removed_projections_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_removed_projections_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_removed_projections_shader
      DEPENDS "${FYZZXYZZY_D3D12_REMOVED_PROJECTIONS_DXIL}")
    add_executable(fyzzxyzzy_physx_removed_projections_d3d12
      tests/physx_gpu_removed_projections.cpp)
    target_compile_definitions(fyzzxyzzy_physx_removed_projections_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_removed_projections_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_removed_projections_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_removed_projections_d3d12
      fyzzxyzzy_d3d12_removed_arena_shader
      fyzzxyzzy_d3d12_removed_projections_shader)
    add_test(NAME fyzzxyzzy_physx_removed_projections_d3d12
      COMMAND fyzzxyzzy_physx_removed_projections_d3d12
        "${FYZZXYZZY_D3D12_REMOVED_ARENA_DXIL}"
        "${FYZZXYZZY_D3D12_REMOVED_PROJECTIONS_DXIL}")
    set(FYZZXYZZY_D3D12_CREATED_PAIRS_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_created_pairs_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_CREATED_PAIRS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_D3D12_CREATED_PAIRS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_created_pairs_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_created_pairs_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_created_pairs_shader
      DEPENDS "${FYZZXYZZY_D3D12_CREATED_PAIRS_DXIL}")
    add_executable(fyzzxyzzy_physx_created_pairs_d3d12
      tests/physx_gpu_created_pairs.cpp)
    target_compile_definitions(fyzzxyzzy_physx_created_pairs_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_created_pairs_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_created_pairs_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_created_pairs_d3d12
      fyzzxyzzy_d3d12_created_pairs_shader)
    add_test(NAME fyzzxyzzy_physx_created_pairs_d3d12
      COMMAND fyzzxyzzy_physx_created_pairs_d3d12
        "${FYZZXYZZY_D3D12_CREATED_PAIRS_DXIL}")

    # PhysX radixSortImpl.cu multi-block count stage (32x3x1024).
    set(FYZZXYZZY_D3D12_RADIX_COUNT_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_radix_count_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_RADIX_COUNT_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_D3D12_RADIX_COUNT_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_count_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_count_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_radix_count_shader
      DEPENDS "${FYZZXYZZY_D3D12_RADIX_COUNT_DXIL}")
    set(FYZZXYZZY_D3D12_RADIX_COUNT_WAVE_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_radix_count_wave_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_RADIX_COUNT_WAVE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_6 -E main
        -Fo "${FYZZXYZZY_D3D12_RADIX_COUNT_WAVE_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_count_wave_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_count_wave_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_radix_count_wave_shader
      DEPENDS "${FYZZXYZZY_D3D12_RADIX_COUNT_WAVE_DXIL}")
    add_executable(fyzzxyzzy_physx_radix_count_d3d12
      tests/physx_gpu_radix_count.cpp)
    target_compile_definitions(fyzzxyzzy_physx_radix_count_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_radix_count_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_radix_count_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_radix_count_d3d12
      fyzzxyzzy_d3d12_radix_count_shader fyzzxyzzy_d3d12_radix_count_wave_shader)
    add_test(NAME fyzzxyzzy_physx_radix_count_d3d12
      COMMAND fyzzxyzzy_physx_radix_count_d3d12
        "${FYZZXYZZY_D3D12_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_COUNT_WAVE_DXIL}")

    # PhysX radixSortImpl.cu::radixSortMultiCalculateRanksLaunchWithCount.
    set(FYZZXYZZY_D3D12_RADIX_SCATTER_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_radix_scatter_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_RADIX_SCATTER_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_D3D12_RADIX_SCATTER_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_scatter_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_scatter_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_radix_scatter_shader
      DEPENDS "${FYZZXYZZY_D3D12_RADIX_SCATTER_DXIL}")

    set(FYZZXYZZY_D3D12_RADIX_SCATTER_WAVE_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_radix_scatter_wave_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_RADIX_SCATTER_WAVE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_6 -E main
        -Fo "${FYZZXYZZY_D3D12_RADIX_SCATTER_WAVE_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_scatter_wave_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_radix_scatter_wave_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_radix_scatter_wave_shader
      DEPENDS "${FYZZXYZZY_D3D12_RADIX_SCATTER_WAVE_DXIL}")
    add_executable(fyzzxyzzy_physx_radix_scatter_d3d12
      tests/physx_gpu_radix_scatter.cpp)
    target_compile_definitions(fyzzxyzzy_physx_radix_scatter_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_radix_scatter_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_radix_scatter_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_radix_scatter_d3d12
      fyzzxyzzy_d3d12_radix_count_shader fyzzxyzzy_d3d12_radix_count_wave_shader
      fyzzxyzzy_d3d12_radix_scatter_shader
      fyzzxyzzy_d3d12_radix_scatter_wave_shader)
    add_test(NAME fyzzxyzzy_physx_radix_scatter_d3d12
      COMMAND fyzzxyzzy_physx_radix_scatter_d3d12
        "${FYZZXYZZY_D3D12_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_COUNT_WAVE_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_SCATTER_WAVE_DXIL}")

    # broadphase.cu::initializeSapBox1DLaunch (ceil(2 * handles / 256)).
    set(FYZZXYZZY_D3D12_SAP_BOX_INDICES_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_box_indices_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_SAP_BOX_INDICES_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_D3D12_SAP_BOX_INDICES_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_box_indices_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_box_indices_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_sap_box_indices_shader
      DEPENDS "${FYZZXYZZY_D3D12_SAP_BOX_INDICES_DXIL}")
    add_executable(fyzzxyzzy_physx_sap_box_indices_d3d12
      tests/physx_gpu_sap_box_indices.cpp)
    target_compile_definitions(fyzzxyzzy_physx_sap_box_indices_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_sap_box_indices_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_sap_box_indices_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_sap_box_indices_d3d12
      fyzzxyzzy_d3d12_created_pairs_shader
      fyzzxyzzy_d3d12_radix_count_shader
      fyzzxyzzy_d3d12_radix_scatter_shader
      fyzzxyzzy_d3d12_sap_box_indices_shader
        fyzzxyzzy_d3d12_radix_count_wave_shader
      fyzzxyzzy_d3d12_radix_scatter_wave_shader
)
    add_test(NAME fyzzxyzzy_physx_sap_box_indices_d3d12
      COMMAND fyzzxyzzy_physx_sap_box_indices_d3d12
        "${FYZZXYZZY_D3D12_CREATED_PAIRS_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_BOX_INDICES_DXIL}"
          "${FYZZXYZZY_D3D12_RADIX_COUNT_WAVE_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_SCATTER_WAVE_DXIL}"
)

    # PhysX WR: initializeRadixRanks → eight radix passes → updateHandles.
    foreach(_sap_kernel IN ITEMS initialize_ranks update_handles)
      set(FYZZXYZZY_D3D12_SAP_${_sap_kernel}_DXIL
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_${_sap_kernel}_arena.dxil")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_D3D12_SAP_${_sap_kernel}_DXIL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_D3D12_SAP_${_sap_kernel}_DXIL}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_${_sap_kernel}_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_${_sap_kernel}_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_d3d12_sap_${_sap_kernel}_shader
        DEPENDS "${FYZZXYZZY_D3D12_SAP_${_sap_kernel}_DXIL}")
    endforeach()
    add_executable(fyzzxyzzy_physx_sap_ranked_sort_d3d12
      tests/physx_gpu_sap_ranked_sort.cpp)
    target_compile_definitions(fyzzxyzzy_physx_sap_ranked_sort_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_sap_ranked_sort_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_sap_ranked_sort_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_sap_ranked_sort_d3d12
      fyzzxyzzy_d3d12_radix_count_shader
      fyzzxyzzy_d3d12_radix_scatter_shader
      fyzzxyzzy_d3d12_sap_initialize_ranks_shader
      fyzzxyzzy_d3d12_sap_update_handles_shader
        fyzzxyzzy_d3d12_radix_count_wave_shader
      fyzzxyzzy_d3d12_radix_scatter_wave_shader
)
    add_test(NAME fyzzxyzzy_physx_sap_ranked_sort_d3d12
      COMMAND fyzzxyzzy_physx_sap_ranked_sort_d3d12
        "${FYZZXYZZY_D3D12_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_initialize_ranks_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_update_handles_DXIL}"
          "${FYZZXYZZY_D3D12_RADIX_COUNT_WAVE_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_SCATTER_WAVE_DXIL}"
)

    # Original BP_COMPUTE_ENDPT_HISTOGRAM / BP_OUTPUT_ENDPT_HISTOGRAM.
    foreach(_hist_pass IN ITEMS first second)
      set(FYZZXYZZY_D3D12_SAP_ENDPOINT_${_hist_pass}_DXIL
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_endpoint_histogram_${_hist_pass}_arena.dxil")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_D3D12_SAP_ENDPOINT_${_hist_pass}_DXIL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_D3D12_SAP_ENDPOINT_${_hist_pass}_DXIL}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_endpoint_histogram_${_hist_pass}_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_endpoint_histogram_${_hist_pass}_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_d3d12_sap_endpoint_${_hist_pass}_shader
        DEPENDS "${FYZZXYZZY_D3D12_SAP_ENDPOINT_${_hist_pass}_DXIL}")
    endforeach()

    # Source packed 10-bit first-pass wave shader, SM6.6 native path.
    set(FYZZXYZZY_D3D12_SAP_ENDPOINT_WAVE_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_endpoint_histogram_wave.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_SAP_ENDPOINT_WAVE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_6 -E main
        -Fo "${FYZZXYZZY_D3D12_SAP_ENDPOINT_WAVE_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_endpoint_histogram_wave_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_endpoint_histogram_wave_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_sap_endpoint_wave_shader
      DEPENDS "${FYZZXYZZY_D3D12_SAP_ENDPOINT_WAVE_DXIL}")
    add_executable(fyzzxyzzy_physx_sap_endpoint_histogram_d3d12
      tests/physx_gpu_sap_endpoint_histogram.cpp)
    target_compile_definitions(fyzzxyzzy_physx_sap_endpoint_histogram_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_sap_endpoint_histogram_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_sap_endpoint_histogram_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_sap_endpoint_histogram_d3d12
      fyzzxyzzy_d3d12_sap_endpoint_first_shader
      fyzzxyzzy_d3d12_sap_endpoint_second_shader
      fyzzxyzzy_d3d12_sap_endpoint_wave_shader)
    add_test(NAME fyzzxyzzy_physx_sap_endpoint_histogram_d3d12
      COMMAND fyzzxyzzy_physx_sap_endpoint_histogram_d3d12
        "${FYZZXYZZY_D3D12_SAP_ENDPOINT_first_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_ENDPOINT_second_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_ENDPOINT_WAVE_DXIL}")

    # PhysX BP_COMPUTE_INCREMENTAL_CMP_COUNTS1 / COUNTS2 (32x256).
    foreach(_comparison_pass IN ITEMS first second)
      set(FYZZXYZZY_D3D12_SAP_COMPARISONS_${_comparison_pass}_DXIL
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_incremental_comparisons_${_comparison_pass}_arena.dxil")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_D3D12_SAP_COMPARISONS_${_comparison_pass}_DXIL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_D3D12_SAP_COMPARISONS_${_comparison_pass}_DXIL}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_comparisons_${_comparison_pass}_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_comparisons_${_comparison_pass}_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_d3d12_sap_comparisons_${_comparison_pass}_shader
        DEPENDS "${FYZZXYZZY_D3D12_SAP_COMPARISONS_${_comparison_pass}_DXIL}")
    endforeach()

    # Three-axis native-wave comparison histogram first-pass fast path.
    set(FYZZXYZZY_D3D12_SAP_COMPARISONS_WAVE_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_incremental_comparisons_wave.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_SAP_COMPARISONS_WAVE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_6 -E main
        -Fo "${FYZZXYZZY_D3D12_SAP_COMPARISONS_WAVE_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_comparisons_wave_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_comparisons_wave_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_sap_comparisons_wave_shader
      DEPENDS "${FYZZXYZZY_D3D12_SAP_COMPARISONS_WAVE_DXIL}")
    add_executable(fyzzxyzzy_physx_sap_incremental_comparisons_d3d12
      tests/physx_gpu_sap_incremental_comparisons.cpp)
    target_compile_definitions(fyzzxyzzy_physx_sap_incremental_comparisons_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_sap_incremental_comparisons_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_sap_incremental_comparisons_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_sap_incremental_comparisons_d3d12
      fyzzxyzzy_d3d12_sap_endpoint_first_shader
      fyzzxyzzy_d3d12_sap_endpoint_second_shader
      fyzzxyzzy_d3d12_sap_comparisons_first_shader
      fyzzxyzzy_d3d12_sap_comparisons_second_shader
        fyzzxyzzy_d3d12_sap_endpoint_wave_shader
        fyzzxyzzy_d3d12_sap_comparisons_wave_shader)
    add_test(NAME fyzzxyzzy_physx_sap_incremental_comparisons_d3d12
      COMMAND fyzzxyzzy_physx_sap_incremental_comparisons_d3d12
        "${FYZZXYZZY_D3D12_SAP_ENDPOINT_first_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_ENDPOINT_second_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_COMPARISONS_first_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_COMPARISONS_second_DXIL}"
          "${FYZZXYZZY_D3D12_SAP_ENDPOINT_WAVE_DXIL}"
          "${FYZZXYZZY_D3D12_SAP_COMPARISONS_WAVE_DXIL}")

    # PhysX BP_INCREMENTAL_SAP: 256 x 3 x 256 source dispatch.
    set(FYZZXYZZY_D3D12_SAP_INCREMENTAL_PAIRS_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_incremental_pairs_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_SAP_INCREMENTAL_PAIRS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_D3D12_SAP_INCREMENTAL_PAIRS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_pairs_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_pairs_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_sap_incremental_pairs_shader
      DEPENDS "${FYZZXYZZY_D3D12_SAP_INCREMENTAL_PAIRS_DXIL}")
        # Native packed found/lost wave scan, original 256x3x256 dispatch.
    set(FYZZXYZZY_D3D12_SAP_INCREMENTAL_PAIRS_WAVE_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_incremental_pairs_wave_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_SAP_INCREMENTAL_PAIRS_WAVE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}"  -T cs_6_6 -E main
        -Fo "${FYZZXYZZY_D3D12_SAP_INCREMENTAL_PAIRS_WAVE_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_pairs_wave_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_incremental_pairs_wave_arena.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_d3d12_sap_incremental_pairs_wave_shader DEPENDS "${FYZZXYZZY_D3D12_SAP_INCREMENTAL_PAIRS_WAVE_DXIL}")
    add_executable(fyzzxyzzy_physx_sap_incremental_pairs_d3d12
      tests/physx_gpu_sap_incremental_pairs.cpp)
    target_compile_definitions(fyzzxyzzy_physx_sap_incremental_pairs_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_sap_incremental_pairs_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_sap_incremental_pairs_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_sap_incremental_pairs_d3d12
      fyzzxyzzy_d3d12_sap_endpoint_first_shader
      fyzzxyzzy_d3d12_sap_endpoint_second_shader
      fyzzxyzzy_d3d12_sap_comparisons_first_shader
      fyzzxyzzy_d3d12_sap_comparisons_second_shader
      fyzzxyzzy_d3d12_sap_incremental_pairs_shader
        fyzzxyzzy_d3d12_sap_endpoint_wave_shader
        fyzzxyzzy_d3d12_sap_comparisons_wave_shader
        fyzzxyzzy_d3d12_sap_incremental_pairs_wave_shader)
    add_test(NAME fyzzxyzzy_physx_sap_incremental_pairs_d3d12
      COMMAND fyzzxyzzy_physx_sap_incremental_pairs_d3d12
        "${FYZZXYZZY_D3D12_SAP_ENDPOINT_first_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_ENDPOINT_second_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_COMPARISONS_first_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_COMPARISONS_second_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_INCREMENTAL_PAIRS_DXIL}"
          "${FYZZXYZZY_D3D12_SAP_ENDPOINT_WAVE_DXIL}"
          "${FYZZXYZZY_D3D12_SAP_COMPARISONS_WAVE_DXIL}"
          "${FYZZXYZZY_D3D12_SAP_INCREMENTAL_PAIRS_WAVE_DXIL}")

    # PhysX clearNewFlagLaunch: 32 x 256, after new-pair generation.
    set(FYZZXYZZY_D3D12_SAP_CLEAR_NEW_FLAGS_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_clear_new_flags_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_SAP_CLEAR_NEW_FLAGS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_D3D12_SAP_CLEAR_NEW_FLAGS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_clear_new_flags_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_clear_new_flags_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_sap_clear_new_flags_shader
      DEPENDS "${FYZZXYZZY_D3D12_SAP_CLEAR_NEW_FLAGS_DXIL}")
    add_executable(fyzzxyzzy_physx_sap_clear_new_flags_d3d12
      tests/physx_gpu_sap_clear_new_flags.cpp)
    target_compile_definitions(fyzzxyzzy_physx_sap_clear_new_flags_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_sap_clear_new_flags_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_sap_clear_new_flags_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_sap_clear_new_flags_d3d12
      fyzzxyzzy_d3d12_created_pairs_shader
      fyzzxyzzy_d3d12_radix_count_shader
      fyzzxyzzy_d3d12_radix_scatter_shader
      fyzzxyzzy_d3d12_sap_box_indices_shader
      fyzzxyzzy_d3d12_sap_clear_new_flags_shader
        fyzzxyzzy_d3d12_radix_count_wave_shader
      fyzzxyzzy_d3d12_radix_scatter_wave_shader
)
    add_test(NAME fyzzxyzzy_physx_sap_clear_new_flags_d3d12
      COMMAND fyzzxyzzy_physx_sap_clear_new_flags_d3d12
        "${FYZZXYZZY_D3D12_CREATED_PAIRS_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_BOX_INDICES_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_CLEAR_NEW_FLAGS_DXIL}"
          "${FYZZXYZZY_D3D12_RADIX_COUNT_WAVE_DXIL}"
        "${FYZZXYZZY_D3D12_RADIX_SCATTER_WAVE_DXIL}"
)

    # PhysX broadphase.cu::copyReports: 64 x 256 publication stage.
    set(FYZZXYZZY_D3D12_BP_COPY_REPORTS_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_broadphase_copy_reports_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_BP_COPY_REPORTS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_D3D12_BP_COPY_REPORTS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_copy_reports_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_copy_reports_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_bp_copy_reports_shader
      DEPENDS "${FYZZXYZZY_D3D12_BP_COPY_REPORTS_DXIL}")
    add_executable(fyzzxyzzy_physx_bp_copy_reports_d3d12
      tests/physx_gpu_broadphase_copy_reports.cpp)
    target_compile_definitions(fyzzxyzzy_physx_bp_copy_reports_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_bp_copy_reports_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_bp_copy_reports_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_bp_copy_reports_d3d12
      fyzzxyzzy_d3d12_bp_copy_reports_shader)
    add_test(NAME fyzzxyzzy_physx_bp_copy_reports_d3d12
      COMMAND fyzzxyzzy_physx_bp_copy_reports_d3d12
        "${FYZZXYZZY_D3D12_BP_COPY_REPORTS_DXIL}")

    # Pinned PhysX accumulateReportsStage_1 / _2: 32 x 4 x 256 each.
    foreach(_report_pass IN ITEMS first second)
      set(FYZZXYZZY_D3D12_BP_REPORT_CLASS_${_report_pass}_DXIL
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_broadphase_report_classification_${_report_pass}_arena.dxil")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_${_report_pass}_DXIL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_${_report_pass}_DXIL}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_report_classification_${_report_pass}_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_report_classification_${_report_pass}_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_d3d12_bp_report_class_${_report_pass}_shader
        DEPENDS "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_${_report_pass}_DXIL}")
    endforeach()
    # Same wave implementation as DXIL, selected only on SM6.6 WaveOps.
    foreach(_report_pass IN ITEMS first second)
      set(FYZZXYZZY_D3D12_BP_REPORT_CLASS_${_report_pass}_WAVE_DXIL
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_broadphase_report_classification_${_report_pass}_wave.dxil")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_${_report_pass}_WAVE_DXIL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T cs_6_6 -E main
          -Fo "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_${_report_pass}_WAVE_DXIL}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_report_classification_${_report_pass}_wave_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_broadphase_report_classification_${_report_pass}_wave_arena.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_d3d12_bp_report_class_${_report_pass}_wave_shader
        DEPENDS "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_${_report_pass}_WAVE_DXIL}")
    endforeach()
    add_executable(fyzzxyzzy_physx_bp_report_classification_d3d12
      tests/physx_gpu_broadphase_report_classification.cpp)
    target_compile_definitions(fyzzxyzzy_physx_bp_report_classification_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_bp_report_classification_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_bp_report_classification_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_bp_report_classification_d3d12
      fyzzxyzzy_d3d12_bp_report_class_first_shader
      fyzzxyzzy_d3d12_bp_report_class_second_shader
      fyzzxyzzy_d3d12_bp_report_class_first_wave_shader
      fyzzxyzzy_d3d12_bp_report_class_second_wave_shader
      fyzzxyzzy_d3d12_bp_copy_reports_shader)
    add_test(NAME fyzzxyzzy_physx_bp_report_classification_d3d12
      COMMAND fyzzxyzzy_physx_bp_report_classification_d3d12
        "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_first_DXIL}"
        "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_second_DXIL}"
        "${FYZZXYZZY_D3D12_BP_COPY_REPORTS_DXIL}"
        "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_first_WAVE_DXIL}"
        "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_second_WAVE_DXIL}")

    # PhysX BP_CREATE_REGIONS: original 32 x 256 region-range producer.
    set(FYZZXYZZY_D3D12_SAP_REGION_PRODUCER_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_region_producer_arena.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_D3D12_SAP_REGION_PRODUCER_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_D3D12_SAP_REGION_PRODUCER_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_producer_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_producer_arena.hlsl"
      VERBATIM
    )
    add_custom_target(fyzzxyzzy_d3d12_sap_region_producer_shader
      DEPENDS "${FYZZXYZZY_D3D12_SAP_REGION_PRODUCER_DXIL}")
    add_executable(fyzzxyzzy_physx_sap_region_producer_d3d12
      tests/physx_gpu_sap_region_producer.cpp)
    target_compile_definitions(fyzzxyzzy_physx_sap_region_producer_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_sap_region_producer_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_sap_region_producer_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_sap_region_producer_d3d12
      fyzzxyzzy_d3d12_sap_region_producer_shader)
    add_test(NAME fyzzxyzzy_physx_sap_region_producer_d3d12
      COMMAND fyzzxyzzy_physx_sap_region_producer_d3d12
        "${FYZZXYZZY_D3D12_SAP_REGION_PRODUCER_DXIL}")

    # Four distinct PhysX region histogram kernels, in source order.
    foreach(_region_kind IN ITEMS start region)
      foreach(_region_pass IN ITEMS first second)
        if(_region_kind STREQUAL "start")
          set(_region_flag 1)
        else()
          set(_region_flag 0)
        endif()
        set(FYZZXYZZY_D3D12_SAP_REGION_${_region_kind}_${_region_pass}_DXIL
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_region_${_region_kind}_histogram_${_region_pass}.dxil")
        add_custom_command(
          OUTPUT "${FYZZXYZZY_D3D12_SAP_REGION_${_region_kind}_${_region_pass}_DXIL}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main -DFYZZXYZZY_REGION_START=${_region_flag}
            -Fo "${FYZZXYZZY_D3D12_SAP_REGION_${_region_kind}_${_region_pass}_DXIL}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_histogram_${_region_pass}_arena.hlsl"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_histogram_${_region_pass}_arena.hlsl"
          VERBATIM
        )
        add_custom_target(fyzzxyzzy_d3d12_sap_region_${_region_kind}_${_region_pass}_shader
          DEPENDS "${FYZZXYZZY_D3D12_SAP_REGION_${_region_kind}_${_region_pass}_DXIL}")
      endforeach()
    endforeach()
    # Fast source-style warp-prefix lowering; 32x256 four-pass schedule.
    foreach(_region_kind IN ITEMS start region)
      if(_region_kind STREQUAL "start")
        set(_region_flag 1)
      else()
        set(_region_flag 0)
      endif()
      set(FYZZXYZZY_D3D12_SAP_REGION_${_region_kind}_WAVE_DXIL
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_region_${_region_kind}_histogram_wave.dxil")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_D3D12_SAP_REGION_${_region_kind}_WAVE_DXIL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T cs_6_6 -E main -DFYZZXYZZY_REGION_START=${_region_flag}
          -Fo "${FYZZXYZZY_D3D12_SAP_REGION_${_region_kind}_WAVE_DXIL}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_histogram_wave_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_region_histogram_wave_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_d3d12_sap_region_${_region_kind}_wave_shader
        DEPENDS "${FYZZXYZZY_D3D12_SAP_REGION_${_region_kind}_WAVE_DXIL}")
    endforeach()
    add_executable(fyzzxyzzy_physx_sap_region_histograms_d3d12
      tests/physx_gpu_sap_region_producer.cpp)
    target_compile_definitions(fyzzxyzzy_physx_sap_region_histograms_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_sap_region_histograms_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_sap_region_histograms_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_sap_region_histograms_d3d12
      fyzzxyzzy_d3d12_sap_region_producer_shader
      fyzzxyzzy_d3d12_sap_region_start_first_shader
      fyzzxyzzy_d3d12_sap_region_start_second_shader
      fyzzxyzzy_d3d12_sap_region_region_first_shader
      fyzzxyzzy_d3d12_sap_region_region_second_shader
      fyzzxyzzy_d3d12_sap_region_start_wave_shader
      fyzzxyzzy_d3d12_sap_region_region_wave_shader)
    add_test(NAME fyzzxyzzy_physx_sap_region_histograms_d3d12
      COMMAND fyzzxyzzy_physx_sap_region_histograms_d3d12
        "${FYZZXYZZY_D3D12_SAP_REGION_PRODUCER_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_REGION_start_first_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_REGION_start_second_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_REGION_region_first_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_REGION_region_second_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_REGION_start_WAVE_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_REGION_region_WAVE_DXIL}")

    # Three PhysX start/active-region kernels: 32x256, 64x512, 64x256.
    foreach(_active_pass IN ITEMS writeout compute_wave compute_portable output)
        if(_active_pass STREQUAL "compute_wave")
          set(_active_source compute)
          set(_active_flag 1)
          set(_active_profile cs_6_6)
        elseif(_active_pass STREQUAL "compute_portable")
          set(_active_source compute)
          set(_active_flag 0)
          set(_active_profile cs_6_0)
        else()
          set(_active_source ${_active_pass})
          set(_active_flag 0)
          set(_active_profile cs_6_0)
        endif()
      set(FYZZXYZZY_D3D12_SAP_ACTIVE_${_active_pass}_DXIL
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_active_region_${_active_pass}.dxil")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_D3D12_SAP_ACTIVE_${_active_pass}_DXIL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T ${_active_profile} -E main -DFYZZXYZZY_ACTIVE_REGION_WAVE=${_active_flag}
          -Fo "${FYZZXYZZY_D3D12_SAP_ACTIVE_${_active_pass}_DXIL}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_active_region_${_active_source}_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_active_region_${_active_source}_arena.hlsl"
        VERBATIM
      )
      add_custom_target(fyzzxyzzy_d3d12_sap_active_${_active_pass}_shader
        DEPENDS "${FYZZXYZZY_D3D12_SAP_ACTIVE_${_active_pass}_DXIL}")
    endforeach()
    add_executable(fyzzxyzzy_physx_sap_active_region_lists_d3d12
      tests/physx_gpu_sap_active_region_lists.cpp)
    target_compile_definitions(fyzzxyzzy_physx_sap_active_region_lists_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_sap_active_region_lists_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_sap_active_region_lists_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_sap_active_region_lists_d3d12
      fyzzxyzzy_d3d12_sap_active_writeout_shader
      fyzzxyzzy_d3d12_sap_active_compute_wave_shader
        fyzzxyzzy_d3d12_sap_active_compute_portable_shader
      fyzzxyzzy_d3d12_sap_active_output_shader)
    add_test(NAME fyzzxyzzy_physx_sap_active_region_lists_d3d12
      COMMAND fyzzxyzzy_physx_sap_active_region_lists_d3d12
        "${FYZZXYZZY_D3D12_SAP_ACTIVE_writeout_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_ACTIVE_compute_wave_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_ACTIVE_compute_portable_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_ACTIVE_output_DXIL}")
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
