# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)
project(FyzzXYZZY VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(FetchContent)
find_package(Threads REQUIRED)

# PhysX is a transitional CPU oracle and source-port test dependency, not a
# prerequisite for the native SDK core or standalone GPU compute runtimes.
# OFF must never fetch or compile the reference PhysX repository.
option(FYZZXYZZY_ENABLE_PHYSX_SOURCE
  "Build the transitional PhysX CPU reference and source-port conformance fixtures" ON)
# Run the 48-frame coupled friction/rotation source oracle by default.
# Windows and Linux source-enabled CI explicitly require this gate.
# PhysX-independent builds never compile or link the reference fixture.
option(FYZZXYZZY_STRICT_RIGID_FRICTION_CONFORMANCE
  "Require 48-frame native/PhysX CPU coupled friction trajectory parity" ON)

if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
# CPU PhysX-derived execution remains available; upstream CUDA is disabled.
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

# Transitional source bootstrap only: follows the current PhysX
# development branch when a local checkout is not supplied. The engine must
# ultimately own its PhysX-derived implementation rather than impose an
# external version-locked provider/factory dependency.
FetchContent_Declare(fyzzxyzzy_physx
  GIT_REPOSITORY https://github.com/thomthehound/PhysX.git
  GIT_TAG main
  SOURCE_SUBDIR physx
  GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(fyzzxyzzy_physx)

if(NOT TARGET physx_lib)
  message(FATAL_ERROR "Expected PhysX 5.11 CMake interface target 'physx_lib'")
endif()
endif() # FYZZXYZZY_ENABLE_PHYSX_SOURCE

# Require C++23 for every FyzzXYZZY-owned native target, including tests and
# the Godot adapter. Upstream PhysX is configured above with its own source
# contract; do not unilaterally change that third-party build requirement.
# The required property prevents silently downgrading to an older dialect.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Independently enforce the SDK's C++23 standard-library contract at
# configure time, including production builds with BUILD_TESTING=OFF.
# Compile AND link the same probe used by the optional runtime CTest,
# so the two validation paths cannot drift apart.
try_compile(FYZZXYZZY_CXX23_LIBRARY_SUPPORTED
  "${CMAKE_CURRENT_BINARY_DIR}/fyzzxyzzy-cxx23-probe"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp23_contract.cpp"
  CMAKE_FLAGS
    "-DCMAKE_CXX_STANDARD:STRING=23"
    "-DCMAKE_CXX_STANDARD_REQUIRED:BOOL=ON"
    "-DCMAKE_CXX_EXTENSIONS:BOOL=OFF"
  OUTPUT_VARIABLE FYZZXYZZY_CXX23_PROBE_LOG)
if(NOT FYZZXYZZY_CXX23_LIBRARY_SUPPORTED)
  message(FATAL_ERROR
    "FyzzXYZZY requires a compiler AND standard library supporting C++23 "
    "(including std::expected and if consteval). "
    "The C++23 configuration probe failed:\n"
    "${FYZZXYZZY_CXX23_PROBE_LOG}")
endif()

# PhysX static consumers must not import symbols from PhysX DLLs on Windows.
# The upstream static build defines this internally but does not export it
# through the physx_lib interface target.
if(FYZZXYZZY_ENABLE_PHYSX_SOURCE AND PX_GENERATE_STATIC_LIBRARIES)
  target_compile_definitions(physx_lib INTERFACE PX_PHYSX_STATIC_LIB)
endif()

if(BUILD_TESTING)
  add_executable(fyzzxyzzy_cpp23_contract tests/cpp23_contract.cpp)
  add_test(NAME fyzzxyzzy_cpp23_contract COMMAND fyzzxyzzy_cpp23_contract)
endif()

if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
add_library(fyzzxyzzy_reference_physx_cpu STATIC src/reference/physx_cpu_world.cpp)
add_library(FyzzXYZZY::ReferencePhysXCpu ALIAS fyzzxyzzy_reference_physx_cpu)
target_compile_features(fyzzxyzzy_reference_physx_cpu PUBLIC cxx_std_23)
target_include_directories(fyzzxyzzy_reference_physx_cpu PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(fyzzxyzzy_reference_physx_cpu PUBLIC physx_lib)
set_target_properties(fyzzxyzzy_reference_physx_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(fyzzxyzzy_core STATIC
  src/world.cpp
  src/native/scene.cpp
  src/native/GuContactBoxBox.cpp
  src/native/GuPCMContactBoxBox.cpp
  src/native/GuPCMGjkEpa.cpp)
add_library(FyzzXYZZY::Core ALIAS fyzzxyzzy_core)
target_compile_features(fyzzxyzzy_core PUBLIC cxx_std_23)
target_link_libraries(fyzzxyzzy_core PUBLIC Threads::Threads)
target_include_directories(fyzzxyzzy_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_include_directories(fyzzxyzzy_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(fyzzxyzzy_core PRIVATE
  FYZZXYZZY_ENABLE_PHYSX_SOURCE=$<BOOL:${FYZZXYZZY_ENABLE_PHYSX_SOURCE}>)
if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
  target_link_libraries(fyzzxyzzy_core PRIVATE fyzzxyzzy_reference_physx_cpu)
endif()
set_target_properties(fyzzxyzzy_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_options(fyzzxyzzy_core PRIVATE /W4)
else()
  target_compile_options(fyzzxyzzy_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING AND FYZZXYZZY_ENABLE_PHYSX_SOURCE)
  add_executable(fyzzxyzzy_headless_smoke tests/headless_smoke.cpp)
  target_link_libraries(fyzzxyzzy_headless_smoke PRIVATE FyzzXYZZY::Core)
  # Strict source regression may compare public PhysX output with the
  # production native Gu helper using identical input transforms.
  target_include_directories(fyzzxyzzy_headless_smoke PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_definitions(fyzzxyzzy_headless_smoke PRIVATE
    FYZZXYZZY_STRICT_RIGID_FRICTION_CONFORMANCE=$<BOOL:${FYZZXYZZY_STRICT_RIGID_FRICTION_CONFORMANCE}>)
  add_test(NAME fyzzxyzzy_headless_smoke COMMAND fyzzxyzzy_headless_smoke)
endif()

# A reference-free build tests native core admission without loading or
# linking a PhysX CPU runtime. The native engine is not declared complete.
if(BUILD_TESTING AND NOT FYZZXYZZY_ENABLE_PHYSX_SOURCE)
  add_executable(fyzzxyzzy_native_boundary_smoke tests/native_boundary_smoke.cpp)
  target_link_libraries(fyzzxyzzy_native_boundary_smoke PRIVATE FyzzXYZZY::Core)
  add_test(NAME fyzzxyzzy_native_boundary_smoke
    COMMAND fyzzxyzzy_native_boundary_smoke)
endif()

if(BUILD_TESTING)
  add_executable(fyzzxyzzy_native_scene_smoke tests/native_scene_smoke.cpp)
  target_include_directories(fyzzxyzzy_native_scene_smoke PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(fyzzxyzzy_native_scene_smoke PRIVATE FyzzXYZZY::Core)
  target_compile_definitions(fyzzxyzzy_native_scene_smoke PRIVATE
    FYZZXYZZY_ENABLE_PHYSX_SOURCE=$<BOOL:${FYZZXYZZY_ENABLE_PHYSX_SOURCE}>)
  if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
    target_link_libraries(fyzzxyzzy_native_scene_smoke PRIVATE physx_lib)
  endif()
  add_test(NAME fyzzxyzzy_native_scene_smoke
    COMMAND fyzzxyzzy_native_scene_smoke)
  add_executable(fyzzxyzzy_native_sap_benchmark
    tests/native_sap_benchmark.cpp)
  target_include_directories(fyzzxyzzy_native_sap_benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(fyzzxyzzy_native_sap_benchmark PRIVATE FyzzXYZZY::Core)
  add_executable(fyzzxyzzy_native_third_pass_benchmark
    tests/native_third_pass_benchmark.cpp)
  target_include_directories(fyzzxyzzy_native_third_pass_benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(fyzzxyzzy_native_third_pass_benchmark PRIVATE
    FyzzXYZZY::Core)
endif()

# Internal translation of PhysX's CUdeviceptr-style host addresses into
# native backend buffers. This is not a public CUDA ABI or a replacement
# host scheduler, and is testable without a GPU or CUDA SDK.
add_library(fyzzxyzzy_physx_cuda_host_address STATIC
  src/physx_gpu/cuda_device_address_map.cpp)
target_compile_features(fyzzxyzzy_physx_cuda_host_address PUBLIC cxx_std_23)
target_include_directories(fyzzxyzzy_physx_cuda_host_address PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src")
if(BUILD_TESTING)
  add_executable(fyzzxyzzy_physx_cuda_address_test
    tests/physx_gpu_cuda_device_address_map.cpp)
  target_include_directories(fyzzxyzzy_physx_cuda_address_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(fyzzxyzzy_physx_cuda_address_test PRIVATE
    fyzzxyzzy_physx_cuda_host_address)
  add_test(NAME fyzzxyzzy_physx_cuda_address_test
    COMMAND fyzzxyzzy_physx_cuda_address_test)
endif()

if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
# Original PhysX GPU buddy heap source, owned by this independent engine.
# This is a source port with an injected generic virtual root callback and
# abort sink; it is NOT a second allocation algorithm or an external patch.
add_library(fyzzxyzzy_physx_original_gpu_heap STATIC
  src/physx_port/gpucommon/PxgHeapMemoryAllocator.cpp
  src/physx_port/gpucommon/PxgHeapMemoryAllocatorManager.cpp
  src/physx_port/gpucommon/PxgCudaBuffer.cpp
  src/physx_port/gpubroadphase/PxgSapPreBroadPhaseNative.cpp
  src/physx_port/gpubroadphase/PxgSapGpuDMAUpHandlesNative.cpp
  src/physx_port/gpubroadphase/PxgSapHistoryGrowthNative.cpp
  src/physx_port/gpubroadphase/PxgSapPostAxisBuffersNative.cpp
  src/physx_port/gpubroadphase/PxgSapRadixDescriptorsNative.cpp
  src/physx_port/gpubroadphase/PxgSapBroadphaseDescriptorNative.cpp
  src/physx_port/gpubroadphase/PxgSapDmaUpUploadsNative.cpp
  src/physx_port/gpubroadphase/PxgSapDmaUpFrameNative.cpp
  src/physx_port/gpubroadphase/PxgSapUpdateFetchNative.cpp)
target_compile_features(fyzzxyzzy_physx_original_gpu_heap PUBLIC cxx_std_23)
target_include_directories(fyzzxyzzy_physx_original_gpu_heap PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port/gpucommon"
  "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
  "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
  "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
  "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include"
  "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src")
target_link_libraries(fyzzxyzzy_physx_original_gpu_heap PUBLIC physx_lib)
endif() # source-port PhysX heap and SAP host fixtures

# Native compute resources and operations do not depend on Godot or PhysX.
option(FYZZXYZZY_BUILD_VULKAN "Build the standalone Vulkan compute backend" OFF)
if(FYZZXYZZY_BUILD_VULKAN)
  find_package(Vulkan REQUIRED)
  add_library(fyzzxyzzy_compute STATIC src/compute_vulkan.cpp)
  add_library(FyzzXYZZY::Compute ALIAS fyzzxyzzy_compute)
  target_compile_features(fyzzxyzzy_compute PUBLIC cxx_std_23)
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
    target_compile_features(fyzzxyzzy_vulkan_world PUBLIC cxx_std_23)
    target_include_directories(fyzzxyzzy_vulkan_world PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(fyzzxyzzy_vulkan_world PUBLIC
      FyzzXYZZY::Core FyzzXYZZY::Compute
    )
  endif()

  if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
  # Translated PhysX GPU stages. These operations preserve upstream data
  # contracts; the PhysX scene integration is introduced incrementally.
  add_library(fyzzxyzzy_physx_gpu_stages STATIC
    src/physx_gpu/aabb_encoding.cpp
    src/physx_gpu/cuda_aabb_launch_lowering.cpp
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
    src/physx_gpu/source_sap_report_multi_root.cpp
    src/physx_gpu/sap_host_frame_roles.cpp
    src/physx_gpu/sap_region_producer_arena.cpp
    src/physx_gpu/sap_region_histograms_arena.cpp
    src/physx_gpu/sap_active_region_lists_arena.cpp
    src/physx_gpu/sap_overlap_work_arena.cpp
    src/physx_gpu/source_overlap_work_native.cpp
    src/physx_gpu/source_overlap_prefix_native.cpp
    src/physx_gpu/source_newbounds_pairs_native.cpp
    src/physx_gpu/source_newbounds_sequence_native.cpp
    src/physx_gpu/source_clear_new_flags_native.cpp
    src/physx_gpu/source_sapbox_indices_native.cpp
    src/physx_gpu/source_sap_removed_native.cpp
    src/physx_gpu/source_sap_updated_native.cpp
    src/physx_gpu/source_sap_created_native.cpp
    src/physx_gpu/source_sap_radix_native.cpp
    src/physx_gpu/source_sap_incremental_counts_native.cpp
    src/physx_gpu/source_sap_endpoint_histograms_native.cpp
    src/physx_gpu/source_sap_incremental_pairs_native.cpp
    src/physx_gpu/source_sap_update_native.cpp
    src/physx_gpu/source_fill_d32_native.cpp
    src/physx_gpu/source_region_create_native.cpp
    src/physx_gpu/source_region_histograms_native.cpp
    src/physx_gpu/source_active_region_lists_native.cpp
    src/physx_gpu/sap_overlap_prefix_arena.cpp
    src/physx_gpu/sap_newbounds_pairs_arena.cpp
    src/physx_gpu/sap_newbounds_sequence_arena.cpp
    src/physx_gpu/broadphase_completion.cpp
  )
  target_compile_features(fyzzxyzzy_physx_gpu_stages PUBLIC cxx_std_23)
  target_include_directories(fyzzxyzzy_physx_gpu_stages PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include")
  target_link_libraries(fyzzxyzzy_physx_gpu_stages PUBLIC
    FyzzXYZZY::Compute physx_lib fyzzxyzzy_physx_cuda_host_address
    fyzzxyzzy_physx_original_gpu_heap)
  set_target_properties(fyzzxyzzy_physx_gpu_stages
    PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if(MSVC)
    target_compile_options(fyzzxyzzy_compute PRIVATE /W4)
  else()
    target_compile_options(fyzzxyzzy_compute PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(BUILD_TESTING)
    add_executable(fyzzxyzzy_vulkan_memory_smoke tests/vulkan_memory_smoke.cpp)
    target_link_libraries(fyzzxyzzy_vulkan_memory_smoke PRIVATE FyzzXYZZY::Compute)
    add_test(NAME fyzzxyzzy_vulkan_memory_smoke COMMAND fyzzxyzzy_vulkan_memory_smoke)
    # Exercise CUdeviceptr+offset host lowering against a real Vulkan
    # device-local upload/copy and submission-safe buffer retirement.
    add_executable(fyzzxyzzy_physx_cuda_host_transfer
      tests/physx_gpu_cuda_host_transfer.cpp)
    target_include_directories(fyzzxyzzy_physx_cuda_host_transfer PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
    target_link_libraries(fyzzxyzzy_physx_cuda_host_transfer PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_physx_cuda_host_address fyzzxyzzy_physx_original_gpu_heap physx_lib)
    add_test(NAME fyzzxyzzy_physx_cuda_host_transfer
      COMMAND fyzzxyzzy_physx_cuda_host_transfer)


    add_executable(fyzzxyzzy_physx_sap_broadphase_descriptor_native
      tests/physx_gpu_sap_broadphase_descriptor_native.cpp)
    target_include_directories(fyzzxyzzy_physx_sap_broadphase_descriptor_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
    target_link_libraries(fyzzxyzzy_physx_sap_broadphase_descriptor_native PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_physx_cuda_host_address fyzzxyzzy_physx_original_gpu_heap physx_lib)
    add_test(NAME fyzzxyzzy_physx_sap_broadphase_descriptor_native COMMAND fyzzxyzzy_physx_sap_broadphase_descriptor_native)

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
      set(FYZZXYZZY_MULTI_RESOURCE_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/native_multi_resource_bindings.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_MULTI_RESOURCE_SPV}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main -Fo "${FYZZXYZZY_MULTI_RESOURCE_SPV}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_multi_resource_bindings.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_multi_resource_bindings.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_native_multi_resource_spv
        DEPENDS "${FYZZXYZZY_MULTI_RESOURCE_SPV}")
      add_executable(fyzzxyzzy_native_multi_resource
        tests/physx_gpu_native_multi_resource.cpp)
      target_link_libraries(fyzzxyzzy_native_multi_resource PRIVATE FyzzXYZZY::Compute)
      add_dependencies(fyzzxyzzy_native_multi_resource fyzzxyzzy_native_multi_resource_spv)
      add_test(NAME fyzzxyzzy_native_multi_resource
        COMMAND fyzzxyzzy_native_multi_resource "${FYZZXYZZY_MULTI_RESOURCE_SPV}")
      set_tests_properties(fyzzxyzzy_native_multi_resource PROPERTIES SKIP_RETURN_CODE 77)
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
      # Original CUDA _launch ABI: 11 argument pointers, original 256-thread
      # block layout, CUdeviceptr interior offsets, native shader lowering.
      add_executable(fyzzxyzzy_physx_cuda_aabb_launch
        tests/physx_gpu_cuda_aabb_launch.cpp)
      target_include_directories(fyzzxyzzy_physx_cuda_aabb_launch PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include")
      target_link_libraries(fyzzxyzzy_physx_cuda_aabb_launch PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_cuda_host_address)
      add_dependencies(fyzzxyzzy_physx_cuda_aabb_launch fyzzxyzzy_physx_aabb_shader)
      add_test(NAME fyzzxyzzy_physx_cuda_aabb_launch
        COMMAND fyzzxyzzy_physx_cuda_aabb_launch
          "${FYZZXYZZY_PHYSX_AABB_SHADER}")

      # Execute the source PxgBroadPhaseDesc overlap-work producer with
      # independent native heap roots; retain original per-field ownership.
      set(FYZZXYZZY_SOURCE_OVERLAP_NATIVE_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_work_native.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_SPV}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main -Fo "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_SPV}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_native.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_native.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_physx_source_overlap_native_spv
        DEPENDS "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_SPV}")
      add_executable(fyzzxyzzy_physx_source_overlap_native
        tests/physx_gpu_source_overlap_native.cpp)
      target_include_directories(fyzzxyzzy_physx_source_overlap_native PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
      target_link_libraries(fyzzxyzzy_physx_source_overlap_native PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_cuda_host_address)
      add_dependencies(fyzzxyzzy_physx_source_overlap_native
        fyzzxyzzy_physx_source_overlap_native_spv)
      add_test(NAME fyzzxyzzy_physx_source_overlap_native
        COMMAND fyzzxyzzy_physx_source_overlap_native
          "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_SPV}")

      # Original 32x256 overlap histogram compute/output scans, source-owned
      # independent roots and GPU-resident descriptor counters.
      set(FYZZXYZZY_SOURCE_SCAN_SPV_FILES)
      foreach(_phase first second)
        set(_shader "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_${_phase}_native.hlsl")
        set(_spv "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_prefix_${_phase}_native.spv")
        add_custom_command(OUTPUT "${_spv}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
            -T cs_6_0 -E main -Fo "${_spv}" "${_shader}"
          DEPENDS "${_shader}" VERBATIM)
        list(APPEND FYZZXYZZY_SOURCE_SCAN_SPV_FILES "${_spv}")
      endforeach()
      add_custom_target(fyzzxyzzy_physx_source_scan_spv
        DEPENDS ${FYZZXYZZY_SOURCE_SCAN_SPV_FILES})
      add_executable(fyzzxyzzy_physx_source_scan_native
        tests/physx_gpu_source_overlap_prefix_native.cpp)
      target_include_directories(fyzzxyzzy_physx_source_scan_native PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
      target_link_libraries(fyzzxyzzy_physx_source_scan_native PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_cuda_host_address)
      add_dependencies(fyzzxyzzy_physx_source_scan_native
        fyzzxyzzy_physx_source_scan_spv)
      add_test(NAME fyzzxyzzy_physx_source_scan_native
        COMMAND fyzzxyzzy_physx_source_scan_native
          ${FYZZXYZZY_SOURCE_SCAN_SPV_FILES})
      set_tests_properties(fyzzxyzzy_physx_source_scan_native PROPERTIES SKIP_RETURN_CODE 77)

      # BP_GENERATE_FOUNDPAIR_NEWBOUNDS: original 64x256, GPU descriptor
      # counters and native source heap-root views without arena repacking.
      set(FYZZXYZZY_SOURCE_PAIRS_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_newbounds_pairs_wave_native.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_SOURCE_PAIRS_SPV}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main -Fo "${FYZZXYZZY_SOURCE_PAIRS_SPV}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_native.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_native.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_physx_source_pairs_spv
        DEPENDS "${FYZZXYZZY_SOURCE_PAIRS_SPV}")
      add_executable(fyzzxyzzy_physx_source_pairs_native
        tests/physx_gpu_source_newbounds_pairs_native.cpp)
      target_include_directories(fyzzxyzzy_physx_source_pairs_native PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
      target_link_libraries(fyzzxyzzy_physx_source_pairs_native PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_cuda_host_address)
      add_dependencies(fyzzxyzzy_physx_source_pairs_native
        fyzzxyzzy_physx_source_pairs_spv)
      add_test(NAME fyzzxyzzy_physx_source_pairs_native
        COMMAND fyzzxyzzy_physx_source_pairs_native
          "${FYZZXYZZY_SOURCE_PAIRS_SPV}")
      set_tests_properties(fyzzxyzzy_physx_source_pairs_native PROPERTIES SKIP_RETURN_CODE 77)

      # PxgCudaBroadPhaseSap::generateNewPairsKernel - four ordered source
      # dispatches, one GPU batch, no intermediate host data transfers.
      add_executable(fyzzxyzzy_physx_source_sequence_native
        tests/physx_gpu_source_newbounds_sequence_native.cpp)
      target_include_directories(fyzzxyzzy_physx_source_sequence_native PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
      target_link_libraries(fyzzxyzzy_physx_source_sequence_native PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_cuda_host_address)
      add_dependencies(fyzzxyzzy_physx_source_sequence_native
        fyzzxyzzy_physx_source_overlap_native_spv
        fyzzxyzzy_physx_source_scan_spv
        fyzzxyzzy_physx_source_pairs_spv)
      add_test(NAME fyzzxyzzy_physx_source_sequence_native
        COMMAND fyzzxyzzy_physx_source_sequence_native
          "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_SPV}"
          ${FYZZXYZZY_SOURCE_SCAN_SPV_FILES}
          "${FYZZXYZZY_SOURCE_PAIRS_SPV}")
      set_tests_properties(fyzzxyzzy_physx_source_sequence_native PROPERTIES
        SKIP_RETURN_CODE 77)

      # Lower PhysX mStream memsetD32Async into an ordered native GPU
      # dispatch. Original CUdeviceptr slices remain source-owned.
      set(FYZZXYZZY_SOURCE_D32_FILL_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_fill_d32_native.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_SOURCE_D32_FILL_SPV}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main -Fo "${FYZZXYZZY_SOURCE_D32_FILL_SPV}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_fill_d32_native.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_fill_d32_native.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_physx_source_d32_fill_spv
        DEPENDS "${FYZZXYZZY_SOURCE_D32_FILL_SPV}")
      add_executable(fyzzxyzzy_physx_source_d32_fill_native
        tests/physx_gpu_source_fill_d32_native.cpp)
      target_include_directories(fyzzxyzzy_physx_source_d32_fill_native PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_source_d32_fill_native PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_cuda_host_address)
      add_dependencies(fyzzxyzzy_physx_source_d32_fill_native
        fyzzxyzzy_physx_source_d32_fill_spv)
      add_test(NAME fyzzxyzzy_physx_source_d32_fill_native
        COMMAND fyzzxyzzy_physx_source_d32_fill_native
          "${FYZZXYZZY_SOURCE_D32_FILL_SPV}")
      set_tests_properties(fyzzxyzzy_physx_source_d32_fill_native PROPERTIES
        SKIP_RETURN_CODE 77)

      # First three operations of source computeRegionHistogramKernel:
      # two original D32 clears, then BP_CREATE_REGIONS over native source buffers.
      set(FYZZXYZZY_SOURCE_REGION_CREATE_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_create_native.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_SOURCE_REGION_CREATE_SPV}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main -Fo "${FYZZXYZZY_SOURCE_REGION_CREATE_SPV}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_create_native.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_create_native.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_physx_source_region_create_spv
        DEPENDS "${FYZZXYZZY_SOURCE_REGION_CREATE_SPV}")
      add_executable(fyzzxyzzy_physx_source_region_create_native
        tests/physx_gpu_source_region_create_native.cpp)
      target_include_directories(fyzzxyzzy_physx_source_region_create_native PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
      target_link_libraries(fyzzxyzzy_physx_source_region_create_native PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_cuda_host_address)
      add_dependencies(fyzzxyzzy_physx_source_region_create_native
        fyzzxyzzy_physx_source_d32_fill_spv fyzzxyzzy_physx_source_region_create_spv)
      add_test(NAME fyzzxyzzy_physx_source_region_create_native
        COMMAND fyzzxyzzy_physx_source_region_create_native
          "${FYZZXYZZY_SOURCE_D32_FILL_SPV}" "${FYZZXYZZY_SOURCE_REGION_CREATE_SPV}"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_first_native.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_wave_native.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_second_native.spv")
      set_tests_properties(fyzzxyzzy_physx_source_region_create_native PROPERTIES
        SKIP_RETURN_CODE 77)
      add_test(NAME fyzzxyzzy_physx_ten_operation_chain
        COMMAND fyzzxyzzy_physx_source_region_create_native
          "${FYZZXYZZY_SOURCE_D32_FILL_SPV}"
          "${FYZZXYZZY_SOURCE_REGION_CREATE_SPV}"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_first_native.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_wave_native.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_second_native.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_writeout.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_compute_wave.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_compute_portable.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_output.spv")
      set_tests_properties(fyzzxyzzy_physx_ten_operation_chain PROPERTIES
        SKIP_RETURN_CODE 77)
      # Source native 7+3 region stages and four new-bounds kernels share
      # original PxgBroadPhaseDesc, CUDA allocations, and one submission.
      add_dependencies(fyzzxyzzy_physx_source_region_create_native
        fyzzxyzzy_physx_source_overlap_native_spv
        fyzzxyzzy_physx_source_scan_spv
        fyzzxyzzy_physx_source_pairs_spv)
      add_test(NAME fyzzxyzzy_physx_fourteen_operation_chain
        COMMAND fyzzxyzzy_physx_source_region_create_native
          "${FYZZXYZZY_SOURCE_D32_FILL_SPV}"
          "${FYZZXYZZY_SOURCE_REGION_CREATE_SPV}"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_first_native.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_wave_native.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_second_native.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_writeout.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_compute_wave.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_compute_portable.spv"
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_output.spv"
          "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_SPV}"
          ${FYZZXYZZY_SOURCE_SCAN_SPV_FILES}
          "${FYZZXYZZY_SOURCE_PAIRS_SPV}")
      set_tests_properties(fyzzxyzzy_physx_fourteen_operation_chain PROPERTIES
        SKIP_RETURN_CODE 77)

      # Source-ordered four region histogram scans; scalar totals remain
      # within their original GPU-resident PxgBroadPhaseDesc fields.
      foreach(variant first wave second)
        set(FYZZXYZZY_SOURCE_REGION_HIST_${variant}_SPV
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_${variant}_native.spv")
        add_custom_command(
          OUTPUT "${FYZZXYZZY_SOURCE_REGION_HIST_${variant}_SPV}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
            -T cs_6_0 -E main -Fo "${FYZZXYZZY_SOURCE_REGION_HIST_${variant}_SPV}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_${variant}_native.hlsl"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_${variant}_native.hlsl"
          VERBATIM)
      endforeach()
      add_custom_target(fyzzxyzzy_physx_source_region_histograms_spv
        DEPENDS "${FYZZXYZZY_SOURCE_REGION_HIST_first_SPV}"
                "${FYZZXYZZY_SOURCE_REGION_HIST_wave_SPV}"
                "${FYZZXYZZY_SOURCE_REGION_HIST_second_SPV}")
      add_dependencies(fyzzxyzzy_physx_source_region_create_native
        fyzzxyzzy_physx_source_region_histograms_spv
        fyzzxyzzy_physx_source_active_writeout_shader
        fyzzxyzzy_physx_source_active_compute_wave_shader
        fyzzxyzzy_physx_source_active_compute_portable_shader
        fyzzxyzzy_physx_source_active_output_shader)
      add_executable(fyzzxyzzy_physx_source_region_histograms_native
        tests/physx_gpu_source_region_histograms_native.cpp)
      target_include_directories(fyzzxyzzy_physx_source_region_histograms_native PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
      target_link_libraries(fyzzxyzzy_physx_source_region_histograms_native PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_cuda_host_address)
      add_dependencies(fyzzxyzzy_physx_source_region_histograms_native
        fyzzxyzzy_physx_source_region_histograms_spv)
      add_test(NAME fyzzxyzzy_physx_source_region_histograms_native
        COMMAND fyzzxyzzy_physx_source_region_histograms_native
          "${FYZZXYZZY_SOURCE_REGION_HIST_first_SPV}"
          "${FYZZXYZZY_SOURCE_REGION_HIST_wave_SPV}"
          "${FYZZXYZZY_SOURCE_REGION_HIST_second_SPV}")
      set_tests_properties(fyzzxyzzy_physx_source_region_histograms_native PROPERTIES
        SKIP_RETURN_CODE 77)

      # Original PxgBroadPhaseDesc fields and source CUDA virtual pointers,
      # native persistent backing growth and descriptor-only re-materialization.
      # Source pointer -> single-backing arena is a feasibility test,
      # not the production solution for PhysX's multiple heap roots.
      add_executable(fyzzxyzzy_physx_source_descriptor
        tests/physx_gpu_source_descriptor.cpp
        src/physx_gpu/source_newbounds_descriptor.cpp)
      target_include_directories(fyzzxyzzy_physx_source_descriptor PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
      target_link_libraries(fyzzxyzzy_physx_source_descriptor PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_cuda_host_address)
      add_test(NAME fyzzxyzzy_physx_source_descriptor
        COMMAND fyzzxyzzy_physx_source_descriptor)



      # PhysX PxgAABBManager::postBroadPhase result-publication boundary:
      # GPU completion precedes found/lost-pair task reference release.
      add_executable(fyzzxyzzy_physx_broadphase_completion
        tests/physx_gpu_broadphase_completion.cpp)
      target_include_directories(fyzzxyzzy_physx_broadphase_completion PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
      target_link_libraries(fyzzxyzzy_physx_broadphase_completion PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_original_gpu_heap)
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

    # Original PxgCudaBroadPhaseSap::initializeSapBoxKernel source allocations.
    set(FYZZXYZZY_SOURCE_SAPBOX_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sapbox_indices_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAPBOX_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAPBOX_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sapbox_indices_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sapbox_indices_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_sapbox_spv DEPENDS "${FYZZXYZZY_SOURCE_SAPBOX_SPV}")
    add_executable(fyzzxyzzy_physx_source_sapbox_indices_native tests/physx_gpu_source_sapbox_indices_native.cpp)
    target_include_directories(fyzzxyzzy_physx_source_sapbox_indices_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sapbox_indices_native PRIVATE fyzzxyzzy_physx_gpu_stages)
    add_dependencies(fyzzxyzzy_physx_source_sapbox_indices_native fyzzxyzzy_source_sapbox_spv)
    add_test(NAME fyzzxyzzy_physx_source_sapbox_indices_native COMMAND fyzzxyzzy_physx_source_sapbox_indices_native "${FYZZXYZZY_SOURCE_SAPBOX_SPV}")

    # Original BP_MARK_DELETEDPAIRS and BP_UPDATE_DELETEDPAIRS on
    # PxgBroadPhaseDesc's separate source GPU allocations.
    set(FYZZXYZZY_SOURCE_SAP_REMOVED_MARK_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_mark_removed_native.spv")
    set(FYZZXYZZY_SOURCE_SAP_REMOVED_PROJ_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_remove_projections_native.spv")
    foreach(part IN ITEMS mark_removed remove_projections)
      set(shader_src "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_${part}_native.hlsl")
      set(shader_out "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_${part}_native.spv")
      add_custom_command(OUTPUT "${shader_out}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main -Fo "${shader_out}" "${shader_src}"
        DEPENDS "${shader_src}" VERBATIM)
    endforeach()
    add_custom_target(fyzzxyzzy_source_sap_removed_spv
      DEPENDS "${FYZZXYZZY_SOURCE_SAP_REMOVED_MARK_SPV}" "${FYZZXYZZY_SOURCE_SAP_REMOVED_PROJ_SPV}")
    add_executable(fyzzxyzzy_physx_source_sap_removed_native
      tests/physx_gpu_source_sap_removed_native.cpp)
    target_include_directories(fyzzxyzzy_physx_source_sap_removed_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_removed_native PRIVATE
      fyzzxyzzy_physx_gpu_stages)
    add_dependencies(fyzzxyzzy_physx_source_sap_removed_native
      fyzzxyzzy_source_sap_removed_spv)
    add_test(NAME fyzzxyzzy_physx_source_sap_removed_native
      COMMAND fyzzxyzzy_physx_source_sap_removed_native
        "${FYZZXYZZY_SOURCE_SAP_REMOVED_MARK_SPV}"
        "${FYZZXYZZY_SOURCE_SAP_REMOVED_PROJ_SPV}")

    # Original GPU AABB-manager changed/added/removed bitmap projection update.
    set(FYZZXYZZY_SOURCE_SAP_UPDATED_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_updated_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_UPDATED_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
        -T cs_6_0 -E main -Fo "${FYZZXYZZY_SOURCE_SAP_UPDATED_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_updated_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_updated_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_updated_spv
      DEPENDS "${FYZZXYZZY_SOURCE_SAP_UPDATED_SPV}")
    add_executable(fyzzxyzzy_physx_source_sap_updated_native
      tests/physx_gpu_source_sap_updated_native.cpp)
    target_include_directories(fyzzxyzzy_physx_source_sap_updated_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_updated_native PRIVATE
      fyzzxyzzy_physx_gpu_stages)
    add_dependencies(fyzzxyzzy_physx_source_sap_updated_native
      fyzzxyzzy_source_sap_updated_spv)
    add_test(NAME fyzzxyzzy_physx_source_sap_updated_native
      COMMAND fyzzxyzzy_physx_source_sap_updated_native "${FYZZXYZZY_SOURCE_SAP_UPDATED_SPV}")

    # Source BP_UPDATE_CREATEDPAIRS writes into source-owned projections,
    # current endpoint handles and old integer bounds before WOR sorting.
    set(FYZZXYZZY_SOURCE_SAP_CREATED_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_created_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_CREATED_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
        -T cs_6_0 -E main -Fo "${FYZZXYZZY_SOURCE_SAP_CREATED_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_created_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_created_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_created_spv
      DEPENDS "${FYZZXYZZY_SOURCE_SAP_CREATED_SPV}")
    add_executable(fyzzxyzzy_physx_source_sap_created_native
      tests/physx_gpu_source_sap_created_native.cpp)
    target_include_directories(fyzzxyzzy_physx_source_sap_created_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_created_native PRIVATE
      fyzzxyzzy_physx_gpu_stages)
    add_dependencies(fyzzxyzzy_physx_source_sap_created_native
      fyzzxyzzy_source_sap_created_spv)
    add_test(NAME fyzzxyzzy_physx_source_sap_created_native
      COMMAND fyzzxyzzy_physx_source_sap_created_native
        "${FYZZXYZZY_SOURCE_SAP_CREATED_SPV}")

    # Original WR/WOR radix pointers, eight stable LSD passes, preserved handles.
    set(FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_init_ranks_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_init_ranks_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_init_ranks_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_count_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_scatter_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_gather_handles_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_gather_handles_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_gather_handles_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_count_wave_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_wave_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_scatter_wave_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_wave_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_radix_spv DEPENDS "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_SPV}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_SPV}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_SPV}" "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_SPV}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_SPV}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_SPV}")
    add_executable(fyzzxyzzy_physx_source_sap_radix_native tests/physx_gpu_source_sap_radix_native.cpp)
    target_include_directories(fyzzxyzzy_physx_source_sap_radix_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_radix_native PRIVATE fyzzxyzzy_physx_gpu_stages)
    add_dependencies(fyzzxyzzy_physx_source_sap_radix_native fyzzxyzzy_source_sap_radix_spv)
    add_test(NAME fyzzxyzzy_physx_source_sap_radix_native COMMAND fyzzxyzzy_physx_source_sap_radix_native "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_SPV}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_SPV}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_SPV}" "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_SPV}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_SPV}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_SPV}")

    # Source incremental comparison prefixes publish into the original GPU descriptor.
    set(FYZZXYZZY_SOURCE_SAP_INCREMENTAL_FIRST_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_counts_first_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_FIRST_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_FIRST_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_first_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_INCREMENTAL_SECOND_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_counts_second_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_SECOND_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_SECOND_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_second_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_incremental_counts_spv DEPENDS "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_FIRST_SPV}" "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_SECOND_SPV}")
    add_executable(fyzzxyzzy_physx_source_sap_incremental_counts_native tests/physx_gpu_source_sap_incremental_counts_native.cpp)
    target_include_directories(fyzzxyzzy_physx_source_sap_incremental_counts_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_incremental_counts_native PRIVATE fyzzxyzzy_physx_gpu_stages)
    add_dependencies(fyzzxyzzy_physx_source_sap_incremental_counts_native fyzzxyzzy_source_sap_incremental_counts_spv)
    add_test(NAME fyzzxyzzy_physx_source_sap_incremental_counts_native COMMAND fyzzxyzzy_physx_source_sap_incremental_counts_native "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_FIRST_SPV}" "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_SECOND_SPV}")

    # Original two 32x3x256 SAP endpoint histogram launches (48 source views).
    set(FYZZXYZZY_SOURCE_SAP_HIST_FIRST_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_endpoint_histograms_first_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_HIST_FIRST_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_HIST_FIRST_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_first_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_HIST_SECOND_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_endpoint_histograms_second_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_HIST_SECOND_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_HIST_SECOND_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_second_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_hist_spv DEPENDS "${FYZZXYZZY_SOURCE_SAP_HIST_FIRST_SPV}" "${FYZZXYZZY_SOURCE_SAP_HIST_SECOND_SPV}")
    add_executable(fyzzxyzzy_physx_source_sap_endpoint_histograms_native tests/physx_gpu_source_sap_endpoint_histograms_native.cpp)
    target_include_directories(fyzzxyzzy_physx_source_sap_endpoint_histograms_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_endpoint_histograms_native PRIVATE fyzzxyzzy_physx_gpu_stages)
    add_dependencies(fyzzxyzzy_physx_source_sap_endpoint_histograms_native fyzzxyzzy_source_sap_hist_spv)
    add_test(NAME fyzzxyzzy_physx_source_sap_endpoint_histograms_native COMMAND fyzzxyzzy_physx_source_sap_endpoint_histograms_native "${FYZZXYZZY_SOURCE_SAP_HIST_FIRST_SPV}" "${FYZZXYZZY_SOURCE_SAP_HIST_SECOND_SPV}")

    # Source BP_INCREMENTAL_SAP consumes original per-axis comparisons
    # and writes original GPU PxgBroadPhaseDesc found/lost report indices.
    set(FYZZXYZZY_SOURCE_SAP_INCREMENTAL_PAIRS_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_pairs_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_PAIRS_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_PAIRS_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_pairs_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_pairs_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_incremental_pairs_spv DEPENDS "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_PAIRS_SPV}")
    add_executable(fyzzxyzzy_physx_source_sap_incremental_pairs_native tests/physx_gpu_source_sap_incremental_pairs_native.cpp)
    target_include_directories(fyzzxyzzy_physx_source_sap_incremental_pairs_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_incremental_pairs_native PRIVATE fyzzxyzzy_physx_gpu_stages)
    add_dependencies(fyzzxyzzy_physx_source_sap_incremental_pairs_native fyzzxyzzy_source_sap_incremental_pairs_spv)
    add_test(NAME fyzzxyzzy_physx_source_sap_incremental_pairs_native COMMAND fyzzxyzzy_physx_source_sap_incremental_pairs_native "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_PAIRS_SPV}")







    
    # PhysX BP_CLEAR_NEWFLAG on the original seven GPU allocations.
    set(FYZZXYZZY_SOURCE_CLEAR_NEW_VULKAN "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_clear_new_flags_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_CLEAR_NEW_VULKAN}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_CLEAR_NEW_VULKAN}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_clear_new_flags_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_clear_new_flags_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_clear_new_flags_spv DEPENDS "${FYZZXYZZY_SOURCE_CLEAR_NEW_VULKAN}")
    add_executable(fyzzxyzzy_physx_source_clear_new_flags_native
      tests/physx_gpu_source_clear_new_flags_native.cpp)
    target_include_directories(fyzzxyzzy_physx_source_clear_new_flags_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_clear_new_flags_native PRIVATE fyzzxyzzy_physx_gpu_stages)
    add_dependencies(fyzzxyzzy_physx_source_clear_new_flags_native fyzzxyzzy_source_clear_new_flags_spv)
    add_test(NAME fyzzxyzzy_physx_source_clear_new_flags_native COMMAND fyzzxyzzy_physx_source_clear_new_flags_native "${FYZZXYZZY_SOURCE_CLEAR_NEW_VULKAN}")
    
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
      # Existing source report kernels, now bound directly to PhysX heap roots.
      set(FYZZXYZZY_SOURCE_REPORT_NATIVE_SPV_OUTPUTS "")
      foreach(_native_report IN ITEMS first second copy first_wave second_wave)
        if(_native_report MATCHES "wave")
          set(_native_shader_model cs_6_6)
        else()
          set(_native_shader_model cs_6_0)
        endif()
        set(_native_source
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_report_${_native_report}_native.hlsl")
        set(_native_binary
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_report_${_native_report}_native.spv")
        add_custom_command(
          OUTPUT "${_native_binary}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
            -T "${_native_shader_model}" -E main -Fo "${_native_binary}" "${_native_source}"
          DEPENDS "${_native_source}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_report_multi_root_common.hlsli"
          VERBATIM)
        list(APPEND FYZZXYZZY_SOURCE_REPORT_NATIVE_SPV_OUTPUTS "${_native_binary}")
      endforeach()
      add_custom_target(fyzzxyzzy_physx_source_report_multi_root_spv
        DEPENDS ${FYZZXYZZY_SOURCE_REPORT_NATIVE_SPV_OUTPUTS})

      add_executable(fyzzxyzzy_physx_bp_report_classification
        tests/physx_gpu_broadphase_report_classification.cpp)
      target_include_directories(fyzzxyzzy_physx_bp_report_classification PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
      target_link_libraries(fyzzxyzzy_physx_bp_report_classification PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_original_gpu_heap)
      add_dependencies(fyzzxyzzy_physx_bp_report_classification
        fyzzxyzzy_physx_bp_report_class_first_shader
        fyzzxyzzy_physx_bp_report_class_second_shader
        fyzzxyzzy_physx_bp_report_class_first_wave_shader
        fyzzxyzzy_physx_bp_report_class_second_wave_shader
        fyzzxyzzy_physx_bp_copy_reports_shader
        fyzzxyzzy_physx_source_report_multi_root_spv)
      add_test(NAME fyzzxyzzy_physx_bp_report_classification
        COMMAND fyzzxyzzy_physx_bp_report_classification
          "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_first_SHADER}"
          "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_second_SHADER}"
          "${FYZZXYZZY_PHYSX_BP_COPY_REPORTS_SHADER}"
          "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_first_WAVE_SHADER}"
          "${FYZZXYZZY_PHYSX_BP_REPORT_CLASS_second_WAVE_SHADER}"
          ${FYZZXYZZY_SOURCE_REPORT_NATIVE_SPV_OUTPUTS})

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
      # Original PhysX active region allocations (separate from the
      # single-packed-arena diagnostic shaders above).
      foreach(_native_pass IN ITEMS writeout compute_wave compute_portable output)
        if(_native_pass STREQUAL "compute_wave")
          set(_native_source compute)
          set(_native_flag 1)
          set(_native_profile cs_6_6)
        elseif(_native_pass STREQUAL "compute_portable")
          set(_native_source compute)
          set(_native_flag 0)
          set(_native_profile cs_6_0)
        else()
          set(_native_source ${_native_pass})
          set(_native_flag 0)
          set(_native_profile cs_6_0)
        endif()
        set(FYZZXYZZY_SOURCE_ACTIVE_${_native_pass}_SPV
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_${_native_pass}.spv")
        add_custom_command(
          OUTPUT "${FYZZXYZZY_SOURCE_ACTIVE_${_native_pass}_SPV}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
            -T ${_native_profile} -E main -DFYZZXYZZY_ACTIVE_REGION_WAVE=${_native_flag}
            -Fo "${FYZZXYZZY_SOURCE_ACTIVE_${_native_pass}_SPV}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_active_${_native_source}_native.hlsl"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_active_${_native_source}_native.hlsl"
          VERBATIM)
        add_custom_target(fyzzxyzzy_physx_source_active_${_native_pass}_shader
          DEPENDS "${FYZZXYZZY_SOURCE_ACTIVE_${_native_pass}_SPV}")
      endforeach()
      add_executable(fyzzxyzzy_physx_sap_active_region_lists
        tests/physx_gpu_sap_active_region_lists.cpp)
      target_include_directories(fyzzxyzzy_physx_sap_active_region_lists PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
      target_link_libraries(fyzzxyzzy_physx_sap_active_region_lists PRIVATE
        fyzzxyzzy_physx_gpu_stages fyzzxyzzy_physx_cuda_host_address)
      add_dependencies(fyzzxyzzy_physx_sap_active_region_lists
        fyzzxyzzy_physx_sap_active_writeout_shader
        fyzzxyzzy_physx_sap_active_compute_wave_shader
        fyzzxyzzy_physx_sap_active_compute_portable_shader
        fyzzxyzzy_physx_sap_active_output_shader
        fyzzxyzzy_physx_source_active_writeout_shader
        fyzzxyzzy_physx_source_active_compute_wave_shader
        fyzzxyzzy_physx_source_active_compute_portable_shader
        fyzzxyzzy_physx_source_active_output_shader)
      add_test(NAME fyzzxyzzy_physx_sap_active_region_lists
        COMMAND fyzzxyzzy_physx_sap_active_region_lists
          "${FYZZXYZZY_PHYSX_SAP_ACTIVE_writeout_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ACTIVE_compute_wave_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ACTIVE_compute_portable_SHADER}"
          "${FYZZXYZZY_PHYSX_SAP_ACTIVE_output_SHADER}"
          "${FYZZXYZZY_SOURCE_ACTIVE_writeout_SPV}"
          "${FYZZXYZZY_SOURCE_ACTIVE_compute_wave_SPV}"
          "${FYZZXYZZY_SOURCE_ACTIVE_compute_portable_SPV}"
          "${FYZZXYZZY_SOURCE_ACTIVE_output_SPV}")

      set(FYZZXYZZY_SAP_OVERLAP_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_work.spv")
      add_custom_command(OUTPUT "${FYZZXYZZY_SAP_OVERLAP_SPV}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_SAP_OVERLAP_SPV}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_arena.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_sap_overlap_shader DEPENDS "${FYZZXYZZY_SAP_OVERLAP_SPV}")
      add_executable(fyzzxyzzy_sap_overlap_work tests/physx_gpu_sap_overlap_work.cpp)
      target_include_directories(fyzzxyzzy_sap_overlap_work PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_sap_overlap_work PRIVATE fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_sap_overlap_work fyzzxyzzy_sap_overlap_shader)
      add_test(NAME fyzzxyzzy_sap_overlap_work
        COMMAND fyzzxyzzy_sap_overlap_work "${FYZZXYZZY_SAP_OVERLAP_SPV}")

      # Pinned PhysX compute/output overlap histogram pair. Both kernels
      # require 64-bit subgroup operations (Vulkan 1.2 extended types).
      foreach(_pass IN ITEMS first second)
        set(FYZZXYZZY_OVERLAP_PREFIX_${_pass}_SPV
          "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_prefix_${_pass}.spv")
        add_custom_command(
          OUTPUT "${FYZZXYZZY_OVERLAP_PREFIX_${_pass}_SPV}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
          COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.2
            -T cs_6_6 -E main
            -Fo "${FYZZXYZZY_OVERLAP_PREFIX_${_pass}_SPV}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_${_pass}_wave.hlsl"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_${_pass}_wave.hlsl"
          VERBATIM)
        add_custom_target(fyzzxyzzy_overlap_prefix_${_pass}_spv
          DEPENDS "${FYZZXYZZY_OVERLAP_PREFIX_${_pass}_SPV}")
      endforeach()
      add_executable(fyzzxyzzy_sap_overlap_prefix
        tests/physx_gpu_sap_overlap_prefix.cpp)
      target_include_directories(fyzzxyzzy_sap_overlap_prefix PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_sap_overlap_prefix PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_sap_overlap_prefix
        fyzzxyzzy_overlap_prefix_first_spv fyzzxyzzy_overlap_prefix_second_spv)
      add_test(NAME fyzzxyzzy_sap_overlap_prefix
        COMMAND fyzzxyzzy_sap_overlap_prefix
          "${FYZZXYZZY_OVERLAP_PREFIX_first_SPV}"
          "${FYZZXYZZY_OVERLAP_PREFIX_second_SPV}")
      set_tests_properties(fyzzxyzzy_sap_overlap_prefix PROPERTIES
        SKIP_RETURN_CODE 77)
      # PhysX BP_GENERATE_FOUNDPAIR_NEWBOUNDS, fixed 64x256 native-wave scan.
      set(FYZZXYZZY_NEWBOUNDS_PAIRS_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_newbounds_pairs.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_NEWBOUNDS_PAIRS_SPV}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.2
          -T cs_6_6 -E main -Fo "${FYZZXYZZY_NEWBOUNDS_PAIRS_SPV}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_arena.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_arena.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_newbounds_pairs_spv
        DEPENDS "${FYZZXYZZY_NEWBOUNDS_PAIRS_SPV}")
      add_executable(fyzzxyzzy_sap_newbounds_pairs
        tests/physx_gpu_sap_newbounds_pairs.cpp)
      target_include_directories(fyzzxyzzy_sap_newbounds_pairs PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_sap_newbounds_pairs PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_sap_newbounds_pairs
        fyzzxyzzy_newbounds_pairs_spv)
      add_test(NAME fyzzxyzzy_sap_newbounds_pairs
        COMMAND fyzzxyzzy_sap_newbounds_pairs
          "${FYZZXYZZY_NEWBOUNDS_PAIRS_SPV}")
      set_tests_properties(fyzzxyzzy_sap_newbounds_pairs PROPERTIES
        SKIP_RETURN_CODE 77)
      # One ordered PhysX new-bounds producer / prefix / report batch.
      add_executable(fyzzxyzzy_sap_newbounds_sequence
        tests/physx_gpu_sap_newbounds_sequence.cpp)
      target_include_directories(fyzzxyzzy_sap_newbounds_sequence PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_sap_newbounds_sequence PRIVATE
        fyzzxyzzy_physx_gpu_stages)
      add_dependencies(fyzzxyzzy_sap_newbounds_sequence
        fyzzxyzzy_sap_overlap_shader
        fyzzxyzzy_overlap_prefix_first_spv
        fyzzxyzzy_overlap_prefix_second_spv
        fyzzxyzzy_newbounds_pairs_spv)
      add_test(NAME fyzzxyzzy_sap_newbounds_sequence
        COMMAND fyzzxyzzy_sap_newbounds_sequence
          "${FYZZXYZZY_SAP_OVERLAP_SPV}"
          "${FYZZXYZZY_OVERLAP_PREFIX_first_SPV}"
          "${FYZZXYZZY_OVERLAP_PREFIX_second_SPV}"
          "${FYZZXYZZY_NEWBOUNDS_PAIRS_SPV}")
      set_tests_properties(fyzzxyzzy_sap_newbounds_sequence PROPERTIES
        SKIP_RETURN_CODE 77)



    endif()

  endif()
  endif() # source-port GPU stages and source-based conformance tests
  if(BUILD_TESTING AND FYZZXYZZY_ENABLE_GPU_PHYSICS_PROTOTYPES)
    find_program(FYZZXYZZY_GLSLANG_VALIDATOR NAMES glslangValidator REQUIRED)
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
  
      # This oracle comparison needs the transitional reference CPU scene.
      if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
      add_executable(fyzzxyzzy_vulkan_world_smoke tests/vulkan_world_smoke.cpp)
      target_link_libraries(fyzzxyzzy_vulkan_world_smoke PRIVATE FyzzXYZZY::VulkanWorld)
      add_dependencies(fyzzxyzzy_vulkan_world_smoke fyzzxyzzy_floor_shader)
      add_test(NAME fyzzxyzzy_vulkan_world_smoke
        COMMAND fyzzxyzzy_vulkan_world_smoke "${FYZZXYZZY_FLOOR_SHADER}")
      endif()
    endif()
  endif()
  if(BUILD_TESTING AND NOT FYZZXYZZY_ENABLE_PHYSX_SOURCE)
    add_executable(fyzzxyzzy_vulkan_memory_smoke tests/vulkan_memory_smoke.cpp)
    target_link_libraries(fyzzxyzzy_vulkan_memory_smoke PRIVATE FyzzXYZZY::Compute)
    add_test(NAME fyzzxyzzy_vulkan_memory_smoke
      COMMAND fyzzxyzzy_vulkan_memory_smoke)
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
  target_compile_features(fyzzxyzzy_compute_d3d12 PUBLIC cxx_std_23)
  target_include_directories(fyzzxyzzy_compute_d3d12 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_link_libraries(fyzzxyzzy_compute_d3d12 PUBLIC d3d12 dxgi d3dcompiler)
  set_target_properties(fyzzxyzzy_compute_d3d12 PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  if(MSVC)
    target_compile_options(fyzzxyzzy_compute_d3d12 PRIVATE /W4)
  endif()

  if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
  add_library(fyzzxyzzy_physx_gpu_stages_d3d12 STATIC
    src/physx_gpu/aabb_encoding.cpp
    src/physx_gpu/cuda_aabb_launch_lowering.cpp
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
    src/physx_gpu/source_sap_report_multi_root.cpp
    src/physx_gpu/sap_host_frame_roles.cpp
    src/physx_gpu/sap_region_producer_arena.cpp
    src/physx_gpu/sap_region_histograms_arena.cpp
    src/physx_gpu/sap_active_region_lists_arena.cpp
    src/physx_gpu/sap_overlap_work_arena.cpp
    src/physx_gpu/source_overlap_work_native.cpp
    src/physx_gpu/source_overlap_prefix_native.cpp
    src/physx_gpu/source_newbounds_pairs_native.cpp
    src/physx_gpu/source_newbounds_sequence_native.cpp
    src/physx_gpu/source_clear_new_flags_native.cpp
    src/physx_gpu/source_sapbox_indices_native.cpp
    src/physx_gpu/source_sap_removed_native.cpp
    src/physx_gpu/source_sap_updated_native.cpp
    src/physx_gpu/source_sap_created_native.cpp
    src/physx_gpu/source_sap_radix_native.cpp
    src/physx_gpu/source_sap_incremental_counts_native.cpp
    src/physx_gpu/source_sap_endpoint_histograms_native.cpp
    src/physx_gpu/source_sap_incremental_pairs_native.cpp
    src/physx_gpu/source_sap_update_native.cpp
    src/physx_gpu/source_fill_d32_native.cpp
    src/physx_gpu/source_region_create_native.cpp
    src/physx_gpu/source_region_histograms_native.cpp
    src/physx_gpu/source_active_region_lists_native.cpp
    src/physx_gpu/sap_overlap_prefix_arena.cpp
    src/physx_gpu/sap_newbounds_pairs_arena.cpp
    src/physx_gpu/sap_newbounds_sequence_arena.cpp
    src/physx_gpu/broadphase_completion.cpp)
  target_compile_features(fyzzxyzzy_physx_gpu_stages_d3d12 PUBLIC cxx_std_23)
  target_include_directories(fyzzxyzzy_physx_gpu_stages_d3d12 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include")
  target_link_libraries(fyzzxyzzy_physx_gpu_stages_d3d12 PUBLIC
    FyzzXYZZY::ComputeD3D12 physx_lib fyzzxyzzy_physx_cuda_host_address
    fyzzxyzzy_physx_original_gpu_heap)


  if(BUILD_TESTING)
    add_executable(fyzzxyzzy_physx_cuda_host_transfer_d3d12
      tests/physx_gpu_cuda_host_transfer.cpp)
    target_compile_definitions(fyzzxyzzy_physx_cuda_host_transfer_d3d12
      PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_cuda_host_transfer_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
    target_link_libraries(fyzzxyzzy_physx_cuda_host_transfer_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_physx_cuda_host_address fyzzxyzzy_physx_original_gpu_heap physx_lib)
    add_test(NAME fyzzxyzzy_physx_cuda_host_transfer_d3d12
      COMMAND fyzzxyzzy_physx_cuda_host_transfer_d3d12)
    add_executable(fyzzxyzzy_physx_sap_broadphase_descriptor_native_d3d12
      tests/physx_gpu_sap_broadphase_descriptor_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_sap_broadphase_descriptor_native_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_sap_broadphase_descriptor_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
    target_link_libraries(fyzzxyzzy_physx_sap_broadphase_descriptor_native_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_physx_cuda_host_address fyzzxyzzy_physx_original_gpu_heap physx_lib)
    add_test(NAME fyzzxyzzy_physx_sap_broadphase_descriptor_native_d3d12 COMMAND fyzzxyzzy_physx_sap_broadphase_descriptor_native_d3d12)
  endif()

  option(FYZZXYZZY_BUILD_D3D12_AABB_TEST
    "Enable native D3D12 PhysX AABB conformance test (requires DXC)" OFF)
  if(BUILD_TESTING AND FYZZXYZZY_BUILD_D3D12_AABB_TEST)
    find_program(FYZZXYZZY_DXC NAMES dxc REQUIRED)
    set(FYZZXYZZY_MULTI_RESOURCE_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/native_multi_resource_bindings.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_MULTI_RESOURCE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_MULTI_RESOURCE_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_multi_resource_bindings.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_multi_resource_bindings.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_native_multi_resource_dxil
      DEPENDS "${FYZZXYZZY_MULTI_RESOURCE_DXIL}")
    add_executable(fyzzxyzzy_native_multi_resource_d3d12
      tests/physx_gpu_native_multi_resource.cpp)
    target_compile_definitions(fyzzxyzzy_native_multi_resource_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_link_libraries(fyzzxyzzy_native_multi_resource_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12)
    add_dependencies(fyzzxyzzy_native_multi_resource_d3d12
      fyzzxyzzy_native_multi_resource_dxil)
    add_test(NAME fyzzxyzzy_native_multi_resource_d3d12
      COMMAND fyzzxyzzy_native_multi_resource_d3d12 "${FYZZXYZZY_MULTI_RESOURCE_DXIL}")
    set_tests_properties(fyzzxyzzy_native_multi_resource_d3d12 PROPERTIES SKIP_RETURN_CODE 77)
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
    add_executable(fyzzxyzzy_physx_cuda_aabb_launch_d3d12
      tests/physx_gpu_cuda_aabb_launch.cpp)
    target_compile_definitions(fyzzxyzzy_physx_cuda_aabb_launch_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_cuda_aabb_launch_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include")
    target_link_libraries(fyzzxyzzy_physx_cuda_aabb_launch_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_cuda_host_address)
    add_dependencies(fyzzxyzzy_physx_cuda_aabb_launch_d3d12
      fyzzxyzzy_d3d12_aabb_shader)
    add_test(NAME fyzzxyzzy_physx_cuda_aabb_launch_d3d12
      COMMAND fyzzxyzzy_physx_cuda_aabb_launch_d3d12
        "${FYZZXYZZY_D3D12_AABB_DXIL}")

    set(FYZZXYZZY_SOURCE_OVERLAP_NATIVE_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_work_native.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_physx_source_overlap_native_dxil
      DEPENDS "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_DXIL}")
    add_executable(fyzzxyzzy_physx_source_overlap_native_d3d12
      tests/physx_gpu_source_overlap_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_overlap_native_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_overlap_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_overlap_native_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_cuda_host_address)
    add_dependencies(fyzzxyzzy_physx_source_overlap_native_d3d12
      fyzzxyzzy_physx_source_overlap_native_dxil)
    add_test(NAME fyzzxyzzy_physx_source_overlap_native_d3d12
      COMMAND fyzzxyzzy_physx_source_overlap_native_d3d12
        "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_DXIL}")

    set(FYZZXYZZY_SOURCE_SCAN_DXIL_FILES)
    foreach(_phase first second)
      set(_shader "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_${_phase}_native.hlsl")
      set(_dxil "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_prefix_${_phase}_native.dxil")
      add_custom_command(OUTPUT "${_dxil}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main -Fo "${_dxil}" "${_shader}"
        DEPENDS "${_shader}" VERBATIM)
      list(APPEND FYZZXYZZY_SOURCE_SCAN_DXIL_FILES "${_dxil}")
    endforeach()
    add_custom_target(fyzzxyzzy_physx_source_scan_dxil
      DEPENDS ${FYZZXYZZY_SOURCE_SCAN_DXIL_FILES})
    add_executable(fyzzxyzzy_physx_source_scan_native_d3d12
      tests/physx_gpu_source_overlap_prefix_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_scan_native_d3d12
      PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_scan_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_scan_native_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_cuda_host_address)
    add_dependencies(fyzzxyzzy_physx_source_scan_native_d3d12
      fyzzxyzzy_physx_source_scan_dxil)
    add_test(NAME fyzzxyzzy_physx_source_scan_native_d3d12
      COMMAND fyzzxyzzy_physx_source_scan_native_d3d12
        ${FYZZXYZZY_SOURCE_SCAN_DXIL_FILES})
    set_tests_properties(fyzzxyzzy_physx_source_scan_native_d3d12 PROPERTIES SKIP_RETURN_CODE 77)

    set(FYZZXYZZY_SOURCE_PAIRS_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_newbounds_pairs_wave_native.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_SOURCE_PAIRS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_PAIRS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_physx_source_pairs_dxil
      DEPENDS "${FYZZXYZZY_SOURCE_PAIRS_DXIL}")
    add_executable(fyzzxyzzy_physx_source_pairs_native_d3d12
      tests/physx_gpu_source_newbounds_pairs_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_pairs_native_d3d12
      PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_pairs_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_pairs_native_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_cuda_host_address)
    add_dependencies(fyzzxyzzy_physx_source_pairs_native_d3d12
      fyzzxyzzy_physx_source_pairs_dxil)
    add_test(NAME fyzzxyzzy_physx_source_pairs_native_d3d12
      COMMAND fyzzxyzzy_physx_source_pairs_native_d3d12
        "${FYZZXYZZY_SOURCE_PAIRS_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_pairs_native_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    add_executable(fyzzxyzzy_physx_source_sequence_native_d3d12
      tests/physx_gpu_source_newbounds_sequence_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_sequence_native_d3d12
      PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_sequence_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sequence_native_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_cuda_host_address)
    add_dependencies(fyzzxyzzy_physx_source_sequence_native_d3d12
      fyzzxyzzy_physx_source_overlap_native_dxil
      fyzzxyzzy_physx_source_scan_dxil
      fyzzxyzzy_physx_source_pairs_dxil)
    add_test(NAME fyzzxyzzy_physx_source_sequence_native_d3d12
      COMMAND fyzzxyzzy_physx_source_sequence_native_d3d12
        "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_DXIL}"
        ${FYZZXYZZY_SOURCE_SCAN_DXIL_FILES}
        "${FYZZXYZZY_SOURCE_PAIRS_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_sequence_native_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    set(FYZZXYZZY_SOURCE_D32_FILL_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_fill_d32_native.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_SOURCE_D32_FILL_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_D32_FILL_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_fill_d32_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_fill_d32_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_physx_source_d32_fill_dxil
      DEPENDS "${FYZZXYZZY_SOURCE_D32_FILL_DXIL}")
    add_executable(fyzzxyzzy_physx_source_d32_fill_native_d3d12
      tests/physx_gpu_source_fill_d32_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_d32_fill_native_d3d12
      PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_d32_fill_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_source_d32_fill_native_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_cuda_host_address)
    add_dependencies(fyzzxyzzy_physx_source_d32_fill_native_d3d12
      fyzzxyzzy_physx_source_d32_fill_dxil)
    add_test(NAME fyzzxyzzy_physx_source_d32_fill_native_d3d12
      COMMAND fyzzxyzzy_physx_source_d32_fill_native_d3d12
        "${FYZZXYZZY_SOURCE_D32_FILL_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_d32_fill_native_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    set(FYZZXYZZY_SOURCE_REGION_CREATE_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_create_native.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_SOURCE_REGION_CREATE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_REGION_CREATE_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_create_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_create_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_physx_source_region_create_dxil
      DEPENDS "${FYZZXYZZY_SOURCE_REGION_CREATE_DXIL}")
    add_executable(fyzzxyzzy_physx_source_region_create_native_d3d12
      tests/physx_gpu_source_region_create_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_region_create_native_d3d12
      PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_region_create_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_region_create_native_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_cuda_host_address)
    add_dependencies(fyzzxyzzy_physx_source_region_create_native_d3d12
      fyzzxyzzy_physx_source_d32_fill_dxil fyzzxyzzy_physx_source_region_create_dxil)
    add_test(NAME fyzzxyzzy_physx_source_region_create_native_d3d12
      COMMAND fyzzxyzzy_physx_source_region_create_native_d3d12
        "${FYZZXYZZY_SOURCE_D32_FILL_DXIL}" "${FYZZXYZZY_SOURCE_REGION_CREATE_DXIL}"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_first_native.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_wave_native.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_second_native.dxil")
    set_tests_properties(fyzzxyzzy_physx_source_region_create_native_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_test(NAME fyzzxyzzy_physx_ten_operation_chain_d3d12
      COMMAND fyzzxyzzy_physx_source_region_create_native_d3d12
        "${FYZZXYZZY_SOURCE_D32_FILL_DXIL}"
        "${FYZZXYZZY_SOURCE_REGION_CREATE_DXIL}"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_first_native.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_wave_native.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_second_native.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_writeout.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_compute_wave.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_compute_portable.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_output.dxil")
    set_tests_properties(fyzzxyzzy_physx_ten_operation_chain_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_dependencies(fyzzxyzzy_physx_source_region_create_native_d3d12
      fyzzxyzzy_physx_source_overlap_native_dxil
      fyzzxyzzy_physx_source_scan_dxil
      fyzzxyzzy_physx_source_pairs_dxil)
    add_test(NAME fyzzxyzzy_physx_fourteen_operation_chain_d3d12
      COMMAND fyzzxyzzy_physx_source_region_create_native_d3d12
        "${FYZZXYZZY_SOURCE_D32_FILL_DXIL}"
        "${FYZZXYZZY_SOURCE_REGION_CREATE_DXIL}"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_first_native.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_wave_native.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_second_native.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_writeout.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_compute_wave.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_compute_portable.dxil"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_output.dxil"
        "${FYZZXYZZY_SOURCE_OVERLAP_NATIVE_DXIL}"
        ${FYZZXYZZY_SOURCE_SCAN_DXIL_FILES}
        "${FYZZXYZZY_SOURCE_PAIRS_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_fourteen_operation_chain_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    foreach(variant first wave second)
      set(FYZZXYZZY_SOURCE_REGION_HIST_${variant}_DXIL
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_${variant}_native.dxil")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_SOURCE_REGION_HIST_${variant}_DXIL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
          -Fo "${FYZZXYZZY_SOURCE_REGION_HIST_${variant}_DXIL}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_${variant}_native.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_${variant}_native.hlsl"
        VERBATIM)
    endforeach()
    add_custom_target(fyzzxyzzy_physx_source_region_histograms_dxil
      DEPENDS "${FYZZXYZZY_SOURCE_REGION_HIST_first_DXIL}"
              "${FYZZXYZZY_SOURCE_REGION_HIST_wave_DXIL}"
              "${FYZZXYZZY_SOURCE_REGION_HIST_second_DXIL}")
    add_dependencies(fyzzxyzzy_physx_source_region_create_native_d3d12
      fyzzxyzzy_physx_source_region_histograms_dxil
      fyzzxyzzy_physx_source_active_writeout_dxil
      fyzzxyzzy_physx_source_active_compute_wave_dxil
      fyzzxyzzy_physx_source_active_compute_portable_dxil
      fyzzxyzzy_physx_source_active_output_dxil)
    add_executable(fyzzxyzzy_physx_source_region_histograms_native_d3d12
      tests/physx_gpu_source_region_histograms_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_region_histograms_native_d3d12
      PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_region_histograms_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_region_histograms_native_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_cuda_host_address)
    add_dependencies(fyzzxyzzy_physx_source_region_histograms_native_d3d12
      fyzzxyzzy_physx_source_region_histograms_dxil)
    add_test(NAME fyzzxyzzy_physx_source_region_histograms_native_d3d12
      COMMAND fyzzxyzzy_physx_source_region_histograms_native_d3d12
        "${FYZZXYZZY_SOURCE_REGION_HIST_first_DXIL}"
        "${FYZZXYZZY_SOURCE_REGION_HIST_wave_DXIL}"
        "${FYZZXYZZY_SOURCE_REGION_HIST_second_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_region_histograms_native_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    add_executable(fyzzxyzzy_physx_source_descriptor_d3d12
      tests/physx_gpu_source_descriptor.cpp
      src/physx_gpu/source_newbounds_descriptor.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_descriptor_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_descriptor_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_descriptor_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_cuda_host_address)
    add_test(NAME fyzzxyzzy_physx_source_descriptor_d3d12
      COMMAND fyzzxyzzy_physx_source_descriptor_d3d12)



    add_executable(fyzzxyzzy_physx_broadphase_completion_d3d12
      tests/physx_gpu_broadphase_completion.cpp)
    target_compile_definitions(fyzzxyzzy_physx_broadphase_completion_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_broadphase_completion_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
    target_link_libraries(fyzzxyzzy_physx_broadphase_completion_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_original_gpu_heap)
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

    # Original PxgCudaBroadPhaseSap::initializeSapBoxKernel source allocations.
    set(FYZZXYZZY_SOURCE_SAPBOX_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sapbox_indices_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAPBOX_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAPBOX_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sapbox_indices_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sapbox_indices_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_sapbox_dxil DEPENDS "${FYZZXYZZY_SOURCE_SAPBOX_DXIL}")
    add_executable(fyzzxyzzy_physx_source_sapbox_indices_native_d3d12 tests/physx_gpu_source_sapbox_indices_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_sapbox_indices_native_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_sapbox_indices_native_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sapbox_indices_native_d3d12 PRIVATE fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_source_sapbox_indices_native_d3d12 fyzzxyzzy_source_sapbox_dxil)
    add_test(NAME fyzzxyzzy_physx_source_sapbox_indices_native_d3d12 COMMAND fyzzxyzzy_physx_source_sapbox_indices_native_d3d12 "${FYZZXYZZY_SOURCE_SAPBOX_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_sapbox_indices_native_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    set(FYZZXYZZY_SOURCE_SAP_REMOVED_MARK_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_mark_removed_native.dxil")
    set(FYZZXYZZY_SOURCE_SAP_REMOVED_PROJ_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_remove_projections_native.dxil")
    foreach(part IN ITEMS mark_removed remove_projections)
      set(shader_src "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_${part}_native.hlsl")
      set(shader_out "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_${part}_native.dxil")
      add_custom_command(OUTPUT "${shader_out}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
          -Fo "${shader_out}" "${shader_src}"
        DEPENDS "${shader_src}" VERBATIM)
    endforeach()
    add_custom_target(fyzzxyzzy_source_sap_removed_dxil
      DEPENDS "${FYZZXYZZY_SOURCE_SAP_REMOVED_MARK_DXIL}" "${FYZZXYZZY_SOURCE_SAP_REMOVED_PROJ_DXIL}")
    add_executable(fyzzxyzzy_physx_source_sap_removed_native_d3d12
      tests/physx_gpu_source_sap_removed_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_sap_removed_native_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_sap_removed_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_removed_native_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_source_sap_removed_native_d3d12
      fyzzxyzzy_source_sap_removed_dxil)
    add_test(NAME fyzzxyzzy_physx_source_sap_removed_native_d3d12
      COMMAND fyzzxyzzy_physx_source_sap_removed_native_d3d12
        "${FYZZXYZZY_SOURCE_SAP_REMOVED_MARK_DXIL}"
        "${FYZZXYZZY_SOURCE_SAP_REMOVED_PROJ_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_sap_removed_native_d3d12
      PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    set(FYZZXYZZY_SOURCE_SAP_UPDATED_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_updated_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_UPDATED_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_UPDATED_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_updated_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_updated_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_updated_dxil
      DEPENDS "${FYZZXYZZY_SOURCE_SAP_UPDATED_DXIL}")
    add_executable(fyzzxyzzy_physx_source_sap_updated_native_d3d12
      tests/physx_gpu_source_sap_updated_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_sap_updated_native_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_sap_updated_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_updated_native_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_source_sap_updated_native_d3d12
      fyzzxyzzy_source_sap_updated_dxil)
    add_test(NAME fyzzxyzzy_physx_source_sap_updated_native_d3d12
      COMMAND fyzzxyzzy_physx_source_sap_updated_native_d3d12
        "${FYZZXYZZY_SOURCE_SAP_UPDATED_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_sap_updated_native_d3d12
      PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    set(FYZZXYZZY_SOURCE_SAP_CREATED_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_created_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_CREATED_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_CREATED_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_created_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_created_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_created_dxil
      DEPENDS "${FYZZXYZZY_SOURCE_SAP_CREATED_DXIL}")
    add_executable(fyzzxyzzy_physx_source_sap_created_native_d3d12
      tests/physx_gpu_source_sap_created_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_sap_created_native_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_sap_created_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_created_native_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_source_sap_created_native_d3d12
      fyzzxyzzy_source_sap_created_dxil)
    add_test(NAME fyzzxyzzy_physx_source_sap_created_native_d3d12
      COMMAND fyzzxyzzy_physx_source_sap_created_native_d3d12
        "${FYZZXYZZY_SOURCE_SAP_CREATED_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_sap_created_native_d3d12
      PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    # Original WR/WOR radix pointers, eight stable LSD passes, preserved handles.
    set(FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_init_ranks_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_init_ranks_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_init_ranks_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_count_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_scatter_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_gather_handles_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_gather_handles_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_gather_handles_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_count_wave_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_wave_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_scatter_wave_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_wave_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_radix_dxil DEPENDS "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_DXIL}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_DXIL}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_DXIL}" "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_DXIL}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_DXIL}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_DXIL}")
    add_executable(fyzzxyzzy_physx_source_sap_radix_native_d3d12 tests/physx_gpu_source_sap_radix_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_sap_radix_native_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_sap_radix_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_radix_native_d3d12 PRIVATE fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_source_sap_radix_native_d3d12 fyzzxyzzy_source_sap_radix_dxil)
    add_test(NAME fyzzxyzzy_physx_source_sap_radix_native_d3d12 COMMAND fyzzxyzzy_physx_source_sap_radix_native_d3d12 "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_DXIL}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_DXIL}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_DXIL}" "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_DXIL}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_DXIL}" "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_sap_radix_native_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    # Source incremental comparison prefixes publish into the original GPU descriptor.
    set(FYZZXYZZY_SOURCE_SAP_INCREMENTAL_FIRST_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_counts_first_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_FIRST_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_FIRST_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_first_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_INCREMENTAL_SECOND_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_counts_second_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_SECOND_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_SECOND_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_second_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_incremental_counts_dxil DEPENDS "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_FIRST_DXIL}" "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_SECOND_DXIL}")
    add_executable(fyzzxyzzy_physx_source_sap_incremental_counts_native_d3d12 tests/physx_gpu_source_sap_incremental_counts_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_sap_incremental_counts_native_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_sap_incremental_counts_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_incremental_counts_native_d3d12 PRIVATE fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_source_sap_incremental_counts_native_d3d12 fyzzxyzzy_source_sap_incremental_counts_dxil)
    add_test(NAME fyzzxyzzy_physx_source_sap_incremental_counts_native_d3d12 COMMAND fyzzxyzzy_physx_source_sap_incremental_counts_native_d3d12 "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_FIRST_DXIL}" "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_SECOND_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_sap_incremental_counts_native_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    # Original two 32x3x256 SAP endpoint histogram launches (48 source views).
    set(FYZZXYZZY_SOURCE_SAP_HIST_FIRST_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_endpoint_histograms_first_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_HIST_FIRST_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_HIST_FIRST_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_first_native.hlsl" VERBATIM)
    set(FYZZXYZZY_SOURCE_SAP_HIST_SECOND_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_endpoint_histograms_second_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_HIST_SECOND_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_HIST_SECOND_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_second_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_hist_dxil DEPENDS "${FYZZXYZZY_SOURCE_SAP_HIST_FIRST_DXIL}" "${FYZZXYZZY_SOURCE_SAP_HIST_SECOND_DXIL}")
    add_executable(fyzzxyzzy_physx_source_sap_endpoint_histograms_native_d3d12 tests/physx_gpu_source_sap_endpoint_histograms_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_sap_endpoint_histograms_native_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_sap_endpoint_histograms_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_endpoint_histograms_native_d3d12 PRIVATE fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_source_sap_endpoint_histograms_native_d3d12 fyzzxyzzy_source_sap_hist_dxil)
    add_test(NAME fyzzxyzzy_physx_source_sap_endpoint_histograms_native_d3d12 COMMAND fyzzxyzzy_physx_source_sap_endpoint_histograms_native_d3d12 "${FYZZXYZZY_SOURCE_SAP_HIST_FIRST_DXIL}" "${FYZZXYZZY_SOURCE_SAP_HIST_SECOND_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_sap_endpoint_histograms_native_d3d12
      PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1" TIMEOUT 120)

    # Source BP_INCREMENTAL_SAP consumes original per-axis comparisons
    # and writes original GPU PxgBroadPhaseDesc found/lost report indices.
    set(FYZZXYZZY_SOURCE_SAP_INCREMENTAL_PAIRS_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_pairs_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_PAIRS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_PAIRS_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_pairs_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_pairs_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_incremental_pairs_dxil DEPENDS "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_PAIRS_DXIL}")
    add_executable(fyzzxyzzy_physx_source_sap_incremental_pairs_native_d3d12 tests/physx_gpu_source_sap_incremental_pairs_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_sap_incremental_pairs_native_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_sap_incremental_pairs_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_incremental_pairs_native_d3d12 PRIVATE fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_source_sap_incremental_pairs_native_d3d12 fyzzxyzzy_source_sap_incremental_pairs_dxil)
    add_test(NAME fyzzxyzzy_physx_source_sap_incremental_pairs_native_d3d12 COMMAND fyzzxyzzy_physx_source_sap_incremental_pairs_native_d3d12 "${FYZZXYZZY_SOURCE_SAP_INCREMENTAL_PAIRS_DXIL}")
    set_tests_properties(fyzzxyzzy_physx_source_sap_incremental_pairs_native_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1" TIMEOUT 120)







    
    # PhysX BP_CLEAR_NEWFLAG on the original seven GPU allocations.
    set(FYZZXYZZY_SOURCE_CLEAR_NEW_D3D12 "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_clear_new_flags_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_CLEAR_NEW_D3D12}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_CLEAR_NEW_D3D12}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_clear_new_flags_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_clear_new_flags_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_clear_new_flags_dxil DEPENDS "${FYZZXYZZY_SOURCE_CLEAR_NEW_D3D12}")
    add_executable(fyzzxyzzy_physx_source_clear_new_flags_native_d3d12
      tests/physx_gpu_source_clear_new_flags_native.cpp)
    target_compile_definitions(fyzzxyzzy_physx_source_clear_new_flags_native_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_source_clear_new_flags_native_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_source_clear_new_flags_native_d3d12 PRIVATE fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_physx_source_clear_new_flags_native_d3d12 fyzzxyzzy_source_clear_new_flags_dxil)
    add_test(NAME fyzzxyzzy_physx_source_clear_new_flags_native_d3d12 COMMAND fyzzxyzzy_physx_source_clear_new_flags_native_d3d12 "${FYZZXYZZY_SOURCE_CLEAR_NEW_D3D12}")
    set_tests_properties(fyzzxyzzy_physx_source_clear_new_flags_native_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    
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
    set(FYZZXYZZY_SOURCE_REPORT_NATIVE_DXIL_OUTPUTS "")
    foreach(_native_report IN ITEMS first second copy first_wave second_wave)
      if(_native_report MATCHES "wave")
        set(_native_shader_model cs_6_6)
      else()
        set(_native_shader_model cs_6_0)
      endif()
      set(_native_source
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_report_${_native_report}_native.hlsl")
      set(_native_binary
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_report_${_native_report}_native.dxil")
      add_custom_command(
        OUTPUT "${_native_binary}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T "${_native_shader_model}" -E main
          -Fo "${_native_binary}" "${_native_source}"
        DEPENDS "${_native_source}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_report_multi_root_common.hlsli"
        VERBATIM)
      list(APPEND FYZZXYZZY_SOURCE_REPORT_NATIVE_DXIL_OUTPUTS "${_native_binary}")
    endforeach()
    add_custom_target(fyzzxyzzy_physx_source_report_multi_root_dxil
      DEPENDS ${FYZZXYZZY_SOURCE_REPORT_NATIVE_DXIL_OUTPUTS})

    add_executable(fyzzxyzzy_physx_bp_report_classification_d3d12
      tests/physx_gpu_broadphase_report_classification.cpp)
    target_compile_definitions(fyzzxyzzy_physx_bp_report_classification_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_bp_report_classification_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
         "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
    target_link_libraries(fyzzxyzzy_physx_bp_report_classification_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_original_gpu_heap)
    add_dependencies(fyzzxyzzy_physx_bp_report_classification_d3d12
      fyzzxyzzy_d3d12_bp_report_class_first_shader
      fyzzxyzzy_d3d12_bp_report_class_second_shader
      fyzzxyzzy_d3d12_bp_report_class_first_wave_shader
      fyzzxyzzy_d3d12_bp_report_class_second_wave_shader
      fyzzxyzzy_d3d12_bp_copy_reports_shader
      fyzzxyzzy_physx_source_report_multi_root_dxil)
    add_test(NAME fyzzxyzzy_physx_bp_report_classification_d3d12
      COMMAND fyzzxyzzy_physx_bp_report_classification_d3d12
        "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_first_DXIL}"
        "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_second_DXIL}"
        "${FYZZXYZZY_D3D12_BP_COPY_REPORTS_DXIL}"
        "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_first_WAVE_DXIL}"
        "${FYZZXYZZY_D3D12_BP_REPORT_CLASS_second_WAVE_DXIL}"
        ${FYZZXYZZY_SOURCE_REPORT_NATIVE_DXIL_OUTPUTS})

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
    foreach(_native_pass IN ITEMS writeout compute_wave compute_portable output)
      if(_native_pass STREQUAL "compute_wave")
        set(_native_source compute)
        set(_native_flag 1)
        set(_native_profile cs_6_6)
      elseif(_native_pass STREQUAL "compute_portable")
        set(_native_source compute)
        set(_native_flag 0)
        set(_native_profile cs_6_0)
      else()
        set(_native_source ${_native_pass})
        set(_native_flag 0)
        set(_native_profile cs_6_0)
      endif()
      set(FYZZXYZZY_SOURCE_ACTIVE_${_native_pass}_DXIL
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_${_native_pass}.dxil")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_SOURCE_ACTIVE_${_native_pass}_DXIL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T ${_native_profile} -E main
          -DFYZZXYZZY_ACTIVE_REGION_WAVE=${_native_flag}
          -Fo "${FYZZXYZZY_SOURCE_ACTIVE_${_native_pass}_DXIL}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_active_${_native_source}_native.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_active_${_native_source}_native.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_physx_source_active_${_native_pass}_dxil
        DEPENDS "${FYZZXYZZY_SOURCE_ACTIVE_${_native_pass}_DXIL}")
    endforeach()
    add_executable(fyzzxyzzy_physx_sap_active_region_lists_d3d12
      tests/physx_gpu_sap_active_region_lists.cpp)
    target_compile_definitions(fyzzxyzzy_physx_sap_active_region_lists_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12)
    target_include_directories(fyzzxyzzy_physx_sap_active_region_lists_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include")
    target_link_libraries(fyzzxyzzy_physx_sap_active_region_lists_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12 fyzzxyzzy_physx_cuda_host_address)
    add_dependencies(fyzzxyzzy_physx_sap_active_region_lists_d3d12
      fyzzxyzzy_d3d12_sap_active_writeout_shader
      fyzzxyzzy_d3d12_sap_active_compute_wave_shader
        fyzzxyzzy_d3d12_sap_active_compute_portable_shader
      fyzzxyzzy_d3d12_sap_active_output_shader
      fyzzxyzzy_physx_source_active_writeout_dxil
      fyzzxyzzy_physx_source_active_compute_wave_dxil
      fyzzxyzzy_physx_source_active_compute_portable_dxil
      fyzzxyzzy_physx_source_active_output_dxil)
    add_test(NAME fyzzxyzzy_physx_sap_active_region_lists_d3d12
      COMMAND fyzzxyzzy_physx_sap_active_region_lists_d3d12
        "${FYZZXYZZY_D3D12_SAP_ACTIVE_writeout_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_ACTIVE_compute_wave_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_ACTIVE_compute_portable_DXIL}"
        "${FYZZXYZZY_D3D12_SAP_ACTIVE_output_DXIL}"
        "${FYZZXYZZY_SOURCE_ACTIVE_writeout_DXIL}"
        "${FYZZXYZZY_SOURCE_ACTIVE_compute_wave_DXIL}"
        "${FYZZXYZZY_SOURCE_ACTIVE_compute_portable_DXIL}"
        "${FYZZXYZZY_SOURCE_ACTIVE_output_DXIL}")

    set(FYZZXYZZY_SAP_OVERLAP_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_work.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SAP_OVERLAP_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SAP_OVERLAP_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_arena.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_d3d12_sap_overlap_shader DEPENDS "${FYZZXYZZY_SAP_OVERLAP_DXIL}")
    add_executable(fyzzxyzzy_sap_overlap_work_d3d12 tests/physx_gpu_sap_overlap_work.cpp)
    target_compile_definitions(fyzzxyzzy_sap_overlap_work_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_sap_overlap_work_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_sap_overlap_work_d3d12 PRIVATE fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_sap_overlap_work_d3d12 fyzzxyzzy_d3d12_sap_overlap_shader)
    add_test(NAME fyzzxyzzy_sap_overlap_work_d3d12
      COMMAND fyzzxyzzy_sap_overlap_work_d3d12 "${FYZZXYZZY_SAP_OVERLAP_DXIL}")

    foreach(_pass IN ITEMS first second)
      set(FYZZXYZZY_OVERLAP_PREFIX_${_pass}_DXIL
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_prefix_${_pass}.dxil")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_OVERLAP_PREFIX_${_pass}_DXIL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T cs_6_6 -E main
          -Fo "${FYZZXYZZY_OVERLAP_PREFIX_${_pass}_DXIL}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_${_pass}_wave.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_${_pass}_wave.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_overlap_prefix_${_pass}_dxil
        DEPENDS "${FYZZXYZZY_OVERLAP_PREFIX_${_pass}_DXIL}")
    endforeach()
    add_executable(fyzzxyzzy_sap_overlap_prefix_d3d12
      tests/physx_gpu_sap_overlap_prefix.cpp)
    target_compile_definitions(fyzzxyzzy_sap_overlap_prefix_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_sap_overlap_prefix_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_sap_overlap_prefix_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_sap_overlap_prefix_d3d12
      fyzzxyzzy_overlap_prefix_first_dxil fyzzxyzzy_overlap_prefix_second_dxil)
    add_test(NAME fyzzxyzzy_sap_overlap_prefix_d3d12
      COMMAND fyzzxyzzy_sap_overlap_prefix_d3d12
        "${FYZZXYZZY_OVERLAP_PREFIX_first_DXIL}"
        "${FYZZXYZZY_OVERLAP_PREFIX_second_DXIL}")
    set_tests_properties(fyzzxyzzy_sap_overlap_prefix_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77)
    set(FYZZXYZZY_NEWBOUNDS_PAIRS_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_newbounds_pairs.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_NEWBOUNDS_PAIRS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_6 -E main
        -Fo "${FYZZXYZZY_NEWBOUNDS_PAIRS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_arena.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_arena.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_newbounds_pairs_dxil
      DEPENDS "${FYZZXYZZY_NEWBOUNDS_PAIRS_DXIL}")
    add_executable(fyzzxyzzy_sap_newbounds_pairs_d3d12
      tests/physx_gpu_sap_newbounds_pairs.cpp)
    target_compile_definitions(fyzzxyzzy_sap_newbounds_pairs_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_sap_newbounds_pairs_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_sap_newbounds_pairs_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_sap_newbounds_pairs_d3d12
      fyzzxyzzy_newbounds_pairs_dxil)
    add_test(NAME fyzzxyzzy_sap_newbounds_pairs_d3d12
      COMMAND fyzzxyzzy_sap_newbounds_pairs_d3d12
        "${FYZZXYZZY_NEWBOUNDS_PAIRS_DXIL}")
    set_tests_properties(fyzzxyzzy_sap_newbounds_pairs_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77)
    add_executable(fyzzxyzzy_sap_newbounds_sequence_d3d12
      tests/physx_gpu_sap_newbounds_sequence.cpp)
    target_compile_definitions(fyzzxyzzy_sap_newbounds_sequence_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_sap_newbounds_sequence_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_sap_newbounds_sequence_d3d12 PRIVATE
      fyzzxyzzy_physx_gpu_stages_d3d12)
    add_dependencies(fyzzxyzzy_sap_newbounds_sequence_d3d12
      fyzzxyzzy_d3d12_sap_overlap_shader
      fyzzxyzzy_overlap_prefix_first_dxil
      fyzzxyzzy_overlap_prefix_second_dxil
      fyzzxyzzy_newbounds_pairs_dxil)
    add_test(NAME fyzzxyzzy_sap_newbounds_sequence_d3d12
      COMMAND fyzzxyzzy_sap_newbounds_sequence_d3d12
        "${FYZZXYZZY_SAP_OVERLAP_DXIL}"
        "${FYZZXYZZY_OVERLAP_PREFIX_first_DXIL}"
        "${FYZZXYZZY_OVERLAP_PREFIX_second_DXIL}"
        "${FYZZXYZZY_NEWBOUNDS_PAIRS_DXIL}")
    set_tests_properties(fyzzxyzzy_sap_newbounds_sequence_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77)



  endif()
  endif() # source-port GPU stages and source-based conformance tests
endif()

# Godot is a first SDK consumer, never a dependency of FyzzXYZZY::Core.
option(FYZZXYZZY_BUILD_GODOT "Build the optional Godot GDExtension consumer" OFF)
if(FYZZXYZZY_BUILD_GODOT)
  if(NOT FYZZXYZZY_ENABLE_PHYSX_SOURCE)
    message(FATAL_ERROR
      "Godot physics acceptance still requires the transitional PhysX CPU reference; "
      "native FyzzXYZZY scene is not yet implemented")
  endif()
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GodotDependency.cmake")

  add_library(fyzzxyzzy_godot SHARED
    godot/src/fyzzxyzzy_godot_world.cpp
    godot/src/register_types.cpp
  )
  target_compile_features(fyzzxyzzy_godot PRIVATE cxx_std_23)
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
if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
add_custom_target(fyzzxyzzy_source_bundle
  COMMAND "${CMAKE_COMMAND}"
    "-DFYZZXYZZY_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DFYZZXYZZY_PHYSX_ROOT=${fyzzxyzzy_physx_SOURCE_DIR}"
    "-DFYZZXYZZY_BUNDLE_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/fyzzxyzzy-source.tar.gz"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package-source.cmake"
  BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/fyzzxyzzy-source.tar.gz"
  VERBATIM
)
endif() # PhysX reference-inclusive source bundle
