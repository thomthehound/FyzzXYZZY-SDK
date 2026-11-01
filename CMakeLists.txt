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
# Exercise source-owned GPU DMA/bitmap kernels without the upstream checkout.
option(FYZZXYZZY_BUILD_SOURCE_GPU_TESTS
  "Test independent source GPU AABB encoding and DMA (requires DXC)" OFF)

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

# The original PhysX broadphase filter is independent of GPU API and the
# reference PhysX binary. Keep its source code and pair-policy regression
# in both independent Vulkan and D3D12 builds, not the CPU oracle.
if(BUILD_TESTING AND NOT FYZZXYZZY_ENABLE_PHYSX_SOURCE)
  add_library(fyzzxyzzy_physx_source_bp_filter STATIC
    src/physx_port/lowlevelaabb/src/BpFiltering.cpp)
  target_compile_features(fyzzxyzzy_physx_source_bp_filter PUBLIC cxx_std_23)
  target_include_directories(fyzzxyzzy_physx_source_bp_filter PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
  add_executable(fyzzxyzzy_physx_source_bp_filter_test
    tests/physx_source_bp_filter_independent.cpp)
  target_include_directories(fyzzxyzzy_physx_source_bp_filter_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
  target_link_libraries(fyzzxyzzy_physx_source_bp_filter_test PRIVATE
    fyzzxyzzy_physx_source_bp_filter)
  add_test(NAME fyzzxyzzy_physx_source_bp_filter_test
    COMMAND fyzzxyzzy_physx_source_bp_filter_test)
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
  src/physx_port/gpubroadphase/PxgSapSourceMembersNative.cpp
  src/physx_port/gpubroadphase/PxgSapUpdateFetchNative.cpp)
target_compile_features(fyzzxyzzy_physx_original_gpu_heap PUBLIC cxx_std_23)
target_include_directories(fyzzxyzzy_physx_original_gpu_heap PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port/gpucommon")
target_link_libraries(fyzzxyzzy_physx_original_gpu_heap PUBLIC physx_lib)
else()
# Source-owned original PhysX Foundation and GPU buddy heap, independent
# of the reference PhysX checkout and CUDA. Both implement the original
# allocation and synchronization contracts.
set(FYZZXYZZY_OWNED_FOUNDATION_SOURCES
  src/physx_port/foundation_source/FdAllocator.cpp
  src/physx_port/foundation_source/FdAssert.cpp
  src/physx_port/foundation_source/FdFoundation.cpp
  src/physx_port/foundation_source/FdMathUtils.cpp
  src/physx_port/foundation_source/FdString.cpp
  src/physx_port/foundation_source/FdTempAllocator.cpp)
if(WIN32)
  list(APPEND FYZZXYZZY_OWNED_FOUNDATION_SOURCES
    src/physx_port/foundation_source/windows/FdWindowsAtomic.cpp
    src/physx_port/foundation_source/windows/FdWindowsFPU.cpp
    src/physx_port/foundation_source/windows/FdWindowsMutex.cpp
    src/physx_port/foundation_source/windows/FdWindowsPrintString.cpp
    src/physx_port/foundation_source/windows/FdWindowsSList.cpp
    src/physx_port/foundation_source/windows/FdWindowsSocket.cpp
    src/physx_port/foundation_source/windows/FdWindowsSync.cpp
    src/physx_port/foundation_source/windows/FdWindowsThread.cpp
    src/physx_port/foundation_source/windows/FdWindowsTime.cpp)
else()
  list(APPEND FYZZXYZZY_OWNED_FOUNDATION_SOURCES
    src/physx_port/foundation_source/unix/FdUnixAtomic.cpp
    src/physx_port/foundation_source/unix/FdUnixFPU.cpp
    src/physx_port/foundation_source/unix/FdUnixMutex.cpp
    src/physx_port/foundation_source/unix/FdUnixPrintString.cpp
    src/physx_port/foundation_source/unix/FdUnixSList.cpp
    src/physx_port/foundation_source/unix/FdUnixSocket.cpp
    src/physx_port/foundation_source/unix/FdUnixSync.cpp
    src/physx_port/foundation_source/unix/FdUnixThread.cpp
    src/physx_port/foundation_source/unix/FdUnixTime.cpp)
endif()
add_library(fyzzxyzzy_physx_source_foundation STATIC
  ${FYZZXYZZY_OWNED_FOUNDATION_SOURCES})
target_compile_features(fyzzxyzzy_physx_source_foundation PUBLIC cxx_std_23)
target_compile_definitions(fyzzxyzzy_physx_source_foundation PUBLIC
  PX_PHYSX_STATIC_LIB)
target_include_directories(fyzzxyzzy_physx_source_foundation
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port/foundation_source")
target_link_libraries(fyzzxyzzy_physx_source_foundation PUBLIC Threads::Threads)

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
  src/physx_port/gpubroadphase/PxgSapSourceMembersNative.cpp
  src/physx_port/gpubroadphase/PxgSapUpdateFetchNative.cpp
  src/physx_port/lowlevelaabb/src/BpFiltering.cpp)
target_compile_features(fyzzxyzzy_physx_original_gpu_heap PUBLIC cxx_std_23)
target_include_directories(fyzzxyzzy_physx_original_gpu_heap PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port/gpucommon")
target_link_libraries(fyzzxyzzy_physx_original_gpu_heap PUBLIC
  fyzzxyzzy_physx_source_foundation)
endif() # source-port PhysX heap and SAP host fixtures

# One process-wide source Foundation, shared by reference CPU Physics and
# the native source GPU allocator. It owns no PhysX scene or GPU device.
add_library(fyzzxyzzy_physx_foundation_lifetime STATIC
  src/physx_foundation_lifetime.cpp)
target_compile_features(fyzzxyzzy_physx_foundation_lifetime PUBLIC cxx_std_23)
target_include_directories(fyzzxyzzy_physx_foundation_lifetime
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
  target_link_libraries(fyzzxyzzy_physx_foundation_lifetime PUBLIC physx_lib)
  target_link_libraries(fyzzxyzzy_reference_physx_cpu PRIVATE
    fyzzxyzzy_physx_foundation_lifetime)
else()
  target_link_libraries(fyzzxyzzy_physx_foundation_lifetime PUBLIC
    fyzzxyzzy_physx_source_foundation)
endif()
set_target_properties(fyzzxyzzy_physx_foundation_lifetime
  PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(BUILD_TESTING AND FYZZXYZZY_ENABLE_PHYSX_SOURCE)
  find_package(Threads REQUIRED)
  add_executable(fyzzxyzzy_physx_foundation_shared_lifetime
    tests/physx_foundation_shared_lifetime.cpp)
  target_link_libraries(fyzzxyzzy_physx_foundation_shared_lifetime PRIVATE
    FyzzXYZZY::Core fyzzxyzzy_physx_foundation_lifetime Threads::Threads)
  add_test(NAME fyzzxyzzy_physx_foundation_shared_lifetime
    COMMAND fyzzxyzzy_physx_foundation_shared_lifetime)
endif()

# Native compute resources and operations do not depend on Godot or PhysX.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/NativeGpuLibraries.cmake")
option(FYZZXYZZY_BUILD_VULKAN "Build the standalone Vulkan compute backend" OFF)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VulkanBackend.cmake")

# Native D3D12 execution of PhysX's AABB stage, using the exact same
# AabbEncodingStage host dispatch and reference fixtures as Vulkan.
# The initial executor supports this kernel's 3 SRV + 1 UAV/8-constant ABI.
# It remains a separate static target pending backend-neutral runtime dispatch.
option(FYZZXYZZY_BUILD_D3D12 "Build the native Windows D3D12 compute backend" OFF)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/D3D12Backend.cmake")

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
