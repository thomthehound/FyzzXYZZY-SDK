# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

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

  # Same portable source operation on native D3D12, no PhysX reference.
  add_library(fyzzxyzzy_source_gpu_primitives_d3d12 STATIC
    src/physx_gpu/source_fill_d32_native.cpp
    src/physx_gpu/aabb_encoding.cpp
    src/physx_gpu/cuda_aabb_launch_lowering.cpp
    src/physx_gpu/source_overlap_work_native.cpp
    src/physx_gpu/source_overlap_prefix_native.cpp
    src/physx_gpu/source_newbounds_pairs_native.cpp
    src/physx_gpu/source_newbounds_sequence_native.cpp
    src/physx_gpu/source_sap_created_native.cpp
    src/physx_gpu/source_sap_incremental_counts_native.cpp
    src/physx_gpu/source_sap_endpoint_histograms_native.cpp
    src/physx_gpu/source_sap_incremental_pairs_native.cpp
    src/physx_gpu/source_sap_updated_native.cpp
    src/physx_gpu/source_sap_removed_native.cpp
    src/physx_gpu/source_sapbox_indices_native.cpp
    src/physx_gpu/source_sap_radix_native.cpp
    src/physx_gpu/source_region_create_native.cpp
    src/physx_gpu/source_region_histograms_native.cpp
    src/physx_gpu/source_active_region_lists_native.cpp
    src/physx_gpu/source_clear_new_flags_native.cpp
    src/physx_gpu/source_sap_report_multi_root.cpp
    src/physx_gpu/source_sap_update_native.cpp
    src/physx_gpu/source_sap_scene_frame_native.cpp)
  target_compile_features(fyzzxyzzy_source_gpu_primitives_d3d12 PUBLIC cxx_std_23)
  target_include_directories(fyzzxyzzy_source_gpu_primitives_d3d12 PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(fyzzxyzzy_source_gpu_primitives_d3d12 PUBLIC
    FyzzXYZZY::ComputeD3D12 fyzzxyzzy_physx_cuda_host_address)

  add_library(fyzzxyzzy_native_gpu_world_d3d12 STATIC
    src/physx_gpu/native_gpu_world.cpp)
  add_library(FyzzXYZZY::NativeGpuWorldD3D12 ALIAS
    fyzzxyzzy_native_gpu_world_d3d12)
  target_compile_features(fyzzxyzzy_native_gpu_world_d3d12 PUBLIC cxx_std_23)
  target_include_directories(fyzzxyzzy_native_gpu_world_d3d12
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
  target_link_libraries(fyzzxyzzy_native_gpu_world_d3d12 PUBLIC
    FyzzXYZZY::Core fyzzxyzzy_physx_original_gpu_heap
    fyzzxyzzy_source_gpu_primitives_d3d12
    fyzzxyzzy_physx_foundation_lifetime)
  set_target_properties(fyzzxyzzy_native_gpu_world_d3d12
    PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if(BUILD_TESTING AND NOT FYZZXYZZY_ENABLE_PHYSX_SOURCE
     AND FYZZXYZZY_BUILD_SOURCE_GPU_TESTS)
    find_program(FYZZXYZZY_DXC NAMES dxc REQUIRED)
    set(FYZZXYZZY_WAVE32_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/native_wave32_contract.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_WAVE32_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_6 -E main
        -Fo "${FYZZXYZZY_WAVE32_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_wave32_contract.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_wave32_contract.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_wave32_dxil DEPENDS "${FYZZXYZZY_WAVE32_DXIL}")
    add_executable(fyzzxyzzy_wave32_contract_d3d12 tests/native_wave32_contract.cpp)
    target_compile_definitions(fyzzxyzzy_wave32_contract_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_link_libraries(fyzzxyzzy_wave32_contract_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12)
    add_dependencies(fyzzxyzzy_wave32_contract_d3d12 fyzzxyzzy_wave32_dxil)
    add_test(NAME fyzzxyzzy_wave32_contract_d3d12
      COMMAND fyzzxyzzy_wave32_contract_d3d12 "${FYZZXYZZY_WAVE32_DXIL}")

    set(FYZZXYZZY_INDEPENDENT_D32_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_fill_d32_native.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_INDEPENDENT_D32_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_INDEPENDENT_D32_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_fill_d32_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_fill_d32_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_independent_source_d32_dxil
      DEPENDS "${FYZZXYZZY_INDEPENDENT_D32_DXIL}")
    add_executable(fyzzxyzzy_source_gpu_d32_independent_d3d12
      tests/physx_gpu_source_fill_d32_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_d32_independent_d3d12
      PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_d32_independent_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_d32_independent_d3d12 PRIVATE
      fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_d32_independent_d3d12
      fyzzxyzzy_independent_source_d32_dxil)
    add_test(NAME fyzzxyzzy_source_gpu_d32_independent_d3d12
      COMMAND fyzzxyzzy_source_gpu_d32_independent_d3d12
        "${FYZZXYZZY_INDEPENDENT_D32_DXIL}")
    set_tests_properties(fyzzxyzzy_source_gpu_d32_independent_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77 ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    set(FYZZXYZZY_INDEPENDENT_AABB_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_translate_aabbs.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_INDEPENDENT_AABB_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_INDEPENDENT_AABB_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_independent_source_aabb_dxil
      DEPENDS "${FYZZXYZZY_INDEPENDENT_AABB_DXIL}")
    add_executable(fyzzxyzzy_source_gpu_aabb_independent_d3d12
      tests/physx_gpu_aabb_encoding.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_aabb_independent_d3d12
      PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_aabb_independent_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_aabb_independent_d3d12 PRIVATE
      fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_aabb_independent_d3d12
      fyzzxyzzy_independent_source_aabb_dxil)
    add_test(NAME fyzzxyzzy_source_gpu_aabb_independent_d3d12
      COMMAND fyzzxyzzy_source_gpu_aabb_independent_d3d12
        "${FYZZXYZZY_INDEPENDENT_AABB_DXIL}")
    set_tests_properties(fyzzxyzzy_source_gpu_aabb_independent_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    # Original PxgBroadPhaseDesc is now SDK-owned, not an upstream fixture.
    set(FYZZXYZZY_OWNED_OVERLAP_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_work_native.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_OWNED_OVERLAP_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_OVERLAP_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_SCAN_FIRST_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_prefix_first_native.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_OWNED_SCAN_FIRST_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_SCAN_FIRST_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_first_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_SCAN_SECOND_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_prefix_second_native.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_OWNED_SCAN_SECOND_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_SCAN_SECOND_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_second_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_PAIRS_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_newbounds_pairs_wave_native.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_OWNED_PAIRS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_PAIRS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_owned_source_sap_shaders_d3d12 DEPENDS
      ${FYZZXYZZY_OWNED_OVERLAP_DXIL}
      ${FYZZXYZZY_OWNED_SCAN_FIRST_DXIL}
      ${FYZZXYZZY_OWNED_SCAN_SECOND_DXIL}
      ${FYZZXYZZY_OWNED_PAIRS_DXIL})
    add_executable(fyzzxyzzy_source_gpu_overlap_independent_d3d12 tests/physx_gpu_source_overlap_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_overlap_independent_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_overlap_independent_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_overlap_independent_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_overlap_independent_d3d12 fyzzxyzzy_owned_source_sap_shaders_d3d12)
    add_test(NAME fyzzxyzzy_source_gpu_overlap_independent_d3d12 COMMAND fyzzxyzzy_source_gpu_overlap_independent_d3d12
      "${FYZZXYZZY_OWNED_OVERLAP_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_source_gpu_overlap_independent_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_source_gpu_scan_independent_d3d12 tests/physx_gpu_source_overlap_prefix_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_scan_independent_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_scan_independent_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_scan_independent_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_scan_independent_d3d12 fyzzxyzzy_owned_source_sap_shaders_d3d12)
    add_test(NAME fyzzxyzzy_source_gpu_scan_independent_d3d12 COMMAND fyzzxyzzy_source_gpu_scan_independent_d3d12
      "${FYZZXYZZY_OWNED_SCAN_FIRST_DXIL}"
      "${FYZZXYZZY_OWNED_SCAN_SECOND_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_source_gpu_scan_independent_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77 ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_source_gpu_pairs_independent_d3d12 tests/physx_gpu_source_newbounds_pairs_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_pairs_independent_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_pairs_independent_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_pairs_independent_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_pairs_independent_d3d12 fyzzxyzzy_owned_source_sap_shaders_d3d12)
    add_test(NAME fyzzxyzzy_source_gpu_pairs_independent_d3d12 COMMAND fyzzxyzzy_source_gpu_pairs_independent_d3d12
      "${FYZZXYZZY_OWNED_PAIRS_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_source_gpu_pairs_independent_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77 ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    # Source-owned SAP incremental/created GPU work uses the original
    # PxgBroadPhaseDesc and native backend, without a reference PhysX ABI.
    set(FYZZXYZZY_OWNED_CREATED_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_created_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_CREATED_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_CREATED_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_created_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_created_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_COUNT_FIRST_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_counts_first_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_COUNT_FIRST_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_COUNT_FIRST_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_first_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_COUNT_SECOND_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_counts_second_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_COUNT_SECOND_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_COUNT_SECOND_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_second_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_HIST_FIRST_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_endpoint_histograms_first_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_first_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_HIST_SECOND_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_endpoint_histograms_second_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_second_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_pairs_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_pairs_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_pairs_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_owned_sap_incremental_shaders_d3d12 DEPENDS
      ${FYZZXYZZY_OWNED_CREATED_DXIL}
      ${FYZZXYZZY_OWNED_COUNT_FIRST_DXIL}
      ${FYZZXYZZY_OWNED_COUNT_SECOND_DXIL}
      ${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}
      ${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}
      ${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_DXIL}
    )
    add_executable(fyzzxyzzy_owned_sap_created_d3d12 tests/physx_gpu_source_sap_created_native.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_created_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_created_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_created_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_created_d3d12 fyzzxyzzy_owned_sap_incremental_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_created_d3d12 COMMAND fyzzxyzzy_owned_sap_created_d3d12
      "${FYZZXYZZY_OWNED_CREATED_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_owned_sap_created_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_owned_sap_counts_d3d12 tests/physx_gpu_source_sap_incremental_counts_native.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_counts_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_counts_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_counts_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_counts_d3d12 fyzzxyzzy_owned_sap_incremental_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_counts_d3d12 COMMAND fyzzxyzzy_owned_sap_counts_d3d12
      "${FYZZXYZZY_OWNED_COUNT_FIRST_DXIL}"
      "${FYZZXYZZY_OWNED_COUNT_SECOND_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_owned_sap_counts_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_owned_sap_hist_d3d12 tests/physx_gpu_source_sap_endpoint_histograms_native.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_hist_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_hist_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_hist_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_hist_d3d12 fyzzxyzzy_owned_sap_incremental_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_hist_d3d12 COMMAND fyzzxyzzy_owned_sap_hist_d3d12
      "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}"
      "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_owned_sap_hist_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77 ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_owned_sap_incremental_pairs_d3d12 tests/physx_gpu_source_sap_incremental_pairs_native.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_incremental_pairs_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_incremental_pairs_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_incremental_pairs_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_incremental_pairs_d3d12 fyzzxyzzy_owned_sap_incremental_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_incremental_pairs_d3d12 COMMAND fyzzxyzzy_owned_sap_incremental_pairs_d3d12
      "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_owned_sap_incremental_pairs_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77 ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    # Original SAP rank, projection, changed and removed handle kernels.
    # All layout declarations and stage implementations belong to this SDK.
    set(FYZZXYZZY_OWNED_SAPBOX_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sapbox_indices_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_SAPBOX_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_SAPBOX_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sapbox_indices_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sapbox_indices_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_REMOVED_MARK_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_mark_removed_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REMOVED_MARK_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REMOVED_MARK_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_mark_removed_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_mark_removed_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_REMOVED_PROJ_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_remove_projections_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REMOVED_PROJ_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REMOVED_PROJ_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_remove_projections_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_remove_projections_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_UPDATED_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_updated_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_UPDATED_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_UPDATED_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_updated_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_updated_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_INIT_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_init_ranks_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_init_ranks_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_init_ranks_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_COUNT_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_count_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_scatter_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_GATHER_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_gather_handles_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_gather_handles_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_gather_handles_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_COUNT_WAVE_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_count_wave_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_COUNT_WAVE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_COUNT_WAVE_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_wave_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_SCATTER_WAVE_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_scatter_wave_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_SCATTER_WAVE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_SCATTER_WAVE_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_wave_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_owned_sap_sort_shaders_d3d12 DEPENDS
      ${FYZZXYZZY_OWNED_SAPBOX_DXIL}
      ${FYZZXYZZY_OWNED_REMOVED_MARK_DXIL}
      ${FYZZXYZZY_OWNED_REMOVED_PROJ_DXIL}
      ${FYZZXYZZY_OWNED_UPDATED_DXIL}
      ${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}
      ${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}
      ${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}
      ${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}
      ${FYZZXYZZY_OWNED_RADIX_COUNT_WAVE_DXIL}
      ${FYZZXYZZY_OWNED_RADIX_SCATTER_WAVE_DXIL}
    )
    add_executable(fyzzxyzzy_owned_sap_sapbox_d3d12 tests/physx_gpu_source_sapbox_indices_native.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_sapbox_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_sapbox_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_sapbox_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_sapbox_d3d12 fyzzxyzzy_owned_sap_sort_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_sapbox_d3d12 COMMAND fyzzxyzzy_owned_sap_sapbox_d3d12
      "${FYZZXYZZY_OWNED_SAPBOX_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_owned_sap_sapbox_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_owned_sap_removed_d3d12 tests/physx_gpu_source_sap_removed_native.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_removed_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_removed_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_removed_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_removed_d3d12 fyzzxyzzy_owned_sap_sort_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_removed_d3d12 COMMAND fyzzxyzzy_owned_sap_removed_d3d12
      "${FYZZXYZZY_OWNED_REMOVED_MARK_DXIL}"
      "${FYZZXYZZY_OWNED_REMOVED_PROJ_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_owned_sap_removed_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_owned_sap_updated_d3d12 tests/physx_gpu_source_sap_updated_native.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_updated_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_updated_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_updated_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_updated_d3d12 fyzzxyzzy_owned_sap_sort_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_updated_d3d12 COMMAND fyzzxyzzy_owned_sap_updated_d3d12
      "${FYZZXYZZY_OWNED_UPDATED_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_owned_sap_updated_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_owned_sap_radix_d3d12 tests/physx_gpu_source_sap_radix_native.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_radix_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_radix_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_radix_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_radix_d3d12 fyzzxyzzy_owned_sap_sort_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_radix_d3d12 COMMAND fyzzxyzzy_owned_sap_radix_d3d12
      "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}"
      "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}"
      "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}"
      "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}"
      "${FYZZXYZZY_OWNED_RADIX_COUNT_WAVE_DXIL}"
      "${FYZZXYZZY_OWNED_RADIX_SCATTER_WAVE_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_owned_sap_radix_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    # Original source GPU region create and histogram kernels.
    set(FYZZXYZZY_OWNED_REGION_CREATE_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_create_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_create_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_create_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_first_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_first_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_wave_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_wave_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_second_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_second_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_owned_region_shaders_d3d12 DEPENDS
      "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}"
    )
    add_executable(fyzzxyzzy_owned_region_histograms_d3d12 tests/physx_gpu_source_region_histograms_native.cpp)
    target_compile_definitions(fyzzxyzzy_owned_region_histograms_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_region_histograms_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_region_histograms_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_region_histograms_d3d12 fyzzxyzzy_owned_region_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_region_histograms_d3d12 COMMAND fyzzxyzzy_owned_region_histograms_d3d12
      "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_owned_region_histograms_d3d12 PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_owned_region_create_d3d12 tests/physx_gpu_source_region_create_native.cpp)
    target_compile_definitions(fyzzxyzzy_owned_region_create_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_region_create_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_region_create_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_region_create_d3d12 fyzzxyzzy_owned_region_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_region_create_d3d12 COMMAND fyzzxyzzy_owned_region_create_d3d12
      "${FYZZXYZZY_INDEPENDENT_D32_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}"
    )
    set_tests_properties(fyzzxyzzy_owned_region_create_d3d12 PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    set(FYZZXYZZY_OWNED_CLEAR_NEW_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_clear_new_flags_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_CLEAR_NEW_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_CLEAR_NEW_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_clear_new_flags_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_clear_new_flags_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_owned_clear_new_shader_d3d12
      DEPENDS "${FYZZXYZZY_OWNED_CLEAR_NEW_DXIL}")
    add_executable(fyzzxyzzy_owned_clear_new_d3d12
      tests/physx_gpu_source_clear_new_flags_native.cpp)
    target_compile_definitions(fyzzxyzzy_owned_clear_new_d3d12
      PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_clear_new_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_clear_new_d3d12 PRIVATE
      fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_clear_new_d3d12
      fyzzxyzzy_owned_clear_new_shader_d3d12)
    add_test(NAME fyzzxyzzy_owned_clear_new_d3d12
      COMMAND fyzzxyzzy_owned_clear_new_d3d12
        "${FYZZXYZZY_OWNED_CLEAR_NEW_DXIL}")
    set_tests_properties(fyzzxyzzy_owned_clear_new_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    set(FYZZXYZZY_OWNED_REPORT_DXIL_OUTPUTS "")
    foreach(_report_pass IN ITEMS first second copy first_wave second_wave)
      if(_report_pass MATCHES "wave")
        set(_report_model cs_6_6)
      else()
        set(_report_model cs_6_0)
      endif()
      set(_report_src "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_report_${_report_pass}_native.hlsl")
      set(_report_bin "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_report_${_report_pass}_native.dxil")
      add_custom_command(OUTPUT "${_report_bin}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T "${_report_model}" -E main
          -Fo "${_report_bin}" "${_report_src}"
        DEPENDS "${_report_src}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_report_multi_root_common.hlsli"
        VERBATIM)
      list(APPEND FYZZXYZZY_OWNED_REPORT_DXIL_OUTPUTS "${_report_bin}")
    endforeach()
    add_custom_target(fyzzxyzzy_owned_report_shaders_d3d12
      DEPENDS ${FYZZXYZZY_OWNED_REPORT_DXIL_OUTPUTS})
    add_executable(fyzzxyzzy_owned_reports_d3d12
      tests/physx_gpu_source_report_multi_root_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_reports_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_reports_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_reports_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_reports_d3d12 fyzzxyzzy_owned_report_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_reports_d3d12 COMMAND fyzzxyzzy_owned_reports_d3d12
      ${FYZZXYZZY_OWNED_REPORT_DXIL_OUTPUTS})
    set_tests_properties(fyzzxyzzy_owned_reports_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")

    set(FYZZXYZZY_OWNED_PAIR_PRODUCER_DXIL_FILES "")
    foreach(_pair_shader IN ITEMS
      physx_sap_overlap_work_native
      physx_sap_overlap_prefix_first_native
      physx_sap_overlap_prefix_second_native
      physx_sap_newbounds_pairs_wave_native)
      set(_source "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/${_pair_shader}.hlsl")
      set(_output "${CMAKE_CURRENT_BINARY_DIR}/kernels/${_pair_shader}_owned.dxil")
      add_custom_command(OUTPUT "${_output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
          -Fo "${_output}" "${_source}"
        DEPENDS "${_source}" VERBATIM)
      list(APPEND FYZZXYZZY_OWNED_PAIR_PRODUCER_DXIL_FILES "${_output}")
    endforeach()
    add_custom_target(fyzzxyzzy_owned_pair_producer_shaders_d3d12
      DEPENDS ${FYZZXYZZY_OWNED_PAIR_PRODUCER_DXIL_FILES})
    add_executable(fyzzxyzzy_owned_pairs_to_sc_d3d12
      tests/physx_gpu_source_pairs_to_sc_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_pairs_to_sc_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_pairs_to_sc_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_pairs_to_sc_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_pairs_to_sc_d3d12
      fyzzxyzzy_owned_pair_producer_shaders_d3d12 fyzzxyzzy_owned_report_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_pairs_to_sc_d3d12
      COMMAND fyzzxyzzy_owned_pairs_to_sc_d3d12
      ${FYZZXYZZY_OWNED_PAIR_PRODUCER_DXIL_FILES} ${FYZZXYZZY_OWNED_REPORT_DXIL_OUTPUTS})
    set_tests_properties(fyzzxyzzy_owned_pairs_to_sc_d3d12 PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 180)

    add_executable(fyzzxyzzy_owned_source_roots_d3d12
      tests/physx_gpu_source_roots_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_source_roots_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_source_roots_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_source_roots_d3d12 PRIVATE
      fyzzxyzzy_source_gpu_primitives_d3d12)
    add_test(NAME fyzzxyzzy_owned_source_roots_d3d12
      COMMAND fyzzxyzzy_owned_source_roots_d3d12)

    add_executable(fyzzxyzzy_original_gpu_heap_independent_d3d12
      tests/physx_original_gpu_heap_independent.cpp)
    target_compile_definitions(fyzzxyzzy_original_gpu_heap_independent_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_original_gpu_heap_independent_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_original_gpu_heap_independent_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_test(NAME fyzzxyzzy_original_gpu_heap_independent_d3d12
      COMMAND fyzzxyzzy_original_gpu_heap_independent_d3d12)

    add_executable(fyzzxyzzy_sap_source_members_independent_d3d12
      tests/physx_gpu_sap_source_members_independent.cpp)
    target_compile_definitions(
      fyzzxyzzy_sap_source_members_independent_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(
      fyzzxyzzy_sap_source_members_independent_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(
      fyzzxyzzy_sap_source_members_independent_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_test(NAME fyzzxyzzy_sap_source_members_independent_d3d12
      COMMAND fyzzxyzzy_sap_source_members_independent_d3d12)

    add_executable(fyzzxyzzy_sap_dma_up_independent_d3d12
      tests/physx_gpu_sap_dma_up_independent.cpp)
    target_compile_definitions(
      fyzzxyzzy_sap_dma_up_independent_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(
      fyzzxyzzy_sap_dma_up_independent_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(
      fyzzxyzzy_sap_dma_up_independent_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_test(NAME fyzzxyzzy_sap_dma_up_independent_d3d12
      COMMAND fyzzxyzzy_sap_dma_up_independent_d3d12)



    set(FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_translate_aabbs_owned.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_owned_translate_aabbs_shader_d3d12
      DEPENDS "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}")
    add_executable(fyzzxyzzy_owned_translate_aabbs_d3d12
      tests/physx_gpu_cuda_aabb_launch.cpp)
    target_compile_definitions(fyzzxyzzy_owned_translate_aabbs_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_translate_aabbs_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_translate_aabbs_d3d12 PRIVATE fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_translate_aabbs_d3d12 fyzzxyzzy_owned_translate_aabbs_shader_d3d12)
    add_test(NAME fyzzxyzzy_owned_translate_aabbs_d3d12
      COMMAND fyzzxyzzy_owned_translate_aabbs_d3d12
      "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}")
    set_tests_properties(fyzzxyzzy_owned_translate_aabbs_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_owned_sap_gpu_created_frame_d3d12
      tests/physx_gpu_sap_owned_created_frame_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_gpu_created_frame_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_created_frame_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_created_frame_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_created_frame_d3d12
      fyzzxyzzy_owned_translate_aabbs_shader_d3d12
      fyzzxyzzy_owned_sap_incremental_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_created_frame_d3d12
      COMMAND fyzzxyzzy_owned_sap_gpu_created_frame_d3d12
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
        "${FYZZXYZZY_OWNED_CREATED_DXIL}")
    set_tests_properties(fyzzxyzzy_owned_sap_gpu_created_frame_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_owned_sap_gpu_endpoint_history_d3d12
      tests/physx_gpu_sap_owned_created_radix_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_gpu_endpoint_history_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_endpoint_history_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_endpoint_history_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_endpoint_history_d3d12
      fyzzxyzzy_owned_translate_aabbs_shader_d3d12
      fyzzxyzzy_owned_sap_incremental_shaders_d3d12
      fyzzxyzzy_owned_sap_sort_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_endpoint_history_d3d12
      COMMAND fyzzxyzzy_owned_sap_gpu_endpoint_history_d3d12
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
        "${FYZZXYZZY_OWNED_CREATED_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}"
        "${FYZZXYZZY_OWNED_SAPBOX_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}")
    set_tests_properties(fyzzxyzzy_owned_sap_gpu_endpoint_history_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_owned_sap_gpu_region_prefix_d3d12
      tests/physx_gpu_sap_owned_region_prefix_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_gpu_region_prefix_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_region_prefix_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_region_prefix_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_region_prefix_d3d12
      fyzzxyzzy_owned_translate_aabbs_shader_d3d12
      fyzzxyzzy_owned_sap_incremental_shaders_d3d12
      fyzzxyzzy_owned_sap_sort_shaders_d3d12
      fyzzxyzzy_owned_region_shaders_d3d12
      fyzzxyzzy_independent_source_d32_dxil)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_region_prefix_d3d12
      COMMAND fyzzxyzzy_owned_sap_gpu_region_prefix_d3d12
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
        "${FYZZXYZZY_OWNED_CREATED_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}"
        "${FYZZXYZZY_OWNED_SAPBOX_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_INDEPENDENT_D32_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}")
    set_tests_properties(fyzzxyzzy_owned_sap_gpu_region_prefix_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    foreach(_pass IN ITEMS writeout compute_wave compute_portable output)
      if(_pass STREQUAL "compute_wave")
        set(_source compute)
        set(_flag 1)
        set(_profile cs_6_6)
      elseif(_pass STREQUAL "compute_portable")
        set(_source compute)
        set(_flag 0)
        set(_profile cs_6_0)
      else()
        set(_source ${_pass})
        set(_flag 0)
        set(_profile cs_6_0)
      endif()
      set(FYZZXYZZY_OWNED_ACTIVE_${_pass}_DXIL
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_${_pass}_owned.dxil")
      add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_ACTIVE_${_pass}_DXIL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -T ${_profile} -E main
          -DFYZZXYZZY_ACTIVE_REGION_WAVE=${_flag}
          -Fo "${FYZZXYZZY_OWNED_ACTIVE_${_pass}_DXIL}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_active_${_source}_native.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_active_${_source}_native.hlsl"
        VERBATIM)
    endforeach()
    add_custom_target(fyzzxyzzy_owned_active_shaders_d3d12 DEPENDS
      "${FYZZXYZZY_OWNED_ACTIVE_writeout_DXIL}"
      "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_DXIL}"
      "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_DXIL}"
      "${FYZZXYZZY_OWNED_ACTIVE_output_DXIL}")
    add_executable(fyzzxyzzy_owned_sap_gpu_new_pairs_d3d12
      tests/physx_gpu_sap_owned_new_pairs_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_gpu_new_pairs_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_new_pairs_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_new_pairs_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_new_pairs_d3d12
      fyzzxyzzy_owned_translate_aabbs_shader_d3d12
      fyzzxyzzy_owned_sap_incremental_shaders_d3d12
      fyzzxyzzy_owned_sap_sort_shaders_d3d12
      fyzzxyzzy_owned_region_shaders_d3d12
      fyzzxyzzy_independent_source_d32_dxil
      fyzzxyzzy_owned_active_shaders_d3d12
      fyzzxyzzy_owned_source_sap_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_new_pairs_d3d12
      COMMAND fyzzxyzzy_owned_sap_gpu_new_pairs_d3d12
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
        "${FYZZXYZZY_OWNED_CREATED_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}"
        "${FYZZXYZZY_OWNED_SAPBOX_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_INDEPENDENT_D32_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_DXIL}"
        "${FYZZXYZZY_OWNED_OVERLAP_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_PAIRS_DXIL}")
    add_executable(fyzzxyzzy_owned_sap_gpu_report_fetch_d3d12
      tests/physx_gpu_sap_owned_report_fetch_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_gpu_report_fetch_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_report_fetch_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_report_fetch_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_report_fetch_d3d12
      fyzzxyzzy_owned_translate_aabbs_shader_d3d12
      fyzzxyzzy_owned_sap_incremental_shaders_d3d12
      fyzzxyzzy_owned_sap_sort_shaders_d3d12
      fyzzxyzzy_owned_region_shaders_d3d12
      fyzzxyzzy_independent_source_d32_dxil
      fyzzxyzzy_owned_active_shaders_d3d12
      fyzzxyzzy_owned_source_sap_shaders_d3d12
      fyzzxyzzy_owned_report_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_report_fetch_d3d12
      COMMAND fyzzxyzzy_owned_sap_gpu_report_fetch_d3d12
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
        "${FYZZXYZZY_OWNED_CREATED_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}"
        "${FYZZXYZZY_OWNED_SAPBOX_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_INDEPENDENT_D32_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_DXIL}"
        "${FYZZXYZZY_OWNED_OVERLAP_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_PAIRS_DXIL}"
        ${FYZZXYZZY_OWNED_REPORT_DXIL_OUTPUTS})
    add_executable(fyzzxyzzy_owned_sap_gpu_removed_history_d3d12
      tests/physx_gpu_sap_owned_removed_history_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_gpu_removed_history_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_removed_history_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_removed_history_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_removed_history_d3d12
      fyzzxyzzy_owned_translate_aabbs_shader_d3d12
      fyzzxyzzy_owned_sap_incremental_shaders_d3d12
      fyzzxyzzy_owned_sap_sort_shaders_d3d12
      fyzzxyzzy_owned_region_shaders_d3d12
      fyzzxyzzy_independent_source_d32_dxil
      fyzzxyzzy_owned_active_shaders_d3d12
      fyzzxyzzy_owned_source_sap_shaders_d3d12
      fyzzxyzzy_owned_report_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_removed_history_d3d12
      COMMAND fyzzxyzzy_owned_sap_gpu_removed_history_d3d12
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
        "${FYZZXYZZY_OWNED_CREATED_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}"
        "${FYZZXYZZY_OWNED_SAPBOX_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_INDEPENDENT_D32_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_DXIL}"
        "${FYZZXYZZY_OWNED_OVERLAP_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_PAIRS_DXIL}"
        ${FYZZXYZZY_OWNED_REPORT_DXIL_OUTPUTS}
        "${FYZZXYZZY_OWNED_REMOVED_MARK_DXIL}"
        "${FYZZXYZZY_OWNED_REMOVED_PROJ_DXIL}")
    add_executable(fyzzxyzzy_owned_sap_gpu_removal_compaction_d3d12
      tests/physx_gpu_sap_owned_removal_compaction_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_gpu_removal_compaction_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_removal_compaction_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_removal_compaction_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_removal_compaction_d3d12
      fyzzxyzzy_owned_translate_aabbs_shader_d3d12
      fyzzxyzzy_owned_sap_incremental_shaders_d3d12
      fyzzxyzzy_owned_sap_sort_shaders_d3d12
      fyzzxyzzy_owned_region_shaders_d3d12
      fyzzxyzzy_independent_source_d32_dxil
      fyzzxyzzy_owned_active_shaders_d3d12
      fyzzxyzzy_owned_source_sap_shaders_d3d12
      fyzzxyzzy_owned_report_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_removal_compaction_d3d12
      COMMAND fyzzxyzzy_owned_sap_gpu_removal_compaction_d3d12
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
        "${FYZZXYZZY_OWNED_CREATED_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}"
        "${FYZZXYZZY_OWNED_SAPBOX_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_INDEPENDENT_D32_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_DXIL}"
        "${FYZZXYZZY_OWNED_OVERLAP_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_PAIRS_DXIL}"
        ${FYZZXYZZY_OWNED_REPORT_DXIL_OUTPUTS}
        "${FYZZXYZZY_OWNED_REMOVED_MARK_DXIL}"
        "${FYZZXYZZY_OWNED_REMOVED_PROJ_DXIL}")
    add_executable(fyzzxyzzy_owned_sap_gpu_live_lost_fetch_d3d12
      tests/physx_gpu_sap_owned_live_lost_fetch_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_gpu_live_lost_fetch_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_live_lost_fetch_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_live_lost_fetch_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_live_lost_fetch_d3d12
      fyzzxyzzy_owned_translate_aabbs_shader_d3d12
      fyzzxyzzy_owned_sap_incremental_shaders_d3d12
      fyzzxyzzy_owned_sap_sort_shaders_d3d12
      fyzzxyzzy_owned_region_shaders_d3d12
      fyzzxyzzy_independent_source_d32_dxil
      fyzzxyzzy_owned_active_shaders_d3d12
      fyzzxyzzy_owned_source_sap_shaders_d3d12
      fyzzxyzzy_owned_report_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_live_lost_fetch_d3d12
      COMMAND fyzzxyzzy_owned_sap_gpu_live_lost_fetch_d3d12
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
        "${FYZZXYZZY_OWNED_CREATED_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}"
        "${FYZZXYZZY_OWNED_SAPBOX_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_INDEPENDENT_D32_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_DXIL}"
        "${FYZZXYZZY_OWNED_OVERLAP_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_PAIRS_DXIL}"
        ${FYZZXYZZY_OWNED_REPORT_DXIL_OUTPUTS}
        "${FYZZXYZZY_OWNED_REMOVED_MARK_DXIL}"
        "${FYZZXYZZY_OWNED_REMOVED_PROJ_DXIL}"
        "${FYZZXYZZY_OWNED_UPDATED_DXIL}"
        "${FYZZXYZZY_OWNED_COUNT_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_COUNT_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_DXIL}")
    add_executable(fyzzxyzzy_owned_sap_gpu_three_frame_retirement_d3d12
      tests/physx_gpu_sap_owned_three_frame_retirement_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_gpu_three_frame_retirement_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_three_frame_retirement_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_three_frame_retirement_d3d12 PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_three_frame_retirement_d3d12
      fyzzxyzzy_owned_translate_aabbs_shader_d3d12
      fyzzxyzzy_owned_sap_incremental_shaders_d3d12
      fyzzxyzzy_owned_sap_sort_shaders_d3d12
      fyzzxyzzy_owned_region_shaders_d3d12
      fyzzxyzzy_independent_source_d32_dxil
      fyzzxyzzy_owned_active_shaders_d3d12
      fyzzxyzzy_owned_source_sap_shaders_d3d12
      fyzzxyzzy_owned_report_shaders_d3d12)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_three_frame_retirement_d3d12
      COMMAND fyzzxyzzy_owned_sap_gpu_three_frame_retirement_d3d12
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
        "${FYZZXYZZY_OWNED_CREATED_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}"
        "${FYZZXYZZY_OWNED_SAPBOX_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_INDEPENDENT_D32_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_DXIL}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_DXIL}"
        "${FYZZXYZZY_OWNED_OVERLAP_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_PAIRS_DXIL}"
        ${FYZZXYZZY_OWNED_REPORT_DXIL_OUTPUTS}
        "${FYZZXYZZY_OWNED_REMOVED_MARK_DXIL}"
        "${FYZZXYZZY_OWNED_REMOVED_PROJ_DXIL}"
        "${FYZZXYZZY_OWNED_UPDATED_DXIL}"
        "${FYZZXYZZY_OWNED_COUNT_FIRST_DXIL}"
        "${FYZZXYZZY_OWNED_COUNT_SECOND_DXIL}"
        "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_DXIL}")
    set(FYZZXYZZY_SC_BOX_BOUNDS_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sc_box_transform_bounds_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SC_BOX_BOUNDS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SC_BOX_BOUNDS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sc_box_transform_bounds_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sc_box_transform_bounds_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_sc_box_bounds_shader_d3d12
      DEPENDS "${FYZZXYZZY_SC_BOX_BOUNDS_DXIL}")
    add_executable(fyzzxyzzy_source_sc_box_bounds_d3d12
      tests/physx_gpu_source_sc_box_transform_bounds_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_sc_box_bounds_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_sc_box_bounds_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_sc_box_bounds_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_sc_box_bounds_d3d12
      fyzzxyzzy_source_sc_box_bounds_shader_d3d12)
    add_test(NAME fyzzxyzzy_source_sc_box_bounds_d3d12
      COMMAND fyzzxyzzy_source_sc_box_bounds_d3d12 "${FYZZXYZZY_SC_BOX_BOUNDS_DXIL}")
    set_tests_properties(fyzzxyzzy_source_sc_box_bounds_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")


    set(FYZZXYZZY_GU_BOX_CONTACTS_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_gu_box_box_contacts_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_GU_BOX_CONTACTS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_GU_BOX_CONTACTS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_gu_box_box_contacts_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_gu_box_box_contacts_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gu_box_contacts_shader_d3d12
      DEPENDS "${FYZZXYZZY_GU_BOX_CONTACTS_DXIL}")
    add_executable(fyzzxyzzy_source_gu_box_contacts_d3d12
      tests/physx_gpu_source_gu_box_box_contacts_native.cpp
      src/native/GuContactBoxBox.cpp)
    target_compile_definitions(fyzzxyzzy_source_gu_box_contacts_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gu_box_contacts_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gu_box_contacts_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gu_box_contacts_d3d12
      fyzzxyzzy_source_sc_box_bounds_shader_d3d12
      fyzzxyzzy_source_gu_box_contacts_shader_d3d12)
    add_test(NAME fyzzxyzzy_source_gu_box_contacts_d3d12
      COMMAND fyzzxyzzy_source_gu_box_contacts_d3d12
        "${FYZZXYZZY_SC_BOX_BOUNDS_DXIL}" "${FYZZXYZZY_GU_BOX_CONTACTS_DXIL}")
    set_tests_properties(fyzzxyzzy_source_gu_box_contacts_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")


    set(FYZZXYZZY_SOURCE_PCM_BOX_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_pcm_box_candidates_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_PCM_BOX_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_PCM_BOX_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_candidates_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_candidates_native.hlsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_gjk_native.hlsli"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_math_native.hlsli"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_pcm_box_candidates_shader_d3d12
      DEPENDS "${FYZZXYZZY_SOURCE_PCM_BOX_DXIL}")
    set(FYZZXYZZY_SOURCE_PCM_EPA_WORK_DXIL_RESET
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_pcm_epa_reset_native.dxil")
    set(FYZZXYZZY_SOURCE_PCM_EPA_WORK_DXIL_COMPACT
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_pcm_epa_compact_native.dxil")
    foreach(ENTRY reset compact)
      if(ENTRY STREQUAL "reset")
        set(EPA_DEFINE "-DSOURCE_PCM_EPA_RESET=1")
        set(EPA_OUTPUT "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_DXIL_RESET}")
      else()
        set(EPA_DEFINE "")
        set(EPA_OUTPUT "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_DXIL_COMPACT}")
      endif()
      add_custom_command(OUTPUT "${EPA_OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" ${EPA_DEFINE} -T cs_6_0 -E "${ENTRY}"
          -Fo "${EPA_OUTPUT}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_epa_work_native.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_epa_work_native.hlsl"
        VERBATIM)
    endforeach()
    add_custom_target(fyzzxyzzy_source_pcm_epa_work_shader_d3d12
      DEPENDS "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_DXIL_RESET}" "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_DXIL_COMPACT}")
    set(FYZZXYZZY_SOURCE_PCM_EPA_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_pcm_box_epa_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_PCM_EPA_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_PCM_EPA_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_epa_native.hlsl"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_epa_native.hlsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_epa_native.hlsli"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_math_native.hlsli"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_gjk_native.hlsli"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_pcm_epa_shader_d3d12
      DEPENDS "${FYZZXYZZY_SOURCE_PCM_EPA_DXIL}")
    set(FYZZXYZZY_SOURCE_PCM_CONTACT_STREAM_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_pcm_sc_contact_stream_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_PCM_CONTACT_STREAM_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_PCM_CONTACT_STREAM_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_sc_contact_stream_native.hlsl"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_sc_contact_stream_native.hlsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_math_native.hlsli"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_pcm_contact_stream_shader_d3d12
      DEPENDS "${FYZZXYZZY_SOURCE_PCM_CONTACT_STREAM_DXIL}")
    set(FYZZXYZZY_SOURCE_DY_DISTANCE_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_dy_distance_pgs_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_DY_DISTANCE_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_6 -E main
        -Fo "${FYZZXYZZY_SOURCE_DY_DISTANCE_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_dy_distance_pgs_native.hlsl"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_dy_distance_pgs_native.hlsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_dy_contact_pgs_native.hlsli"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_dy_distance_shader_d3d12
      DEPENDS "${FYZZXYZZY_SOURCE_DY_DISTANCE_DXIL}")
    add_executable(fyzzxyzzy_source_pcm_box_candidates_d3d12
      tests/physx_gpu_source_pcm_box_candidates_native.cpp
      src/native/GuPCMContactBoxBox.cpp
      src/native/GuPCMGjkEpa.cpp)
    target_compile_definitions(fyzzxyzzy_source_pcm_box_candidates_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_pcm_box_candidates_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_pcm_box_candidates_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_pcm_box_candidates_d3d12
      fyzzxyzzy_source_sc_box_bounds_shader_d3d12
      fyzzxyzzy_source_pcm_box_candidates_shader_d3d12
      fyzzxyzzy_source_pcm_epa_work_shader_d3d12
      fyzzxyzzy_source_pcm_epa_shader_d3d12)
    add_test(NAME fyzzxyzzy_source_pcm_box_candidates_d3d12
      COMMAND fyzzxyzzy_source_pcm_box_candidates_d3d12
        "${FYZZXYZZY_SC_BOX_BOUNDS_DXIL}" "${FYZZXYZZY_SOURCE_PCM_BOX_DXIL}"
        "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_DXIL_RESET}" "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_DXIL_COMPACT}"
        "${FYZZXYZZY_SOURCE_PCM_EPA_DXIL}")
    set_tests_properties(fyzzxyzzy_source_pcm_box_candidates_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")


    # Source aggregate.cu::updateAggregateBounds, original 32-lane
    # aggregate member reduction (4 warps per native workgroup).
    set(FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_bounds_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_bounds_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_bounds_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_SOURCE_AGGREGATE_MARK_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_mark_changed_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_DXIL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_mark_changed_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_mark_changed_native.hlsl"
      VERBATIM)
    add_executable(fyzzxyzzy_source_gpu_aggregate_bounds_d3d12
      tests/physx_gpu_source_aggregate_bounds_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_bounds_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(fyzzxyzzy_source_gpu_aggregate_bounds_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_bounds_d3d12 PRIVATE FyzzXYZZY::ComputeD3D12 fyzzxyzzy_source_gpu_primitives_d3d12)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_bounds_d3d12_shader
      DEPENDS "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_DXIL}"
              "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_DXIL}")
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_bounds_d3d12 fyzzxyzzy_source_gpu_aggregate_bounds_d3d12_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_bounds_d3d12
      COMMAND fyzzxyzzy_source_gpu_aggregate_bounds_d3d12 "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_DXIL}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_DXIL}")
    set(FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_mark_members_history_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_mark_members_history_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_mark_members_history_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_member_history_d3d12_shader
      DEPENDS "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_DXIL}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_member_history_d3d12
      tests/physx_gpu_source_aggregate_member_history_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_aggregate_member_history_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_member_history_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_member_history_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_member_history_d3d12
      fyzzxyzzy_source_gpu_aggregate_member_history_d3d12_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_member_history_d3d12
      COMMAND fyzzxyzzy_source_gpu_aggregate_member_history_d3d12
        "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_DXIL}")
    set_tests_properties(fyzzxyzzy_source_gpu_aggregate_member_history_d3d12
      PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    set(FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_projection_sort_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_projection_sort_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_projection_sort_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_projection_d3d12_shader
      DEPENDS "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_DXIL}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_projections_d3d12
      tests/physx_gpu_source_aggregate_projection_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_aggregate_projections_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_projections_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_projections_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_projections_d3d12
      fyzzxyzzy_source_gpu_aggregate_projection_d3d12_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_projections_d3d12
      COMMAND fyzzxyzzy_source_gpu_aggregate_projections_d3d12
        "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_DXIL}")
    set_tests_properties(fyzzxyzzy_source_gpu_aggregate_projections_d3d12
      PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    # aggregate.cu::doSelfCollision, found/lost member history on GPU.
    set(FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_self_collision_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_self_collision_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_self_collision_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12_shader
      DEPENDS "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_DXIL}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12
      tests/physx_gpu_source_aggregate_self_collision_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12
      fyzzxyzzy_source_gpu_aggregate_projection_d3d12_shader
      fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12
      COMMAND fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12
        "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_DXIL}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_DXIL}")
    set_tests_properties(fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12
      PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    set(FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_copy_reports_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_copy_reports_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_copy_reports_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_copy_report_d3d12_shader
      DEPENDS "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_DXIL}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_report_copy_d3d12
      tests/physx_gpu_source_aggregate_report_copy_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_aggregate_report_copy_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_report_copy_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_report_copy_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_report_copy_d3d12
      fyzzxyzzy_source_gpu_aggregate_copy_report_d3d12_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_report_copy_d3d12
      COMMAND fyzzxyzzy_source_gpu_aggregate_report_copy_d3d12
        "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_DXIL}")
    set_tests_properties(fyzzxyzzy_source_gpu_aggregate_report_copy_d3d12
      PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    # Original aggregate.cu::addAggPairsStage1 and addAggPairsStage2.
    set(FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_pair_admit_native.dxil")
    set(FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_pair_publish_native.dxil")
    add_custom_command(OUTPUT
      "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_DXIL}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E admit
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_admission_native.hlsl"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E publish
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_admission_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_admission_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_pair_admission_d3d12_shaders
      DEPENDS
        "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_DXIL}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_DXIL}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_pair_admission_d3d12
      tests/physx_gpu_source_aggregate_pair_admission_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_aggregate_pair_admission_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_pair_admission_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_pair_admission_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_pair_admission_d3d12
      fyzzxyzzy_source_gpu_aggregate_pair_admission_d3d12_shaders)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_pair_admission_d3d12
      COMMAND fyzzxyzzy_source_gpu_aggregate_pair_admission_d3d12
        "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_DXIL}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_DXIL}")
    set_tests_properties(fyzzxyzzy_source_gpu_aggregate_pair_admission_d3d12
      PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    # aggregate.cu::removeAggPairsStage1/2/3 native GPU translations.
    set(FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_COUNT
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_pair_retire_count_native.dxil")
    set(FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_MARK
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_pair_retire_mark_native.dxil")
    set(FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_PUBLISH
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_pair_retire_publish_native.dxil")
    add_custom_command(OUTPUT
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_COUNT}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_MARK}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_PUBLISH}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E histogram
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_COUNT}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_retirement_native.hlsl"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E mark
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_MARK}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_retirement_native.hlsl"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E publish
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_PUBLISH}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_retirement_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_retirement_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_pair_retirement_d3d12_shaders
      DEPENDS
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_COUNT}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_MARK}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_PUBLISH}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_pair_retirement_d3d12
      tests/physx_gpu_source_aggregate_pair_retirement_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_aggregate_pair_retirement_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_pair_retirement_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_pair_retirement_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_pair_retirement_d3d12
      fyzzxyzzy_source_gpu_aggregate_pair_retirement_d3d12_shaders
      fyzzxyzzy_source_gpu_aggregate_pair_admission_d3d12_shaders)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_pair_retirement_d3d12
      COMMAND fyzzxyzzy_source_gpu_aggregate_pair_retirement_d3d12
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_COUNT}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_MARK}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_PUBLISH}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_DXIL}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_DXIL}")
    set_tests_properties(fyzzxyzzy_source_gpu_aggregate_pair_retirement_d3d12
      PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    # aggregate.cu::doAggPairCollisions, original GPU root pair member sweep.
    set(FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_persistent_collisions_native.dxil")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_persistent_collisions_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_persistent_collisions_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_persistent_collisions_d3d12_shader DEPENDS "${FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_DXIL}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_persistent_collisions_d3d12
      tests/physx_gpu_source_aggregate_persistent_collisions_native.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_aggregate_persistent_collisions_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_persistent_collisions_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_persistent_collisions_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_source_gpu_primitives_d3d12)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_persistent_collisions_d3d12
      fyzzxyzzy_source_gpu_aggregate_persistent_collisions_d3d12_shader
      fyzzxyzzy_source_gpu_aggregate_projection_d3d12_shader
      fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12_shader
      fyzzxyzzy_source_gpu_aggregate_pair_retirement_d3d12_shaders)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_persistent_collisions_d3d12 COMMAND fyzzxyzzy_source_gpu_aggregate_persistent_collisions_d3d12
      "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_DXIL}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_DXIL}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_DXIL}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_COUNT}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_MARK}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_PUBLISH}")
    set_tests_properties(fyzzxyzzy_source_gpu_aggregate_persistent_collisions_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    # The production Sc owner requires the complete twelve-stage original
    # aggregate bundle in the same Context as the existing SAP manifest.
    set(FYZZXYZZY_NATIVE_GPU_AGGREGATE_DXIL_BINARIES
      "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_DXIL}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_DXIL}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_DXIL}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_DXIL}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_DXIL}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_DXIL}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_DXIL}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_DXIL}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_COUNT}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_MARK}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_DXIL_PUBLISH}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_DXIL}")
    set(_aggregate_stage_names
      Bounds
      MarkBounds
      MemberHistory
      Projection
      SelfCollision
      PairAdmit
      PairPublish
      PersistentCollisions
      PairRetireHistogram
      PairRetireMark
      PairRetirePublish
      CopyReports)
    set(_aggregate_manifest "FYZZXYZZY_SOURCE_AGGREGATE DXIL\n")
    foreach(_index RANGE 0 11)
      list(GET _aggregate_stage_names ${_index} _stage)
      list(GET FYZZXYZZY_NATIVE_GPU_AGGREGATE_DXIL_BINARIES ${_index} _binary)
      get_filename_component(_filename "${_binary}" NAME)
      string(APPEND _aggregate_manifest "${_stage} ${_filename}\n")
    endforeach()
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_aggregate_dxil.manifest" CONTENT "${_aggregate_manifest}")
    add_custom_target(fyzzxyzzy_native_gpu_aggregate_shaders_d3d12
      DEPENDS ${FYZZXYZZY_NATIVE_GPU_AGGREGATE_DXIL_BINARIES})
    add_executable(fyzzxyzzy_source_gpu_aggregate_world_bundle_d3d12
      tests/physx_gpu_source_aggregate_world_bundle.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_aggregate_world_bundle_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_world_bundle_d3d12 PRIVATE FyzzXYZZY::NativeGpuWorldD3D12)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_world_bundle_d3d12 fyzzxyzzy_native_gpu_aggregate_shaders_d3d12)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_world_bundle_d3d12 COMMAND fyzzxyzzy_source_gpu_aggregate_world_bundle_d3d12
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_sap_dxil.manifest"
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_aggregate_dxil.manifest")
    set_tests_properties(fyzzxyzzy_source_gpu_aggregate_world_bundle_d3d12 PROPERTIES TIMEOUT 180)
    add_executable(fyzzxyzzy_source_sc_aggregate_elements_d3d12
      tests/physx_gpu_source_sc_aggregate_elements.cpp)
    target_compile_definitions(fyzzxyzzy_source_sc_aggregate_elements_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_sc_aggregate_elements_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_source_sc_aggregate_elements_d3d12 PRIVATE
      FyzzXYZZY::NativeGpuWorldD3D12)
    add_test(NAME fyzzxyzzy_source_sc_aggregate_elements_d3d12 COMMAND fyzzxyzzy_source_sc_aggregate_elements_d3d12)
    set_tests_properties(fyzzxyzzy_source_sc_aggregate_elements_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass_d3d12
      tests/physx_gpu_source_aggregate_aabb_owner_prepass.cpp)
    target_compile_definitions(fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass_d3d12 PRIVATE
      FyzzXYZZY::NativeGpuWorldD3D12)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass_d3d12
      fyzzxyzzy_source_gpu_aggregate_bounds_d3d12_shader
      fyzzxyzzy_source_gpu_aggregate_member_history_d3d12_shader
      fyzzxyzzy_source_gpu_aggregate_projection_d3d12_shader
      fyzzxyzzy_source_gpu_aggregate_self_collision_d3d12_shader
      fyzzxyzzy_source_gpu_aggregate_copy_report_d3d12_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass_d3d12
      COMMAND fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass_d3d12
        "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_DXIL}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_DXIL}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_DXIL}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_DXIL}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_DXIL}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_DXIL}")
    set_tests_properties(fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass_d3d12
      PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    set_tests_properties(fyzzxyzzy_source_gpu_aggregate_bounds_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    add_executable(fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle_d3d12
      tests/physx_gpu_sap_owned_aabb_scene_lifecycle_independent.cpp)
    target_compile_definitions(fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle_d3d12 PRIVATE
      FyzzXYZZY::NativeGpuWorldD3D12)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle_d3d12
      fyzzxyzzy_owned_translate_aabbs_shader_d3d12
      fyzzxyzzy_owned_sap_incremental_shaders_d3d12
      fyzzxyzzy_owned_sap_sort_shaders_d3d12
      fyzzxyzzy_owned_region_shaders_d3d12
      fyzzxyzzy_independent_source_d32_dxil
      fyzzxyzzy_owned_active_shaders_d3d12
      fyzzxyzzy_owned_source_sap_shaders_d3d12
      fyzzxyzzy_owned_report_shaders_d3d12
      fyzzxyzzy_owned_clear_new_shader_d3d12)
    set(FYZZXYZZY_NATIVE_GPU_SAP_DXIL_BINARIES
      "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_DXIL}"
      "${FYZZXYZZY_OWNED_CREATED_DXIL}"
      "${FYZZXYZZY_OWNED_RADIX_INIT_DXIL}"
      "${FYZZXYZZY_OWNED_RADIX_COUNT_DXIL}"
      "${FYZZXYZZY_OWNED_RADIX_SCATTER_DXIL}"
      "${FYZZXYZZY_OWNED_RADIX_GATHER_DXIL}"
      "${FYZZXYZZY_OWNED_SAPBOX_DXIL}"
      "${FYZZXYZZY_OWNED_HIST_FIRST_DXIL}"
      "${FYZZXYZZY_OWNED_HIST_SECOND_DXIL}"
      "${FYZZXYZZY_INDEPENDENT_D32_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_CREATE_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_DXIL}"
      "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_DXIL}"
      "${FYZZXYZZY_OWNED_ACTIVE_writeout_DXIL}"
      "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_DXIL}"
      "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_DXIL}"
      "${FYZZXYZZY_OWNED_ACTIVE_output_DXIL}"
      "${FYZZXYZZY_OWNED_OVERLAP_DXIL}"
      "${FYZZXYZZY_OWNED_SCAN_FIRST_DXIL}"
      "${FYZZXYZZY_OWNED_SCAN_SECOND_DXIL}"
      "${FYZZXYZZY_OWNED_PAIRS_DXIL}"
      ${FYZZXYZZY_OWNED_REPORT_DXIL_OUTPUTS}
      "${FYZZXYZZY_OWNED_REMOVED_MARK_DXIL}"
      "${FYZZXYZZY_OWNED_REMOVED_PROJ_DXIL}"
      "${FYZZXYZZY_OWNED_UPDATED_DXIL}"
      "${FYZZXYZZY_OWNED_COUNT_FIRST_DXIL}"
      "${FYZZXYZZY_OWNED_COUNT_SECOND_DXIL}"
      "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_DXIL}"
      "${FYZZXYZZY_OWNED_CLEAR_NEW_DXIL}"
      "${FYZZXYZZY_SC_BOX_BOUNDS_DXIL}")
    list(LENGTH FYZZXYZZY_NATIVE_GPU_SAP_DXIL_BINARIES _sap_shader_count)
    if(NOT _sap_shader_count EQUAL 35)
      message(FATAL_ERROR "Native DXIL SAP shader bundle does not contain 35 stages")
    endif()
    set(_sap_names
      TranslateAabbs
      CreatedPairs
      RadixInitialize
      RadixCount
      RadixScatter
      RadixGather
      SapBoxIndices
      EndpointHistogramFirst
      EndpointHistogramSecond
      FillD32
      RegionCreate
      RegionHistogramFirst
      RegionHistogramWave
      RegionHistogramSecond
      ActiveWriteout
      ActiveWave
      ActivePortable
      ActiveOutput
      OverlapWork
      OverlapScanFirst
      OverlapScanSecond
      NewBoundsPairs
      ReportFirst
      ReportSecond
      ReportCopy
      ReportWaveFirst
      ReportWaveSecond
      RemovedMark
      RemovedProjections
      UpdatedBounds
      IncrementalCountFirst
      IncrementalCountSecond
      IncrementalPairs
      ClearNewFlags
      BoxTransformBounds
    )
    set(_sap_manifest "FYZZXYZZY_SOURCE_SAP_1 DXIL\n")
    foreach(_idx RANGE 0 34)
      list(GET _sap_names ${_idx} _stage)
      list(GET FYZZXYZZY_NATIVE_GPU_SAP_DXIL_BINARIES ${_idx} _binary)
      get_filename_component(_filename "${_binary}" NAME)
      string(APPEND _sap_manifest "${_stage} ${_filename}\n")
    endforeach()
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_sap_dxil.manifest" CONTENT "${_sap_manifest}")
    add_custom_target(fyzzxyzzy_native_gpu_sap_shaders_d3d12
      DEPENDS ${FYZZXYZZY_NATIVE_GPU_SAP_DXIL_BINARIES})
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_world_bundle_d3d12
      fyzzxyzzy_native_gpu_sap_shaders_d3d12)
    set(_pcm_binaries
      "${FYZZXYZZY_SOURCE_PCM_BOX_DXIL}"
      "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_DXIL_RESET}"
      "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_DXIL_COMPACT}"
      "${FYZZXYZZY_SOURCE_PCM_EPA_DXIL}"
      "${FYZZXYZZY_SOURCE_PCM_CONTACT_STREAM_DXIL}")
    set(_pcm_stages Candidates EpaReset EpaCompact EpaContact ContactStream)
    set(_pcm_manifest "FYZZXYZZY_SOURCE_PCM_1 DXIL\n")
    foreach(_idx RANGE 0 4)
      list(GET _pcm_stages ${_idx} _stage)
      list(GET _pcm_binaries ${_idx} _binary)
      get_filename_component(_name "${_binary}" NAME)
      string(APPEND _pcm_manifest "${_stage} ${_name}\n")
    endforeach()
    file(GENERATE
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_pcm_dxil.manifest"
      CONTENT "${_pcm_manifest}")
    add_custom_target(fyzzxyzzy_native_gpu_pcm_shaders_d3d12
      DEPENDS ${_pcm_binaries})
    add_executable(fyzzxyzzy_source_sc_gpu_pcm_scene_d3d12
      tests/physx_gpu_source_sc_pcm_scene_native.cpp)
    target_include_directories(fyzzxyzzy_source_sc_gpu_pcm_scene_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_compile_definitions(fyzzxyzzy_source_sc_gpu_pcm_scene_d3d12 PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_link_libraries(fyzzxyzzy_source_sc_gpu_pcm_scene_d3d12 PRIVATE
      FyzzXYZZY::NativeGpuWorldD3D12)
    add_dependencies(fyzzxyzzy_source_sc_gpu_pcm_scene_d3d12
      fyzzxyzzy_native_gpu_sap_shaders_d3d12
      fyzzxyzzy_native_gpu_pcm_shaders_d3d12
      fyzzxyzzy_source_dy_distance_shader_d3d12)
    add_test(NAME fyzzxyzzy_source_sc_gpu_pcm_scene_d3d12
      COMMAND fyzzxyzzy_source_sc_gpu_pcm_scene_d3d12
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_sap_dxil.manifest"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_pcm_dxil.manifest"
        "${FYZZXYZZY_SOURCE_DY_DISTANCE_DXIL}")
    set_tests_properties(fyzzxyzzy_source_sc_gpu_pcm_scene_d3d12
      PROPERTIES TIMEOUT 180)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle_d3d12
      COMMAND fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle_d3d12
        ${FYZZXYZZY_NATIVE_GPU_SAP_DXIL_BINARIES}
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_sap_dxil.manifest")
    set_tests_properties(fyzzxyzzy_owned_sap_gpu_new_pairs_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    set_tests_properties(fyzzxyzzy_owned_sap_gpu_report_fetch_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    set_tests_properties(fyzzxyzzy_owned_sap_gpu_removed_history_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    set_tests_properties(fyzzxyzzy_owned_sap_gpu_removal_compaction_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    set_tests_properties(fyzzxyzzy_owned_sap_gpu_live_lost_fetch_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    set_tests_properties(fyzzxyzzy_owned_sap_gpu_three_frame_retirement_d3d12 PROPERTIES
      ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
    set_tests_properties(fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle_d3d12 PROPERTIES
      TIMEOUT 180)









  endif()

  if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
  add_library(fyzzxyzzy_physx_gpu_stages_d3d12 STATIC
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
    src/physx_gpu/sap_overlap_work_arena.cpp
    src/physx_gpu/sap_overlap_prefix_arena.cpp
    src/physx_gpu/sap_newbounds_pairs_arena.cpp
    src/physx_gpu/sap_newbounds_sequence_arena.cpp
    src/physx_gpu/broadphase_completion.cpp)
  target_compile_features(fyzzxyzzy_physx_gpu_stages_d3d12 PUBLIC cxx_std_23)
  target_include_directories(fyzzxyzzy_physx_gpu_stages_d3d12 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
  target_link_libraries(fyzzxyzzy_physx_gpu_stages_d3d12 PUBLIC
    FyzzXYZZY::ComputeD3D12 physx_lib fyzzxyzzy_physx_cuda_host_address
    fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives_d3d12)


  if(BUILD_TESTING)
    add_executable(fyzzxyzzy_physx_cuda_host_transfer_d3d12
      tests/physx_gpu_cuda_host_transfer.cpp)
    target_compile_definitions(fyzzxyzzy_physx_cuda_host_transfer_d3d12
      PRIVATE FYZZXYZZY_TEST_D3D12=1)
    target_include_directories(fyzzxyzzy_physx_cuda_host_transfer_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/simulationcontroller/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
    target_link_libraries(fyzzxyzzy_physx_cuda_host_transfer_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12 fyzzxyzzy_physx_cuda_host_address fyzzxyzzy_physx_original_gpu_heap physx_lib fyzzxyzzy_physx_gpu_stages_d3d12)
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
  if(BUILD_TESTING AND (FYZZXYZZY_BUILD_D3D12_AABB_TEST OR FYZZXYZZY_BUILD_SOURCE_GPU_TESTS))
    find_program(FYZZXYZZY_DXC NAMES dxc REQUIRED)
    set(FYZZXYZZY_WAVE32_DXIL
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/native_wave32_contract.dxil")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_WAVE32_DXIL}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -T cs_6_6 -E main
        -Fo "${FYZZXYZZY_WAVE32_DXIL}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_wave32_contract.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_wave32_contract.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_wave32_dxil DEPENDS "${FYZZXYZZY_WAVE32_DXIL}")
    add_executable(fyzzxyzzy_wave32_contract_d3d12 tests/native_wave32_contract.cpp)
    target_compile_definitions(fyzzxyzzy_wave32_contract_d3d12 PRIVATE
      FYZZXYZZY_TEST_D3D12=1)
    target_link_libraries(fyzzxyzzy_wave32_contract_d3d12 PRIVATE
      FyzzXYZZY::ComputeD3D12)
    add_dependencies(fyzzxyzzy_wave32_contract_d3d12 fyzzxyzzy_wave32_dxil)
    add_test(NAME fyzzxyzzy_wave32_contract_d3d12
      COMMAND fyzzxyzzy_wave32_contract_d3d12 "${FYZZXYZZY_WAVE32_DXIL}")

  endif()
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
    # Explicit source-copy CPU measurement, not a conformance threshold.
    add_custom_target(fyzzxyzzy_native_same_buffer_copy_profile_d3d12
      COMMAND "$<TARGET_FILE:fyzzxyzzy_native_multi_resource_d3d12>"
        "${FYZZXYZZY_MULTI_RESOURCE_DXIL}" --benchmark-self-copy
      DEPENDS fyzzxyzzy_native_multi_resource_d3d12
      USES_TERMINAL VERBATIM)
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
      fyzzxyzzy_source_gpu_primitives_d3d12)
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
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_physx_cuda_aabb_launch_d3d12 PRIVATE
      fyzzxyzzy_source_gpu_primitives_d3d12)
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
      fyzzxyzzy_source_gpu_primitives_d3d12)
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
    add_custom_target(fyzzxyzzy_source_sap_radix_profile_d3d12
      COMMAND "$<TARGET_FILE:fyzzxyzzy_physx_source_sap_radix_native_d3d12>"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_DXIL}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_DXIL}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_DXIL}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_DXIL}"
        --benchmark
      DEPENDS fyzzxyzzy_physx_source_sap_radix_native_d3d12
      USES_TERMINAL VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_radix_profile_alternating_d3d12
      COMMAND "$<TARGET_FILE:fyzzxyzzy_physx_source_sap_radix_native_d3d12>"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_DXIL}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_DXIL}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_DXIL}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_DXIL}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_DXIL}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_DXIL}"
        --benchmark-alternating
      DEPENDS fyzzxyzzy_physx_source_sap_radix_native_d3d12
      USES_TERMINAL VERBATIM)
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

    # Source AABB-manager handles and GPU reports must lower to Sc
    # without a second volume registry or partial late publication.
    add_executable(fyzzxyzzy_physx_source_sap_scene_boundary_d3d12 tests/physx_gpu_source_sap_scene_boundary.cpp)
    target_include_directories(fyzzxyzzy_physx_source_sap_scene_boundary_d3d12 PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_scene_boundary_d3d12 PRIVATE fyzzxyzzy_physx_gpu_stages_d3d12)
    add_test(NAME fyzzxyzzy_physx_source_sap_scene_boundary_d3d12 COMMAND fyzzxyzzy_physx_source_sap_scene_boundary_d3d12)
    set_tests_properties(fyzzxyzzy_physx_source_sap_scene_boundary_d3d12 PROPERTIES ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")








    
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
