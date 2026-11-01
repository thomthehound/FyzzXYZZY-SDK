# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

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

  fyzzxyzzy_add_native_gpu_libraries(
    fyzzxyzzy_source_gpu_primitives
    fyzzxyzzy_native_gpu_world FyzzXYZZY::NativeGpuWorld FyzzXYZZY::Compute)

  if(BUILD_TESTING AND NOT FYZZXYZZY_ENABLE_PHYSX_SOURCE
     AND FYZZXYZZY_BUILD_SOURCE_GPU_TESTS)
    find_program(FYZZXYZZY_DXC NAMES dxc REQUIRED)
      # Verify a physically fixed 32-lane subgroup through actual shader execution.
      set(FYZZXYZZY_WAVE32_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/native_wave32_contract.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_WAVE32_SPV}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_6 -E main -Fo "${FYZZXYZZY_WAVE32_SPV}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_wave32_contract.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_wave32_contract.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_wave32_spv DEPENDS "${FYZZXYZZY_WAVE32_SPV}")
      add_executable(fyzzxyzzy_wave32_contract tests/native_wave32_contract.cpp)
      target_link_libraries(fyzzxyzzy_wave32_contract PRIVATE FyzzXYZZY::Compute)
      add_dependencies(fyzzxyzzy_wave32_contract fyzzxyzzy_wave32_spv)
      add_test(NAME fyzzxyzzy_wave32_contract
        COMMAND fyzzxyzzy_wave32_contract "${FYZZXYZZY_WAVE32_SPV}")

    set(FYZZXYZZY_INDEPENDENT_D32_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_fill_d32_native.spv")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_INDEPENDENT_D32_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
        -T cs_6_0 -E main -Fo "${FYZZXYZZY_INDEPENDENT_D32_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_fill_d32_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_fill_d32_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_independent_source_d32_spv
      DEPENDS "${FYZZXYZZY_INDEPENDENT_D32_SPV}")
    add_executable(fyzzxyzzy_source_gpu_d32_independent
      tests/physx_gpu_source_fill_d32_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_d32_independent PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_d32_independent PRIVATE
      fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_d32_independent
      fyzzxyzzy_independent_source_d32_spv)
    add_test(NAME fyzzxyzzy_source_gpu_d32_independent
      COMMAND fyzzxyzzy_source_gpu_d32_independent
        "${FYZZXYZZY_INDEPENDENT_D32_SPV}")
    set_tests_properties(fyzzxyzzy_source_gpu_d32_independent PROPERTIES
      SKIP_RETURN_CODE 77)
    set(FYZZXYZZY_INDEPENDENT_AABB_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_translate_aabbs.spv")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_INDEPENDENT_AABB_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
        -T cs_6_0 -E main -Fo "${FYZZXYZZY_INDEPENDENT_AABB_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_independent_source_aabb_spv
      DEPENDS "${FYZZXYZZY_INDEPENDENT_AABB_SPV}")
    add_executable(fyzzxyzzy_source_gpu_aabb_independent
      tests/physx_gpu_aabb_encoding.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_aabb_independent PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_aabb_independent PRIVATE
      fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_aabb_independent
      fyzzxyzzy_independent_source_aabb_spv)
    add_test(NAME fyzzxyzzy_source_gpu_aabb_independent
      COMMAND fyzzxyzzy_source_gpu_aabb_independent
        "${FYZZXYZZY_INDEPENDENT_AABB_SPV}")
    # Original PxgBroadPhaseDesc is now SDK-owned, not an upstream fixture.
    set(FYZZXYZZY_OWNED_OVERLAP_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_work_native.spv")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_OWNED_OVERLAP_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_OVERLAP_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_work_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_SCAN_FIRST_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_prefix_first_native.spv")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_OWNED_SCAN_FIRST_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_SCAN_FIRST_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_first_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_SCAN_SECOND_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_overlap_prefix_second_native.spv")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_OWNED_SCAN_SECOND_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_SCAN_SECOND_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_overlap_prefix_second_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_PAIRS_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_sap_newbounds_pairs_wave_native.spv")
    add_custom_command(
      OUTPUT "${FYZZXYZZY_OWNED_PAIRS_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_PAIRS_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_sap_newbounds_pairs_wave_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_owned_source_sap_shaders DEPENDS
      ${FYZZXYZZY_OWNED_OVERLAP_SPV}
      ${FYZZXYZZY_OWNED_SCAN_FIRST_SPV}
      ${FYZZXYZZY_OWNED_SCAN_SECOND_SPV}
      ${FYZZXYZZY_OWNED_PAIRS_SPV})
    add_executable(fyzzxyzzy_source_gpu_overlap_independent tests/physx_gpu_source_overlap_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_overlap_independent PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_overlap_independent PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_overlap_independent fyzzxyzzy_owned_source_sap_shaders)
    add_test(NAME fyzzxyzzy_source_gpu_overlap_independent COMMAND fyzzxyzzy_source_gpu_overlap_independent
      "${FYZZXYZZY_OWNED_OVERLAP_SPV}"
    )
    add_executable(fyzzxyzzy_source_gpu_scan_independent tests/physx_gpu_source_overlap_prefix_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_scan_independent PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_scan_independent PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_scan_independent fyzzxyzzy_owned_source_sap_shaders)
    add_test(NAME fyzzxyzzy_source_gpu_scan_independent COMMAND fyzzxyzzy_source_gpu_scan_independent
      "${FYZZXYZZY_OWNED_SCAN_FIRST_SPV}"
      "${FYZZXYZZY_OWNED_SCAN_SECOND_SPV}"
    )
    set_tests_properties(fyzzxyzzy_source_gpu_scan_independent PROPERTIES
      SKIP_RETURN_CODE 77)
    add_executable(fyzzxyzzy_source_gpu_pairs_independent tests/physx_gpu_source_newbounds_pairs_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_pairs_independent PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_pairs_independent PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_pairs_independent fyzzxyzzy_owned_source_sap_shaders)
    add_test(NAME fyzzxyzzy_source_gpu_pairs_independent COMMAND fyzzxyzzy_source_gpu_pairs_independent
      "${FYZZXYZZY_OWNED_PAIRS_SPV}"
    )
    set_tests_properties(fyzzxyzzy_source_gpu_pairs_independent PROPERTIES
      SKIP_RETURN_CODE 77)
    # Source-owned SAP incremental/created GPU work uses the original
    # PxgBroadPhaseDesc and native backend, without a reference PhysX ABI.
    set(FYZZXYZZY_OWNED_CREATED_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_created_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_CREATED_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_CREATED_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_created_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_created_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_COUNT_FIRST_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_counts_first_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_COUNT_FIRST_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_COUNT_FIRST_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_first_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_COUNT_SECOND_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_counts_second_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_COUNT_SECOND_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_COUNT_SECOND_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_counts_second_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_HIST_FIRST_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_endpoint_histograms_first_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_first_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_HIST_SECOND_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_endpoint_histograms_second_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_endpoint_histograms_second_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_incremental_pairs_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_pairs_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_incremental_pairs_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_owned_sap_incremental_shaders DEPENDS
      ${FYZZXYZZY_OWNED_CREATED_SPV}
      ${FYZZXYZZY_OWNED_COUNT_FIRST_SPV}
      ${FYZZXYZZY_OWNED_COUNT_SECOND_SPV}
      ${FYZZXYZZY_OWNED_HIST_FIRST_SPV}
      ${FYZZXYZZY_OWNED_HIST_SECOND_SPV}
      ${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_SPV}
    )
    add_executable(fyzzxyzzy_owned_sap_created tests/physx_gpu_source_sap_created_native.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_created PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_created PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_created fyzzxyzzy_owned_sap_incremental_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_created COMMAND fyzzxyzzy_owned_sap_created
      "${FYZZXYZZY_OWNED_CREATED_SPV}"
    )
    add_executable(fyzzxyzzy_owned_sap_counts tests/physx_gpu_source_sap_incremental_counts_native.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_counts PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_counts PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_counts fyzzxyzzy_owned_sap_incremental_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_counts COMMAND fyzzxyzzy_owned_sap_counts
      "${FYZZXYZZY_OWNED_COUNT_FIRST_SPV}"
      "${FYZZXYZZY_OWNED_COUNT_SECOND_SPV}"
    )
    add_executable(fyzzxyzzy_owned_sap_hist tests/physx_gpu_source_sap_endpoint_histograms_native.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_hist PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_hist PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_hist fyzzxyzzy_owned_sap_incremental_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_hist COMMAND fyzzxyzzy_owned_sap_hist
      "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}"
      "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}"
    )
    set_tests_properties(fyzzxyzzy_owned_sap_hist PROPERTIES
      SKIP_RETURN_CODE 77)
    add_executable(fyzzxyzzy_owned_sap_incremental_pairs tests/physx_gpu_source_sap_incremental_pairs_native.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_incremental_pairs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_incremental_pairs PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_incremental_pairs fyzzxyzzy_owned_sap_incremental_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_incremental_pairs COMMAND fyzzxyzzy_owned_sap_incremental_pairs
      "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_SPV}"
    )
    set_tests_properties(fyzzxyzzy_owned_sap_incremental_pairs PROPERTIES
      SKIP_RETURN_CODE 77)
    # Original SAP rank, projection, changed and removed handle kernels.
    # All layout declarations and stage implementations belong to this SDK.
    set(FYZZXYZZY_OWNED_SAPBOX_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sapbox_indices_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_SAPBOX_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_SAPBOX_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sapbox_indices_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sapbox_indices_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_REMOVED_MARK_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_mark_removed_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REMOVED_MARK_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REMOVED_MARK_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_mark_removed_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_mark_removed_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_REMOVED_PROJ_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_remove_projections_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REMOVED_PROJ_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REMOVED_PROJ_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_remove_projections_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_remove_projections_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_UPDATED_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_updated_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_UPDATED_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_UPDATED_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_updated_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_updated_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_INIT_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_init_ranks_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_init_ranks_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_init_ranks_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_COUNT_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_count_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_SCATTER_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_scatter_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_GATHER_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_gather_handles_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_gather_handles_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_gather_handles_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_COUNT_WAVE_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_count_wave_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_COUNT_WAVE_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_COUNT_WAVE_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_count_wave_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_OWNED_RADIX_SCATTER_WAVE_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_radix_scatter_wave_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_RADIX_SCATTER_WAVE_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_RADIX_SCATTER_WAVE_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_radix_scatter_wave_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_owned_sap_sort_shaders DEPENDS
      ${FYZZXYZZY_OWNED_SAPBOX_SPV}
      ${FYZZXYZZY_OWNED_REMOVED_MARK_SPV}
      ${FYZZXYZZY_OWNED_REMOVED_PROJ_SPV}
      ${FYZZXYZZY_OWNED_UPDATED_SPV}
      ${FYZZXYZZY_OWNED_RADIX_INIT_SPV}
      ${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}
      ${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}
      ${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}
      ${FYZZXYZZY_OWNED_RADIX_COUNT_WAVE_SPV}
      ${FYZZXYZZY_OWNED_RADIX_SCATTER_WAVE_SPV}
    )
    add_executable(fyzzxyzzy_owned_sap_sapbox tests/physx_gpu_source_sapbox_indices_native.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_sapbox PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_sapbox PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_sapbox fyzzxyzzy_owned_sap_sort_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_sapbox COMMAND fyzzxyzzy_owned_sap_sapbox
      "${FYZZXYZZY_OWNED_SAPBOX_SPV}"
    )
    add_executable(fyzzxyzzy_owned_sap_removed tests/physx_gpu_source_sap_removed_native.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_removed PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_removed PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_removed fyzzxyzzy_owned_sap_sort_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_removed COMMAND fyzzxyzzy_owned_sap_removed
      "${FYZZXYZZY_OWNED_REMOVED_MARK_SPV}"
      "${FYZZXYZZY_OWNED_REMOVED_PROJ_SPV}"
    )
    add_executable(fyzzxyzzy_owned_sap_updated tests/physx_gpu_source_sap_updated_native.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_updated PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_updated PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_updated fyzzxyzzy_owned_sap_sort_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_updated COMMAND fyzzxyzzy_owned_sap_updated
      "${FYZZXYZZY_OWNED_UPDATED_SPV}"
    )
    add_executable(fyzzxyzzy_owned_sap_radix tests/physx_gpu_source_sap_radix_native.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_radix PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_sap_radix PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_radix fyzzxyzzy_owned_sap_sort_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_radix COMMAND fyzzxyzzy_owned_sap_radix
      "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}"
      "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}"
      "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}"
      "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}"
      "${FYZZXYZZY_OWNED_RADIX_COUNT_WAVE_SPV}"
      "${FYZZXYZZY_OWNED_RADIX_SCATTER_WAVE_SPV}"
    )
    # Original source GPU region create and histogram kernels.
    set(FYZZXYZZY_OWNED_REGION_CREATE_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_create_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_create_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_create_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_first_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_first_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_first_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_wave_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_wave_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_wave_native.hlsl" VERBATIM)
    set(FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_region_histogram_second_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_second_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_region_histogram_second_native.hlsl" VERBATIM)
    add_custom_target(fyzzxyzzy_owned_region_shaders DEPENDS
      "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}"
      "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
      "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
      "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}"
    )
    add_executable(fyzzxyzzy_owned_region_histograms tests/physx_gpu_source_region_histograms_native.cpp)
    target_include_directories(fyzzxyzzy_owned_region_histograms PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_region_histograms PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_region_histograms fyzzxyzzy_owned_region_shaders)
    add_test(NAME fyzzxyzzy_owned_region_histograms COMMAND fyzzxyzzy_owned_region_histograms
      "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
      "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
      "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}"
    )
    set_tests_properties(fyzzxyzzy_owned_region_histograms PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(fyzzxyzzy_owned_region_create tests/physx_gpu_source_region_create_native.cpp)
    target_include_directories(fyzzxyzzy_owned_region_create PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_region_create PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_region_create fyzzxyzzy_owned_region_shaders)
    add_test(NAME fyzzxyzzy_owned_region_create COMMAND fyzzxyzzy_owned_region_create
      "${FYZZXYZZY_INDEPENDENT_D32_SPV}"
      "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}"
      "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
      "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
      "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}"
    )
    set_tests_properties(fyzzxyzzy_owned_region_create PROPERTIES SKIP_RETURN_CODE 77)

    set(FYZZXYZZY_OWNED_CLEAR_NEW_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_clear_new_flags_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_CLEAR_NEW_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
        -T cs_6_0 -E main -Fo "${FYZZXYZZY_OWNED_CLEAR_NEW_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_clear_new_flags_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_clear_new_flags_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_owned_clear_new_shader
      DEPENDS "${FYZZXYZZY_OWNED_CLEAR_NEW_SPV}")
    add_executable(fyzzxyzzy_owned_clear_new
      tests/physx_gpu_source_clear_new_flags_native.cpp)
    target_include_directories(fyzzxyzzy_owned_clear_new PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_clear_new PRIVATE
      fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_clear_new fyzzxyzzy_owned_clear_new_shader)
    add_test(NAME fyzzxyzzy_owned_clear_new COMMAND fyzzxyzzy_owned_clear_new
      "${FYZZXYZZY_OWNED_CLEAR_NEW_SPV}")

    set(FYZZXYZZY_OWNED_REPORT_SPV_OUTPUTS "")
    foreach(_report_pass IN ITEMS first second copy first_wave second_wave)
      if(_report_pass MATCHES "wave")
        set(_report_model cs_6_6)
      else()
        set(_report_model cs_6_0)
      endif()
      set(_report_src "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_report_${_report_pass}_native.hlsl")
      set(_report_bin "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sap_report_${_report_pass}_native.spv")
      add_custom_command(OUTPUT "${_report_bin}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T "${_report_model}" -E main
          -Fo "${_report_bin}" "${_report_src}"
        DEPENDS "${_report_src}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sap_report_multi_root_common.hlsli"
        VERBATIM)
      list(APPEND FYZZXYZZY_OWNED_REPORT_SPV_OUTPUTS "${_report_bin}")
    endforeach()
    add_custom_target(fyzzxyzzy_owned_report_shaders
      DEPENDS ${FYZZXYZZY_OWNED_REPORT_SPV_OUTPUTS})
    add_executable(fyzzxyzzy_owned_reports
      tests/physx_gpu_source_report_multi_root_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_reports PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_reports PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_reports fyzzxyzzy_owned_report_shaders)
    add_test(NAME fyzzxyzzy_owned_reports COMMAND fyzzxyzzy_owned_reports
      ${FYZZXYZZY_OWNED_REPORT_SPV_OUTPUTS})

    set(FYZZXYZZY_OWNED_PAIR_PRODUCER_SPV_FILES "")
    foreach(_pair_shader IN ITEMS
      physx_sap_overlap_work_native
      physx_sap_overlap_prefix_first_native
      physx_sap_overlap_prefix_second_native
      physx_sap_newbounds_pairs_wave_native)
      set(_source "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/${_pair_shader}.hlsl")
      set(_output "${CMAKE_CURRENT_BINARY_DIR}/kernels/${_pair_shader}_owned.spv")
      add_custom_command(OUTPUT "${_output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_0 -E main -Fo "${_output}" "${_source}"
        DEPENDS "${_source}" VERBATIM)
      list(APPEND FYZZXYZZY_OWNED_PAIR_PRODUCER_SPV_FILES "${_output}")
    endforeach()
    add_custom_target(fyzzxyzzy_owned_pair_producer_shaders
      DEPENDS ${FYZZXYZZY_OWNED_PAIR_PRODUCER_SPV_FILES})
    add_executable(fyzzxyzzy_owned_pairs_to_sc
      tests/physx_gpu_source_pairs_to_sc_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_pairs_to_sc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_pairs_to_sc PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_pairs_to_sc
      fyzzxyzzy_owned_pair_producer_shaders fyzzxyzzy_owned_report_shaders)
    add_test(NAME fyzzxyzzy_owned_pairs_to_sc COMMAND fyzzxyzzy_owned_pairs_to_sc
      ${FYZZXYZZY_OWNED_PAIR_PRODUCER_SPV_FILES} ${FYZZXYZZY_OWNED_REPORT_SPV_OUTPUTS})

    add_executable(fyzzxyzzy_owned_source_roots
      tests/physx_gpu_source_roots_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_source_roots PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_source_roots PRIVATE
      fyzzxyzzy_source_gpu_primitives)
    add_test(NAME fyzzxyzzy_owned_source_roots
      COMMAND fyzzxyzzy_owned_source_roots)

    add_executable(fyzzxyzzy_original_gpu_heap_independent
      tests/physx_original_gpu_heap_independent.cpp)
    target_include_directories(fyzzxyzzy_original_gpu_heap_independent PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_original_gpu_heap_independent PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_test(NAME fyzzxyzzy_original_gpu_heap_independent
      COMMAND fyzzxyzzy_original_gpu_heap_independent)

    add_executable(fyzzxyzzy_sap_source_members_independent
      tests/physx_gpu_sap_source_members_independent.cpp)
    target_include_directories(fyzzxyzzy_sap_source_members_independent PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_sap_source_members_independent PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_test(NAME fyzzxyzzy_sap_source_members_independent
      COMMAND fyzzxyzzy_sap_source_members_independent)

    add_executable(fyzzxyzzy_sap_dma_up_independent
      tests/physx_gpu_sap_dma_up_independent.cpp)
    target_include_directories(fyzzxyzzy_sap_dma_up_independent PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_sap_dma_up_independent PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_test(NAME fyzzxyzzy_sap_dma_up_independent
      COMMAND fyzzxyzzy_sap_dma_up_independent)

    set_tests_properties(fyzzxyzzy_owned_pairs_to_sc PROPERTIES SKIP_RETURN_CODE 77)


    set(FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_translate_aabbs_owned.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
        -T cs_6_0 -E main -Fo "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_translate_aabbs.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_owned_translate_aabbs_shader
      DEPENDS "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}")
    add_executable(fyzzxyzzy_owned_translate_aabbs
      tests/physx_gpu_cuda_aabb_launch.cpp)
    target_include_directories(fyzzxyzzy_owned_translate_aabbs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_owned_translate_aabbs PRIVATE fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_translate_aabbs fyzzxyzzy_owned_translate_aabbs_shader)
    add_test(NAME fyzzxyzzy_owned_translate_aabbs COMMAND fyzzxyzzy_owned_translate_aabbs
      "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}")
    add_executable(fyzzxyzzy_owned_sap_gpu_created_frame
      tests/physx_gpu_sap_owned_created_frame_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_created_frame PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_created_frame PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_created_frame
      fyzzxyzzy_owned_translate_aabbs_shader
      fyzzxyzzy_owned_sap_incremental_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_created_frame
      COMMAND fyzzxyzzy_owned_sap_gpu_created_frame
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
        "${FYZZXYZZY_OWNED_CREATED_SPV}")
    add_executable(fyzzxyzzy_owned_sap_gpu_endpoint_history
      tests/physx_gpu_sap_owned_created_radix_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_endpoint_history PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_endpoint_history PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_endpoint_history
      fyzzxyzzy_owned_translate_aabbs_shader
      fyzzxyzzy_owned_sap_incremental_shaders
      fyzzxyzzy_owned_sap_sort_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_endpoint_history
      COMMAND fyzzxyzzy_owned_sap_gpu_endpoint_history
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
        "${FYZZXYZZY_OWNED_CREATED_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}"
        "${FYZZXYZZY_OWNED_SAPBOX_SPV}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}")
    add_executable(fyzzxyzzy_owned_sap_gpu_region_prefix
      tests/physx_gpu_sap_owned_region_prefix_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_region_prefix PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_region_prefix PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_region_prefix
      fyzzxyzzy_owned_translate_aabbs_shader
      fyzzxyzzy_owned_sap_incremental_shaders
      fyzzxyzzy_owned_sap_sort_shaders
      fyzzxyzzy_owned_region_shaders
      fyzzxyzzy_independent_source_d32_spv)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_region_prefix
      COMMAND fyzzxyzzy_owned_sap_gpu_region_prefix
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
        "${FYZZXYZZY_OWNED_CREATED_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}"
        "${FYZZXYZZY_OWNED_SAPBOX_SPV}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}"
        "${FYZZXYZZY_INDEPENDENT_D32_SPV}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}")
    # Original source active-list compute variants, independent of PhysX source.
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
      set(FYZZXYZZY_OWNED_ACTIVE_${_pass}_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_active_${_pass}_owned.spv")
      add_custom_command(OUTPUT "${FYZZXYZZY_OWNED_ACTIVE_${_pass}_SPV}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T ${_profile} -E main -DFYZZXYZZY_ACTIVE_REGION_WAVE=${_flag}
          -Fo "${FYZZXYZZY_OWNED_ACTIVE_${_pass}_SPV}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_active_${_source}_native.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_active_${_source}_native.hlsl"
        VERBATIM)
    endforeach()
    add_custom_target(fyzzxyzzy_owned_active_shaders DEPENDS
      "${FYZZXYZZY_OWNED_ACTIVE_writeout_SPV}"
      "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_SPV}"
      "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_SPV}"
      "${FYZZXYZZY_OWNED_ACTIVE_output_SPV}")
    add_executable(fyzzxyzzy_owned_sap_gpu_new_pairs
      tests/physx_gpu_sap_owned_new_pairs_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_new_pairs PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_new_pairs PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_new_pairs
      fyzzxyzzy_owned_translate_aabbs_shader
      fyzzxyzzy_owned_sap_incremental_shaders
      fyzzxyzzy_owned_sap_sort_shaders
      fyzzxyzzy_owned_region_shaders
      fyzzxyzzy_independent_source_d32_spv
      fyzzxyzzy_owned_active_shaders
      fyzzxyzzy_owned_source_sap_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_new_pairs
      COMMAND fyzzxyzzy_owned_sap_gpu_new_pairs
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
        "${FYZZXYZZY_OWNED_CREATED_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}"
        "${FYZZXYZZY_OWNED_SAPBOX_SPV}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}"
        "${FYZZXYZZY_INDEPENDENT_D32_SPV}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_SPV}"
        "${FYZZXYZZY_OWNED_OVERLAP_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_PAIRS_SPV}")
    add_executable(fyzzxyzzy_owned_sap_gpu_report_fetch
      tests/physx_gpu_sap_owned_report_fetch_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_report_fetch PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_report_fetch PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_report_fetch
      fyzzxyzzy_owned_translate_aabbs_shader
      fyzzxyzzy_owned_sap_incremental_shaders
      fyzzxyzzy_owned_sap_sort_shaders
      fyzzxyzzy_owned_region_shaders
      fyzzxyzzy_independent_source_d32_spv
      fyzzxyzzy_owned_active_shaders
      fyzzxyzzy_owned_source_sap_shaders
      fyzzxyzzy_owned_report_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_report_fetch
      COMMAND fyzzxyzzy_owned_sap_gpu_report_fetch
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
        "${FYZZXYZZY_OWNED_CREATED_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}"
        "${FYZZXYZZY_OWNED_SAPBOX_SPV}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}"
        "${FYZZXYZZY_INDEPENDENT_D32_SPV}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_SPV}"
        "${FYZZXYZZY_OWNED_OVERLAP_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_PAIRS_SPV}"
        ${FYZZXYZZY_OWNED_REPORT_SPV_OUTPUTS})
    add_executable(fyzzxyzzy_owned_sap_gpu_removed_history
      tests/physx_gpu_sap_owned_removed_history_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_removed_history PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_removed_history PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_removed_history
      fyzzxyzzy_owned_translate_aabbs_shader
      fyzzxyzzy_owned_sap_incremental_shaders
      fyzzxyzzy_owned_sap_sort_shaders
      fyzzxyzzy_owned_region_shaders
      fyzzxyzzy_independent_source_d32_spv
      fyzzxyzzy_owned_active_shaders
      fyzzxyzzy_owned_source_sap_shaders
      fyzzxyzzy_owned_report_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_removed_history
      COMMAND fyzzxyzzy_owned_sap_gpu_removed_history
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
        "${FYZZXYZZY_OWNED_CREATED_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}"
        "${FYZZXYZZY_OWNED_SAPBOX_SPV}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}"
        "${FYZZXYZZY_INDEPENDENT_D32_SPV}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_SPV}"
        "${FYZZXYZZY_OWNED_OVERLAP_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_PAIRS_SPV}"
        ${FYZZXYZZY_OWNED_REPORT_SPV_OUTPUTS}
        "${FYZZXYZZY_OWNED_REMOVED_MARK_SPV}"
        "${FYZZXYZZY_OWNED_REMOVED_PROJ_SPV}")
    add_executable(fyzzxyzzy_owned_sap_gpu_removal_compaction
      tests/physx_gpu_sap_owned_removal_compaction_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_removal_compaction PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_removal_compaction PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_removal_compaction
      fyzzxyzzy_owned_translate_aabbs_shader
      fyzzxyzzy_owned_sap_incremental_shaders
      fyzzxyzzy_owned_sap_sort_shaders
      fyzzxyzzy_owned_region_shaders
      fyzzxyzzy_independent_source_d32_spv
      fyzzxyzzy_owned_active_shaders
      fyzzxyzzy_owned_source_sap_shaders
      fyzzxyzzy_owned_report_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_removal_compaction
      COMMAND fyzzxyzzy_owned_sap_gpu_removal_compaction
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
        "${FYZZXYZZY_OWNED_CREATED_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}"
        "${FYZZXYZZY_OWNED_SAPBOX_SPV}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}"
        "${FYZZXYZZY_INDEPENDENT_D32_SPV}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_SPV}"
        "${FYZZXYZZY_OWNED_OVERLAP_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_PAIRS_SPV}"
        ${FYZZXYZZY_OWNED_REPORT_SPV_OUTPUTS}
        "${FYZZXYZZY_OWNED_REMOVED_MARK_SPV}"
        "${FYZZXYZZY_OWNED_REMOVED_PROJ_SPV}")
    add_executable(fyzzxyzzy_owned_sap_gpu_live_lost_fetch
      tests/physx_gpu_sap_owned_live_lost_fetch_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_live_lost_fetch PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_live_lost_fetch PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_live_lost_fetch
      fyzzxyzzy_owned_translate_aabbs_shader
      fyzzxyzzy_owned_sap_incremental_shaders
      fyzzxyzzy_owned_sap_sort_shaders
      fyzzxyzzy_owned_region_shaders
      fyzzxyzzy_independent_source_d32_spv
      fyzzxyzzy_owned_active_shaders
      fyzzxyzzy_owned_source_sap_shaders
      fyzzxyzzy_owned_report_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_live_lost_fetch
      COMMAND fyzzxyzzy_owned_sap_gpu_live_lost_fetch
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
        "${FYZZXYZZY_OWNED_CREATED_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}"
        "${FYZZXYZZY_OWNED_SAPBOX_SPV}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}"
        "${FYZZXYZZY_INDEPENDENT_D32_SPV}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_SPV}"
        "${FYZZXYZZY_OWNED_OVERLAP_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_PAIRS_SPV}"
        ${FYZZXYZZY_OWNED_REPORT_SPV_OUTPUTS}
        "${FYZZXYZZY_OWNED_REMOVED_MARK_SPV}"
        "${FYZZXYZZY_OWNED_REMOVED_PROJ_SPV}"
        "${FYZZXYZZY_OWNED_UPDATED_SPV}"
        "${FYZZXYZZY_OWNED_COUNT_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_COUNT_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_SPV}")
    add_executable(fyzzxyzzy_owned_sap_gpu_three_frame_retirement
      tests/physx_gpu_sap_owned_three_frame_retirement_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_three_frame_retirement PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_three_frame_retirement PRIVATE
      fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_three_frame_retirement
      fyzzxyzzy_owned_translate_aabbs_shader
      fyzzxyzzy_owned_sap_incremental_shaders
      fyzzxyzzy_owned_sap_sort_shaders
      fyzzxyzzy_owned_region_shaders
      fyzzxyzzy_independent_source_d32_spv
      fyzzxyzzy_owned_active_shaders
      fyzzxyzzy_owned_source_sap_shaders
      fyzzxyzzy_owned_report_shaders)
    add_test(NAME fyzzxyzzy_owned_sap_gpu_three_frame_retirement
      COMMAND fyzzxyzzy_owned_sap_gpu_three_frame_retirement
        "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
        "${FYZZXYZZY_OWNED_CREATED_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}"
        "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}"
        "${FYZZXYZZY_OWNED_SAPBOX_SPV}"
        "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}"
        "${FYZZXYZZY_INDEPENDENT_D32_SPV}"
        "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
        "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_writeout_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_SPV}"
        "${FYZZXYZZY_OWNED_ACTIVE_output_SPV}"
        "${FYZZXYZZY_OWNED_OVERLAP_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_SCAN_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_PAIRS_SPV}"
        ${FYZZXYZZY_OWNED_REPORT_SPV_OUTPUTS}
        "${FYZZXYZZY_OWNED_REMOVED_MARK_SPV}"
        "${FYZZXYZZY_OWNED_REMOVED_PROJ_SPV}"
        "${FYZZXYZZY_OWNED_UPDATED_SPV}"
        "${FYZZXYZZY_OWNED_COUNT_FIRST_SPV}"
        "${FYZZXYZZY_OWNED_COUNT_SECOND_SPV}"
        "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_SPV}")
    # Source Sc box transform-cache and shared bounds update.
    set(FYZZXYZZY_SC_BOX_BOUNDS_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_sc_box_transform_bounds_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SC_BOX_BOUNDS_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SC_BOX_BOUNDS_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sc_box_transform_bounds_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_sc_box_transform_bounds_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_sc_box_bounds_shader
      DEPENDS "${FYZZXYZZY_SC_BOX_BOUNDS_SPV}")
    add_executable(fyzzxyzzy_source_sc_box_bounds
      tests/physx_gpu_source_sc_box_transform_bounds_native.cpp)
    target_include_directories(fyzzxyzzy_source_sc_box_bounds PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_sc_box_bounds PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_sc_box_bounds
      fyzzxyzzy_source_sc_box_bounds_shader)
    add_test(NAME fyzzxyzzy_source_sc_box_bounds
      COMMAND fyzzxyzzy_source_sc_box_bounds "${FYZZXYZZY_SC_BOX_BOUNDS_SPV}")


    set(FYZZXYZZY_GU_BOX_CONTACTS_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_gu_box_box_contacts_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_GU_BOX_CONTACTS_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_GU_BOX_CONTACTS_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_gu_box_box_contacts_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_gu_box_box_contacts_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gu_box_contacts_shader
      DEPENDS "${FYZZXYZZY_GU_BOX_CONTACTS_SPV}")
    add_executable(fyzzxyzzy_source_gu_box_contacts
      tests/physx_gpu_source_gu_box_box_contacts_native.cpp
      src/native/GuContactBoxBox.cpp)
    target_include_directories(fyzzxyzzy_source_gu_box_contacts PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gu_box_contacts PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gu_box_contacts
      fyzzxyzzy_source_sc_box_bounds_shader
      fyzzxyzzy_source_gu_box_contacts_shader)
    add_test(NAME fyzzxyzzy_source_gu_box_contacts
      COMMAND fyzzxyzzy_source_gu_box_contacts
        "${FYZZXYZZY_SC_BOX_BOUNDS_SPV}" "${FYZZXYZZY_GU_BOX_CONTACTS_SPV}")


    set(FYZZXYZZY_SOURCE_PCM_BOX_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_pcm_box_candidates_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_PCM_BOX_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_PCM_BOX_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_candidates_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_candidates_native.hlsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_gjk_native.hlsli"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_math_native.hlsli"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_pcm_box_candidates_shader
      DEPENDS "${FYZZXYZZY_SOURCE_PCM_BOX_SPV}")
    set(FYZZXYZZY_SOURCE_PCM_EPA_WORK_SPV_RESET
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_pcm_epa_reset_native.spv")
    set(FYZZXYZZY_SOURCE_PCM_EPA_WORK_SPV_COMPACT
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_pcm_epa_compact_native.spv")
    foreach(ENTRY reset compact)
      if(ENTRY STREQUAL "reset")
        set(EPA_DEFINE "-DSOURCE_PCM_EPA_RESET=1")
        set(EPA_OUTPUT "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_SPV_RESET}")
      else()
        set(EPA_DEFINE "")
        set(EPA_OUTPUT "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_SPV_COMPACT}")
      endif()
      add_custom_command(OUTPUT "${EPA_OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" ${EPA_DEFINE} -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E "${ENTRY}"
          -Fo "${EPA_OUTPUT}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_epa_work_native.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_epa_work_native.hlsl"
        VERBATIM)
    endforeach()
    add_custom_target(fyzzxyzzy_source_pcm_epa_work_shader
      DEPENDS "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_SPV_RESET}" "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_SPV_COMPACT}")
    set(FYZZXYZZY_SOURCE_PCM_EPA_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_pcm_box_epa_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_PCM_EPA_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_PCM_EPA_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_epa_native.hlsl"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_epa_native.hlsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_epa_native.hlsli"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_math_native.hlsli"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_gjk_native.hlsli"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_pcm_epa_shader
      DEPENDS "${FYZZXYZZY_SOURCE_PCM_EPA_SPV}")
    set(FYZZXYZZY_SOURCE_PCM_CONTACT_STREAM_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_pcm_sc_contact_stream_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_PCM_CONTACT_STREAM_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_PCM_CONTACT_STREAM_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_sc_contact_stream_native.hlsl"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_sc_contact_stream_native.hlsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_pcm_box_math_native.hlsli"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_pcm_contact_stream_shader
      DEPENDS "${FYZZXYZZY_SOURCE_PCM_CONTACT_STREAM_SPV}")
    set(FYZZXYZZY_SOURCE_DY_DISTANCE_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_dy_distance_pgs_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_DY_DISTANCE_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_6 -E main
        -Fo "${FYZZXYZZY_SOURCE_DY_DISTANCE_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_dy_distance_pgs_native.hlsl"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_dy_distance_pgs_native.hlsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_dy_contact_pgs_native.hlsli"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_dy_distance_shader
      DEPENDS "${FYZZXYZZY_SOURCE_DY_DISTANCE_SPV}")
    add_executable(fyzzxyzzy_source_pcm_box_candidates
      tests/physx_gpu_source_pcm_box_candidates_native.cpp
      src/native/GuPCMContactBoxBox.cpp
      src/native/GuPCMGjkEpa.cpp)
    target_include_directories(fyzzxyzzy_source_pcm_box_candidates PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_pcm_box_candidates PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_pcm_box_candidates
      fyzzxyzzy_source_sc_box_bounds_shader
      fyzzxyzzy_source_pcm_box_candidates_shader
      fyzzxyzzy_source_pcm_epa_work_shader
      fyzzxyzzy_source_pcm_epa_shader)
    add_test(NAME fyzzxyzzy_source_pcm_box_candidates
      COMMAND fyzzxyzzy_source_pcm_box_candidates
        "${FYZZXYZZY_SC_BOX_BOUNDS_SPV}" "${FYZZXYZZY_SOURCE_PCM_BOX_SPV}"
        "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_SPV_RESET}" "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_SPV_COMPACT}"
        "${FYZZXYZZY_SOURCE_PCM_EPA_SPV}")


    # Source aggregate.cu::updateAggregateBounds, original 32-lane
    # aggregate member reduction (4 warps per native workgroup).
    set(FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_bounds_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_bounds_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_bounds_native.hlsl"
      VERBATIM)
    set(FYZZXYZZY_SOURCE_AGGREGATE_MARK_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_mark_changed_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_mark_changed_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_mark_changed_native.hlsl"
      VERBATIM)
    add_executable(fyzzxyzzy_source_gpu_aggregate_bounds
      tests/physx_gpu_source_aggregate_bounds_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_bounds PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_bounds PRIVATE FyzzXYZZY::Compute fyzzxyzzy_source_gpu_primitives)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_bounds_shader
      DEPENDS "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_SPV}"
              "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_SPV}")
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_bounds fyzzxyzzy_source_gpu_aggregate_bounds_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_bounds
      COMMAND fyzzxyzzy_source_gpu_aggregate_bounds "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_SPV}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_SPV}")
    # Original aggregate.cu added/deleted member integer-AABB history.
    set(FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_mark_members_history_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_mark_members_history_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_mark_members_history_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_member_history_shader
      DEPENDS "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_SPV}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_member_history
      tests/physx_gpu_source_aggregate_member_history_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_member_history PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_member_history PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_member_history
      fyzzxyzzy_source_gpu_aggregate_member_history_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_member_history
      COMMAND fyzzxyzzy_source_gpu_aggregate_member_history
        "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_SPV}")
    # Source aggregate.cu::sortAndUpdateAggregateProjections.
    set(FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_projection_sort_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_projection_sort_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_projection_sort_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_projection_shader
      DEPENDS "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_SPV}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_projections
      tests/physx_gpu_source_aggregate_projection_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_projections PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_projections PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_projections
      fyzzxyzzy_source_gpu_aggregate_projection_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_projections
      COMMAND fyzzxyzzy_source_gpu_aggregate_projections
        "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_SPV}")
    # aggregate.cu::doSelfCollision, found/lost member history on GPU.
    set(FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_self_collision_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_self_collision_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_self_collision_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_self_collision_shader
      DEPENDS "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_SPV}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_self_collision
      tests/physx_gpu_source_aggregate_self_collision_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_self_collision PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_self_collision PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_self_collision
      fyzzxyzzy_source_gpu_aggregate_projection_shader
      fyzzxyzzy_source_gpu_aggregate_self_collision_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_self_collision
      COMMAND fyzzxyzzy_source_gpu_aggregate_self_collision
        "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_SPV}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_SPV}")
    set(FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_copy_reports_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_copy_reports_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_copy_reports_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_copy_report_shader
      DEPENDS "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_SPV}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_report_copy
      tests/physx_gpu_source_aggregate_report_copy_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_report_copy PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_report_copy PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_report_copy
      fyzzxyzzy_source_gpu_aggregate_copy_report_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_report_copy
      COMMAND fyzzxyzzy_source_gpu_aggregate_report_copy
        "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_SPV}")
    # Original aggregate.cu::addAggPairsStage1 and addAggPairsStage2.
    set(FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_pair_admit_native.spv")
    set(FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_pair_publish_native.spv")
    add_custom_command(OUTPUT
      "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_SPV}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E admit
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_admission_native.hlsl"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E publish
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_admission_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_admission_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_pair_admission_shaders
      DEPENDS
        "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_SPV}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_SPV}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_pair_admission
      tests/physx_gpu_source_aggregate_pair_admission_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_pair_admission PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_pair_admission PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_pair_admission
      fyzzxyzzy_source_gpu_aggregate_pair_admission_shaders)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_pair_admission
      COMMAND fyzzxyzzy_source_gpu_aggregate_pair_admission
        "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_SPV}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_SPV}")
    # aggregate.cu::removeAggPairsStage1/2/3 native GPU translations.
    set(FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_COUNT
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_pair_retire_count_native.spv")
    set(FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_MARK
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_pair_retire_mark_native.spv")
    set(FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_PUBLISH
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_pair_retire_publish_native.spv")
    add_custom_command(OUTPUT
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_COUNT}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_MARK}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_PUBLISH}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E histogram
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_COUNT}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_retirement_native.hlsl"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E mark
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_MARK}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_retirement_native.hlsl"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E publish
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_PUBLISH}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_retirement_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_pair_retirement_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_pair_retirement_shaders
      DEPENDS
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_COUNT}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_MARK}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_PUBLISH}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_pair_retirement
      tests/physx_gpu_source_aggregate_pair_retirement_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_pair_retirement PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_pair_retirement PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_pair_retirement
      fyzzxyzzy_source_gpu_aggregate_pair_retirement_shaders
      fyzzxyzzy_source_gpu_aggregate_pair_admission_shaders)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_pair_retirement
      COMMAND fyzzxyzzy_source_gpu_aggregate_pair_retirement
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_COUNT}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_MARK}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_PUBLISH}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_SPV}"
        "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_SPV}")
    # aggregate.cu::doAggPairCollisions, original GPU root pair member sweep.
    set(FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_SPV
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/physx_source_aggregate_persistent_collisions_native.spv")
    add_custom_command(OUTPUT "${FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_SPV}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
      COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1 -T cs_6_0 -E main
        -Fo "${FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_SPV}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_persistent_collisions_native.hlsl"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/physx_source_aggregate_persistent_collisions_native.hlsl"
      VERBATIM)
    add_custom_target(fyzzxyzzy_source_gpu_aggregate_persistent_collisions_shader DEPENDS "${FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_SPV}")
    add_executable(fyzzxyzzy_source_gpu_aggregate_persistent_collisions
      tests/physx_gpu_source_aggregate_persistent_collisions_native.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_persistent_collisions PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_persistent_collisions PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_source_gpu_primitives)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_persistent_collisions
      fyzzxyzzy_source_gpu_aggregate_persistent_collisions_shader
      fyzzxyzzy_source_gpu_aggregate_projection_shader
      fyzzxyzzy_source_gpu_aggregate_self_collision_shader
      fyzzxyzzy_source_gpu_aggregate_pair_retirement_shaders)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_persistent_collisions COMMAND fyzzxyzzy_source_gpu_aggregate_persistent_collisions
      "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_SPV}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_SPV}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_SPV}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_COUNT}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_MARK}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_PUBLISH}")
    # The production Sc owner requires the complete twelve-stage original
    # aggregate bundle in the same Context as the existing SAP manifest.
    set(FYZZXYZZY_NATIVE_GPU_AGGREGATE_SPV_BINARIES
      "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_SPV}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_SPV}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_SPV}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_SPV}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_SPV}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_ADMIT_SPV}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_PUBLISH_SPV}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_COLLISION_SPV}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_COUNT}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_MARK}"
      "${FYZZXYZZY_SOURCE_AGG_PAIR_RETIRE_SPV_PUBLISH}"
      "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_SPV}")
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
    set(_aggregate_manifest "FYZZXYZZY_SOURCE_AGGREGATE SPIRV\n")
    foreach(_index RANGE 0 11)
      list(GET _aggregate_stage_names ${_index} _stage)
      list(GET FYZZXYZZY_NATIVE_GPU_AGGREGATE_SPV_BINARIES ${_index} _binary)
      get_filename_component(_filename "${_binary}" NAME)
      string(APPEND _aggregate_manifest "${_stage} ${_filename}\n")
    endforeach()
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_aggregate_spirv.manifest" CONTENT "${_aggregate_manifest}")
    add_custom_target(fyzzxyzzy_native_gpu_aggregate_shaders
      DEPENDS ${FYZZXYZZY_NATIVE_GPU_AGGREGATE_SPV_BINARIES})
    add_executable(fyzzxyzzy_source_gpu_aggregate_world_bundle
      tests/physx_gpu_source_aggregate_world_bundle.cpp)
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_world_bundle PRIVATE FyzzXYZZY::NativeGpuWorld)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_world_bundle fyzzxyzzy_native_gpu_aggregate_shaders)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_world_bundle COMMAND fyzzxyzzy_source_gpu_aggregate_world_bundle
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_sap_spirv.manifest"
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_aggregate_spirv.manifest")
    add_executable(fyzzxyzzy_source_sc_aggregate_elements
      tests/physx_gpu_source_sc_aggregate_elements.cpp)
    target_include_directories(fyzzxyzzy_source_sc_aggregate_elements PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_source_sc_aggregate_elements PRIVATE
      FyzzXYZZY::NativeGpuWorld)
    add_test(NAME fyzzxyzzy_source_sc_aggregate_elements COMMAND fyzzxyzzy_source_sc_aggregate_elements)
    add_executable(fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass
      tests/physx_gpu_source_aggregate_aabb_owner_prepass.cpp)
    target_include_directories(fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass PRIVATE
      FyzzXYZZY::NativeGpuWorld)
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass
      fyzzxyzzy_source_gpu_aggregate_bounds_shader
      fyzzxyzzy_source_gpu_aggregate_member_history_shader
      fyzzxyzzy_source_gpu_aggregate_projection_shader
      fyzzxyzzy_source_gpu_aggregate_self_collision_shader
      fyzzxyzzy_source_gpu_aggregate_copy_report_shader)
    add_test(NAME fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass
      COMMAND fyzzxyzzy_source_gpu_aggregate_aabb_owner_prepass
        "${FYZZXYZZY_SOURCE_AGGREGATE_BOUNDS_SPV}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_MARK_SPV}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_MEMBER_HISTORY_SPV}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_PROJECTION_SPV}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_SELF_COLLISION_SPV}"
        "${FYZZXYZZY_SOURCE_AGGREGATE_COPY_REPORT_SPV}")
    add_executable(fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle
      tests/physx_gpu_sap_owned_aabb_scene_lifecycle_independent.cpp)
    target_include_directories(fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    target_link_libraries(fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle PRIVATE
      FyzzXYZZY::NativeGpuWorld)
    add_dependencies(fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle
      fyzzxyzzy_owned_translate_aabbs_shader
      fyzzxyzzy_owned_sap_incremental_shaders
      fyzzxyzzy_owned_sap_sort_shaders
      fyzzxyzzy_owned_region_shaders
      fyzzxyzzy_independent_source_d32_spv
      fyzzxyzzy_owned_active_shaders
      fyzzxyzzy_owned_source_sap_shaders
      fyzzxyzzy_owned_report_shaders
      fyzzxyzzy_owned_clear_new_shader)
    set(FYZZXYZZY_NATIVE_GPU_SAP_SPIRV_BINARIES
      "${FYZZXYZZY_OWNED_TRANSLATE_AABBS_SPV}"
      "${FYZZXYZZY_OWNED_CREATED_SPV}"
      "${FYZZXYZZY_OWNED_RADIX_INIT_SPV}"
      "${FYZZXYZZY_OWNED_RADIX_COUNT_SPV}"
      "${FYZZXYZZY_OWNED_RADIX_SCATTER_SPV}"
      "${FYZZXYZZY_OWNED_RADIX_GATHER_SPV}"
      "${FYZZXYZZY_OWNED_SAPBOX_SPV}"
      "${FYZZXYZZY_OWNED_HIST_FIRST_SPV}"
      "${FYZZXYZZY_OWNED_HIST_SECOND_SPV}"
      "${FYZZXYZZY_INDEPENDENT_D32_SPV}"
      "${FYZZXYZZY_OWNED_REGION_CREATE_SPV}"
      "${FYZZXYZZY_OWNED_REGION_HIST_FIRST_SPV}"
      "${FYZZXYZZY_OWNED_REGION_HIST_WAVE_SPV}"
      "${FYZZXYZZY_OWNED_REGION_HIST_SECOND_SPV}"
      "${FYZZXYZZY_OWNED_ACTIVE_writeout_SPV}"
      "${FYZZXYZZY_OWNED_ACTIVE_compute_wave_SPV}"
      "${FYZZXYZZY_OWNED_ACTIVE_compute_portable_SPV}"
      "${FYZZXYZZY_OWNED_ACTIVE_output_SPV}"
      "${FYZZXYZZY_OWNED_OVERLAP_SPV}"
      "${FYZZXYZZY_OWNED_SCAN_FIRST_SPV}"
      "${FYZZXYZZY_OWNED_SCAN_SECOND_SPV}"
      "${FYZZXYZZY_OWNED_PAIRS_SPV}"
      ${FYZZXYZZY_OWNED_REPORT_SPV_OUTPUTS}
      "${FYZZXYZZY_OWNED_REMOVED_MARK_SPV}"
      "${FYZZXYZZY_OWNED_REMOVED_PROJ_SPV}"
      "${FYZZXYZZY_OWNED_UPDATED_SPV}"
      "${FYZZXYZZY_OWNED_COUNT_FIRST_SPV}"
      "${FYZZXYZZY_OWNED_COUNT_SECOND_SPV}"
      "${FYZZXYZZY_OWNED_INCREMENTAL_PAIRS_SPV}"
      "${FYZZXYZZY_OWNED_CLEAR_NEW_SPV}"
      "${FYZZXYZZY_SC_BOX_BOUNDS_SPV}")
    list(LENGTH FYZZXYZZY_NATIVE_GPU_SAP_SPIRV_BINARIES _sap_shader_count)
    if(NOT _sap_shader_count EQUAL 35)
      message(FATAL_ERROR "Native SPIRV SAP shader bundle does not contain 35 stages")
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
    set(_sap_manifest "FYZZXYZZY_SOURCE_SAP_1 SPIRV\n")
    foreach(_idx RANGE 0 34)
      list(GET _sap_names ${_idx} _stage)
      list(GET FYZZXYZZY_NATIVE_GPU_SAP_SPIRV_BINARIES ${_idx} _binary)
      get_filename_component(_filename "${_binary}" NAME)
      string(APPEND _sap_manifest "${_stage} ${_filename}\n")
    endforeach()
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_sap_spirv.manifest" CONTENT "${_sap_manifest}")
    add_custom_target(fyzzxyzzy_native_gpu_sap_shaders
      DEPENDS ${FYZZXYZZY_NATIVE_GPU_SAP_SPIRV_BINARIES})
    add_dependencies(fyzzxyzzy_source_gpu_aggregate_world_bundle
      fyzzxyzzy_native_gpu_sap_shaders)
    set(_pcm_binaries
      "${FYZZXYZZY_SOURCE_PCM_BOX_SPV}"
      "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_SPV_RESET}"
      "${FYZZXYZZY_SOURCE_PCM_EPA_WORK_SPV_COMPACT}"
      "${FYZZXYZZY_SOURCE_PCM_EPA_SPV}"
      "${FYZZXYZZY_SOURCE_PCM_CONTACT_STREAM_SPV}")
    set(_pcm_stages Candidates EpaReset EpaCompact EpaContact ContactStream)
    set(_pcm_manifest "FYZZXYZZY_SOURCE_PCM_1 SPIRV\n")
    foreach(_idx RANGE 0 4)
      list(GET _pcm_stages ${_idx} _stage)
      list(GET _pcm_binaries ${_idx} _binary)
      get_filename_component(_name "${_binary}" NAME)
      string(APPEND _pcm_manifest "${_stage} ${_name}\n")
    endforeach()
    file(GENERATE
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_pcm_spv.manifest"
      CONTENT "${_pcm_manifest}")
    add_custom_target(fyzzxyzzy_native_gpu_pcm_shaders
      DEPENDS ${_pcm_binaries})
    add_executable(fyzzxyzzy_source_sc_gpu_pcm_scene
      tests/physx_gpu_source_sc_pcm_scene_native.cpp)
    target_include_directories(fyzzxyzzy_source_sc_gpu_pcm_scene PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
    
    target_link_libraries(fyzzxyzzy_source_sc_gpu_pcm_scene PRIVATE
      FyzzXYZZY::NativeGpuWorld)
    add_dependencies(fyzzxyzzy_source_sc_gpu_pcm_scene
      fyzzxyzzy_native_gpu_sap_shaders
      fyzzxyzzy_native_gpu_pcm_shaders
      fyzzxyzzy_source_dy_distance_shader)
    add_test(NAME fyzzxyzzy_source_sc_gpu_pcm_scene
      COMMAND fyzzxyzzy_source_sc_gpu_pcm_scene
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_sap_spirv.manifest"
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_pcm_spv.manifest"
        "${FYZZXYZZY_SOURCE_DY_DISTANCE_SPV}")
    
    add_test(NAME fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle
      COMMAND fyzzxyzzy_owned_sap_gpu_aabb_scene_lifecycle
        ${FYZZXYZZY_NATIVE_GPU_SAP_SPIRV_BINARIES}
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/fyzzxyzzy_native_sap_spirv.manifest")









  endif()

  if(FYZZXYZZY_ENABLE_PHYSX_SOURCE)
  # Translated PhysX GPU stages. These operations preserve upstream data
  # contracts; the PhysX scene integration is introduced incrementally.
  add_library(fyzzxyzzy_physx_gpu_stages STATIC
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
    src/physx_gpu/broadphase_completion.cpp
  )
  target_compile_features(fyzzxyzzy_physx_gpu_stages PUBLIC cxx_std_23)
  target_include_directories(fyzzxyzzy_physx_gpu_stages PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
  target_link_libraries(fyzzxyzzy_physx_gpu_stages PUBLIC
    FyzzXYZZY::Compute physx_lib fyzzxyzzy_physx_cuda_host_address
    fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_source_gpu_primitives)
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
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/simulationcontroller/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
    target_link_libraries(fyzzxyzzy_physx_cuda_host_transfer PRIVATE
      FyzzXYZZY::Compute fyzzxyzzy_physx_cuda_host_address fyzzxyzzy_physx_original_gpu_heap physx_lib fyzzxyzzy_physx_gpu_stages)
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
    if(FYZZXYZZY_BUILD_PHYSX_GPU_AABB_TEST OR FYZZXYZZY_BUILD_SOURCE_GPU_TESTS)
      find_program(FYZZXYZZY_DXC NAMES dxc REQUIRED)
      # Verify a physically fixed 32-lane subgroup through actual shader execution.
      set(FYZZXYZZY_WAVE32_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/kernels/native_wave32_contract.spv")
      add_custom_command(
        OUTPUT "${FYZZXYZZY_WAVE32_SPV}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${FYZZXYZZY_DXC}" -spirv -fspv-target-env=vulkan1.1
          -T cs_6_6 -E main -Fo "${FYZZXYZZY_WAVE32_SPV}"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_wave32_contract.hlsl"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_gpu/kernels/native_wave32_contract.hlsl"
        VERBATIM)
      add_custom_target(fyzzxyzzy_wave32_spv DEPENDS "${FYZZXYZZY_WAVE32_SPV}")
      add_executable(fyzzxyzzy_wave32_contract tests/native_wave32_contract.cpp)
      target_link_libraries(fyzzxyzzy_wave32_contract PRIVATE FyzzXYZZY::Compute)
      add_dependencies(fyzzxyzzy_wave32_contract fyzzxyzzy_wave32_spv)
      add_test(NAME fyzzxyzzy_wave32_contract
        COMMAND fyzzxyzzy_wave32_contract "${FYZZXYZZY_WAVE32_SPV}")

    endif()
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
        PRIVATE fyzzxyzzy_source_gpu_primitives)
      add_dependencies(fyzzxyzzy_physx_aabb_encoding fyzzxyzzy_physx_aabb_shader)
      add_test(NAME fyzzxyzzy_physx_aabb_encoding
        COMMAND fyzzxyzzy_physx_aabb_encoding "${FYZZXYZZY_PHYSX_AABB_SHADER}")
      # Original CUDA _launch ABI: 11 argument pointers, original 256-thread
      # block layout, CUdeviceptr interior offsets, native shader lowering.
      add_executable(fyzzxyzzy_physx_cuda_aabb_launch
        tests/physx_gpu_cuda_aabb_launch.cpp)
      target_include_directories(fyzzxyzzy_physx_cuda_aabb_launch PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_link_libraries(fyzzxyzzy_physx_cuda_aabb_launch PRIVATE
        fyzzxyzzy_source_gpu_primitives)
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
        fyzzxyzzy_source_gpu_primitives)
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
    # Explicit, hardware-dependent profiling; never part of CTest or CI
    # pass/fail thresholds. Compile the real source WR radix shaders.
    add_custom_target(fyzzxyzzy_source_sap_radix_profile_vulkan
      COMMAND "$<TARGET_FILE:fyzzxyzzy_physx_source_sap_radix_native>"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_SPV}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_SPV}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_SPV}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_SPV}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_SPV}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_SPV}"
        --benchmark
      DEPENDS fyzzxyzzy_physx_source_sap_radix_native
      USES_TERMINAL VERBATIM)
    add_custom_target(fyzzxyzzy_source_sap_radix_profile_alternating_vulkan
      COMMAND "$<TARGET_FILE:fyzzxyzzy_physx_source_sap_radix_native>"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_INIT_RANKS_SPV}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_SPV}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_SPV}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_GATHER_HANDLES_SPV}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_COUNT_WAVE_SPV}"
        "${FYZZXYZZY_SOURCE_SAP_RADIX_RADIX_SCATTER_WAVE_SPV}"
        --benchmark-alternating
      DEPENDS fyzzxyzzy_physx_source_sap_radix_native
      USES_TERMINAL VERBATIM)

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

    # Source AABB-manager handles and GPU reports must lower to Sc
    # without a second volume registry or partial late publication.
    add_executable(fyzzxyzzy_physx_source_sap_scene_boundary tests/physx_gpu_source_sap_scene_boundary.cpp)
    target_include_directories(fyzzxyzzy_physx_source_sap_scene_boundary PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
      "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include")
    target_link_libraries(fyzzxyzzy_physx_source_sap_scene_boundary PRIVATE fyzzxyzzy_physx_gpu_stages)
    add_test(NAME fyzzxyzzy_physx_source_sap_scene_boundary COMMAND fyzzxyzzy_physx_source_sap_scene_boundary)








    
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
