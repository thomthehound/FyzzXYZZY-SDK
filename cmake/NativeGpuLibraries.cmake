# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

# The source GPU primitives and World use identical sources and interfaces
# on both compute backends. Backend-specific targets supply the executor.
function(fyzzxyzzy_add_native_gpu_libraries primitives world world_alias compute)
  add_library(${primitives} STATIC
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
  target_compile_features(${primitives} PUBLIC cxx_std_23)
  target_include_directories(${primitives} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(${primitives} PUBLIC
    ${compute} fyzzxyzzy_physx_cuda_host_address)
  set_target_properties(${primitives} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_library(${world} STATIC src/physx_gpu/native_gpu_world.cpp)
  add_library(${world_alias} ALIAS ${world})
  target_compile_features(${world} PUBLIC cxx_std_23)
  target_include_directories(${world}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/physx_port")
  target_link_libraries(${world} PUBLIC
    FyzzXYZZY::Core ${primitives}
    fyzzxyzzy_physx_original_gpu_heap fyzzxyzzy_physx_foundation_lifetime)
  set_target_properties(${world} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# Compute backends share the same source-stage implementation and dependencies.
function(fyzzxyzzy_add_gpu_source_stages target compute primitives)
  add_library(${target} STATIC
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
  target_compile_features(${target} PUBLIC cxx_std_23)
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpucommon/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/gpubroadphase/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevelaabb/include"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/common/src"
    "${fyzzxyzzy_physx_SOURCE_DIR}/physx/source/lowlevel/software/include")
  target_link_libraries(${target} PUBLIC
    ${compute} physx_lib fyzzxyzzy_physx_cuda_host_address
    fyzzxyzzy_physx_original_gpu_heap ${primitives})
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()
