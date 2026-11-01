# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

# Each source Sc scenario has its own timeout and identifies a specific
# failure without repeating shader builds or dropping the coupled Dy gates.
function(fyzzxyzzy_register_sc_gpu_pcm_tests target sap_manifest pcm_manifest dy_shader)
  foreach(_scenario IN ITEMS
      scene sparse_contacts coupled_stack coupled_solver distance_solver)
    if(_scenario STREQUAL "scene")
      set(_name "${target}")
    else()
      set(_name "${target}_${_scenario}")
    endif()
    add_test(NAME "${_name}"
      COMMAND "${target}" "${sap_manifest}" "${pcm_manifest}"
        "${dy_shader}" "${_scenario}")
    set_tests_properties("${_name}" PROPERTIES LABELS "native_gpu_scene")
    if("${target}" MATCHES "_d3d12$")
      set_tests_properties("${_name}" PROPERTIES TIMEOUT 360)
      # The small publication case exercises D3D12 validation. Full
      # multiframe WARP parity runs without the debug-layer overhead.
      if(_scenario STREQUAL "scene")
        set_tests_properties("${_name}" PROPERTIES
          ENVIRONMENT "FYZZXYZZY_D3D12_DEBUG=1")
      endif()
    endif()
  endforeach()
endfunction()
