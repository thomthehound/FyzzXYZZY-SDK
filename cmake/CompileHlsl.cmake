# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

# One shader compilation contract for both native GPU formats.
function(fyzzxyzzy_compile_hlsl output source profile entry)
  cmake_parse_arguments(PARSE_ARGV 4 HLSL "" "SPIRV_ENV" "FLAGS;DEPENDS")
  if(HLSL_UNPARSED_ARGUMENTS OR HLSL_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "Invalid HLSL compilation arguments for ${output}")
  endif()
  get_filename_component(_extension "${output}" EXT)
  if(_extension STREQUAL ".spv")
    if(HLSL_SPIRV_ENV)
      set(_environment "${HLSL_SPIRV_ENV}")
    else()
      set(_environment "vulkan1.1")
    endif()
    set(_format_flags -spirv "-fspv-target-env=${_environment}")
  elseif(_extension STREQUAL ".dxil")
    if(HLSL_SPIRV_ENV)
      message(FATAL_ERROR "DXIL target cannot request a SPIR-V environment")
    endif()
    set(_format_flags)
  else()
    message(FATAL_ERROR "Unrecognized HLSL output format: ${output}")
  endif()

  add_custom_command(
    OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
    COMMAND "${FYZZXYZZY_DXC}" ${HLSL_FLAGS} ${_format_flags}
      -T "${profile}" -E "${entry}" -Fo "${output}" "${source}"
    DEPENDS "${source}" ${HLSL_DEPENDS}
    VERBATIM)
endfunction()
