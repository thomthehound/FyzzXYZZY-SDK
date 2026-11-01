# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

# A GPU fixture declares its compiled sources, shader dependencies, and
# runtime inputs together. Test properties remain local to each fixture.
function(fyzzxyzzy_add_gpu_test name)
  cmake_parse_arguments(PARSE_ARGV 1 TEST ""
    "" "SOURCES;DEFINITIONS;INCLUDES;LIBRARIES;DEPENDS;ARGS")
  if(TEST_UNPARSED_ARGUMENTS OR TEST_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "Invalid GPU test declaration: ${name}")
  endif()
  if(NOT TEST_SOURCES OR NOT TEST_LIBRARIES)
    message(FATAL_ERROR "GPU test must specify sources and libraries: ${name}")
  endif()

  add_executable(${name} ${TEST_SOURCES})
  if(TEST_DEFINITIONS)
    target_compile_definitions(${name} PRIVATE ${TEST_DEFINITIONS})
  endif()
  if(TEST_INCLUDES)
    target_include_directories(${name} PRIVATE ${TEST_INCLUDES})
  endif()
  target_link_libraries(${name} PRIVATE ${TEST_LIBRARIES})
  if(TEST_DEPENDS)
    add_dependencies(${name} ${TEST_DEPENDS})
  endif()
  add_test(NAME ${name} COMMAND ${name} ${TEST_ARGS})
endfunction()
