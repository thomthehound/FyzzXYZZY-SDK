# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

foreach(required_var FYZZXYZZY_SOURCE_ROOT FYZZXYZZY_PHYSX_ROOT FYZZXYZZY_BUNDLE_OUTPUT)
  if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
    message(FATAL_ERROR "Missing ${required_var}")
  endif()
endforeach()

if(NOT EXISTS "${FYZZXYZZY_PHYSX_ROOT}/physx/CMakeLists.txt" OR
   NOT EXISTS "${FYZZXYZZY_PHYSX_ROOT}/LICENSE.md" OR
   NOT EXISTS "${FYZZXYZZY_PHYSX_ROOT}/physx/LICENSE.md")
  message(FATAL_ERROR "PhysX SDK source and its two license files are required")
endif()

get_filename_component(bundle_output_dir "${FYZZXYZZY_BUNDLE_OUTPUT}" DIRECTORY)
set(bundle_stage "${bundle_output_dir}/fyzzxyzzy-source-stage")
set(bundle_root "${bundle_stage}/FyzzXYZZY-SDK")
file(REMOVE_RECURSE "${bundle_stage}")
file(MAKE_DIRECTORY "${bundle_root}/vendor/PhysX")

# Copy the SDK source, consumer sources, tests, and build documentation.
foreach(relative_path
    CMakeLists.txt LICENSE.md README.md .gitignore
    cmake docs include src tests godot)
  if(NOT EXISTS "${FYZZXYZZY_SOURCE_ROOT}/${relative_path}")
    message(FATAL_ERROR "Missing FyzzXYZZY source path: ${relative_path}")
  endif()
  file(COPY "${FYZZXYZZY_SOURCE_ROOT}/${relative_path}"
    DESTINATION "${bundle_root}"
    PATTERN ".git" EXCLUDE
    PATTERN ".godot" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
    PATTERN "bin" EXCLUDE)
endforeach()

# Preserve the PhysX SDK tree and both upstream license documents.
file(COPY "${FYZZXYZZY_PHYSX_ROOT}/LICENSE.md"
  DESTINATION "${bundle_root}/vendor/PhysX")
file(COPY "${FYZZXYZZY_PHYSX_ROOT}/physx"
  DESTINATION "${bundle_root}/vendor/PhysX"
  PATTERN ".git" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN "bin" EXCLUDE)

foreach(relative_path
    CMakeLists.txt LICENSE.md
    vendor/PhysX/LICENSE.md
    vendor/PhysX/physx/LICENSE.md
    vendor/PhysX/physx/CMakeLists.txt
    vendor/PhysX/physx/source/compiler/cmake/CMakeLists.txt)
  if(NOT EXISTS "${bundle_root}/${relative_path}")
    message(FATAL_ERROR "Source distribution missing: ${relative_path}")
  endif()
endforeach()

file(REMOVE "${FYZZXYZZY_BUNDLE_OUTPUT}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E tar czf "${FYZZXYZZY_BUNDLE_OUTPUT}" FyzzXYZZY-SDK
  WORKING_DIRECTORY "${bundle_stage}"
  RESULT_VARIABLE archive_status)
if(NOT archive_status EQUAL 0)
  message(FATAL_ERROR "Source archive creation failed: ${archive_status}")
endif()
file(REMOVE_RECURSE "${bundle_stage}")
message(STATUS "Source archive: ${FYZZXYZZY_BUNDLE_OUTPUT}")
