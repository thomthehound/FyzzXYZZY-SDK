#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0
"""Enforce the native World/reference PhysX source boundary at review time."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent

def main() -> int:
    problems = []
    for relative in ("include/fyzzxyzzy/world.h", "src/world.cpp"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        if re.search(r'^\s*#\s*include\s*[<"](?:Px|physx/)', source, re.MULTILINE):
            problems.append(f"{relative}: reference PhysX header in native API/core")
        if "physx::" in source:
            problems.append(f"{relative}: native API/core refers to PhysX namespace")

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    if not re.search(r'add_library\(fyzzxyzzy_reference_physx_cpu\s+STATIC\s+src/reference/physx_cpu_world\.cpp\)', cmake):
        problems.append("CMakeLists.txt: missing separately compiled reference CPU owner")
    if not re.search(r'target_link_libraries\(fyzzxyzzy_core\s+PRIVATE\s+fyzzxyzzy_reference_physx_cpu\)', cmake):
        problems.append("CMakeLists.txt: native core must link reference through its named adapter")
    if re.search(r'target_link_libraries\(fyzzxyzzy_core\s+PRIVATE\s+physx_lib\)', cmake):
        problems.append("CMakeLists.txt: native core directly links PhysX reference source")
    if not (ROOT / "src/reference/physx_cpu_world.cpp").is_file():
        problems.append("src/reference/physx_cpu_world.cpp: reference backend absent")

    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        return 1
    print("Native World/PhysX reference source boundary PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
