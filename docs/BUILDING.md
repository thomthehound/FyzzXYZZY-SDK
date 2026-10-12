<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M0 headless build (provisional)

M0 deliberately uses **upstream PhysX CPU** as its physics implementation, through the
Godot-independent `FizzX::Core` C++ target. CUDA is disabled. The shader translation and
portable GPU runtimes are separate milestones.

## Source layout and provenance

- `thomthehound/PhysX` at commit `6dc53585d7db07346f53d0598a5fd3bfdce47d8d`
  is the **temporary pinned M0 source dependency**.
- CMake will obtain that checkout over Git if `FIZZX_PHYSX_SOURCE_DIR` is unset.
- For offline or editable builds, point `FIZZX_PHYSX_SOURCE_DIR` at the **repository
  root** of a local PhysX checkout (the folder containing `physx/CMakeLists.txt`).
- Importing needed PhysX source into FizzX, with its attribution and licenses, is
  part of M0 packaging.

## Linux: Ninja

Prerequisites: CMake 3.21+, Ninja, supported C++ compiler, and Git (only if fetching source).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target fizzx_headless_smoke
ctest --test-dir build --output-on-failure
```

For a local PhysX checkout, add
`-DFIZZX_PHYSX_SOURCE_DIR=/absolute/path/to/PhysX` to the configure command.

## Windows: Developer PowerShell

Prerequisites: Visual Studio 2022 C++ tools, CMake 3.21+, Ninja, and Git (only if
fetching source). Invoke from a Developer PowerShell environment where `cl` is available.

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target fizzx_headless_smoke
ctest --test-dir build --output-on-failure
```

For a local checkout, append
`-DFIZZX_PHYSX_SOURCE_DIR="C:/path/to/PhysX"` to configuration.

## Expected M0 smoke result

The test creates a floor and dynamic box, advances 180 fixed 60 Hz steps, and checks
that the box falls, contacts the floor and settles.

The Linux and Windows CI jobs build and execute this test. The Godot adapter
consumes the same FizzX core. Both Godot acceptance jobs execute the headless
runtime script directly and require a successful completion marker.

## Headers

New FizzX-authored C++ and CMake files use:
`SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound` and
`SPDX-License-Identifier: Apache-2.0`.
Upstream and third-party files retain their original license and provenance.

## Optional Godot 4.7.2 native consumer

The `FizzX::Core` library remains independently buildable. To compile the
GDExtension and copy it into the sample project, add
`-DFIZZX_BUILD_GODOT=ON` to the usual CMake configuration, then build the
`fizzx_godot` target.

godot-cpp 10.0.0 is compiled against the Godot 4.7 API. The resulting
library is copied to `godot/test_project/bin`, and the extension descriptor
is `godot/test_project/addons/fizzx/fizzx.gdextension`.

After installing Godot 4.7.2, execute the headless acceptance test:

```sh
godot --headless --path godot/test_project --script res://tests/headless_smoke.gd
```

The test loads `res://addons/fizzx/fizzx.gdextension` explicitly when the
Godot extension registry has not yet been generated. On Windows, use the
console executable for command-line testing and process exit status.

The Godot acceptance test constructs the same type of FizzX-managed rigid
scene as the standalone test, advances 180 physics steps, and checks the
contact/settling outcome. Godot does not perform its physics computation.
The adapter exposes box creation, simulation stepping, position and velocity.
This is the initial SDK-consumer test, not the complete Godot PhysicsServer3D
backend.

## Godot editor import coverage

The Linux editor-import run exited with SIGABRT (134) after project initialization
on the first integration build. This remains an open editor-integration defect.
The cross-platform M0 acceptance gate tests the runtime extension directly.
Linux job diagnostics include a stack trace on a runtime crash when GDB is available.
