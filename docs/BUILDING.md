<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CPU SDK build and test

The CPU implementation uses **upstream PhysX CPU** as its physics implementation, through the
Godot-independent `FyzzXYZZY::Core` C++ target. CUDA is disabled. The shader translation and
portable GPU runtimes are separate milestones.

## Source layout and provenance

- During the transition to an in-tree PhysX-derived engine, CMake uses
  `thomthehound/PhysX` **main** as a source bootstrap only; no PhysX
  revision is pinned to the product.
- CMake fetches the current development branch when
  `FYZZXYZZY_PHYSX_SOURCE_DIR` is unset. For offline or reproducible
  development, supply a local checkout and manage its contents yourself.
  Upstream revision tracking belongs to individual test reports, not to a
  required public provider ABI.
- For offline or editable builds, point `FYZZXYZZY_PHYSX_SOURCE_DIR` at the **repository
  root** of a local PhysX checkout (the folder containing `physx/CMakeLists.txt`).
- The `fyzzxyzzy_source_bundle` target produces an offline-configurable CPU source
  archive containing the PhysX SDK tree and both upstream license documents.
- Repository-level source vendoring remains a separate source-integration task.

## Linux: Ninja

Prerequisites: CMake 3.21+, Ninja, supported C++ compiler, and Git (only if fetching source).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target fyzzxyzzy_headless_smoke fyzzxyzzy_physx_cuda_address_test
ctest --test-dir build --output-on-failure
```

For a local PhysX checkout, add
`-DFYZZXYZZY_PHYSX_SOURCE_DIR=/absolute/path/to/PhysX` to the configure command.

## Windows: Developer PowerShell

Prerequisites: Visual Studio 2022 C++ tools, CMake 3.21+, Ninja, and Git (only if
fetching source). Invoke from a Developer PowerShell environment where `cl` is available.

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target fyzzxyzzy_headless_smoke fyzzxyzzy_physx_cuda_address_test
ctest --test-dir build --output-on-failure
```

For a local checkout, append
`-DFYZZXYZZY_PHYSX_SOURCE_DIR="C:/path/to/PhysX"` to configuration.

## Source distribution

Build the offline CPU source archive after CMake configuration:

```sh
cmake --build build --target fyzzxyzzy_source_bundle
```

The resulting `build/fyzzxyzzy-source.tar.gz` contains the FyzzXYZZY SDK
and `vendor/PhysX/physx`. It also preserves the upstream repository-level
BSD 3-Clause `vendor/PhysX/LICENSE.md` and PhysX SDK Apache-2.0
`vendor/PhysX/physx/LICENSE.md`.

Extract the archive and configure the CPU library without network access:

```sh
cmake -E tar xzf build/fyzzxyzzy-source.tar.gz
cmake -S FyzzXYZZY-SDK -B FyzzXYZZY-SDK/build -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build FyzzXYZZY-SDK/build --target fyzzxyzzy_headless_smoke fyzzxyzzy_physx_cuda_address_test
ctest --test-dir FyzzXYZZY-SDK/build --output-on-failure
```

The bundled PhysX source takes precedence over Git fetching unless
`FYZZXYZZY_PHYSX_SOURCE_DIR` is specified. Optional Godot builds require
godot-cpp separately and are not offline-complete in this archive.

## Expected CPU smoke result

The test creates a floor and dynamic box, advances 180 fixed 60 Hz steps, and checks
that the box falls, contacts the floor and settles.

The CPU test suite also checks the internal 64-bit CUDA-style host
device-address registry, including >4 GiB virtual offsets, allocation bounds,
and stale pointer rejection (without allocating GPU memory). Both executables
are explicitly built in the commands above, including offline builds.
When Vulkan or D3D12 is enabled, the additional CUDA-host transfer tests
exercise actual GPU memory copies and deferred resource retirement.

The Linux and Windows CI jobs build and execute this test. The Godot adapter
consumes the same FyzzXYZZY core. Both Godot acceptance jobs execute the headless
runtime script directly and require a successful completion marker.

## Headers

New FyzzXYZZY-authored C++ and CMake files use:
`SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound` and
`SPDX-License-Identifier: Apache-2.0`.
Upstream and third-party files retain their original license and provenance.

## Optional Godot 4.7.2 native consumer

The `FyzzXYZZY::Core` library remains independently buildable. To compile the
GDExtension and copy it into the sample project, add
`-DFYZZXYZZY_BUILD_GODOT=ON` to the usual CMake configuration, then build the
`fyzzxyzzy_godot` target.

godot-cpp 10.0.0 is compiled against the Godot 4.7 API. The resulting
library is copied to `godot/test_project/bin`, and the extension descriptor
is `godot/test_project/addons/fyzzxyzzy/fyzzxyzzy.gdextension`.

After installing Godot 4.7.2, execute the headless acceptance test:

```sh
godot --headless --path godot/test_project --script res://tests/headless_smoke.gd
```

The test loads `res://addons/fyzzxyzzy/fyzzxyzzy.gdextension` explicitly when the
Godot extension registry has not yet been generated. On Windows, use the
console executable for command-line testing and process exit status.

The Godot acceptance test constructs the same type of FyzzXYZZY-managed rigid
scene as the standalone test, advances 180 physics steps, and checks the
contact/settling outcome. Godot does not perform its physics computation.
The adapter exposes box creation, simulation stepping, position and velocity.
This is the initial SDK-consumer test, not the complete Godot PhysicsServer3D
backend.

## Godot 4.7.2 editor import

For a fresh sample-project cache, bootstrap headless editor import in recovery
mode, followed by the regular editor-import pass:

```sh
godot --headless --recovery-mode --editor --import --path godot/test_project
godot --headless --editor --import --path godot/test_project
```

This sequence addresses a Godot 4.7.2 first-run headless-editor shutdown defect
when the project discovers a GDExtension registering a native class. See
[godotengine/godot#123511](https://github.com/godotengine/godot/issues/123511).
The issue was closed without a demonstrated engine fix. Recovery mode prepares
the extension registry for the subsequent normal editor import.

CI validates this sequence on Linux and Windows, and then runs the FyzzXYZZY
simulation in the prepared project. The separate runtime acceptance test remains
required.
