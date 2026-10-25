<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CPU SDK build and test

The optional CPU **reference backend** uses PhysX CPU through
`FyzzXYZZY::ReferencePhysXCpu`. `FyzzXYZZY::Core` provides an independent
native CPU rigid scene by default. Reference execution must be explicitly
selected and requires `FYZZXYZZY_ENABLE_PHYSX_SOURCE=ON`. CUDA is disabled.
Native GPU execution remains a separate migration milestone.


## Reference-free native build (intermediate ownership proof)

Set `FYZZXYZZY_ENABLE_PHYSX_SOURCE=OFF` to build the native core and
standalone compute backend without a reference-runtime dependency.
The default `World` runs the admitted independent CPU rigid simulation;
explicit reference requests and native GPU execution remain unavailable
without their corresponding backends.

The source option defaults to `ON` when `BUILD_TESTING=ON`, so source
conformance fixtures remain available. It defaults to `OFF` for production
builds with `BUILD_TESTING=OFF`. The admitted CPU rigid subset does not
imply complete CPU shape coverage or native GPU simulation.

Linux (Vulkan SDK installed):

```sh
cmake -S . -B build-native -G Ninja -DBUILD_TESTING=ON \\
  -DFYZZXYZZY_ENABLE_PHYSX_SOURCE=OFF \\
  -DFYZZXYZZY_BUILD_VULKAN=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build build-native --parallel 4
ctest --test-dir build-native --output-on-failure
```

Windows (Visual Studio C++23 and Windows SDK):

```powershell
cmake -S . -B build-native -A x64 -DBUILD_TESTING=ON `
  -DFYZZXYZZY_ENABLE_PHYSX_SOURCE=OFF `
  -DFYZZXYZZY_BUILD_D3D12=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build build-native --config Release --parallel 4
ctest --test-dir build-native -C Release --output-on-failure
```

## Native CPU SAP profiling

The `fyzzxyzzy_native_sap_benchmark` target is available in test builds,
including reference-free configurations. It records CPU query time and
the scene-owned full-sort/local-shift and incremental-pair-update
counters for unchanged AABBs, adjacent-key motion, repeated genuine
contact-manager transitions, wide teleports and scenes with many retired
BodyIds. Sparse candidate pairs originate from three-axis
min/max endpoint crossings, with orthogonal overlap tests and
preservation of prior pairs requiring Lost publication. The source-style
hashed `SapPairManager` retains finite broadphase pairs independently
of Sc contact-manager EdgeIndices. Multi-volume endpoint updates use
coalesced activity pockets. Additional profiles cover persistent
pair sets, static-only membership changes and activity-pocket updates.
The profiler reports endpoint crossings, activity pockets and pair
transitions; timings are diagnostic, not conformance thresholds.

Linux (after configuring `build-native` above):

```sh
cmake --build build-native --target fyzzxyzzy_native_sap_benchmark
./build-native/fyzzxyzzy_native_sap_benchmark
```

Windows (after configuring `build-native` above):

```powershell
cmake --build build-native --config Release --target fyzzxyzzy_native_sap_benchmark
.\\build-native\\Release\\fyzzxyzzy_native_sap_benchmark.exe
```

A benchmark row measures the full native collision-only query, including
source pair and Gu publication; it is not a GPU kernel timer or a pure
endpoint-sort timer. Compare matching scenarios and optimized builds on
the same hardware and toolchain. CI runner timings are diagnostic only,
never pass/fail performance thresholds.

## C++ toolchain requirement

FyzzXYZZY-owned native libraries, test executables and the Godot GDExtension
require **C++23**, including a standard library with `std::expected`.
CMake must be 3.21 or newer. The SDK uses standard C++23 mode, not GNU
dialect extensions; the build fails rather than falling back to C++20/17.
The `FyzzXYZZY::Core` target propagates its C++23 compile requirement to
downstream native consumers. This does not change the upstream PhysX
source's independently configured language requirements.

An unconditional CMake configure-time compile-and-link check of
`tests/cpp23_contract.cpp` verifies C++23 `if consteval` and
`std::expected`, including in production with `BUILD_TESTING=OFF`.
When testing is enabled, the same file is also built and executed as
`fyzzxyzzy_cpp23_contract` in CTest. Both platform CI jobs run the
test before longer shader and full SDK builds; Linux's offline source
distribution CI also configures a `BUILD_TESTING=OFF` production build.

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

Prerequisites: CMake 3.21+, Ninja, a C++23-capable compiler and standard library (GCC 13+ is used by Linux CI), and Git (only if fetching source).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target fyzzxyzzy_cpp23_contract fyzzxyzzy_headless_smoke fyzzxyzzy_physx_cuda_address_test
ctest --test-dir build --output-on-failure
```

For a local PhysX checkout, add
`-DFYZZXYZZY_PHYSX_SOURCE_DIR=/absolute/path/to/PhysX` to the configure command.

## Windows: Developer PowerShell

Prerequisites: current Visual Studio 2022 C++ tools with C++23 preview/latest mode, CMake 3.21+, Ninja, and Git (only if
fetching source). Invoke from a Developer PowerShell environment where `cl` is available.

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target fyzzxyzzy_cpp23_contract fyzzxyzzy_headless_smoke fyzzxyzzy_physx_cuda_address_test
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
cmake --build FyzzXYZZY-SDK/build --target fyzzxyzzy_cpp23_contract fyzzxyzzy_headless_smoke fyzzxyzzy_physx_cuda_address_test
ctest --test-dir FyzzXYZZY-SDK/build --output-on-failure
```

The bundled PhysX source takes precedence over Git fetching unless
`FYZZXYZZY_PHYSX_SOURCE_DIR` is specified. Optional Godot builds require
godot-cpp separately and are not offline-complete in this archive.

## Expected CPU smoke result

The test creates a floor and dynamic box, advances 180 fixed 60 Hz steps, and checks
that the box falls, contacts the floor and settles.

The C++23 language/library contract is compiled and executed before full CI builds.
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
