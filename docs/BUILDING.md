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
will consume the same FizzX core.

## Headers

New FizzX-authored C++ and CMake files use:
`SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound` and
`SPDX-License-Identifier: Apache-2.0`.
Upstream and third-party files retain their original license and provenance.
