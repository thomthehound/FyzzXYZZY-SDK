<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Portable compute backend

The native Vulkan backend is a standalone C++ library (`FizzX::Compute`)
without Godot or PhysX dependencies. Public resources use opaque identifiers;
Vulkan types do not appear in `include/fizzx/compute.h`.

## Build

Requires CMake, a C++17 compiler, Vulkan headers and loader, and a Vulkan
driver with a compute-capable queue.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DFIZZX_BUILD_VULKAN=ON
cmake --build build --target fizzx_vulkan_memory_smoke
ctest --test-dir build -R fizzx_vulkan_memory_smoke --output-on-failure
```

Vulkan is opt-in; CPU-only builds require no Vulkan SDK. Linux CI validates
the Vulkan implementation using a software Vulkan device. That verifies API
correctness but does not establish AMD GPU compatibility or performance.

## Implemented

- Enumerate compute-capable devices and their relevant resource/dispatch limits.
- Select a device and create a compute queue and command pool.
- Allocate and destroy host-visible coherent storage buffers.
- Upload/download byte ranges with bounds checks.
- Submit buffer-copy commands and synchronize their completion.

The current context is synchronous and not thread-safe. Buffer-copy offsets
and sizes must be 4-byte aligned. Device-local memory, staging, descriptor
bindings, compute pipelines, shader dispatch, barriers, fences, and graphics
interop are not implemented yet. No GPU physics computation is claimed.
