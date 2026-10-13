<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Portable compute backend

The native Vulkan backend is a standalone C++ library (`FyzzXYZZY::Compute`)
without Godot or PhysX dependencies. Public resources use opaque identifiers;
Vulkan types do not appear in `include/fyzzxyzzy/compute.h`.

## Build

Requires CMake, a C++17 compiler, Vulkan headers and loader, glslangValidator
(for compiling the test shader), and a Vulkan compute-capable driver.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DFYZZXYZZY_BUILD_VULKAN=ON
cmake --build build --target fyzzxyzzy_vulkan_memory_smoke fyzzxyzzy_vulkan_dispatch_smoke
ctest --test-dir build -R fyzzxyzzy_vulkan_ --output-on-failure
```

Vulkan is opt-in; CPU-only builds require no Vulkan SDK. Linux CI validates
the Vulkan implementation using a software Vulkan device. That verifies API
correctness but does not establish AMD GPU compatibility or performance.

## Implemented

- Enumerate compute-capable devices and their relevant resource/dispatch limits.
- Select a device and create a compute queue and command pool.
- Allocate host-visible coherent or device-local storage buffers.
- Upload and download host-visible buffers directly; stage device-local transfers
  through temporary host-visible buffers, with bounds and alignment validation.
- Submit buffer-copy commands and synchronize their completion.
- Create SPIR-V compute pipelines with storage-buffer bindings and push constants.
- Dispatch 1D/2D/3D workgroups with validated device limits and descriptor ranges.
- Synchronize shader writes for subsequent host reads.

The current context is synchronous and not thread-safe. Buffer copies and
device-local upload/download ranges require 4-byte-aligned offsets and sizes;
host-visible memory transfers permit arbitrary byte ranges. Descriptor offsets
must satisfy the device's storage-buffer alignment limits. Pipeline inputs
supply compiled shader binaries and explicit binding layouts. Device-local
allocations prefer non-mappable memory where available; staged transfers work
on devices with unified memory as well. Asynchronous submission, persistent
staging allocation, fences, shader reflection, and graphics interop are not
implemented. The kernel test dispatches real compute work using device-local
buffers but does not simulate physics.
