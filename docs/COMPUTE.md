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
- Submit compute dispatches and buffer copies with per-submission fences,
  query completion, and wait on an individual submission or all submissions.
- Submit ordered batches of copies and dispatches in one command buffer,
  descriptor pool, and completion fence, avoiding per-operation queue submissions.

The context is not thread-safe. `copy_buffer` and `dispatch` wait for their
own submissions; `submit_copy_buffer` and `submit_dispatch` return a fence-backed
submission ID without waiting. Operations execute on the same compute queue,
in submission order. `submission_complete` checks a submitted fence;
`wait_submission` waits and releases its command resources, invalidating the
ID. Buffer and pipeline destruction, and host transfers, wait for outstanding
work to avoid premature resource reuse.

`submit_batch` accepts an ordered vector of `CopyCommand` and
`DispatchCommand` operations. Dispatch commands own their push-constant bytes
and buffer bindings; the batch validates every operation before submission.
The returned submission ID covers the entire sequence. `execute_batch`
performs the same operation and waits for completion. A batch cannot be empty
or contain a zero-length or in-place copy. Multiple dispatches share one
descriptor pool (with separate descriptor sets); copy-only batches use none.

Buffer copies and device-local upload/download ranges require 4-byte-aligned
offsets and sizes; host-visible memory transfers permit arbitrary byte ranges.
Descriptor offsets must satisfy storage-buffer alignment limits. Device-local
allocations prefer non-mappable memory where available, with staged transfers
on both discrete and unified-memory devices. Each dispatch still creates its
own descriptor pool, and staged transfers still allocate temporary buffers.
Shader reflection and graphics interop are not implemented. The compute
tests execute GPU integer arithmetic, not a physics simulation.
