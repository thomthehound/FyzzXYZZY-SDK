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

## Physics compute acceptance

`src/kernels/rigid_floor.comp` is an initial bounded GPU physics workload:
fixed-step semi-implicit linear integration under gravity, per-body static-floor
contact projection, normal restitution, and tangential impact damping. It uses
two GPU-resident arrays of `vec4`: position xyz plus half-height, and velocity
xyz plus restitution. It processes independent axis-aligned bodies with one
shared floor; it does not handle body-body contacts, rotation, or constraints.

The `fyzz::compute::FloorContactKernel` API in
`include/fyzzxyzzy/floor_contact.h` wraps this shader as an SDK operation.
It accepts a caller-provided compiled `ShaderBinary`, creates the pipeline,
and exposes `append_step`, `submit_step`, and `step`. The public
`FloorBodyVector` and `FloorStepParameters` types define its binary data
contract. The caller owns both state buffers and the `Context`, which must
outlive the kernel. `append_step` participates in the existing ordered batch
API; it does not own a parallel scene.

For example, compile `src/kernels/rigid_floor.comp` to SPIR-V with
`glslangValidator -V -S comp`, load the bytes into `ShaderBinary`, then
construct `FloorContactKernel(context, std::move(shader))`. Each body uses
two `FloorBodyVector` entries: center + half-height, and velocity +
restitution. Both arrays must have at least `count` entries. The kernel
checks step parameters; the context checks buffer bounds at submission.
This is a compute building block rather than an automatic physics-backend
selection or renderer integration.

The `fyzzxyzzy_vulkan_floor_physics` acceptance test runs 513 bodies for
160 batched steps, plus two single-step API paths. It checks floor penetration
and compares state against a CPU reference after each batch.

## Opt-in Vulkan World backend

`fyzz::World` owns one `WorldBackend`. Its default backend remains the
existing PhysX CPU implementation. The separate `FyzzXYZZY::VulkanWorld`
target supplies `fyzz::compute::create_floor_world(shader)`, which constructs
a `World` whose authoritative body state lives in Vulkan device-local buffers.

The Vulkan implementation is **explicitly limited** to one infinite horizontal
floor and independent axis-aligned dynamic boxes. It simulates gravity, linear
integration, restitution, and floor contact. It rejects static boxes, additional
floors, and stepping dynamic bodies before adding a floor; it **does not**
implement box-to-box collision, rotation, joints, or the rest of PhysX scene
features. Use the default CPU backend for those capabilities. GPU and CPU
results are compared for plausible behavior, not numerical equivalence.

`World::add_static_floor(height)` creates an infinite floor on the CPU backend
and selects the floor plane on the Vulkan backend. Body IDs belong to the
chosen World, and `body_state` reads GPU state when using Vulkan. New bodies
may be added after stepping; device-local buffers grow through GPU copies
rather than reconstructing previous positions on the CPU. Per-body readback
and synchronous stepping are functional but not optimized.

The factory accepts a caller-supplied `ShaderBinary` compiled from
`src/kernels/rigid_floor.comp`. For example:

```cpp
fyzz::compute::ShaderBinary shader; // Fill shader.bytes with compiled SPIR-V
fyzz::World world = fyzz::compute::create_floor_world(std::move(shader));
world.add_static_floor(0.0f);
const auto body = world.add_dynamic_box({0.25f, 0.25f, 0.25f},
                                        {0.0f, 2.0f, 0.0f}, 1.0f);
world.step(1.0f / 120.0f);
auto state = world.body_state(body);
```

`fyzzxyzzy_vulkan_world_smoke` tests this World integration against basic
CPU PhysX behavior, unsupported-operation rejection, stable floor contact,
body identity, and buffer growth after stepping. The optional Godot consumer
still constructs the default CPU backend; selecting the Vulkan backend in
Godot is a separate integration task.

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
Shader reflection and graphics interop are not implemented. The compute tests cover both arithmetic and the restricted GPU physics workload.
