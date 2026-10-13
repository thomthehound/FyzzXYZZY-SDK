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

`fyzz::World` owns one `WorldBackend`. PhysX remains the default CPU
implementation. The optional `FyzzXYZZY::VulkanWorld` target supplies
`fyzz::compute::create_floor_world(floor_shader, contact_shader)`, producing
one `World` whose authoritative simulation state remains on the GPU.

The experimental backend supports one infinite horizontal floor and
axis-aligned, **nonrotating** dynamic boxes. Gravity, linear integration,
floor collision, and vertical box-to-box contact correction/impulses execute
on Vulkan. Horizontal position and velocity are unchanged by the solver;
a collision with a side face is **not** resolved in 3D. Static boxes,
additional floors, rotation, joints, arbitrary collider shapes, and continuous
collision detection remain unsupported. Those features require the normal
PhysX backend until more complete GPU functionality is implemented.

For the restricted vertical-contact model, the CPU builds disjoint islands
from overlapping X/Z box footprints when bodies are added. The island indices
and ranges are uploaded to GPU buffers. Each island is assigned to one
invocation of `src/kernels/vertical_contacts.comp`, which resolves vertical
position and velocity impulses for that island in eight iterations. This
avoids GPU data races between bodies sharing a contact island, including
stacked bodies. All collision **response** runs on the GPU. The first pass
uses the existing floor-integration kernel, and the contact pass follows
in the same ordered Vulkan command batch. The implementation is intentionally
a correctness-oriented vertical slice: CPU island construction is O(N²),
per-island GPU pair processing is O(M²) per iteration, and there is no
broadphase acceleration or spatially general contact handling yet.

`World::add_static_floor(height)` creates an infinite floor on either backend.
Body IDs are owned by their selected World. The Vulkan implementation holds
body positions, velocities, inverse masses and contact-island tables in
device-local memory. The mass estimate uses box volume times density; stable
settling is checked across bodies with different masses. Adding bodies after
simulation rebuilds island membership and grows buffers through device-local
copies without reconstructing previous positions on the CPU.

Applications must supply compiled shader binaries for
`src/kernels/rigid_floor.comp` and `src/kernels/vertical_contacts.comp`:

```cpp
fyzz::compute::ShaderBinary floor_shader;    // Compiled SPIR-V bytes
fyzz::compute::ShaderBinary contact_shader;  // Compiled SPIR-V bytes
fyzz::World world = fyzz::compute::create_floor_world(
    std::move(floor_shader), std::move(contact_shader));
world.add_static_floor(0.0f);
auto lower = world.add_dynamic_box({0.25f, 0.25f, 0.25f},
                                   {0.0f, 1.0f, 0.0f}, 1.0f);
auto upper = world.add_dynamic_box({0.25f, 0.25f, 0.25f},
                                   {0.0f, 2.0f, 0.0f}, 2.0f);
world.step(1.0f / 120.0f);
auto state = world.body_state(upper);
```

`fyzzxyzzy_vulkan_world_smoke` compares the GPU world to basic PhysX
behavior, checks multi-body settling and separation, body identity, contact
island rebuilding, and state preservation during buffer growth. It is a
functional acceptance test, not a performance claim. Per-body readback and
synchronous stepping remain expensive. The Godot adapter continues to use
the default CPU backend; explicit GPU selection through Godot is future work.

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
