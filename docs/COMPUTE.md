<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Portable compute backend

The native Vulkan backend is a standalone C++ library (`FyzzXYZZY::Compute`)
without Godot or PhysX dependencies. Public resources use opaque identifiers;
Vulkan types do not appear in `include/fyzzxyzzy/compute.h`.

The production compute target contains device and execution infrastructure,
not a GPU physics solver. The standalone floor physics kernel and `VulkanWorld`
are independent experimental demonstrations, not ports of PhysX GPU algorithms
or scene orchestration. They are excluded from ordinary Vulkan builds.

## Build

Requires CMake, a C++17 compiler, Vulkan headers and loader, glslangValidator
(for compiling the test shader), and a Vulkan compute-capable driver.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DFYZZXYZZY_BUILD_VULKAN=ON
cmake --build build --target fyzzxyzzy_vulkan_memory_smoke fyzzxyzzy_vulkan_dispatch_smoke
ctest --test-dir build -R fyzzxyzzy_vulkan_ --output-on-failure
```

The ordinary Vulkan smoke targets do **not** require DXC. PhysX-derived AABB
conformance testing additionally requires DXC with SPIR-V support. It is
deliberately opt-in so unrelated compute configurations remain usable:

```sh
cmake -S . -B build-physx-aabb -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DFYZZXYZZY_BUILD_VULKAN=ON -DFYZZXYZZY_BUILD_PHYSX_GPU_AABB_TEST=ON
cmake --build build-physx-aabb --target fyzzxyzzy_physx_aabb_encoding
ctest --test-dir build-physx-aabb -R fyzzxyzzy_physx_aabb_encoding --output-on-failure
```

The specific PhysX test option requires DXC at configuration time. Linux CI
enables the option explicitly, while ordinary Vulkan tests keep only their
original glslangValidator dependency.

To build and test the isolated floor and world prototypes, configure separately
with `-DFYZZXYZZY_ENABLE_GPU_PHYSICS_PROTOTYPES=ON`. These prototypes are not
part of the production PhysX GPU implementation.

```sh
cmake -S . -B build-prototypes -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DFYZZXYZZY_BUILD_VULKAN=ON -DFYZZXYZZY_ENABLE_GPU_PHYSICS_PROTOTYPES=ON
cmake --build build-prototypes --target fyzzxyzzy_vulkan_floor_physics fyzzxyzzy_vulkan_world_smoke
ctest --test-dir build-prototypes -R 'fyzzxyzzy_vulkan_(floor_physics|world_smoke)' --output-on-failure
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
- Track source-declared per-buffer read/write intent for more precise Vulkan
  batch dependencies, while retaining conservative behavior for callers that
  have not yet been annotated.

### Explicit access intent

`BufferBinding::access` optionally declares `Read`, `Write`, or
`ReadWrite` for the **actual shader operation**, not the resource type.
The default `Unspecified` means conservatively read/write, preserving
existing API callers. The source-derived AABB stage marks its three input
bindings read-only and encoded output write-only; the incremental SAP
single-arena stage declares read/write. Vulkan merges all bindings for
the same allocation before placing barriers, so overlapping read/write
aliases are treated as writes. For now the dependency granularity is the
**whole resource**, not the declared byte subranges.

The current D3D12 proof retains its **fixed** SRV/UAV pipeline layouts,
not a general descriptor allocator. It checks that a bound SRV is not
incorrectly declared writable; access declarations do not change the
DXIL resource type or make arbitrary shaders writable. Eventually shader
reflection or source-derived manifests should cross-check intent so
incorrect declarations cannot silently remove necessary barriers.
The PhysX task graph, cross-submission dependencies, and actual lifetime
rules remain authoritative. Do not infer performance gains from software
Vulkan or WARP CI.

### Nonblocking upload staging

`Context::submit_upload_buffer(id, offset, data, bytes)` accepts nonempty
4-byte-aligned input, **copies the source bytes before returning**, and
enqueues a GPU transfer without waiting for completion. The returned
`SubmissionId` is handled by `submission_complete`, `wait_submission`
and `wait_all` just like a device-to-device copy. Both backends retain
the host-visible upload allocation until the relevant ordered-queue
submission completes, including when callers wait on a later submission
first. A caller may immediately modify, free or reuse its original
CPU buffer without changing the submitted data.

This is the first **ownership-correct** asynchronous upload path, not
yet a high-throughput staging ring: uploads currently allocate one host
staging resource per call, so next steps include batching and pooling
source transfers, preserving source-specific dirty/unchanged behavior.
The existing synchronous `upload` remains available, as do synchronous
downloads. No hardware throughput claim is implied by CI.

### Ordered HtoD upload batches

`Context::submit_upload_batch(vector<UploadSpan>)` accepts several
nonempty 4-byte-aligned destination spans and CPU sources. It **snapshots
all source bytes synchronously** into one Context-owned host-visible
allocation, then records all GPU copies in **one ordered queue submission**
with one completion token. The original CPU pointers may be modified or
freed as soon as this method returns. Overlapping writes to the same
destination retain caller order, and a subsequent GPU dispatch or copy on
the Context queue can consume these results without a CPU wait.

Vulkan records these as one `submit_batch` of `CopyCommand` operations
and uses existing access/hazard bookkeeping. Its internal staging buffer
has transfer-source usage only and is **not** subject to Vulkan's
`maxStorageBufferRange` shader-descriptor limit: the aggregate snapshot
may be larger than each individual public storage buffer, subject to
available device memory and Vulkan's buffer-allocation rules. Public
storage buffers remain range-limited and transfer-only staging cannot
be bound as shader storage. The temporary HostVisible GPU buffer is retired
against the final submission fence. D3D12 records
`CopyBufferRegion` operations against a single upload heap and attaches
that heap directly to the submission's lifetime, releasing it on GPU
completion. Each destination transitions into COPY_DEST once and returns
to COMMON after the batch. Invalid destination IDs, null pointers,
misalignment and out-of-bounds ranges are rejected before submission.

**Current limitations:** This is a first *batching and ownership* path,
not an implementation of the upstream `PxgCopyManager::dispatchCopy`
GPU balanced-copy kernel, and not an allocation subpool or staging ring.
Each batch still allocates one staging resource; completed D3D12/Vulkan
command resources are independently reused. This is enough to preserve
producer/consumer ordering when translating source-derived PhysX host
launch groups. Submission-token to PhysX task-continuation publication
remains the next integration step.

### Deferred GPU buffer retirement

`Context::retire_buffer(id)` immediately invalidates the buffer handle
but **does not block** waiting for GPU execution. The Context owns the old
allocation until the last outstanding submission on its ordered queue
completes. On Vulkan this is attached to the last pending submission ID;
on D3D12 it is attached to the last pending queue-fence value. Completion
of a later submission on the same queue makes preceding operations safe
to release even if their completion tokens have not been explicitly
waited in ID order. Waiting and profiling a submission both trigger
eligible retirement; Context destruction drains and releases remaining
allocations.

The pre-existing `destroy_buffer` retains its synchronous semantics.
`retire_buffer` does not yet implement GPU-side arena suballocation,
asynchronous host-to-device staging reuse, buffer growth, or result
publication into the PhysX scene. It supplies **one necessary lifetime
primitive** for faithful porting of PhysX's
`PxgCudaBuffer::allocateCopyOldDataAsync`, preserving the old allocation
until an in-flight GPU copy finishes. Ordinary device-to-device copies
are already nonblocking when submitted through `submit_copy_buffer`.
The backend's single-queue behavior is currently a conservative
execution boundary, not a replacement for the upstream PhysX task graph.

## PhysX-derived GPU stages

The `fyzzxyzzy_physx_gpu_stages` target contains the first translated PhysX
broadphase operation, `AabbEncodingStage`. Its SPIR-V shader implements
upstream `translateAABBsLaunch` from `gpubroadphase/src/CUDA/broadphase.cu`
and its host wrapper retains the original six-coordinate bounds representation,
per-box contact distance, optional environment IDs and per-axis bit settings.
It operates only on buffers owned by its caller and does not implement or select
a simulation world. The `fyzzxyzzy_physx_aabb_encoding` test verifies encoded
bounds against the upstream algorithm for a range of configurations.

The source-derived AABB translation now has **one HLSL implementation**
at `src/physx_gpu/kernels/physx_translate_aabbs.hlsl`. DXC compiles it to
SPIR-V on Linux and DXIL on Windows. Linux CI runs the Vulkan AABB numerical
conformance test, and Windows CI now executes the same reference fixtures on
native D3D12 using WARP, including ordered dispatch/copy/dispatch behavior.
This verifies a bounded D3D12 compute stage, **not** AMD hardware throughput,
general-purpose pointer addressing, or PhysX scene integration.
The `__spirv__` branch changes only descriptor bindings and the 32-byte
parameter block (Vulkan push constants versus D3D12 constant buffer).
See [GPU port architecture](PHYSX_GPU_PORT.md).

This stage is not integrated into the PhysX `Sc::Scene` pipeline yet. A
complete portable broadphase must also port PhysX's host orchestration,
incremental SAP, pairs, aggregation, resource ownership and task dependencies
before claiming executable GPU broadphase support. See
[GPU port architecture](PHYSX_GPU_PORT.md).

## Native D3D12 AABB bring-up (Windows)

A Windows-only native D3D12 executor is available behind
`FYZZXYZZY_BUILD_D3D12=ON`. It executes the **same**
`physx_translate_aabbs.hlsl` DXIL shader, `AabbEncodingStage` host wrapper,
and exact integer reference fixtures as the Vulkan/SPIR-V test. Configure
with a Windows SDK DXC path (or place `dxc` on PATH):

```powershell
cmake -S . -B build-d3d12 -A x64 -DBUILD_TESTING=ON -DFYZZXYZZY_BUILD_D3D12=ON -DFYZZXYZZY_BUILD_D3D12_AABB_TEST=ON "-DFYZZXYZZY_DXC=C:\path\to\dxc.exe"
cmake --build build-d3d12 --config Release --target fyzzxyzzy_physx_aabb_encoding_d3d12
$env:FYZZXYZZY_D3D12_FORCE_WARP = "1"
ctest --test-dir build-d3d12 -C Release -R fyzzxyzzy_physx_aabb_encoding_d3d12 --output-on-failure
```

Windows CI runs the DXIL test on D3D12's software WARP adapter before the
full Godot build. Remove the environment variable to prefer a hardware
adapter; WARP remains a fallback. WARP conformance does **not** establish
AMD device performance or hardware-specific wave behavior.

**Current limits:** the D3D12 backend is initially a separate static
implementation of `fyzz::compute::Context`, exposing 3 structured
SRVs (t0..t2), one structured UAV (u0), and 8 root constants (b0)
for this source-derived PhysX stage. It does not yet allow both backends
to be linked into a single executable for runtime selection. Its
`HostVisible` requests use device-local resources with staging rather
than a directly mapped device buffer. This is an explicit bring-up
limitation, not a general-purpose or scene-integrated D3D12 GPU backend.
The next architecture step is a backend-neutral context implementation
with interface-owned command, descriptor and task-completion semantics.
Memory/addressing candidates and real-hardware benchmark gates are documented
in [GPU memory/addressing RFC](GPU_MEMORY_ADDRESSING_RFC.md).

## Experimental physics compute acceptance

`src/kernels/rigid_floor.comp` is an original, bounded GPU physics experiment (not a translated PhysX kernel):
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

## Experimental Vulkan World backend

`fyzz::World` owns one `WorldBackend`. Its default backend remains the
existing PhysX CPU implementation. With `FYZZXYZZY_ENABLE_GPU_PHYSICS_PROTOTYPES=ON`, the separate
`FyzzXYZZY::VulkanWorld` target supplies `fyzz::compute::create_floor_world(shader)`, which constructs
a `World` whose authoritative body state lives in Vulkan device-local buffers.

The Vulkan implementation is **explicitly limited** to one infinite horizontal
floor and independent axis-aligned dynamic boxes. It simulates gravity, linear
integration, restitution, and floor contact. It rejects static boxes, additional floors, and stepping dynamic bodies
before adding a floor. Because this kernel cannot resolve body-to-body
contacts, it also rejects new boxes whose horizontal (X/Z) footprints
overlap those of existing dynamic boxes, even if initially at different
heights. With no horizontal velocity or acceleration in this implementation,
accepted footprints remain disjoint. It **does not** implement box-to-box
collision, rotation, joints, or the rest of PhysX scene features. Use the
default CPU backend for those capabilities. GPU and CPU results are compared
for plausible behavior, not numerical equivalence.

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
