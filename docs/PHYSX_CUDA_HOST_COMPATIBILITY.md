<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Returning to the PhysX host execution path

**Status:** October 2026. Source-grounded feasibility work; the engine still has
no live portable GPU `PxScene`. This is not a new public API proposal.

## Non-negotiable product contract

FyzzXYZZY is an **independent PhysX-derived engine**, not an integration plugin
requiring consumers to patch another PhysX repository. Preserve public PhysX
scene/actor/shape/constraint interfaces and their expected lifecycle wherever
practical. Do not make an external PhysX fork, exact source revision, callback
table or NVIDIA approval a consumer prerequisite.

PhysX source is authoritative for algorithm, host control flow, allocation
ownership, update/fetch task graph and publication. Native Vulkan/D3D12
implementations are the compute lowerings. CUDA compatibility is a *possible
internal mechanism*, not a promise of arbitrary CUDA program execution.

### Source audit: existing interfaces worth preserving

These are actual PhysX interfaces, **not new FyzzXYZZY shims for applications**.

| PhysX source | Existing host behavior | Feasibility obligation |
| --- | --- | --- |
| `physx/include/cudamanager/PxCudaContextManager.h` | Acquires contexts, reports cached device capabilities, supplies CUDA context, memory allocators, streams and module table | Retain lifecycle/caching behavior; do not allocate a fake NVIDIA device or query features on every launch |
| `physx/include/cudamanager/PxCudaContext.h` | `CUdeviceptr` allocation, host/device and device/device copies, streams, events, module/function lookup, kernel launch, abort mode | Translate the **used** semantics explicitly to native buffers, queues, submissions and pipelines; preserve failures/ordering |
| `physx/source/cudamanager/include/CudaKernelWrangler.h` | Caches `CUfunction` handles from CUDA modules | Provide a kernel-index/name-to-compiled-HLSL registry at load/setup time, not an arbitrary PTX/CUBIN loader |
| `physx/source/gpucommon/include/PxgDevicePointer.h` | `CUdeviceptr`-sized fields with raw device address arithmetic | Preserve 64-bit host arithmetic and descriptor relationships without assuming a Vulkan/D3D12 buffer handle is a real device address |
| `physx/source/gpucommon/include/PxgCudaBuffer.h` | Persistent allocations, growth, copy-old-data, buffer swizzling and ownership | Lower onto `compute::Context` buffers, asynchronous copies, `retire_buffer` and the original host lifetime decisions |
| `physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp` | Stream submissions, imported CUDA kernels, region counts, current/previous buffers, GPU DMA upload/download, update/fetch | Preserve launch order and all intermediate state, **including** the fully translated new-bounds four-kernel chain |
| `physx/source/gpubroadphase/src/PxgAABBManager.cpp` | Shape/aggregate state, pair reporting, found/lost continuation references | Preserve original scene publication and task completion; no second AABB scheduler |

`PxCudaContext` exposes dozens of virtual operations; implementing an
*apparently complete* fake CUDA API before establishing the subset exercised
by PhysX would be a large and poorly justified diversion. Map **actual call
sites first**. CUDA device properties expressed as NVIDIA SM generations are
not intrinsically portable; don't claim support unless a required algorithm
and resource contract is satisfied.

## Feasibility increment: device address arithmetic

`src/physx_gpu/cuda_device_address_map.{h,cpp}` maintains a private
`CUdeviceptr`-style 64-bit virtual address space over existing
`compute::BufferId` allocations. The host registry:

- Preserves arithmetic on interior pointers, including offsets beyond 4 GiB.
- Rejects invalid, out-of-range and cross-allocation accesses; gives stale
  addresses no chance to alias a newly registered allocation.
- Preserves exact-base deallocation and returns the underlying native buffer
  handle for the caller to retire through `compute::Context::retire_buffer`.
- Does **not** allocate device memory, reinterpret virtual addresses as native
  GPU physical addresses, or dereference them inside shaders.
- Has no device capability queries or per-GPU-dispatch work.

`fyzzxyzzy_physx_cuda_address_test` validates these host properties without
requiring CUDA, hardware wave support, or allocating multiple gigabytes.
This unit test is necessary groundwork, **not** a successful CUDA adapter.

The `fyzzxyzzy_physx_cuda_host_transfer` and
`fyzzxyzzy_physx_cuda_host_transfer_d3d12` tests additionally exercise
**actual Vulkan and D3D12 device-local allocations**. They resolve an
interior CUDA-style address to a native buffer, enqueue a host-to-device
upload and device-to-device copy without intermediate waits, immediately
remove the source virtual address and call `retire_buffer`, then verify
destination bytes and untouched canaries. These are concrete ownership and
DMA prerequisites for PhysX's original host operations, not proof that
translated shaders can dereference CUDA pointers or that stream events
are implemented.

### Still required before source-compatible GPU execution

1. **Kernel/descriptor lowering.** PhysX `CUfunction` handles and its
   pointer-containing uploaded descriptors must select translated kernels and
   bind/offset the correct native storage resources. A host pointer map alone
   cannot make arbitrary GPU pointer chasing work. This is the highest-risk
   feasibility point; test with a source-derived GPU descriptor, not a toy
   `CUfunction` implementation.
2. **Streams and events.** Inventory all used stream creation, sequencing,
   `streamWaitEvent`, record/query/wait, DMA and failure operations. The
   Vulkan/D3D12 `SubmissionId` is not automatically equivalent to a CUDA
   stream/event; preserve the original task dependency edges. Do not add
   per-stage CPU fences to pretend otherwise.
3. **Ownership and reallocation.** Port the original GPU host allocators and
   buffer-growth policy against our existing native context. Resolve lifetime
   and virtual-address aliasing across asynchronous work and descriptor swaps.
4. **Actual scene integration.** Move the required PhysX source into
   FyzzXYZZY's owned implementation and preserve existing public PhysX
   factories/interfaces. No `PxSceneDesc` extension or externally patched
   `PxPhysXGpu` factory is required as an architectural premise.
5. **Performance and acceptance.** Benchmark native execution on supported
   hardware; do not ship a scan fallback merely to run on unsupported cards.
   Require a real multi-frame `PxScene::simulate/fetchResults` overlap and
   contact test (create/update/remove/idle), comparing normal found/lost
   reporting and deferred resource lifetimes.

If the CUDA-compatible internal interface requires reconstructing nearly the
entire CUDA driver, stop and replace **that private host boundary** with
direct PhysX-internal calls to the compute backend. That change must not
affect public PhysX simulation interfaces or resurrect a foreign GPU-provider
ABI.

## Source/build lifecycle

The current SDK can use `FYZZXYZZY_PHYSX_SOURCE_DIR` for an editable local
checkout. As a *transitional build bootstrap*, CMake otherwise fetches the
current main branch of its PhysX source family. It no longer requires a
specific external PhysX revision. This is **not the intended final product
packaging**: consolidate the PhysX-derived source into the engine's own
release/build so consumers aren't dependent on a separate fork.

The earlier provider-injection experiment was **closed unmerged**:
[PhysX #1](https://github.com/thomthehound/PhysX/pull/1) and
[FyzzXYZZY-SDK #58](https://github.com/thomthehound/FyzzXYZZY-SDK/pull/58).
It does not define our production API or ABI.


## First original CUDA kernel launch lowering: BP_TRANSLATE_AABBS

`src/physx_gpu/cuda_aabb_launch_lowering.{h,cpp}` is an **internal**
lowering of the actual `PxgCudaBroadPhaseSap::translateAABBsKernel`
call site, NOT a generic CUDA driver implementation. Original
`PxgKernelIds::BP_TRANSLATE_AABBS` takes exactly eleven
`PX_CUDA_KERNEL_PARAM2` value pointers, in this order:
`updateData_fpBounds`, `newIntegerBounds`,
`updateData_contactDistances`, `updateData_envIDs`,
`mUpdateData_BoxesCapacity`, XYZ projection shifts, and XYZ environment
shift widths. The six shift arguments are **PxU8**, not PxU32; the
adapter reads each argument as exactly one byte and promotes it. The
native shader packs the six byte shifts into two 32-bit constants,
retaining its original eight-DWORD root/push constant footprint.

The original CUDA interior pointers are lowered to device-aligned
descriptor starts using the selected Vulkan `minStorageBufferOffsetAlignment`
(or D3D12's native structured-buffer alignment). Four residual
32-bit **element biases** are passed through those same eight constants.
The shader applies each bias to its input, distance, environment and output
array accesses. No compatibility buffer copies are inserted. The alignment
is queried once at device admission, not per PhysX launch. Tests exercise
four-byte interior CUDA offsets and separately force descriptor alignment
of 256 and 1024 bytes independent of CI hardware.

The source kernel's 256-thread CUDA blocks process 32 boxes;
the translated shader's native 128-thread blocks process 16. The adapter
verifies the original launch grid instead of falsely passing it through
as the translated grid.

The adapter reads only those original arguments and resolves the four
CUDA-style 64-bit pointer values with the existing private
`CudaDeviceAddressMap`; it binds the resulting **interior buffer views**
(with native offsets and Read/Write access) directly to our existing
`AabbEncodingStage` pipeline. No per-launch intermediate buffer copy,
CPU readback, pipeline recreation, per-frame device-feature probe or
new physics algorithm. In particular, the previously zero-offset-only
AABB stage now exposes a shared buffer-view lowering internally, so native
callers and CUDA-derived callers use the **same** shader and parameter ABI.

Vulkan and D3D12 tests pass the original eleven address-of-argument
pointers and source kernel ID, use nonzero interior offsets in four
independent allocations, check incorrect kernel identity/grid/block/arg
count rejection, execute an environment-enabled launch, copy a snapshot,
execute an environment-disabled launch, and validate both translated GPU
results and complete untouched view canaries. The two kernels and GPU copy
share one original-order submission, without a CPU wait between them.

**Remaining limits:** the adapter currently knows only this one PhysX
kernel, and it does not load arbitrary CUDA binaries or dereference
pointer-bearing `PxgBroadPhaseDesc` inside a GPU shader. The next
host task is to bring the actual descriptor upload fields and original
kernel ID/name lookups through the same source-backed lowering approach.
After that, prove the original scene stream/event and AABB manager
publication chain. This experiment does not yet establish that an
unmodified `PxScene` runs GPU broadphase.
