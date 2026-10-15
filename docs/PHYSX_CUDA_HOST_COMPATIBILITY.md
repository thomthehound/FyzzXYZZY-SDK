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

## Source-native D32 fills at PhysX's existing stream call sites

`PxgCudaBroadPhaseSap::computeRegionHistogramKernel()` performs two
`mCudaContext->memsetD32Async` operations on `mStream` before starting the
histogram launches, clearing `mActiveRegionTotalBuf` and
`mStartRegionsTotalBuf` for the current `totalNbProjectionRegions`.
This ordering is part of the real PhysX host work graph, not a request to
reconstruct a separate CUDA runtime or introduce a CPU staging pass.

`SourceFillD32NativeStage` lowers the source's exact `CUdeviceptr`, uint32
fill pattern, and word count into GPU dispatches appended to the
**caller-owned ordered command list**. When a source allocation exceeds the
backend's maximum storage-buffer view range, the stage splits the same
allocation into directly bound views without allocating, copying, or inserting
host waits; it validates all chunks before committing any commands. It reuses `CudaDeviceAddressMap`
and `source_native_view`; the original pointer can be interior to a
source-owned heap root and only four-byte aligned. The physical RAW GPU
view begins at the backend-compliant aligned base; the push constant
retains the original scalar displacement. Zero-word calls append no
command. The stage neither allocates a persistent fill resource, owns a
stream, copies zeros from the host, submits, nor waits.

Source fill and following histogram launches **must remain in the same
source stream order**. The primitive on its own does not implement the
complete `mStream` interface or claim portable `PxScene` support.
The Vulkan and D3D12 tests validate ordered overlapping writes,
per-allocation canaries, native alignment, and invalid CUDA pointers.
The tiny-view planning oracle additionally forces descriptor-window
splitting without requiring multi-gigabyte CI allocations.
This is the first device-side stream operation required at the original
source call sites beyond compute dispatch and buffer copying.

**Performance decision still open.** Vulkan provides native
`vkCmdFillBuffer`, which supports four-byte-aligned source offsets and
may be preferable to a compute dispatch. D3D12 has
`ClearUnorderedAccessViewUint`, but it needs both CPU and GPU UAV
descriptors, explicit UAV ordering around dispatches, and careful
handling of four-byte interior offsets under the 16-byte RAW descriptor
base constraint. The shared shader is a correct portable lowering,
**not evidence that it beats either native fill path**. Before declaring
the host primitive production-optimal, compare fill and following
kernel performance using representative small and large PhysX
histograms and consider backend-native lowering without duplicating
the source host scheduler.

## Source native region creation: original first three host operations

The source `computeRegionHistogramKernel()` checks
`mUpdateData_CreatedHandleSize` and then enqueues, on the SAME original
`mStream`: (1) `memsetD32Async(mActiveRegionTotalBuf)`,
(2) `memsetD32Async(mStartRegionsTotalBuf)`,
(3) `BP_CREATE_REGIONS` at 32 × 256 threads. This code lives in
`PxgCudaBroadPhaseSap.cpp`, immediately before four source region-prefix
kernels and the active-region histogram sequence.

`SourceRegionCreateNativeStage` appends those exact three operations
using the existing `SourceFillD32NativeStage` twice and a new
source-addressed version of `broadphase.cu::createRegionsKernel`.
It reads the original `PxgBroadPhaseDesc` pointers for X sorted handles,
three SAP position buffers, three all-end histograms and the three output
allocations, without staging or repacking them into a temporary GPU arena.
The region-table clears derive their word count from the source's
`(numPreviousHandles + numCreatedHandles)*2*64` expression (current
boxes plus temporarily retained removed projections); the producer
works on `(previous + created - removed)` current boxes. The host
validates both dimensions against caller-owned source allocation capacities.

The integration test begins with twelve separate device-local
allocations at distinct four-byte interior positions, with nonzero
region tables; it submits both clears and source region generation
in one ordered native GPU batch, then checks every output against a
separate per-box CPU oracle and untouched allocation canaries.
No new host scheduler, scene task, or CPU synchronization point is
introduced between those three source operations. The four following
region histogram scans and the actual `PxScene` host integration
remain work to do; passing this test is not a claim of a working scene.

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


## Original PxgBroadPhaseDesc GPU allocation/descriptor handoff

`source_newbounds_descriptor.{h,cpp}` now consumes the real
`physx::PxgBroadPhaseDesc` type and resolves the new-bounds sequence's
source-named pointer fields: original integer bounds, collision groups,
optional environment IDs, X SAP handles/positions, per-box region extents,
region histograms and ordered handles, uint64 overlap counts, per-work-item
handle/region pairs, block scratch, region prefix and bounded pair report
storage. Importantly, **`regionAccumTotal`**, **`overlapChecksTotalRegion`**
and **`sharedFoundPairIndex`** are already scalars *inside the GPU
descriptor*; the lowering points the translated shader at their exact
`offsetof(PxgBroadPhaseDesc, ...)` byte positions, rather than inventing
separate counter allocations. **The host copy of `regionAccumTotal` is not
authoritative:** the GPU produces this count, and no CPU readback occurs
during descriptor lowering. The host supplies the original allocation
capacities of `overlapChecksRegion` and `overlapChecksHandleRegiones`
instead, to validate their backing ranges. This preserves the GPU-owned
runtime count and avoids a hidden synchronization point.

This source descriptor is still uploaded by the original host when GPU
buffer ownership or input state changes, **not reconstructed at every
kernel dispatch**. The native pair/scan/work descriptors are small views
of its source GPU allocations and share the same persistent backing.
The original host's actual projection-slot count is supplied explicitly,
including projection slots for removed handles; it must not be inferred
from a stale box capacity.

`CudaDeviceAddressMap` now allows **distinct logical CUDA allocations
inside the same physical compute buffer**: `register_slice` creates a
stable virtual CUDA pointer with its own bounds/liveness; `resolve`
returns the backing offset; `relocate_slice` preserves the virtual
pointer when source-ordered allocation growth copies data to new backing
storage; and `unregister_slice` invalidates that logical allocation
without claiming ownership of the shared native buffer. The original
PhysX allocator/stream owner decides *when* to copy, retire, and reuse
physical memory. These methods themselves neither launch kernels nor
allocate or synchronize the GPU.

The Vulkan/D3D12 `fyzzxyzzy_physx_source_descriptor` tests construct
the actual source descriptor from separately bounded CUDA-virtual
allocations sharing one device-local backing. They check **every mapped
pointer and all three inline scalar aliases**, asynchronous source and
translated descriptor uploads, a single ordered GPU copy of the arena
on growth, an independently moved source-group allocation, descriptor
re-materialization **at relocation**, and deferred retirement of the
original backing without a CPU fence. They reject disjoint physical
backing, missing/stale source allocations, impossible sizes and
out-of-range inline fields, and check GPU-resident bytes after the
copy/upload chain has completed. The CPU address-map test separately
covers duplicate native backing with independently freed logical slices,
64-bit interior pointers and stable virtual identity during relocation.

**Performance contract:** This is a zero-per-kernel-copy *persistent
shared-backing feasibility test*, compiled **only into the two conformance
executables**, not into the shipping translated GPU stage libraries. Source-allocation growth can require source-
ordered GPU copies, like PhysX's existing `allocateCopyOldDataAsync`,
but there is no per-dispatch inter-allocation packing or host readback.
It is not a claim that unrelated GPU buffers are magically contiguous:
mappings backed by separate native buffers currently **fail explicitly**
rather than silently copying them. Nor is it a production memory pool or
a working GPU `PxScene` yet. The next engineering work is tying the
**original PhysX heap allocation/growth/free decisions** to this backing
and handling the capacity/fragmentation limits of an arena, or extending
kernel binding for disjoint native resources where preferable. Existing
arena shaders use 32-bit offsets and one storage-buffer descriptor, so
this slice is bounded by native storage-buffer-range constraints; do not
advertise unlimited device memory support. Real scene admission and
multi-frame found/lost reporting remain the acceptance milestones.


### Native GPU memory binding architecture review

**Decision:** Do not treat the single native arena as the general PhysX
allocator or the runtime broadphase integration interface. This is a
test-only proof that source pointer fields and inline counters can be
lowered *when* they share a native resource. That special case is useful,
but PhysX's real `PxgHeapMemoryAllocator` has a buddy-style page/root
allocator, deferred frees and exceptional allocations. A broadphase
scene may legitimately allocate the source fields on **different GPU
buffers**. Failing on those resources is truthful; requiring them to
share a root as an engine invariant would be an architectural error.

**Concrete correctness findings repaired:**

1. Original `regionAccumTotal` is GPU-produced. Host-side descriptor
   lowering now validates the allocator-provided capacities, never
   reads back the GPU value. GPU kernels continue using the resident
   `offsetof(PxgBroadPhaseDesc, regionAccumTotal)` location.
2. A source frame's handle count is
   `numPreviousHandles + numCreatedHandles - numRemovedHandles`,
   while sorted/histogram projection storage includes temporarily
   removed handles. Both relationships are checked.
3. Dedicated `CudaDeviceAddressMap::register_allocation` ownership and
   borrowed `register_slice` references can no longer coexist on
   the same native buffer. Neither registration order nor relocation
   can turn a live borrowed slice into a silently retired buffer.
   Multiple non-owning slices of one backing remain supported.
4. The native copy fixture now uses nonuniform source bytes, checks
   exact relocated-copy contents and preserves all outside-descriptor
   bytes. The overlap scratch allocations match the original source
   allocation policy of 64 entries per current box, rather than an
   arbitrary small fixture capacity.
5. The source 64-bit overlap-prefix/scratch arrays and inline 64-bit
   total are now checked against their **eight-byte ABI alignment**.
   A 4-byte-shifted GPU allocation is explicitly rejected even though
   ordinary 32-bit array accesses at that offset would be legal.

**Unresolved correctness/engineering risks, NOT passed by PR #61:**

- The conformance test submits original and lowered *descriptor bytes*
  and ordered GPU copies, but does **not** run the four native shaders
  against that source-derived memory. The earlier four-kernel test
  exercises a different specially constructed arena. The next
  integrated test must join the two, verify original GPU-produced
  inline counters, found-pair reports, capacities and canaries, and
  exercise created/removed/idle frame transitions.
- The source heap's *actual* multi-root and exceptional allocation
  behavior is not integrated. Production descriptor binding must be
  able to represent a **table of independent native resource views**,
  with a tested Vulkan/D3D12 lowering strategy, before claiming a
  generally usable GPU broadphase. Shader descriptor indexing,
  multi-view bindings, or other native GPU-address mechanisms are
  options to evaluate; no per-launch heap packing or GPU readbacks.
  A coalesced single-root specialization may remain an optimization,
  not the required allocator policy.
- There is no original PhysX `PxgCudaBroadPhaseSap` allocation
  owner, event/stream translation, original task-graph dispatcher or
  live `PxScene` wired to these views. A passed native memory test
  does not establish those contracts.
- The address map does not independently know the *physical* backing
  buffer's byte size, concurrent submission lifetimes or the source
  allocator's free-list reuse policy. It is a resolution table, not
  the GPU heap allocator. The owning host must validate physical
  extents and defer reuse until dependent submissions complete.
- The source `PxgCudaBuffer` distinguishes the raw allocator-returned
  `mPtr` from its 128-byte-aligned `getDevicePtr()` view. Its
  `swapBuffer` and heap deallocation deliberately operate on **raw**
  allocation pointers, not the aligned view. Our fixture's 256-aligned
  slices make those two addresses coincide by construction and do NOT
  exercise that distinction. The real allocator integration must
  maintain raw ownership/free identities separately from shader-visible
  aligned CUDA pointers, or demonstrate an allocator alignment guarantee
  that genuinely makes the two equivalent.
- A single arena uses 32-bit byte offsets and a whole-buffer storage
  binding; those are real range/scale limits, not unlimited GPU memory.
  Device capability and binding-range validation belongs at resource
  setup, not per kernel. The true multi-root path must support large
  scenes without copying an entire previous arena each growth cycle.
- Our GPU compute `Context` currently has an ordered queue and
  completion tokens, not all CUDA stream/event semantics. Cross-stream
  copy, launch and report-publication dependencies require their own
  original-source audit and integrated tests.

**Implementation boundary:** the reusable logical-slice bookkeeping can stand as
internal infrastructure once validated; the source arena converter remains
**test-only**. Do not move it into the runtime by making a single-root backing
an allocation requirement.

The compute API already represents buffer identity, byte range and access intent
through `compute::BufferBinding`. Its Vulkan implementation accepts multiple
storage bindings up to device limits, whereas the current D3D12 implementation
hard-codes either one arena UAV or three SRVs and one UAV. A production
multi-resource path must first generalize and validate the D3D12 binding
contract, while keeping equivalent explicit dependency tracking on both
backends. This is a native compute runtime change, not a new public PhysX ABI.

Per-field static resource views offer a simple correctness baseline. A compact
root table can reduce per-kernel descriptor pressure if the required device
features and measured performance justify it. Source kernel ID and its actual
field-access manifest determine the resources bound for each dispatch; allocation
location must not change the simulated algorithm. Source-visible 64-bit virtual
pointers must never be truncated merely to fit a 32-bit arena offset.

The first acceptance milestone is execution of the **existing four new-bounds
kernels** against the source-derived `PxgBroadPhaseDesc` data, including
independent physical resources and GPU-owned counters. Follow it with original
stream/event, deferred-free and scene publication integration. The concrete
work items and cross-backend acceptance criteria are tracked in
[PhysX GPU host front-end: multi-resource execution](https://github.com/thomthehound/FyzzXYZZY-SDK/issues/62).
