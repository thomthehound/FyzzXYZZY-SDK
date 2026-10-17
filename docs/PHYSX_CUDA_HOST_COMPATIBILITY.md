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

## Original PxgCudaBuffer raw pointer and aligned device view

Original PhysX source (`gpucommon/include/PxgCudaBuffer.h`,
`gpucommon/src/PxgCudaBuffer.cpp`) distinguishes the heap allocator's
**raw** `mPtr` from `getDevicePtr() = (mPtr + 127) & ~127`. Heap free
and deferred free use the **raw** pointer; `swapBuffer` swaps raw
pointer identities and sizes, while `assign` releases the destination's
raw allocation and takes the source's raw identity. Most importantly,
`allocateCopyOldDataAsync` copies the original `mSize` bytes
**raw-to-raw** onto `mStream`, doubles capacity when necessary, and
defers reclamation of the old allocation until a later simulation step.

Previously `CudaDeviceAddressMap::register_slice` produced only
256-byte-aligned virtual allocation bases, obscuring that distinction.
Its existing behavior remains intact. The added source-specific
`register_source_slice` keeps the source allocator's **raw pointer
modulo 128** in the emulated `CUdeviceptr`, while keeping the same
borrowed native GPU root, range bounds and exact raw-pointer unmap
semantics. It does *not* manufacture a physical CUDA pointer or
change source allocator ownership.

The plain host unit test checks nonaligned raw addresses, rounded
device addresses, failed free via the aligned pointer, valid/invalid
padding reads, non-recycled virtual addresses and retained root
ownership. Linux Vulkan and Windows D3D12/WARP GPU host-transfer
tests additionally perform the **original raw-to-raw growth copy**
on a shared GPU root, with distinct old/new raw residues, deferred
free after GPU completion, `swapBuffer` pointer/size rotation and
`assign` ownership transfer; they compare the entire backing
buffer including canaries and reject overlapping ranges through both
single-copy and batched APIs.

**Native same-root copying:** the source allocator can allocate the
old and new raw blocks within one GPU root. The original
`memcpyDtoDAsync` can therefore legitimately request a disjoint
same-resource copy. Vulkan permits this directly through
`vkCmdCopyBuffer` with one combined read/write transfer resource
dependency. The current legacy-barrier D3D12 path cannot put one
resource in both `COPY_SOURCE` and `COPY_DEST` states simultaneously.
It lowers the same source command to two queue-ordered
`CopyBufferRegion` operations via a *device-local*, fence-owned
temporary buffer. Neither path performs an intermediate host wait or
CPU staging/readback. The D3D12 lowering adds an extra GPU copy and
temporary allocation **on the growth path only**; benchmark this and
consider capability-gated Enhanced Barriers or reuse of small scratch
allocations when optimizing the actual allocator integration.

These are source-derived **contract tests**,
not a live `PxgCudaBuffer` backend or the production heap allocator.

What remains: actually connect original `PxgHeapMemoryAllocator`
buddy root / exceptional allocations and their deferred reuse queue;
wire original buffer calls to nonblocking GPU copy/submission and
mStream/event completion, preserve historical resource identity
across scene frames, and handle failures under PhysX's task graph.
This focused work does not add an independent frame scheduler or
claim G2/G3 acceptance.

## Connected original three-pass GPU reporting into the PhysX mapped host heap

The source `BP_ACCUMULATE_REPORT_STAGE_1`,
`BP_ACCUMULATE_REPORT_STAGE_2`, and `BP_COPY_REPORTS` kernels
already operate through `SourceSapReportNativeStage`, which accepts
the *original* `PxgBroadPhaseDesc` and resolves each source pointer
independently (multiple original PhysX roots, not a copied global
arena). Previously their conformance fed fabricated GPU storage slices
for `foundPairReportMap`/`lostPairReportMap`.

The connected source conformance (PR #88) now constructs the **original**
`PxgHeapMemoryAllocatorManager` and its **original**
`PxgHeapMemoryAllocator` for mapped host allocation. Those same
`PxgBroadPhasePair` CPU pointers have exact GPU aliases in the
`CudaDeviceAddressMap`. The three real native report GPU kernels
write into those aliases, and `BroadPhaseCompletionGate` waits their
original ordered `SourceSapDmaNative` stream, fetches the descriptor
and original PhysX pair arrays through `fetch_source_sap_results`,
rotates the seven source history families, and releases the original
found/lost `PxBaseTask` references.

Conformance runs on Linux/Vulkan and Windows/D3D12/WARP using
direct host mapping when available and the staged fallback by force.
The original source classification and copy behavior is checked
against a CPU oracle for actor/aggregate pairs, stable order, overflow
and reports. The PhysX heap manager, native GPU buffers and CPU
report roots are released through the original allocator lifetimes;
no report-list shadow copy or alternate simulation scheduler is
introduced.

**Still pending:** the actual `PxgCudaBroadPhaseSap` and
`PxgAABBManager` source objects and their `ScPipeline` scene task
integration. This is real GPU shader execution with the original
host heap and completion gate, not yet a GPU-enabled `PxScene`.
Hardware timing comparisons remain deferred.

## Connected original broadphase fetch and task publication

`fetch_source_sap_results()` now optionally consumes the actual
`SourcePhysxMappedReportRoots` callback supplied to the original PhysX
mapped-host heap (PR #87). At `BroadPhaseCompletionGate::fetch_and_release`,
the original source stream GPU fence has completed. The existing
`PxgBroadPhaseDesc` is downloaded and its original aggregate counts,
actor count clamping, overflow and abort behavior are unchanged.

Before publishing either array, the fetch validates **both** original
GPU-report pointers, host capacities, and exact CPU ↔ virtual-GPU aliases
of the original PhysX buddy-allocated actor-pair arrays. If the aliases
do not match, it fails *before* mutating `host_descriptor`, swapping
the seven history buffer families, or releasing either PhysX found/lost
task. A corrected host-array selection may be retried at the same
completed frame without scheduling new GPU work.

In the direct mapped path, the GPU has already written into the
original PhysX array itself; **no duplicate GPU→CPU copy is issued**.
On unsupported host-visible memory, the same original callback performs
the staged publication after the same source fence. Caller-provided
unmapped host destinations retain the old guarded readback path.
The existing `BroadPhaseCompletionGate` continues to install source
continuations and release original found/lost tasks *only after*
successful fetch and history rotation.

Cross-backend tests exercise the actual original GPU heap manager,
source-stream native GPU report writes, the original descriptor's
found/lost mapped pointers, correct count/overflow metadata,
sequential buffer-history rotation, alias-rejection and task-release
ordering. Both direct-when-supported and forced-staged variants are
tested; GPU hardware performance measurements are deferred.

**Still pending:** constructing a live `PxgCudaBroadPhaseSap` and
`PxgAABBManager` in a GPU-enabled scene, and binding the rest of
their original kernel update paths to these allocations and source
task/stream dependencies. This test integrates their host fetch
boundary; it is not yet live GPU `PxScene` simulation.

## Source GPU fences and mapped actor-pair report allocation

The original `PxgCudaBroadPhaseSap` allocates
`mFoundActorPairsMapped` and `mLostActorPairsMapped` through
`PxgAllocatorDesc::hostMappedAlloc` with pageable fallback disabled.
CUDA `getMappedDevicePtr()` provides a device alias of the *same host
memory* and the CPU consumes the arrays after `gpuDMABack` completes.

`SourcePhysxMappedReportRoots` is now a single original
`Cm::VirtualAllocatorCallback` serving this role through the unchanged
source `PxgHeapMemoryAllocatorManager` and original buddy allocator.
It has two lower-level paths, selected by native capabilities and an
explicit conformance-mode override:

* **Direct mapped:** Request `MemoryPlacement::HostVisible`, map the
  GPU-writable allocation *once* with `HostMapAccess::ReadWrite`, and return
  the suitably aligned CPU pointer to the source heap. Its original
  buddy-block offsets resolve to the same native GPU resource and bytes.
  The GPU writes directly into that resource. After the source stream
  fence, `publish_at_fetch` requires **no copy, shadow report list,
  staging allocation or readback**. The low-level allocation callback
  unmaps and retires the native resource only when the original PhysX
  heap releases the entire host root.
* **Staged fallback:** If the backend lacks genuinely GPU-writable,
  CPU-mappable memory (or conformance explicitly requests fallback),
  retain the source-compatible aligned CPU heap root with one
  device-local GPU root. Publish only the requested report bytes to
  that *same original CPU allocation* after the existing source stream
  fence. This path is not described as zero-copy.

Native `HostVisible` resources may have stricter pointer alignment
than the original heap needs. The callback allocates up to 255
additional bytes, aligns the returned CPU root to 256 bytes, and maps
that same offset into the registered native GPU address. It preserves
the original host-root and GPU-root ownership; all original buddy
offsets continue to resolve inside the one backing allocation.
It distinguishes source buddy-block lifetimes from the physical root
lifetime and invalidates the exact GPU virtual root only at original
heap destruction. No CUDA API or second heap algorithm is exposed.

The original manager aliases `hostAlloc` with `hostMappedAlloc`
in normal mode. That means the same root must additionally support
CPU-authored pinned descriptors subsequently read by GPU source stages.
The callback therefore maps bidirectionally, and native conformance
verifies CPU-write/GPU-read traffic from a third original buddy block
alongside the GPU-write/CPU-read found/lost report arrays.

Linux/Vulkan and Windows/D3D12 conformance now exercise **both**
forced staged and capability-driven direct paths through the actual
source manager: found/lost report pairs in the same original buddy
root, two native GPU-written output regions, correct aligned address
aliases, GPU/CPU data agreement only after the source fence,
source-owned deferred frees, and stale address invalidation.
Tests deliberately **do not read direct mapped CPU memory before its
GPU fence**; the staged path also proves there is no premature copy.

The existing `BroadPhaseCompletionGate` uses PhysX's source stream
submission fence. Vulkan `vkWaitForFences` and D3D12
`ID3D12Fence::SetEventOnCompletion` + an OS event wait avoid CPU
busy-waiting. This source-port bridge does **not** reproduce
the original `spinWait(mEventMapped)` flag poll; the remaining live
scene call sites must instead use that completion gate. No new
scheduler or polling loop is introduced.

**Remaining integration:** assemble the real `PxgCudaBroadPhaseSap`
and `PxgAABBManager` source call paths with this callback, connect
the source descriptor's GPU pair aliases and original fetch/resize
continuations, and remove the actual `mEventMapped` polling site.
The original `fetch_source_sap_results()` path remains for
caller-owned CPU destination arrays. Performance comparisons between
direct and staged paths, including cache behavior on Radeon 680M
and discrete adapters, are **deliberately deferred** until an
integrated GPU scene exists. Direct mapping is not claimed to be
universally faster simply because it avoids copying.

## Original PhysX GPU heap manager with native callback selection

The original `PxgHeapMemoryAllocatorManager` is now implemented beside
the source-derived `PxgHeapMemoryAllocator`. Its **original ownership and
routing policy** is preserved: one device heap; separate ordinary pinned
and GPU-mapped pinned host heaps in zero-capacity/debug mode; one *shared
mapped-host* heap exposed through both host callback slots in ordinary
mode. Only the original manager's CUDA-bound `PxgMemoryManager` lookup
is replaced with three caller-provided
`Cm::VirtualAllocatorCallback&` instances and the neutral abort sink.
The original `getDeviceMemorySize`, `getDeviceHeapStats`, and
`flushDeferredDeallocs` behavior is retained. Partial construction is
transactional: unsuccessful host heap construction releases the
already-created device heap without leaking its native roots.

The conformance test constructs the **actual source manager**, verifies
normal alias identity and debug separation, correct host-callback
selection, device heap accounting and deferred-flush ownership, and
destruction of every native and host root. A failing host allocator
exercises exceptional setup cleanup. Those tests use ordinary CPU host
allocations **only to prove callback selection and ownership**. They
do not claim that the test host pointers are page-locked or GPU-mapped.

**A live GPU scene still requires genuine backend-specific pinned and
mapped host implementations.** An ordinary CPU allocation is not
substitutable for PhysX's `CU_MEMHOSTALLOC_DEVICEMAP` behavior. Native
scene construction must provide GPU-visible host report views with
correct synchronization and lifetimes, alongside the source GPU task
and stream orchestration. These host facilities remain internal
infrastructure, not a CUDA requirement on FyzzXYZZY applications.

## Original PhysX GPU buddy heap, now backend-neutral

The original `Block` and `PxgHeapMemoryAllocator` implementation from
`physx/source/gpucommon` has been brought **into the FyzzXYZZY engine source**
under `src/physx_port/gpucommon/`, retaining its NVIDIA attribution and
Apache-2.0 license. This source remains the authoritative buddy allocator;
there is **no additional independent heap implementation**.

The port's narrow source delta removes the requirement that the heap's
lower-level root callback derive from CUDA-specific
`PxgCudaAllocatorCallbackBase`. It now receives the **original**
`Cm::VirtualAllocatorCallback&` plus a required backend-neutral abort
callback and its owning context. The original CUDA variant reports
duplicate allocation keys through `PxCudaContext::setAbortMode`;
the neutral port reports the same condition to the native scene via the
injected callback, without pretending a CUDA context exists. GPU device
addresses are stored as opaque 64-bit values; source root+offset arithmetic
uses `uintptr_t`, not CPU pointer arithmetic on non-dereferenceable addresses.

The original GPU-heap manager factory (which still creates CUDA pinned/mapped
allocators) is deliberately **not** imported. It must be lowered as part of
native scene construction, alongside the original pinned-host report
buffers and abort-mode wiring, rather than smuggling CUDA allocator state
back into this neutral heap. The original source algorithms for block
splitting, hash key tracking, exceptional allocations, buddy coalescence,
deferred free and heap statistics otherwise remain intact.

Linux Vulkan and Windows D3D12 conformance now construct the actual source
`PxgHeapMemoryAllocator` with `SourcePhysxDeviceRootCallback`, allocate
buddy and exceptional blocks, enqueue GPU DMA growth/copy chains across
the original heap resources, retire an exceptional root while work is
in flight, exercise the original `deallocateDeferred`/flush path, and
verify physical/virtual root cleanup. This is **real original heap code**,
not the earlier callback-only fixture.

**Still pending:** the GPU-specific `PxgHeapMemoryAllocatorManager` factory
and pinned-host allocator lowering, construction of original
`PxgCudaBroadPhaseSap`/scene objects against this native heap, and
live simulation. The shared compute engine's public `fyzzxyzzy/`
API stays independent of CUDA, PhysX GPU internals, and any single
vendor backend.

## Exact source virtual callback surface for PhysX heap roots

The original `physx::Cm::VirtualAllocatorCallback` interface in
`CmVirtualAllocatorCallback.h` defines `allocate(size_t,int,const char*,int)`
and `deallocate(void*)`. `SourcePhysxDeviceRootCallback` now implements
**that exact virtual interface**, translating only lower-level PhysX device
root requests to `SourcePhysxHeapRoots`. Allocations return the opaque
CUDA-style virtual root that the original source heap expects to split via
root+offset arithmetic; frees must use the exact original root identity.
An associated original `SourceSapDmaNative` stream is submitted before
root retirement, without a CPU wait.

Vulkan and D3D12 conformance invoke the callback through a real
`physx::Cm::VirtualAllocatorCallback&`, request ordinary and exceptional
roots, verify source buddy-style offsets, perform asynchronous GPU
growth/copy operations, free roots while work is queued, and reject
interior-pointer frees. The original GPU data and retirement rules—not a
new independent heap allocator—remain authoritative.

**Remaining source integration constraint:** `PxgHeapMemoryAllocator` currently
takes the narrower `PxgCudaAllocatorCallbackBase&`, a CUDA-context-aware
subclass of `Cm::VirtualAllocatorCallback`. The SDK's new callback
implements the underlying virtual interface but is **not** a
`PxgCudaAllocatorCallbackBase` and cannot yet be passed to that original
constructor. A real engine source integration must lower that CUDA-aware
constructor/abort-mode dependency while keeping the source heap's original
buddy logic, allocation statistics, and deferred flush decisions. Do not
claim the virtual callback test is a live PhysX heap or GPU-backed scene.

## Native backing at the original PhysX heap ROOT callback

PhysX already implements the essential GPU allocation policy in
`PxgHeapMemoryAllocator`, not in `PxgCudaBuffer`: its buddy allocator
splits 128-byte blocks from root pages, handles exceptional allocations,
tracks exact raw allocation keys, defers freed blocks and flushes them at
its existing simulation-step boundary. Reimplementing that bookkeeping
inside the Vulkan/D3D12 port would create competing lifetime decisions.

`SourcePhysxHeapRoots` therefore provides the **lower-level allocation
callback boundary** for the original heap's device roots. Each callback
allocation creates one native `compute::BufferId` and registers one
monotonic CUDA-style virtual root in `CudaDeviceAddressMap`. The original
PhysX heap can perform its own root-plus-offset buddy arithmetic without
registering individual suballocations or requiring all roots to share one
native buffer. Exceptional allocations also receive their own roots.
Pointers returned from this boundary are opaque GPU addresses, never
CPU-dereferenceable pointers.

At the original ROOT deallocation boundary, optionally flush pending
source-native stream commands to their **existing ordered queue, not a CPU
wait**, then invalidate the root's virtual address and retire the native
backing with `Context::retire_buffer`. Outstanding submitted copies and
dispatches retain the physical resource until the queue fence completes.
Do **not** call root release for an in-heap buddy block, or while the
original source heap still owns live suballocations of that root. The
original allocator, not this lower-level adapter, still decides when a
root may actually be returned.

The existing Vulkan/D3D12 native CUDA host transfer conformance now
covers root addresses with buddy-style offsets, exceptional roots,
original raw grow-copy dependencies over multiple GPU resources,
asynchronous root retirement during those copies, exact-root free
validation and stale-address rejection. The shader/report conformance
from PR #79 already consumes `CudaDeviceAddressMap` multi-root views.

**Lifetime and failure contract:** normal scene shutdown calls
`SourcePhysxHeapRoots::close(sourceStream)` while the existing PhysX stream is
available; it submits (does not wait for) pending native commands before
retiring the allocator-owned roots. `close()` is idempotent, and a stream
from another scene/address map is rejected before touching the roots. If
scene initialization or teardown throws, the owner destructor provides a
`noexcept` fallback that retires **all remaining** native roots and
unregisters their device-address mappings. This fallback assumes that
any operations referring to those roots have already been submitted;
only the source stream owner can submit pending CPU-side command lists.
The context and address map must outlive the heap root owner.

The same Vulkan/D3D12 native host tests cover implicit destructor teardown
with GPU transfers in flight, exception unwinding with multiple live roots,
stale-address rejection, and explicit/repeated `close()` operations.

**Integration qualification:** this is the native backing operation that
a future internal `PxgCudaAllocatorCallbackBase` implementation will
call. It is not yet installed into the actual PhysX memory manager, nor
is a live `PxScene` using it. Host source callback wiring, failure/abort
translation, original heap destruction ordering and mapping of GPU-pinned
host reports are still required. No new independent PhysX buddy allocator
or game-engine integration API has been introduced.

## Original PhysX report allocations across multiple GPU heap roots

`SourceSapReportNativeStage` lowers original
`BP_ACCUMULATE_REPORT_STAGE_1`, `BP_ACCUMULATE_REPORT_STAGE_2` and
`BP_COPY_REPORTS` into bounded views of **the original PhysX allocations**.
Each CUDA-style source pointer is independently mapped to its native resource
and view-local byte offset. Report arrays, 32-block scratch, source counters,
volume metadata, mapped actor outputs and the descriptor's overflow field
can occupy different GPU heap roots—or share buddy-heap slices—with no
per-launch arena repacking, allocation or staging copy.

The original two 32 × 4 × 256 classification launches and 64 × 256
publication launch remain in the existing source stream. Both portable and
native-wave classification variants operate on original
`Bp::VolumeData::mAggregate` within actual source record strides.
`SourceSapDmaNative`, `BroadPhaseCompletionGate` and
`fetch_source_sap_results` retain the source's GPU completion, host fetch,
history-buffer rotation and original found/lost task release ordering.

Linux Vulkan and Windows D3D12/WARP conformance run the connected GPU
report production→original descriptor→host fetch→task publication path
with separate native roots and one shared root. No scene-owned pair lists
are fabricated between the source kernels and their fetch. The previous
single-root view lowering remains as a legacy conformance reference,
not a requirement of the new native report stage.

**Still outside scope:** actual `PxgCudaBroadPhaseSap` call-site rewiring,
PhysX buddy-heap allocation/retirement integration and native CUDA-mapped
host report interoperability. Those remain prerequisites to shipping a
real native-backed `PxScene` GPU broadphase.

## Source GPU report production connected to native fetch

`lower_source_sap_report_descriptors()` resolves original
`PxgBroadPhaseDesc` raw/aggregate counters, report pointers, block
counters, mapped actor-pair outputs and overflow flag to **native views of
the same caller-owned source allocations**. The existing native
`BP_ACCUMULATE_REPORT_STAGE_1`, `BP_ACCUMULATE_REPORT_STAGE_2`, and
`BP_COPY_REPORTS` launches then write the original GPU descriptor fields
and mapped actor reports consumed by `fetch_source_sap_results()`.
Submission and completion use `SourceSapDmaNative` and the existing
`BroadPhaseCompletionGate`, not an independent frame driver.

Source `Bp::VolumeData` embeds its `mAggregate` field after
`mUserData`; the report-stage ABI now accepts a record stride and
aggregate-member byte offset, so native shaders classify directly from
the original structure-of-records. A zero stride maintains the compact
tag-array convention used by earlier isolated conformance fixtures.
Both portable and native-wave report stages use the same data source.

Cross-backend tests run all three report kernels into original
`PxgBroadPhaseDesc` counters and source-mapped actor report output,
then read and publish those results into the original found/lost task
gate. Aggregate and nonaggregate populations and overflow cases compare
against independent CPU pair classification.

**Remaining qualification:** the adapter currently supports report
allocations sharing one native heap root. Real PhysX heaps can span
multiple roots, and original mapped host pair memory will need native
interop/transfer lowering. Thus this closes a real connected
**source-shaped GPU dataflow**, not yet the live `PxScene` source path.

## Source SAP GPU fetch / original pair-task publication

The native SAP host path can now hand the borrowed `SourceSapDmaNative`
stream directly to `BroadPhaseCompletionGate::arm_source_stream()`, keeping
one original source-stream completion and task-release boundary. It submits
the remaining native GPU command segment without waiting. The completion
gate queries readiness of the last native submission and reaps source
upload/copy/dispatch tokens only in the original fetch phase.

`fetch_source_sap_results()` consumes a real GPU-resident
`PxgBroadPhaseDesc` via its registered source CUDA-style pointer and
downloads source `foundPairReportMap` / `lostPairReportMap` records into
the **caller-owned original actor-pair destinations**. It preserves source
count semantics: actor pairs equal the clamped raw pair count minus the
aggregate count; required capacity uses the raw maximum; abort mode
publishes empty actor-pair lists; overflow is exposed without pretending
lost pairs were retained. It also exposes the original region totals.

Only after validating both CPU destinations and both GPU source views,
downloading reports and copying the original descriptor to the host does it
rotate the seven original X/Y/Z history-buffer families. The integer-bound
swap is deliberately left at its earlier `update()` call site. Pass this
fetch operation as the existing completion gate's fetch callback, so the
original `PxgAABBManager` continuations cannot be installed or released
before result publication. Publication errors remain retryable without
double-swapping histories; partial task release remains terminal.

The Linux Vulkan and Windows D3D12/WARP completion regression exercises
the joined path with real native uploads/readbacks, pair-count overflow,
abort, idle frames, invalid GPU counters, failed-fetch retry and original
task release ordering. **This is not yet a live `PxScene` adapter**:
source scene object lifetime, native copy-reports shader output production,
PhysX's original allocator ownership, and real task dispatcher integration
still require direct wiring. No new scene scheduler or fabricated GPU event
chain is introduced.

## Source SAP mStream DMA ordering for native stages

The internal `SourceSapDmaNative` now lowers adjacent
`PxgCudaBroadPhaseSap::preBroadPhase` / `gpuDMAUp`
`memcpyHtoDAsync` transfers, original raw-pointer `memcpyDtoDAsync`
growth copies, and source-native kernel batches onto the existing
`compute::Context` ordered queue. It retains all original
`CudaDeviceAddressMap` allocations and does not introduce a new frame owner,
heap, source descriptor, or event scheduler.

- An adjacent HtoD group resolves and validates all source pointers before
  submission. `Context::submit_upload_batch` snapshots caller memory and
  owns staging until the group's completion token is reaped.
- Raw DtoD copies and existing source-native dispatches are appended to the
  same GPU command segment. If a later HtoD group follows, pending GPU
  commands submit first; **no CPU wait is introduced**.
- Submission tokens remain live through the source fetch boundary.
  `wait_at_fetch` reaps source-ordered submissions without a queue-wide
  `wait_all` or premature staging reclamation.
- The D32 source regression tests cover HtoD → raw growth DtoD → HtoD →
  dispatch → DtoD on actual native GPU resources, including disjoint same-root
  allocation slices, rejected overlap/cross-allocation access, and full-root
  canaries. The connected fourteen-operation fixture also uploads its
  original GPU descriptor through this path before source dispatch.

**Uncompleted integration:** this is a thin, single-source-stream
transfer adapter used in native GPU fixtures. The original
`PxgCudaBroadPhaseSap` caller, heap allocation and deferred-free queue,
multi-stream/event dependencies, DMA-back publication, and live
`PxScene` task lifecycle are not yet connected. Consecutive upload/GPU
command segments may require distinct ordered submissions; evaluate
submission overhead on physical GPUs before adopting this as a production
stream implementation. Nothing in the source host path mandates a wait
between those segments.

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
introduced between those three source operations. Passing this test
is not a claim of a working GPU scene.

## Original four region histogram scans over GPU-resident source allocations

`SourceRegionHistogramsNativeStage` appends the four original PhysX
region histogram launches in source order:
`BP_COMPUTE_START_REGION_HISTOGRAM`,
`BP_OUTPUT_START_REGION_HISTOGRAM`,
`BP_COMPUTE_REGION_HISTOGRAM`, and
`BP_OUTPUT_REGION_HISTOGRAM`. Each has the source's 32 × 256 workgroup
geometry. The original GPU data and 32-element block histograms are
borrowed from the source `PxgBroadPhaseDesc`, using the existing CUDA
virtual address map and physical native binding views.

The second pass writes the GPU totals **into the original resident
`PxgBroadPhaseDesc::startRegionAccumTotal` and
`PxgBroadPhaseDesc::regionAccumTotal` fields**, including exact
structure member offsets and original descriptor address bias. It does
not write an independent/surrogate descriptor or copy GPU histogram
values to the CPU to reconstruct dispatch sizes.

The portable scan uses the established two-bank workgroup prefix
technique. When native wave arithmetic and a compatible subgroup
width are available, the implementation selects the existing
atomic-wave-ID hierarchical scan strategy. Both variants have
conformance tests against independent serial CPU prefix and
per-block subtotal oracles.

Most importantly, the source region-creation conformance fixture now
also appends these four stages after the two D32 fills and
`BP_CREATE_REGIONS` **in one caller-owned GPU batch**. It checks output
arrays, all 32 block totals, the two original descriptor scalar fields,
unchanged memory/canaries, retained removed-projection capacity and
both separate and shared GPU heap-root backing. This is a contiguous
seven-operation segment of the original source host path, not a
standalone rewritten scene scheduler. The subsequent active-region
histogram steps are addressed by the source-native stage below;
the complete live `PxScene` task graph remains unintegrated.

## Source-native active-region histograms and ordered handle lists

The original `PxgCudaBroadPhaseSap::computeStartAndActiveHistogramKernel()`
is distinct from the preceding region-table clearing routine. It
enqueues exactly three more kernels on `mStream`, gated by the same
nonzero created-handle count: `BP_WRITEOUT_ACTIVE_HISTOGRAM` (32×256),
`BP_COMPUTE_ACTIVE_HISTOGRAM` (64×512), and
`BP_OUTPUT_ACTIVE_HISTOGRAM` (64×256).

`SourceActiveRegionListsNativeStage` binds the source descriptor's
**original** sorted X handles, X SAP positions, region ranges, start-region
prefix, active and start histograms, and ordered-output buffers. The first
and third kernels read `startRegionAccumTotal` from the **GPU-resident**
`PxgBroadPhaseDesc` written by the preceding region-prefix stage,
rather than an out-of-date CPU shadow. No synthetic descriptor,
per-buffer repacking, extra stream, or host readback is introduced.

The 64×512 scan reuses the established packed-16+16 active/start prefix
algorithm; native wave arithmetic is preferred when supported, while
the portable shared-memory variant remains available. Both variants
preserve source 64-region strides and exact original kernel geometry.
Each native pass binds only the source allocations it uses, with
alias-safe RAW UAV views for D3D12 buddy-heap slices.

A source-native conformance extension replays the original allocation
fixture used for the packed-arena implementation as **separate CUDA
virtual address slices of the same physical GPU root**, with the scalar
count deliberately different between the host shadow and the uploaded
GPU descriptor. It compares all region lists, histograms and canaries
against the independent logical-handle CPU oracle. The retained
removed-projection case and a completely live `PxScene` remain future
integration boundaries; the current tests demonstrate correctness of
the original source operation independently, not complete scene
execution.

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

## Native original `PxgCudaBuffer` ownership (PR #89)

`src/physx_port/gpucommon/PxgCudaBuffer.{h,cpp}` now owns the original
`physx::PxgCudaBuffer`/typed-buffer/buffer-family behavior against the
ported `PxgHeapMemoryAllocator`. Original 128-byte aligned device
views retain the underlying raw allocator keys through growth, swap,
assignment, and deferred deallocation. The original GPU heap remains
the sole allocator; no second buddy allocator or per-buffer GPU owner
has been introduced.

The source `allocateCopyOldDataAsync` copy boundary is lowered
internally to `SourceSapDmaNative`: copy the entire **old raw allocation**
on the original ordered stream and defer its buddy block until the
source simulation-step flush. Initial/uncopied allocation growth retains
the source doubling **and free-first** policy, avoiding extra peak GPU
memory consumption. Copy-preserving growth allocates replacement storage
before deferring its previous block, as required by the source copy ordering.

The existing Linux/Vulkan and Windows/D3D12 host-transfer conformance
now constructs the actual buffer object on source GPU heap roots, then
exercises growth, same-root ordered copies, deferred reuse, history
swaps, assignment, and teardown. The earlier pointer/size model
remains as a separate address-mapping and alignment oracle.

**Remaining source integration:** the actual `PxgCudaBroadPhaseSap`
constructor/update/fetch calls must adopt this buffer and the portable
context/stream execution seam. `PxCudaContext*, CUstream` is not yet
provided as a generic host-compatible interface; no live GPU `PxScene`
is implied. Source allocator abort/error translation and live
scene teardown require review at that integration boundary.

## Source `PxgCudaBroadPhaseSap::preBroadPhase` native input path (PR #90)

`src/physx_port/gpubroadphase/PxgSapPreBroadPhaseNative.{h,cpp}`
ports the original source `preBroadPhase` allocation and HtoD upload
sequence. It consumes `Bp::BroadPhaseUpdateData` directly and borrows
the original `PxgTypedCudaBuffer` owners, using the existing
`SourceSapDmaNative` stream; it does not create a second scheduler
or manage a second frame state.

The source semantics are retained:

- Allocate distance and group buffers every call when growth is required.
- Allocate environment-ID buffer only when the input provides IDs.
- Allocate/upload FP bounds only when `getStateChanged()` is true.
- Upload distances and groups on every call, and environment IDs
  only when present.
- Preserve existing GPU bounds/environment data across skipped uploads.

The four original consecutive source HtoD operations are submitted as
one ordered native upload group. Native staging copies input bytes before
the caller can reuse its update arrays and retires after GPU completion.

Conformance exercises the original buffer and buddy allocator across
initial, non-dirty, resized and rejected-input frames, checking actual
GPU storage on both native backends.

Next host connection: install these borrowed buffer references in
`PxgCudaBroadPhaseSap::preBroadPhase`, preserving its
`mUpdateData_BoxesCapacity` update and `PxgAABBManager` task ordering.
`gpuDMAUp` and its descriptor/sort buffers remain the next coherent
source-owned transfer sequence.

## Original `PxgCudaBroadPhaseSap::gpuDMAUp` handle/count staging (PR #92)

`src/physx_port/gpubroadphase/PxgSapGpuDMAUpHandlesNative.{h,cpp}`
now covers the first source bookkeeping and later changed-handle HtoD
portions of `gpuDMAUp`, consuming the actual
`Bp::BroadPhaseUpdateData`. The exposed members are borrowed references to
`PxgCudaBroadPhaseSap` counters and its three original
`PxgTypedCudaBuffer<PxU32>` objects, **not** a parallel broadphase.

The helper boundaries deliberately retain the upstream sequence:

1. Prepare created/removed/updated counts, live-box totals and
   pre-removal/padded projection sizes, rejecting invalid counts or overflow
   before mutating the source members.
2. Leave room for **the original persistent history/scratch allocation and
   copy-preserving growth** (not yet ported into this helper).
3. Allocate source created/removed/updated handle buffers using
   `PxgCudaBuffer` and the native buddy allocator.
4. Leave room for the original `updateDescriptor` /
   `updateRadixSortDesc` and mapped-host report pointer setup (pending).
5. Upload created/removed/updated handle arrays on the existing
   `SourceSapDmaNative` GPU stream before the future descriptor uploads.

This ordering matters: the split does **not** move new handle allocations
ahead of source history-preserving copies or descriptor construction.

The host-transfer test covers mixed add/remove/update, 32-bit sizing,
grow-only buffer behavior, CPU-array snapshot after enqueue, zero-change
frames, invalid handle pointers/counts, and original GPU-root teardown on
Vulkan/Linux and D3D12/Windows. These remain integration tests rather than
a live `PxScene` broadphase test; the full source construction,
descriptor population, and shader/task graph attachment are still pending.

