<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GPU broadphase source trace and migration ledger

Scope: **PhysX 5.11 GPU broadphase source behavior and the native FyzzXYZZY port contract**. The original PhysX scene is a reference for dependencies and correctness, not the runtime that will host FyzzXYZZY's GPU execution. See [native migration](NATIVE_ENGINE_MIGRATION.md).
Source family: PhysX GPU broadphase; no specific source revision is required by the SDK.
This page is a focused implementation worksheet; it complements
[the port program](GPU_PORT_PROGRAM.md) and
[scene-wide execution trace](PHYSX_GPU_PORT.md).

## 1. Entry point and scene dependency chain

Observed in `physx/source/simulationcontroller/src/ScPipeline.cpp`:

| Scene task or function | Required input | Observable output or continuation |
| --- | --- | --- |
| `Sc::Scene::rigidBodyNarrowPhase` | task manager, contact managers, shape bounds | schedules `mBroadPhase` through `mPostBroadPhase` chain; pairs and contact-manager tasks have dependencies |
| `setupBroadPhaseFirstAndSecondPassTasks` | GPU/CPU selection | GPU case chains `mBpFirstPass -> mBpSecondPass -> mPreIntegrate -> mBpUpdate` via continuations; CPU case omits that GPU-specific chain |
| `broadPhaseFirstPass` | bounds/groups/dirty maps, contact-distance dirty flag | calls `mAABBManager->updateBPFirstPass` |
| `preIntegrate` | scene `mGravity` / `mDt`, simulation controller | GPU preintegration and update of body/shape bounds when not CCD BP |
| `updateBroadPhase` | task pool, scratch allocator, updated AABB manager state | calls `mAABBManager->updateBPSecondPass`, releases narrowphase unlock, schedules related GPU particle/soft-body work |
| `postBroadPhase` | GPU BP result completion and output buffers | narrowphase `postBroadPhaseUpdateContactManager` and AABB manager `postBroadPhase` |
| `postBroadPhaseContinuation` onward | overlap lists, pair user data | interaction creation/destruction and scene contact state |
| `updateDynamics` / `updateDynamicsPostPartitioning` | contact outputs and completed island partition | solver scheduling, lost-contact continuation |
| `updateSimulationController` / `postSolver` / `afterIntegration` | completed GPU solver, changed bounds, friction streams | published body/core state, reports, swap/cleanup |

The arrows in the GPU broadphase row describe **continuation dependencies**,
not a claim that all GPU work uses one queue. The concrete task counts and
dependencies are defined by PhysX itself. Keep these entry points and user
visible `simulate`/`fetchResults` semantics.

## 2. Exact GPU SAP host update path

Source: `physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp`.

| Order / gate | Source function or operation | Semantic dependency |
| --- | --- | --- |
| 1 | `preBroadPhase` | stage changed float bounds and shared contact distances/groups/environment IDs |
| 2 | `update -> gpuDMAUp` | resize old/new integer bounds, projections, ranks, handles, histograms, aggregate/actor report buffers; retain old contents where required; upload created/removed handle arrays and GPU descriptor |
| 3 | `update` early return | skip unchanged frames unless `mForceUpdate` requires a one-time completion update; this matters for double-buffer/aggregate semantics |
| 4 | `translateAABBsKernel` | produce 6-component integer bounds with shifts, min/max parity and optional environment mapping |
| 5 | `markRemovedPairsKernel` | account for deletions against persistent previous-frame pair/projection state |
| 6 | `markUpdatedPairsKernel` only when `gpuStateChanged` | mark changed pairs using GPU-managed updated handle information |
| 7 | `sortProjectionAndHandlesWRKernel` | initialize ranks, radix sort projections, update handles; uses prior box count |
| 8 | `calculateEndPtHistogramKernel(true)` | generate incremental endpoint histogram data |
| 9 | `performIncrementalSapKernel` when `gpuStateChanged` | compare ranked endpoints and generate incremental found/lost pairs |
| 10 | `initializeSapBoxKernel` and `markRemovedPairsProjectionsKernel` when handles removed | remove old projection participation while preserving correct ranks |
| 11 | `markCreatedPairsKernel`, `sortProjectionAndHandlesWORKernel`, `calculateEndPtHistogramKernel(false)` when created/removed | update current projections, sort/histogram without ranks |
| 12 | `initializeSapBoxKernel(mNumOfBoxes, false)` | prepare final SAP box state |
| 13 | `computeRegionHistogramKernel` | compute region ranges and output histogram passes |
| 14 | `computeStartAndActiveHistogramKernel` | active/start region summary |
| 15 | `generateNewPairsKernel` | found pairs for new/updated projections in region context |
| 16 | `swapBuffer(mNewIntegerBoundsBuf, mOldIntegerBoundsBuf)` | preserve previous integer bounds (in normal incremental path) |
| 17 | `clearNewFlagKernel` | remove per-projection newly-created markers |
| 18 | `runCopyResultsKernel` | pair count/report accumulation and copying |
| 19 | `gpuDMABack` via `fetchBroadPhaseResults` | retrieve counters, completion synchronization, overflow/abort status and found/lost pair sizes |
| 20 | `fetchBroadPhaseResults` | swap each axis's double-buffered projections/handle and histogram families |
| 21 | `PxgAABBManager::processFoundPairs` / `processLostPairs` | purge duplicate reports, translate to overlap events, clear add/remove bitmaps |

**Do not implement this as "AABB encode then generic pair test".**
`runRadixSort` depends on shared CUDA radix operations from
`gpucommon/src/CUDA/radixSortImpl.cu` and `RadixSort.cuh`. This is a
mandatory source-translation dependency; replacing it with a proven
semantically equivalent sort requires documenting tie-order, key/handle
relationship, stability and incremental rank effects.

## 2a. Portable host frame roles and original update gates

Implementation: `src/physx_gpu/sap_host_frame_roles.{h,cpp}`,
tested through the existing `BroadPhaseCompletionGate` GPU fixture.
This is an adapter for the **original PhysX caller**, not a second
simulation loop or autonomous batch scheduler.

The original `PxgCudaBroadPhaseSap::update` calls `gpuDMAUp`
**before** determining whether to return early. For a quiet frame
(no created or removed handles, `stateChanged=false` and
`gpuStateChanged=false`), it dispatches once if `mForceUpdate`
is true, then sets `mForceUpdate=false`. The next quiet frame
returns early, leaving the flag false. Any active frame resets
`mForceUpdate=true` so the next idle frame performs one
additional bounds update. The constructor initializes the flag
to true. Source guards are distinct:

- `gpuStateChanged` gates the incremental SAP pair-production
  kernel; the base WR sort and incremental endpoint histograms
  are still part of an otherwise-dispatched update.
- Removed handles gate `initializeSapBoxKernel(previousBoxes)`
  plus removed-projection maintenance.
- Created **or** removed handles trigger created-projection
  maintenance, WOR sorting and full endpoint histograms.
- Created handles alone gate `clearNewFlagKernel`; a complete
  update otherwise continues through region/new-pair generation.
- `mNumOfBoxes` changes during `gpuDMAUp`; early return does
  not mean this host preamble was skipped.

`decide_sap_update` returns a pure, source-named decision record.
It rejects invalid box-population arithmetic rather than inheriting
unsigned wraparound. It does **not** perform `gpuDMAUp`, enqueue
kernels, manage original `PxgAABBManager` task references or
change the existing `ScPipeline` continuations.

The upstream `update` swaps **old/new integer-bound buffer
identities** after `generateNewPairsKernel`, before
`clearNewFlagKernel` and report accumulation. The separate
`fetchBroadPhaseResults` operation first calls
`gpuDMABack` (including signal/wait and result-size handling),
then swaps exactly **seven** double-buffered resource families
at `axis` ↔ `axis+3`: endpoint handles, block-end
histograms, end histograms, ordered end handles, block-start
histograms, start histograms, and ordered start handles.
Projections, ranks, integer bounds and SAP indices are **not**
included in that fetch-time seven-family swap.

`SapFrameBufferRoles` swaps `compute::BufferId` allocation
identities rather than copying GPU contents; the original
context and PhysX broadphase remain owners. It exposes separate
`swap_bounds_after_generate_new_pairs` and
`swap_histories_after_fetch` operations, with explicit
source-order preconditions. The existing completion gate is
still the operation that waits, fetches CPU-visible reports,
installs the original found/lost task continuations and
releases them once. Resource rotation belongs inside its
*successful* fetch callback, after data becomes visible.
A failed fetch does not rotate, and can retry without stranding
PhysX references.

The new native test checks active/idle/forced update decisions,
every conditional branch, both invalid count transitions, all
42 per-axis history resource identities, independent integer
bounds flipping, completion and failure ordering using a real
Vulkan/D3D12 submission, and an early-return/no-new-GPU-work
fetch. The test does **not** claim the PhysX scene has begun
using this adapter; that still requires integrating
`PxgCudaBroadPhaseSap` into the original factory/task graph,
including resizing, descriptor repacking and asynchronous
CUDA-to-backend transfer adaptation.

## 2b. Original 3D region producer after created-handle processing

The upstream `PxgCudaBroadPhaseSap::computeRegionHistogramKernel`
runs **only when at least one created handle exists**, after the final
nonincremental endpoint histogram and SAP index initialization.
It first clears the dense `activeRegionsHistogram` and
`startRegionsHistogram` arrays and dispatches
`BP_CREATE_REGIONS` followed by four dedicated prefix passes
(start region stage 1/2 and total region stage 1/2).
These two clearing operations are **not** a reset of
`startRegionAccum` or `regionAccum`: those are sparse
per-projection counts produced by `BP_CREATE_REGIONS`.

The source `broadphase.cu::createRegionsKernel` launches
**32 workgroups × 256 threads**, walking contiguous 256-endpoint
partitions with
`ceil(ceil(2*numBoxes/256)/32)` iterations per workgroup.
It reads the *current* X-sorted `boxHandles[0][X]` and each
axis's `boxSapBox1D[axis]` endpoint indices. Every start
endpoint obtains its own start/end **all-endpoint histogram**
counts in all three axes, then divides each count by
`ceil(numBoxes/REGION_SIZE_PER_AXIS)` where
`REGION_SIZE_PER_AXIS = 4`. This yields
`PxgIntegerRegion::minRange/maxRange` (two `uint4`
vectors per handle) in a 4×4×4 region grid.

For a start endpoint the source writes
`startRegionAccum[boxStartX]` as the number of YZ regions
covered; it zeros `startRegionAccum[boxEndX]`.
`regionAccum[xSortedIndex]` is the full XYZ region count
for start endpoints, or zero for end endpoints. The original
per-handle and per-endpoint storage addresses must be preserved
for subsequent region scan and active/start worklist kernels.
Tags are inspected using the upstream `isStartProjection` bit
and handles decoded with `getHandle(tagged) = tagged >> 3`.

`SapRegionProducerArenaStage` ports **only** this source
producer as Vulkan/D3D12 shared HLSL with a caller-owned,
byte-offset subset of `PxgBroadPhaseDesc`. It does not
allocate persistent regions, publish pairs, or replace the
host's created-handle gate, memory clears or later scans.

The independent GPU oracle constructs three-axis sorted
endpoints, original all-endpoint exclusive histograms, and
per-handle SAP min/max positions, then iterates **objects**
rather than sorted X endpoints to calculate expected region
ranges and sparse counts. It compares the **entire arena**,
including untouched tags, histograms, padding and unused
regions, on both native APIs. Cases include zero-created
and empty skipped updates, tied projection keys, asymmetric
axis ordering, new-handle tags, and 4099 boxes which cross
the first 8192-projection iteration boundary.

**Next dependency:** `BP_COMPUTE_START_REGION_HISTOGRAM`,
`BP_OUTPUT_START_REGION_HISTOGRAM`,
`BP_COMPUTE_REGION_HISTOGRAM`, and
`BP_OUTPUT_REGION_HISTOGRAM`, followed by the original
active-region and overlap-check pair work. Do not fold these
into a generic grid overlap engine.

## 2c. Source start-region and total-region four-pass prefix sequence

After `BP_CREATE_REGIONS`, the original
`PxgCudaBroadPhaseSap::computeRegionHistogramKernel` launches
**four separate kernels in this exact order**:
`BP_COMPUTE_START_REGION_HISTOGRAM`,
`BP_OUTPUT_START_REGION_HISTOGRAM`,
`BP_COMPUTE_REGION_HISTOGRAM`, then
`BP_OUTPUT_REGION_HISTOGRAM`.
All four use the original **32 blocks × 256 threads** and are
gated on a nonempty created-handle list. They consume the
previously produced sparse per-projection
`startRegionAccum` and `regionAccum` arrays, without
re-running region generation.

`broadphase.cu::computeHistogram` computes per-block
**exclusive** prefix sums and exactly 32 block totals, with
contiguous `ceil(ceil(2*numBoxes/256)/32)*256`-projection
partitions. `outputHistogram` applies preceding block sums,
writes one grand total to the source
`startRegionAccumTotal` or `regionAccumTotal` descriptor
field and leaves unrelated state untouched. The portable
HLSL lowers CUDA warp and block scans to 256-thread
workgroup scans and preserves the four dispatches rather
than combining them into a new generic GPU pipeline.

`SapRegionHistogramsArenaStage` owns the four shader
pipeline handles but not their persistent data buffers.
It appends four ordered commands transactionally to its
caller's batch. CMake compiles the first and second scan
sources separately for start-region and total-region views,
using the same code on Vulkan SPIR-V and D3D12 DXIL.
The full GPU fixture runs
**BP_CREATE_REGIONS → all four region prefix stages**
in one ordered batch. The independent CPU reference
calculates each region count per object and then computes
linear per-projection prefix sums, 32 block totals and
both grand totals. It compares every arena word, including
unused capacities and source inputs, on zero-work and
1/5/17/129/257/4099/8201-box workloads.

The next original stages are
`BP_WRITEOUT_ACTIVE_HISTOGRAM`,
`BP_COMPUTE_ACTIVE_HISTOGRAM`,
`BP_OUTPUT_ACTIVE_HISTOGRAM`, and the original overlap
count / pair-write sequence. These remain explicit
dependencies; this is not a fabricated replacement
simulation pipeline.

## 2d. Original start/active region worklists

Following the four region prefix passes,
`PxgCudaBroadPhaseSap::computeStartAndActiveHistogramKernel`
launches three CUDA kernels, gated on nonzero created handles:
`BP_WRITEOUT_ACTIVE_HISTOGRAM` (**32×256**),
`BP_COMPUTE_ACTIVE_HISTOGRAM` (**64×512**), then
`BP_OUTPUT_ACTIVE_HISTOGRAM` (**64×256**). The source
implements these in `broadphase.cu` as
`writeOutStartAndActiveRegionHistogram`,
`computeStartAndActiveRegionHistogram`, and
`outputOrderedActiveRegionHistogram`.

The first maps every work item in the previously scanned
`startRegionAccum` to its X start projection and one YZ
region. It marks the source-owned dense
`startRegionsHistogram[region, xProjection]` and sets
`activeRegionsHistogram` only for endpoints with the
original new-projection tag. **PhysX initializes both
dense histogram arrays to zero before `BP_CREATE_REGIONS`.**
They are not created or cleared by the three kernels here.

The middle kernel computes the **two exclusive region prefixes**
from one packed 32-bit value per projection: `start<<16 | active`,
as in the original CUDA code. Each 512-projection chunk contains at
most 512 unit contributions to either field, so the packed addition
cannot cross the 16-bit field boundary; across chunks the source
maintains two independent wide carry accumulators. The first
portable lowering unnecessarily used **two separate 512-wide
Hillis–Steele shared-memory scans** (8 KiB scratch, repeated
workgroup-wide synchronization). That was functionally sound but
a performance regression risk introduced by this port, **not a
requirement of the original PhysX algorithm**.

PR #44 replaces it with a source-packed subgroup scan on devices
reporting compute subgroup arithmetic: a single `WavePrefixSum`
per packed value and `WaveActiveSum` per subgroup, a short
subgroup-total scan, and three full-group barriers per 512-entry
chunk. Wave leaders atomically acquire logical subgroup numbers
**once per workgroup**, so mapping of physical wave lanes to
`SV_GroupIndex` is never assumed. 32/64-lane implementations
can both use the native fast path. The scan requires only
2 KiB of shared wave-subtotal storage versus 8 KiB originally.
A source-packed, **single** shared-memory scan (4 KiB) is retained
strictly as the functional fallback where native subgroup
arithmetic is unavailable. CUDA warp assumptions do not affect
the output order. These are **analytical** operation/scratch
comparisons; **hardware speedup is not yet measured**.

The third kernel traverses the same original work items and
writes compact, X-ordered `orderedStartRegionHandles` and
the newly-created subset `orderedActiveRegionHandles`.
The independent GPU oracle exercises both the native and fallback
implementations on the same full-buffer fixtures, including
multi-512-projection loops. Compiled target for the fast D3D12
kernel is `cs_6_6`, consistent with the planned SM6.6 shipping
minimum; Vulkan requires a separate queried subgroup-arithmetic
capability, not an imagined SM6.6 Vulkan bit.
**All further optimization departures must follow
[GPU_PORT_PERFORMANCE.md](GPU_PORT_PERFORMANCE.md).**

`SapActiveRegionListsArenaStage` retains the three exact
launch shapes, the same create-only host gate, and the
original resource relationships. Its common shader is
compiled to Vulkan SPIR-V and D3D12 DXIL. The descriptor
is a compact offset lowering of the original
`PxgBroadPhaseDesc`, **not** an independent broadphase
owner. It validates that the dense 64-region storage and
rounded projection-count calculations fit its 32-bit
addressing contract.

Its native GPU test seeds source-compatible sorted X
endpoints, `PxgSapBox1D` positions, 4×4×4 per-handle
region ranges, and exclusive start-region work prefixes.
An independent CPU oracle traverses logical X start
objects, marks membership, linearly scans each region
and emits ordered lists, without duplicating the GPU
binary-search work mapping. Entire arenas including
unwritten padding and source input are compared for
empty, quiet, small, and multi-512-projection workloads.

The first five producer/prefix kernels remain separately
tested. This fixture does **not** yet submit all eight
kernels in a live PhysX scene or perform the original
overlap-check and new-pair work. Next are
`BP_WRITEOUT_OVERLAPCHECKS_HISTOGRAM_NEWBOUNDS`,
`BP_COMPUTE_OVERLAPCHECKS_HISTOGRAM`,
`BP_OUTPUT_OVERLAPCHECKS_HISTOGRAM`, and
`BP_GENERATE_FOUNDPAIR_NEWBOUNDS`, with original
host scheduling, ownership and result publication
preserved.



## 2d. Created-handles new-bounds four-kernel handoff (integrated adapter)

The source `PxgCudaBroadPhaseSap::generateNewPairsKernel` calls the
following sequence, **only if `mUpdateData_CreatedHandleSize != 0`**:

1. `BP_WRITEOUT_OVERLAPCHECKS_HISTOGRAM_NEWBOUNDS` (32 × 256)
2. `BP_COMPUTE_OVERLAPCHECKS_HISTOGRAM` (32 × 256)
3. `BP_OUTPUT_OVERLAPCHECKS_HISTOGRAM` (32 × 256)
4. `BP_GENERATE_FOUNDPAIR_NEWBOUNDS` (64 × 256)

`SapNewBoundsSequenceArenaStage` now appends these four existing pipeline
stages, in exactly that order, to the **caller's existing command list**.
It neither allocates an arena nor creates a new scene update loop. The
constructor receives existing stage references; the caller owns their
pipelines and lifetimes. The caller's PhysX frame decision supplies its
current-box count and created-handle count. On a no-created frame it
appends nothing, even if the containing broadphase update continues for
some other reason. It stages all four commands before modifying the
caller's vector, so a later descriptor validation error cannot leave a
half-appended new-bounds sequence.

The per-frame `current_boxes` is pushed directly into the found-pair
kernel, matching the count used for overlap-work generation; a persistently
allocated pair descriptor is **not** authoritative for that changing field.
A regression holds the pair descriptor's old count at two boxes while the
live frame has three, and checks canonical ownership of a valid pair in
region 1. This prevents the stale-stride error reported in Codex's P1.

The three older arena subsets represent the **same** original
`PxgBroadPhaseDesc` fields. To avoid divergent ad hoc bindings,
`make_sap_newbounds_sequence_descriptors` derives the producer and
prefix descriptors from the pair stage's authoritative shared offsets,
plus the upstream per-projection region prefix and 32-block overlap
scratch. The caller uploads those three small descriptor views whenever
its existing arena layout is established/changed. The full sequence
binds one caller-owned GPU arena in read/write mode on all four
commands. The compute backend already orders dependent RW accesses in
one batch using Vulkan compute memory barriers or D3D12 UAV barriers;
no CPU downloads or per-stage fences are introduced between the four.
All 64-bit overlaps and prefixes stay on GPU until final found-pair
reporting. As in CUDA, the pair counter is 32-bit, retained across
other found-pair writers in the frame, and output writes are bounded by
`max_found_lost_pairs` without clamping the total counter.

The new **integrated GPU fixture**, run on Vulkan and D3D12, starts
with X-sorted handles, input region prefix, physically consistent
multi-region membership histograms, ordered region handles, and
integer bounds. It executes all four source kernels in one GPU batch,
compares the producer's handle-region mapping, 64-bit exclusive
comparison prefixes, all 32 block totals, grand total and final
found-pair set to an independently enumerated CPU reference. Cases
exercise new/non-new projection histograms, multi-region canonical
duplicate suppression, group filtering, separated bounds,
environment IDs and wildcard IDs, pre-existing pair count, capacity
clipping, created-handle gating, zero region work, invalid descriptor
rollback and untouched arena canaries. The existing
`BroadPhaseCompletionGate` waits on the batch at the original fetch
boundary; the CPU oracle checks all GPU results inside its publication
callback **before** found/lost-pair task references are released. The
no-created case exercises the zero-submission publication path. The GPU
test therefore checks both **kernel-chain dataflow and source publication
ordering**, not just isolated shader oracles.

**Remaining integration boundary:** the live `PxgCudaBroadPhaseSap`
implementation still owns its CUDA allocations, update/fetch task
chain, buffer swaps, `gpuDMABack` and `PxgAABBManager` report
publication. The new adapter makes the four-kernel submission a tested,
source-aligned unit usable by the portable PhysX host, but it has **not**
replaced those CUDA dispatches in a live PhysX scene. That replacement
requires the broader source-owned scene GPU orchestration and resource
lifetime port described in sections 1–3, not an invented surrogate
simulation pipeline.

## 2e. Source-owned native kernel batch to actual PhysX postBroadPhase (PRs #64–#68)

The arena fixture in section 2d remains a useful regression, but is **not**
the source-resource integration path. The production-direction native stages
read the original `PxgBroadPhaseDesc` and resolve its CUDA allocation pointers
through `CudaDeviceAddressMap` without repacking source allocations:

- `SourceOverlapWorkNativeStage`: 32 × 256, nine original source views.
- `SourceOverlapPrefixNativeStage`: two 32 × 256 launches, four views,
  consumes the GPU-written `regionAccumTotal` and writes the resident
  `overlapChecksTotalRegion`.
- `SourceNewBoundsPairsNativeStage`: 64 × 256, sixteen views and the original
  GPU-resident 32-bit found-pair counter.
- `SourceNewBoundsSequenceNativeStage`: appends exactly those four original
  commands, with created-handle gating and atomic append validation; it has
  no queue, device allocation, DMA or frame-task ownership.

The common `source_native_view` preserves original interior CUDA addresses,
including four-byte scalar fields, while setting 16-byte-aligned RAW view
bases required by D3D12. Its regression ensures address preservation and
rejects malformed D3D12 views before a GPU submission.

In the source, **`PxgCudaBroadPhaseSap::update()`** runs `gpuDMAUp`,
translated/removed/created SAP work, region histogram production, active and
start-region lists, `generateNewPairsKernel()`, bound-buffer swapping,
`clearNewFlagKernel()` and report accumulation/copy, all on `mStream`.
The first-pass manager's `preBpUpdate_GPU()` previously queued bitmap,
group, bounds and aggregate uploads on that *same* source stream. Translating
only `generateNewPairsKernel()` into a standalone submit operation would
lose those producer dependencies: it must be inserted into the native lowering
of this **existing** host execution chain.

The source publication path is:

1. `PxgCudaBroadPhaseSap::gpuDMABack()` enqueues an asynchronous descriptor
   copy, a mapped signal kernel, then flushes/waits on `mStream` at the fetch
   boundary (with stream synchronization fallback); reads the resident counts;
   clips found/lost pair-list sizes and handles abort/overflow.
2. `PxgCudaBroadPhaseSap::fetchBroadPhaseResults()` performs that fetch then
   swaps seven six-slot SAP histogram/projection-history buffer families.
3. `PxgAABBManager::postBroadPhase()` calls `fetchBroadPhaseResults()` and
   `resizeFoundAndLostPairs()`; **only then** it either sets found/lost task
   continuations and releases both references, or—if `continuation == nullptr`—
   calls the two concrete `Cm::Task::runInternal()` methods directly, without
   changing their task reference counts.

`BroadPhaseCompletionGate` represents the GPU wait → fetch/resize → task
release ordering and now also the original null-continuation inline branch.
The GPU integration fixture passes the *actual four-kernel submission token*
to this gate, validates the complete GPU-resident data inside the publication
callback and verifies the two task paths after publication. **These are
source-contract tests, not proof of a live portable GPU PxScene.**

**Next actual implementation boundary:** keep `PxgCudaBroadPhaseSap` and
`PxgAABBManager` as host owners, translate their used `mStream` DMA,
memset, kernel-launch, buffer-grow/swap, event/wait and abort semantics to the
native compute backend; bind native kernels at their existing source launch
sites, and port the original `Sc::Scene` continuation dependencies into
FyzzXYZZY's single native task graph. Do not introduce a competing scene
loop or claim native G3 until the complete
producer/report chain and multi-frame state have been validated.

## 3. Buffer, ownership, and publication worksheet

The precise allocation/free timeline still requires per-buffer tracing.
The following are source-observed starting points; fill the pending fields
as individual stages are implemented.

| State family | Source allocator / owner | Producer | Consumer | Publication / hazard |
| --- | --- | --- | --- | --- |
| `mBoxFpBoundsBuf` / `mBoxContactDistancesBuf` / `mBoxGroupsBuf` / `mBoxEnvIDsBuf` | Broadphase/AABB manager, GPU allocator | `preBroadPhase` / scene/controller bounds | translate, group and pair kernels | updated GPU bounds may bypass CPU upload; preserve dirty flags |
| `mNewIntegerBoundsBuf` / `mOldIntegerBoundsBuf` | `PxgCudaBroadPhaseSap` | AABB encoding / swap | incremental pair comparisons | old/new swap retained across frames |
| `mBoxPtProjectionsBuf`, `mBoxProjectionRanksBuf`, `mBoxPtHandlesBuf[6]` | `PxgCudaBroadPhaseSap` | sort/rank kernels | SAP, region, pair generation | double buffering and six arrays (two histories times three axes) |
| `mStartPtHistogramBuf` / `mEndPtHistogramBuf` (six variants and block buffers) | `PxgCudaBroadPhaseSap` | endpoint histogram kernels | SAP and update | retain/copy historic content on growth |
| `mRegionRangeBuf` / `mActiveRegionTotalBuf` / `mStartRegionsTotalBuf` | `PxgCudaBroadPhaseSap` | region histogram passes | pair generation | zero/fill/reset before cumulative update |
| `mCreatedHandlesBuf` / `mRemovedHandlesBuf` / changed bitmap buffers | `PxgCudaBroadPhaseSap` and `PxgAABBManager` | first-pass CPU dirty state / controller | second pass | bitmap semantics differ from updated-handle array (currently empty GPU path) |
| `mBPDescBuf` (`PxgBroadPhaseDesc`) | `PxgCudaBroadPhaseSap` | host descriptor packing | essentially all BP kernels | contains many typed device pointers; portable layout not defined yet |
| Found/lost actor and aggregate pair arrays | `PxgCudaBroadPhaseSap`, `PxgAABBManager` | incremental SAP, new-pair, aggregate passes | CPU `processFoundPairs` / `processLostPairs` | overflow capacity, abort, deduplication, scene callbacks |
| `mEventMapped` / CUDA `mStream` and `mEvent` | `PxgCudaBroadPhaseSap` | GPU report completion | `gpuDMABack` | CUDA mapped signal + spin/fallback sync must be replaced by portable fence/event protocol |

Note source `gpucommon/include/PxgCudaBuffer.h`: the raw pointer kept
by the allocator is **not** necessarily the aligned pointer returned to
kernels. `swapBuffer` swaps the correct underlying allocation identities.
When adopting abstract buffer views, preserve this lifetime and alignment
invariant rather than reproducing pointer arithmetic unnecessarily.

## 4. Boundary-specific tests before accepting scene integration

1. **Data layout and address test:** verify `PxgBroadPhaseDesc` fields, nested
   pointer/offset lowering, six-component integer bounds, GPU device memory
   views, and dynamic resize without stale references on Vulkan and D3D12.
2. **Pure kernel fixture:** compile the AABB kernel from one HLSL source,
   compare exact integer outputs, unaligned capacities, environment sentinels
   and no-overwrite padding. (First stage exists; keep running it.)
3. **Incremental sort fixture:** collisions and ties in projections across
   three axes; stable/expected ranks and handle remapping on both APIs.
4. **Pairs through frames:** a scene that introduces, updates, removes and
   sleeps objects; compare created/destroyed overlap **sets** and legal
   report ordering against the original CUDA implementation (if available)
   and CPU PhysX general behavior.
5. **Aggregate regression:** empty aggregate excluded; aggregate membership
   changes, actor–aggregate and aggregate–aggregate pairs, duplicate handling.
6. **Capacity and failure:** saturate found/lost pairs and verify warning,
   counters and omission policy; forced update; aborted submission or device
   loss cannot strand native task continuations or publish half-complete frames.
7. **Native scene acceptance:** FyzzXYZZY's own scene selects GPU broadphase
   and native CPU dynamics; its real simulation/fetch lifecycle publishes
   found/lost pairs into native narrowphase across frames. The reference
   PhysX `PxScene` may run independently as an oracle, but never hosts
   the candidate GPU work. **Exactly one native simulation scheduler.**

Until these tests pass, call the stage a **port-in-progress**, not a working
portable GPU broadphase.

## 5. Open source questions for the next audit pass

- Resolve the complete `PxgBroadPhaseDesc` pointer graph: byte layouts,
  nested aliases, pointer lifetimes, host/GPU producers and mutation order.
- Trace each radix-sort stage and its stable tie semantics, including 32-lane
  CUDA assumptions and how it depends on the prior-frame rank data.
- Make the CPU/GPU-updated-bounds split explicit at the exact
  `PxgSimulationController::preIntegrateAndUpdateBound` boundary.
- Map every `CUstream` and `CUevent` edge for broadphase and aggregate work,
  including `gpuDMABack` and the corresponding native completion/publication edge.
- Validate abort-mode and result-capacity handling before generalizing
  allocator or callback semantics.
- Reconcile original CUDA support flags and CPU fallback behavior with
  proposed GPU broadphase factory and context creation; avoid blind replacement
  of `PxCudaContextManager*` parameter types.

For each question, record the original source function/field, a precise
behavioral claim, evidence, new portable representation, and the acceptance
test before declaring it resolved.
