<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PhysX GPU port: execution architecture and traceability

See also [the multi-month GPU port program, blockers and acceptance gates](GPU_PORT_PROGRAM.md)
and the [broadphase source-to-port worksheet](GPU_BROADPHASE_TRACE.md).
**All GPU work is also bound by the [GPU performance contract](GPU_PORT_PERFORMANCE.md)**, which requires explicit review of throughput costs and modern hardware primitives for every deviation from source PhysX.

Status: initial source-grounded execution-path audit. This is a port dependency
map, not a claim that any PhysX GPU subsystem has been implemented on Vulkan or
D3D12. Source reference: the PhysX CUDA and host algorithms, without a fixed external source revision.

## Port boundary

FyzzXYZZY ports PhysX GPU physics **within the PhysX simulation pipeline**.
Kernel translation and host-side orchestration are equally part of the port.
The portable GPU implementation must retain the causal dependencies between
scene updates, collision detection, contact management, constraint preparation,
solving, integration, sleeping, and publication of scene state.

CUDA streams, events, modules, and device pointers are implementation mechanisms
to translate. The stages they connect, ownership and lifetime invariants,
task continuations, contact/constraint representations, and ordering constraints
must first be accounted for. Changes to PhysX's algorithms or execution
architecture require an identified source counterpart and a technical rationale.

## Verified scene-to-GPU path

These entry points and dependencies are present in the audited PhysX source:

1. `physx/source/physx/src/NpScene.cpp`:
   `NpScene::simulate` enters `simulateOrCollide`; `executeScene` invokes
   `mScene.simulate`. The scene state machine also supports separated
   `collide`, `fetchCollision`, `advance`, and result fetching.
2. `physx/source/simulationcontroller/src/ScScene.cpp`:
   `Sc::Scene` selects GPU broadphase and GPU dynamics independently from
   `PxSceneDesc`. When GPU dynamics is enabled, it creates the GPU dynamics
   context, narrowphase implementation (with CPU fallback context), and
   simulation controller using `PxPhysXGpu`; it joins their contact, patch,
   force/index, and friction-patch stream pools with `PxsContext`.
   The PhysX GPU factory is implemented in
   `physx/source/physxgpu/src/PxgPhysXGpu.cpp`.
3. `physx/source/simulationcontroller/src/ScPipeline.cpp`:
   `Sc::Scene::simulate` schedules task continuations; `collideStep` and
   `rigidBodyNarrowPhase` coordinate shape upload, broadphase, and narrowphase
   with task dependency edges. `broadPhaseSecondPass` schedules
   `preIntegrate` before the later broadphase update. Contact manager updates
   and lost/found touches feed island generation and dynamics.
4. In the same `ScPipeline.cpp`, `updateBodies` invokes the simulation
   controller; `updateDynamics` calls `mDynamicsContext->update`, followed
   by `updateDynamicsPostPartitioning` and
   `mDynamicsContext->updatePostPartitioning`. The GPU branch uses different
   task continuations from the CPU-only path. The upstream comments explicitly
   identify partitioning/continuation scheduling as an architectural concern.
5. `physx/source/gpusolver/src/PxgContext.cpp`:
   `PxgGpuContext::update` consumes narrowphase outputs and GPU contact data,
   accesses the island manager, synchronizes with the simulation controller,
   and prepares solver state. `updatePostPartitioning` consumes partition
   arrays, stages solver memory, and synchronizes joint data before solving.
   `PxgCudaSolverCore.cpp` and `PxgTGSCudaSolverCore.cpp` own kernel-launch
   sequences and stream/event dependencies for the respective solver paths.
6. `physx/source/simulationcontroller/src/ScPipeline.cpp`:
   `updateSimulationController` initiates GPU DMA-back and
   `mDynamicsContext->updateBodyCore`; `postSolver` merges results and swaps
   friction streams; `afterIntegration` updates scene body/shape simulation
   state. These steps are part of physics correctness and state visibility,
   not optional presentation or rendering code.

The GPU source also contains distinct CPU fallback paths for unsupported
contact managers and configurations. Port decisions must classify these paths
explicitly, rather than silently substituting simplified physics.

## Port traceability matrix

| Responsibility | PhysX host-side source | GPU algorithm/implementation | Portable work required |
| --- | --- | --- | --- |
| Scene selection, initialization, ownership | `simulationcontroller/src/ScScene.cpp`, `physxgpu/src/PxgPhysXGpu.cpp` | Selected GPU factory products | Preserve scene/controller/context topology; replace CUDA context provision and capability checks |
| Scene task ordering | `physx/src/NpScene.cpp`, `simulationcontroller/src/ScPipeline.cpp` | Task continuations and GPU callbacks | Preserve task graph and stage ordering; bridge to portable queue completion |
| Device memory and kernel registration | `gpucommon/src/PxgMemoryManager.cpp`, `gpucommon/src/PxgCudaMemoryAllocator.cpp`, `gpucommon/src/PxgKernelWrangler.cpp` | CUDA allocations, modules, device pointers | Design typed resource/descriptor indirection, lifetime and allocation contracts |
| Broadphase and pair management | `gpubroadphase/src/PxgCudaBroadPhaseSap.cpp`, `gpubroadphase/src/PxgAABBManager.cpp` | `gpubroadphase/src/CUDA/broadphase.cu` and related kernels | Port GPU SAP/pair pipeline with original pair and bounds contracts |
| Narrowphase and persistent contacts | `gpunarrowphase/src/PxgNphaseImplementationContext.cpp`, `gpunarrowphase/src/PxgNarrowphaseCore.cpp` | `gpunarrowphase/src/CUDA/` | Preserve contact-manager mapping, contact/patch outputs and fallback decisions |
| Scene GPU state and transfers | `gpusimulationcontroller/src/PxgSimulationController.cpp`, `gpusimulationcontroller/src/PxgSimulationCore.cpp` | `gpusimulationcontroller/src/CUDA/` | Preserve shape/body update and DMA-back boundaries; translate stream/event joins |
| Islands, partitioning and constraint state | `gpusolver/src/PxgContext.cpp`, `gpusolver/src/PxgSolverCore.cpp` | `gpusolver/src/CUDA/constraintBlockPrePrep.cu` and constraint-prep kernels | Preserve island and partition representation, contact/joint batches and host continuations |
| PGS/TGS solve and integration | `gpusolver/src/PxgCudaSolverCore.cpp`, `gpusolver/src/PxgTGSCudaSolverCore.cpp` | `gpusolver/src/CUDA/solver.cu`, `integration.cu`, TGS equivalents | Reproduce algorithm and iteration ordering, solver buffers, synchronization and writeback |
| Articulation dependencies | `gpuarticulation/`, `gpusimulationcontroller/src/PxgSimulationController.cpp` | `gpuarticulation/src/CUDA/` | Account for articulation streams and joint/contact preparation before adopting a slice |

Paths are relative to the `physx/source/` prefix unless they begin with
`physx/source/`; CUDA directories are source anchors, not independently
complete implementations.

## Execution requirements discovered so far

- **GPU narrowphase and solver cannot be replaced independently of their data
  interface.** `Sc::Scene` obtains contact, patch and force/index streams from
  `mDynamicsContext`, supplies them to the GPU narrowphase, then exposes the
  same streams through `mLLContext`.
- **Scene flags are not one GPU switch.** GPU broadphase and GPU dynamics have
  separate selection paths. Unsupported primitive handling may go through
  PhysX's own fallback, and the intended portable policy must be explicit.
- **GPU solving is not an independent dispatch.** The scene pipeline prepares
  contacts, islands, bodies and constraints, then schedules dynamics and
  post-partitioning continuations.
- **GPU work uses multiple dependency domains.** Solver, simulation core,
  broadphase, narrowphase and articulation components create and synchronize
  streams/events. One ordered Vulkan command batch is a useful primitive,
  but is not yet a substitute for those inter-stage dependencies.
- **State publication is part of the simulation.** DMA-back, body-core update,
  friction stream swapping, contact reporting and body/shape state updates
  must preserve PhysX's timing and consistency guarantees.
- **PGS and TGS are distinct GPU paths.** Porting a generic replacement solver
  would bypass real PhysX iteration, contact and partitioning behavior.

## Current FyzzXYZZY implementation classification

| Existing code | Classification | Disposition |
| --- | --- | --- |
| `src/world.cpp` | Actual PhysX CPU scene | Keep; CPU conformance baseline |
| `src/compute_vulkan.cpp`, `include/fyzzxyzzy/compute.h` | Portable execution infrastructure | Keep; extend only against observed PhysX GPU requirements |
| `src/kernels/rigid_floor.comp`, `src/floor_contact.cpp` | Independent demonstration, not a PhysX GPU kernel port | Build only with experimental opt-in |
| `src/world_vulkan.cpp` | Independent floor-simulation backend, not a PhysX GPU scene | Build only with experimental opt-in; do not extend as the production solver |
| PR #9 vertical-contact work | Experimental, non-PhysX contact pipeline | Leave unmerged; evaluate separately from the production port |

The current Vulkan compute `Context` and batch API do not yet expose a
PhysX-compatible task/synchronization contract. Do not attach a new physics
algorithm directly to the public `WorldBackend` as a substitute.

## First translated PhysX GPU stage: broadphase bounds encoding

The first executable translation is maintained under `src/physx_gpu/`:

- Source kernel: `physx/source/gpubroadphase/src/CUDA/broadphase.cu`,
  `translateAABBsLaunch` and `encodeFloat`.
- Upstream host caller: `physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp`,
  `translateAABBsKernel`, invoked by `PxgCudaBroadPhaseSap::update`.
- Upstream binary layouts: `PxBounds3` (six contiguous 32-bit floats),
  `PxgIntegerAABB::mMinMax[6]` (six contiguous 32-bit unsigned integers),
  and per-box `PxReal` contact distances / optional `PxU32` environment IDs.
- Shared source: `src/physx_gpu/kernels/physx_translate_aabbs.hlsl`.
  DXC generates Vulkan SPIR-V and D3D12 DXIL from one translated algorithm.
  It preserves the per-component float-bit ordering, inflated bounds, per-axis
  shifts, min-even/max-odd encoding and shared-environment sentinel behavior.
- Host dispatch: `AabbEncodingStage::append` in
  `src/physx_gpu/aabb_encoding.cpp`. A 128-thread workgroup encodes
  16 boxes (eight logical threads per box). Upstream launches 256 threads;
  the reduced physical workgroup size is required for Vulkan core portability
  and preserves the original per-box lane mapping and encoding semantics.
  The stage appends work to the caller's command batch and never owns a
  scene or overrides collision behavior.
- Test: `fyzzxyzzy_physx_aabb_encoding` compares the encoded integer output
  with an upstream-formula CPU reference for default and custom axis shifts,
  environment IDs, absent environment arrays, capacity boundaries, and invalid
  configuration rejection. The Vulkan test is required in Linux CI.

**Current limit:** this stage is not yet called by a PhysX scene. It does not
produce collision pairs or provide GPU broadphase on its own. The Vulkan
pipeline uses DXC-compiled HLSL, and native D3D12 WARP execution is now
verified for this bounded AABB stage. Hardware D3D12 profiling, a general
multi-backend GPU executor, and PhysX scene integration remain outstanding. The data and algorithm
contract belongs to PhysX rather than a new physics subsystem.

### Broadphase host orchestration required to integrate it

The upstream `PxgCudaBroadPhaseSap.cpp` provides a specific call sequence:

1. `preBroadPhase` stages changed floating-point bounds, per-object contact
   distances, filter groups and optional environment IDs.
2. `update` calls `gpuDMAUp` to update GPU buffers and descriptors, tracks
   created/removed/updated handles and sleeping/forced-update state, then
   dispatches `translateAABBsKernel`.
3. It marks removed/updated pairs, maintains incremental SAP projections,
   executes sorts, histogram/scans and comparison passes, and generates found
   pairs. Operations depend on previous-frame state and cannot all be
   indiscriminately recomputed without changing behavior.
4. `runCopyResultsKernel` and `fetchBroadPhaseResults` publish the found/lost
   pair data for scene narrowphase and update double-buffered projections,
   handles and histograms.
5. `PxgAABBManager` owns additional GPU aggregate and dirty-bounds handling
   and its coordination with the broadphase; its origin and synchronization
   dependencies must be preserved.

PhysX's GPU broadphase does **not** require GPU dynamics, but the current
factory and `Sc::Scene` initialization still require the CUDA context manager
and GPU-specific memory allocators. Simply setting `PxBroadPhaseType::eGPU`
in the existing CPU build would not enable this Vulkan stage. The next
integration work must adapt that factory/controller boundary and its resource
lifetime, without introducing a separate simulation loop.

### Source-derived deleted-projection maintenance (next shader)

The upstream source links two distinct removal operations in
`PxgCudaBroadPhaseSap::update`:

1. `markRemovedPairsKernel` invokes `BP_MARK_DELETEDPAIRS` to mark the
   *previous-frame* SAP handles as deleted and empty old bounds.
2. After the intervening sort, rank/handle update and incremental SAP
   operations, `markRemovedPairsProjectionsKernel` invokes
   `BP_UPDATE_DELETEDPAIRS`/`markRemovedPairsProjectionsLaunch`. It uses
   the current `boxSapBox1D[axis]` index pair to set both
   `boxProjections[axis]` endpoints to 0xffffffff and invalidates the
   SAP min/max entries before the second sort shuffles removals away.

Both original host launches use **32 blocks of 256 threads**, irrespective
of removed count; the device loop strides by 8192. The first port had
used a 128-thread dynamically sized grid. Its grid is now corrected,
and the new projection stage matches the original 32x256 shape.

The new `RemovedProjectionsArenaDesc` extends the existing 64-byte
logical descriptor with **three projection offsets** at bytes 64..75;
the source's other pointers and previous/current handle distinction
retain their original meaning. The HLSL shader carries only the source
algorithm, with an arena-offset lowering for its CUDA pointers.
It shares the Vulkan SPIR-V / D3D12 DXIL path.

A source-oracle test exercises the two removal kernels in one ordered
GPU batch against a whole-arena CPU expected result, including
previous-frame handle deletion, old AABB invalidation, projection
invalidation and SAP index invalidation. This is a **focused dependency
fixture**: it holds the omitted intervening PhysX sort/incremental-SAP
state fixed, rather than claiming that the real host update calls these
kernels consecutively. Full execution ordering is governed by the
upstream `update` method and awaits translation of those stages.
The new test is not yet an integrated GPU broadphase or scene simulation.

### Source-derived created-handle SAP insertion

The upstream `PxgCudaBroadPhaseSap::update` calls
`markCreatedPairsKernel` when created or removed handles are present,
but **the kernel's own host wrapper only dispatches when
`mUpdateData_CreatedHandleSize != 0`**. The CUDA operation is
`broadphase.cu::markCreatedPairsLaunch` (`BP_UPDATE_CREATEDPAIRS`),
launched at its source-defined fixed **32 × 256** geometry.

The kernel retains **previous-frame capacity as the insertion base**:
`numPreviousHandles` + the created-handle list index. It writes one
`uint2` (start/end) per axis into the current `boxProjections[axis]`
and `boxHandles[0][axis]` buffers, using the new integer AABB's min/max
coordinates and new-handle tags. It sets the corresponding
`oldIntegerBounds[handle]` empty. When the total count of
previous and newly created handles is odd, it writes one
`0xffffffff` projection pair and one `0x3fffffff` handle pair
for each axis. That is the **padded pair for the next radix-sort pass**,
not a new collidable object.

`CreatedPairsArenaDesc` extends the already established 80-byte
source-logical SAP descriptor prefix with the created count, previous
count, created-array offset and new-integer-AABB offset, for a 112-byte
descriptor. HLSL retains the original writes and grid-stride dispatch
loop, while lowering CUDA pointers to arena byte offsets. Vulkan SPIR-V
and D3D12 DXIL use the same source file.

The cross-backend test checks all six new bounds and old-empty markers,
tagged current-frame handles, preservation of unrelated prior entries,
odd/even padding, no-created fast path, and 8193 created handles to
exercise the second grid-stride iteration. **It does not yet execute
the adjacent source radix sort or join a PhysX scene.** The original
host update's conditional task sequencing remains authoritative.

### Source-derived SAP radix multi-block counting

The upstream `PxgCudaBroadPhaseSap::sortProjectionAndHandlesWORKernel`
sorts `(previousNumOfBoxes + createdHandles) * 2` scalar endpoints,
rounded up to a multiple of four, using `runRadixSort`.
`runRadixSort` submits eight 4-bit passes (bits 0, 4, ..., 28),
each with an `RS_MULTIBLOCK_COUNT` producer followed by an
`RS_CALCULATERANKS_MULTIBLOCK_COUNT` rank/scatter consumer.
There are three axes and two ping-pong descriptor records per axis.
The source-defined geometry for each dispatch is **32 × 3 blocks
with 1024 threads per block**.

The counting producer corresponds to
`gpucommon/src/CUDA/radixSortImpl.cu::radixSortMultiBlockLaunchWithCount`
and `RadixSort.cuh::radixSortSingleBlock`. It assigns consecutive
`uint4` input groups to each of 32 blocks, counts 16 radix buckets,
and emits `radixBlockCounts[radix * 32 + block]`, per axis.
Partial `uint4` tails are sanitized with `0xffffffff` keys.
The portable HLSL count implementation preserves these partitions,
four-key groups, and interleaved output layout using workgroup-shared
atomic counts rather than CUDA warp-specific packed 16-bit scans.
This is a counting-stage implementation, not an independent sorting
algorithm. The rank/scatter consumer and full multi-pass sort
remain to be ported.

`RadixCountArenaDesc` lowers only the counting producer's two
device pointers (input keys and block counts) per axis to byte offsets.
Persistent projection, handle/rank, temporary, and ping-pong storage
remain the responsibility of the original broadphase owner.
The cross-backend full-arena oracle checks zero work, partial
`uint4` padding, equal keys, all-one sentinels, all three axes,
highest and intermediate radix digits, multi-block partitioning,
and preservation of unrelated words. This does not yet integrate
sorting into `PxgCudaBroadPhaseSap::update` or `Sc::Scene`.

### Source-derived SAP radix rank/scatter

The next original dispatch in each `PxgCudaBroadPhaseSap::runRadixSort`
4-bit pass is `RS_CALCULATERANKS_MULTIBLOCK_COUNT`, implemented by
`gpucommon/src/CUDA/radixSortImpl.cu::radixSortMultiCalculateRanksLaunchWithCount`
and `RadixSort.cuh::radixSortCalculateRanks`. It consumes
`radixBlockCounts[digit * 32 + block]` from the preceding count
dispatch and stably permutes each key together with its input rank/payload
to the per-axis output buffers. The source runs each dispatch at
**32 × 3 blocks of 1024 threads**.

`RadixScatterArenaStage` lowers the five source pointers per axis
(input/output keys and ranks plus radix histogram) into arena offsets,
without changing data ownership. The common HLSL implementation
retains the source's consecutive `uint4` block partitioning,
partial-tail sentinels, per-digit block prefix, stable intra-block rank,
per-iteration carry, and independent axes. It replaces warp-specific
CUDA ballot/shuffle/packed scans with deterministic workgroup-local
parallel prefix sums, so subgroup width is not fixed to 32.

The acceptance fixture executes the original **count → scatter**
dependency on both backends, with Vulkan/D3D12 producer-consumer
barriers from the compute batch. It also executes eight consecutive
4-bit passes with alternating key/rank buffers and checks stable sort
against source-derived CPU behavior, including duplicate projections,
handle/payload correspondence, partial `uint4` tails, crossing block
boundaries, multiple iterations per block, and untouched arena ranges.

This establishes an executable source-derived radix sort path but
**does not yet insert the radix chain into the original SAP host update**.
The upstream WR path also initializes ranks and remaps handles; the
WOR path sorts projections and handles directly. Their allocation,
initialization, buffer swaps, and follow-on SAP histogram/region
dependencies remain owned by PhysX.

### Source-derived SAP endpoint index reconstruction

The original `PxgCudaBroadPhaseSap::update` calls
`initializeSapBoxKernel(mNumOfBoxes, false)` after any created/removed
projection sort and before the region histogram and new-pair detection.
The kernel is `broadphase.cu::initializeSapBox1DLaunch`
(`BP_INITIALIZE_SAPBOX`), with host geometry
`ceil(numHandles * 2 / 256)` blocks of **256 threads**.
A separate call with `isNew=true` selects the
`boxNewSapBox1D[axis]` outputs; the normal update writes
`boxSapBox1D[axis]`.

For each sorted endpoint on each of three axes, the kernel decodes
`getHandle(tagged) = tagged >> 3`, uses the low bit to choose the
minimum or maximum index in `PxgSapBox1D::mMinMax[2]`, and writes its
scalar position in the newly sorted endpoint array. Both the new-handle
flag and the deleted-handle flag are preserved in the sorted handle,
but do not affect this index reconstruction. Only
`numHandles * 2` live endpoint entries are processed; odd-count
radix padding is not interpreted as a physical handle.

`SapBoxIndicesArenaStage` lowers the original sorted-handle and two
alternative output pointers per axis into arena offsets, retaining the
original output-buffer selection and dispatch geometry. No substitute
spatial partition, pairing rule, or handle indexing is introduced.

The cross-backend integration oracle executes a source-ordered creation
path in a **single ordered GPU batch**:
`markCreatedPairsLaunch` → eight pairs of radix count/scatter dispatches
→ `initializeSapBox1DLaunch`.
It verifies exact sorted projections and tagged handles against stable
CPU radix ordering; tests both SAP output-buffer modes; confirms old
bounds are marked empty; and checks zero work, padded projection pairs,
noncontiguous handle IDs, repeated endpoint keys, and dispatches across
the 256-thread SAP box boundary. This is a deliberately scoped
**created-handles-only** path. Incremental updates, removals, WR rank
initialization, histogram/region processing, and scene tasks remain
additional pieces of the original PhysX host execution pipeline.

### Existing-handle WR radix sort and handle remapping

The original host's `PxgCudaBroadPhaseSap::update` performs
`sortProjectionAndHandlesWRKernel(previousNumOfBoxes)` **before**
`markCreatedPairsKernel` or the later WOR sort. Unlike WOR, this
earlier path must preserve the previous-frame endpoint identity while
new bounds change projection order.

The WR host wrapper executes, in order:

1. `BP_INITIALIZE_RANKS` → `broadphase.cu::initializeRadixRanks`
   (64 groups × 256 threads with grid-stride iterations).
   It fills `boxProjectionRanks[axis]` with original endpoint positions,
   including padded `uint4` lanes.
2. `runRadixSort`: eight four-bit count/rank-scatter dispatch pairs,
   using the WR `PxgRadixSortDesc` that carries **ranks**, not tagged
   handles, as its payloads.
3. `BP_UDPATE_HANDLES` → `broadphase.cu::updateHandles`
   (64 groups × 256 threads with grid-stride iterations).
   This gathers `boxHandles[0][axis][i] =
   boxHandles[1][axis][boxProjectionRanks[axis][i]]` for the *live*
   previous-frame projections only.

The original `updateRadixSortDesc` and `updateDescriptor` confirm that
`boxHandles[1]` is the previous-frame handle source and is **not**
`mTempBoxPtHandlesBuf`, the radix ping-pong scratch buffer. The
rank map must survive the radix permutation and remains available
to downstream incremental SAP comparisons.

`SapRankedSortArenaStage::append_source_sequence` reproduces that
18-dispatch host schedule with the existing common radix count/scatter
stages and two dedicated HLSL translations for rank initialization
and handle remapping. Descriptor offsets lower the original three
per-axis pointer relationships without changing ownership or
introducing substitute handle identities. The zero-handle gate,
projection padding, three independent axes, and source dispatch
dimensions are preserved.

The composed host append is transactional: it stages the complete
18-dispatch sequence and publishes it to the caller's command batch
only after all delegated radix descriptors validate successfully.
Malformed descriptors on either radix parity leave existing commands
unchanged, rather than exposing partially scheduled PhysX state.

The cross-backend GPU oracle verifies the complete WR dispatch
sequence against a stable key/index reference, including equal keys,
nonmonotonic tagged handles, preserved previous-frame handle buffers,
padded endpoints excluded from handle remapping, and both odd/even
previous-frame box counts. The test supplies an already-existing
previous-frame state; it does not claim that all scene update paths or
the incremental SAP pair-generation kernels have been ported.

### Incremental and full-update endpoint histograms

`PxgCudaBroadPhaseSap::calculateEndPtHistogramKernel(isIncremental)`
schedules two original kernels in strict order:

- `BP_COMPUTE_ENDPT_HISTOGRAM` / `computeEndPtsHistogram`
- `BP_OUTPUT_ENDPT_HISTOGRAM` / `outputEndPtsHistogram`

Both dispatch with a **32 × 3 grid of 256-thread workgroups**.
The original CUDA helpers are `firstPassEndPtsHistogram` and
`secondPassEndPtsHistogram` in `gpubroadphase/src/CUDA/broadphase.cu`.
Each block owns a contiguous slice of the sorted endpoints, including
the source's multi-iteration distribution when populations are large.

The first pass computes local exclusive prefix histograms for start
projections, end projections, and **all** end projections. The start/end
histograms exclude deleted handles, but the all-end histogram counts
deleted end projections too. Each block publishes its three totals.
The second pass adds exclusive prefixes across the original 32 blocks
and builds the ordered start/end handle lists without deleted entries.
The tagged handle is decoded with PhysX's original `getHandle`
(`tagged >> 3`). This is not a collision-pair generator.

In incremental mode, PhysX scans only `numPreviousHandles` in the
current sorted buffers; when any removals occurred, it additionally
scans the previous `boxHandles[1]` buffers into a separate set of
histograms and ordered handle lists. In non-incremental mode, it scans
`numPreviousHandles + numCreatedHandles - numRemovedHandles` and
does not touch the previous-frame results. Only the current view
publishes the all-end histogram.

`SapEndpointHistogramArenaStage` lowers the source pointers to
arena offsets, records **both** kernel dispatches together, and uses
the existing backend submission barriers to publish the first pass's
block counts to the second pass. The shared HLSL scans use workgroup
memory rather than relying on CUDA's 32-lane ballot and packed
per-warp arithmetic. The original partitioning, output indices,
buffer separation, and persistent-frame responsibilities remain.

A dual-backend source-oracle fixture checks each per-projection
exclusive count, the ordered handles, all 32 block totals, untouched
arena regions, zero work, deleted and new flags, incremental/normal
modes, previous-frame retention, and more than one source iteration
per block. The downstream incremental comparison-count kernels still
need the original histogram and rank inputs connected in the proper
host update order. This stage alone does not report found/lost pairs.

### Incremental SAP comparison histogram and global work offsets

`PxgCudaBroadPhaseSap::performIncrementalSapKernel` begins with
`BP_COMPUTE_INCREMENTAL_CMP_COUNTS1` followed by
`BP_COMPUTE_INCREMENTAL_CMP_COUNTS2`, before it dispatches
`BP_INCREMENTAL_SAP`. Both counting kernels use the original
**32 groups × 256 threads** launch, simultaneously processing all
three projection axes in each workgroup. Their CUDA implementations
are `computeIncrementalComparisonHistograms_Stage1` and
`computeIncrementalComparisonHistograms_Stage2` in
`gpubroadphase/src/CUDA/broadphase.cu`.

The first stage consumes `numPreviousHandles * 2` sorted endpoints,
their stable `boxProjectionRanks[axis]`, tagged `boxHandles[0][axis]`,
and the **current and previous** per-axis start/end endpoint histograms.
For each endpoint, comparison direction is
`current_index <= original_rank`. Downward movement uses the previous
histogram and upward movement uses the current one. Start endpoints
count crossed **end** endpoints; end endpoints count crossed
**start** endpoints. This deliberately selects the opposite
endpoint type, as in the upstream `sStartEndAccum[isStartHandle]` table. Deleted
endpoints contribute no comparisons. The exact candidate count is
a difference of exclusive histogram entries over the original
index-to-rank interval, with the previous-frame range clamped to the
last live projection. Each of the 32 source blocks writes its exclusive
local comparison indices and a per-axis block total.

The second stage scans the block totals, adds the exclusive
cross-block prefix to each endpoint comparison index, and writes
`totalIncrementalComparisons[axis]`, which the subsequent
`performIncrementalSAP` kernel uses to distribute its comparison
work. The original `incrementalComparisons` arrays remain owner-visible
prefix maps; this port does **not** substitute a collision query or
generate found/lost pair reports.

`SapIncrementalComparisonsArenaStage` records both source dispatches
transactionally, with per-axis arena descriptors corresponding to
the original rank/handle and current/previous histogram pointers.
The common HLSL uses workgroup prefix scans instead of fixed
CUDA warp primitives, preserving the 32-block partition and
across-block work offsets on both Vulkan and D3D12.

The integration oracle executes
**compute endpoint histogram → output endpoint histogram →
comparison count stage 1 → comparison count stage 2** in one ordered
GPU batch. It creates physically reordered prior/current sorted
endpoint lists with stable position ranks, then compares every
histogram entry, compacted handle, comparison prefix, block total,
global count, and untouched arena word against an independent
CPU reference. It covers empty and small workloads, deletions,
all three axes, more than 256 projections, and multiple iterations
within each source block. With no removals, the previous-frame
histogram is seeded as persistent prior-frame state, as required by
the original host dispatch.

This is source-faithful comparison **work enumeration**, not the
subsequent `BP_INCREMENTAL_SAP` comparison/overlap/report operation.
The broader persistent scene task graph and result publication
remain the province of the original PhysX host pipeline.

### Incremental SAP found/lost pair reports

`PxgCudaBroadPhaseSap::performIncrementalSapKernel` submits
`BP_COMPUTE_INCREMENTAL_CMP_COUNTS1`, then
`BP_COMPUTE_INCREMENTAL_CMP_COUNTS2`, then `BP_INCREMENTAL_SAP`.
The last dispatch runs **256 × 3 × 256 threads** (256 workgroups per
projection axis, 256 threads per workgroup). The source kernel is
`gpubroadphase/src/CUDA/broadphase.cu::performIncrementalSAP`.

This kernel consumes the original source inputs:
`totalIncrementalComparisons[axis]`, its inclusive-endpoint
`incrementalComparisons[axis]` *exclusive prefix*,
`boxProjectionRanks[axis]`, `boxHandles[0][axis]`, both
generations of `startPtHistogram` and `endPtHistogram` and their
ordered `startPointHandles`/`endPointHandles` arrays. Work IDs map
back to their moving endpoint via the last prefix not exceeding
the work ID. For down-sweeps the partner is selected from the
previous-frame endpoint list; up-sweeps use the current list.

The original `willFindPair = (!isStartHandle) ^ downPass` selects
whether the **new** or **old** AABBs define the present overlap. A pair
is reported only if it overlaps on all three axes in that selected
state and was separated on the swept axis in the other state.
`isSeparatedBeforeAxis` suppresses duplicate reports for axes
later than the first separating axis. Groups must differ.
When environment IDs are supplied, they must match or one must be
`PX_INVALID_U32`; absent environment IDs are permitted exactly as
in the `USE_ENV_IDS` source branch.

`PxgBroadPhasePair` stores its two handle IDs in ascending order.
Found and lost streams each atomically reserve separate slots in
their shared counters. The counters retain the **complete count**
even when `max_found_lost_pairs` clips the report writes.
The original source does not guarantee deterministic ordering
across workgroups; neither does the Vulkan/D3D12 translation.
`SapIncrementalPairsArenaStage` keeps this contract, lowering the
upstream pointer relationships to per-axis and global offsets
within host-owned buffers and replacing CUDA warp scans with
wave-width-independent workgroup scans and one pair of atomic
group reservations per populated batch.

The integrated GPU oracle executes the original two endpoint
histogram passes, both incremental comparison-count passes,
and the incremental SAP report pass in one ordered batch. It
builds old/current sorted endpoints and stable old-position ranks
from deterministic integer AABBs, seeds the previous-frame
histogram as persistent state, and independently computes
**actual 3D overlap changes** for all eligible pairs.
It validates found/lost counters and pair multisets, group and
environment filtering, canonical pair order, clipping behavior,
all 32 local histogram blocks, comparison prefixes and totals,
and untouched arena allocations. Atomic output order is normalized
only in the test; the implementation makes no ordering promise.

This is the source-defined incremental pair **producer**, not
the completed PhysX scene task graph. Report accumulation,
actor-vs-aggregate filtering, mapped publication, persistent
bound swapping, created/removed paths, and full host scheduling
remain separate architecture work.

### Clearing newly-created SAP handle tags after pair generation

In the original `PxgCudaBroadPhaseSap::update`, the source sequence
finishes region pair generation, swaps the **old/new integer AABB
buffers** for the next frame, then invokes `clearNewFlagKernel()`,
followed by `runCopyResultsKernel`. The helper conditionally
launches **`BP_CLEAR_NEWFLAG`** only when the created-handle list
is nonempty. Its CUDA implementation is
`gpubroadphase/src/CUDA/broadphase.cu::clearNewFlagLaunch`, with
the original fixed launch of **32 blocks × 256 threads** and a
grid-stride loop for larger created-handle populations.

Each thread reads an original created object handle, looks up the
*current* `boxSapBox1D[axis][handle].mMinMax[0..1]` positions
for all three axes, and **rewrites** `boxHandles[0][axis]` at
those exact positions with
`createHandle(handle, true/false, false)`. This clears the
`isNew` bit without relying on adjacent endpoints being stored
together or rewriting pre-existing/deleted handles. The source
does not touch its handle positions, projection keys, bounds,
histograms, rank buffers, or previous-frame handles.

`SapClearNewFlagsArenaStage` translates this launch into one
common Vulkan SPIR-V / D3D12 DXIL shader operating through
host-owned handle/SAP/list offsets, and preserves the
zero-created no-dispatch gate. Its arena description is *not*
a replacement PhysX scene or private SAP buffer owner.

The cross-backend integration fixture first executes the
already-ported creation → radix sort → SAP index operations
and validates the **original new-bit tags**. After a GPU submission
and a full-arena snapshot, a separate submission executes
`BP_CLEAR_NEWFLAG`. An independent CPU expectation edits only
the two tagged endpoints for each created object and checks
every other arena word unchanged. An additional isolated
**8193-created-handle** fixture verifies the original 8192-thread
grid-stride wrap without re-running a large radix sort. Zero
work, invalid descriptor ranges, padding and all three axes
are also checked.

**Source-order caveat:** the tests intentionally omit the
upstream region pair generation, old/new AABB buffer swap,
and report-copy stages between SAP initialization and
`BP_CLEAR_NEWFLAG`. Those omitted stages do not change the
tag clear's own inputs, but must be restored in the complete
host orchestration. This is a frame-state maintenance stage,
not yet a multi-frame PhysX scene acceptance test.

### Source report copy and overflow-flag publication

In the original `PxgCudaBroadPhaseSap::update`, report
accumulation and copy follow the created-handle flag clearing:
`runCopyResultsKernel` launches
`BP_ACCUMULATE_REPORT_STAGE_1`,
`BP_ACCUMULATE_REPORT_STAGE_2`, then
`BP_COPY_REPORTS`. The final copy kernel is
`gpubroadphase/src/CUDA/broadphase.cu::copyReports`,
at a fixed **64 × 256** launch, including empty frames.

The preceding accumulation pair separates the mixed found/lost
reports into **actor** and **aggregate** streams. This distinction
is essential: `copyReports` publishes only the *actor* streams,
because aggregate pairs have their own processing path.
The device-shared `sharedFoundPairIndex`/
`sharedLostPairIndex` counters record raw pair counts, potentially
exceeding the physical `max_found_lost_pairs` report capacity.
`sharedFoundAggPairIndex`/`sharedLostAggPairIndex` count the
aggregate pairs actually retained within that capacity.
Accordingly, the source computes
`actorCount = min(rawCount, capacity) - aggregateCount` per
found/lost stream. It copies exactly `2 * actorCount` unsigned
32-bit words from `foundActorPairReport` or
`lostActorPairReport` to the mapped output buffers, with a
grid-stride loop of 16384 threads. It writes a single shared
overflow flag if either raw count exceeds capacity.

`BroadphaseCopyReportsArenaStage` preserves the source launch,
pointer relationships, both raw counts, both retained aggregate
counts, the shared overflow semantic and exact wordwise actor-pair
copy. Original CUDA mapped host pointers are represented as
**caller-owned publication staging offsets**. This is an explicit
resource-address lowering, **not** an assertion that device writes
have already become visible to the PhysX caller.
Portable GPU completion, device-to-host transfer, pair resizing,
`fetchBroadPhaseResults` frame swap and original
`PxgAABBManager` task continuations remain at the original
host publication boundary.

The test seeds stage-two actor/aggregate inputs with canonical
PhysX pair words and checks the complete arena against an
independent CPU word-copy oracle. Cases cover 0 and 1 pair,
differing found/lost counts, aggregate subtraction, overflow
from either or both counters, zero-capacity outputs, canary
preservation and 16,385 actor pairs (crossing the kernel's
grid-stride boundary). Both Vulkan SPIR-V and D3D12 DXIL/WARP
run the same fixture.

**Dependency limitation:** the standalone fixture does **not**
pretend that source `BP_ACCUMULATE_REPORT_STAGE_1` and
`BP_ACCUMULATE_REPORT_STAGE_2` have been executed. They are
the next required GPU data producers; this bounded port tests
their consumer before composing the three-stage host sequence.

### Source actor/aggregate report partition and compaction

`PxgCudaBroadPhaseSap::runCopyResultsKernel` schedules three
original GPU kernels: `BP_ACCUMULATE_REPORT_STAGE_1`,
`BP_ACCUMULATE_REPORT_STAGE_2`, then `BP_COPY_REPORTS`.
The first two are `accumulateReportsStage_1` and
`accumulateReportsStage_2` in
`gpubroadphase/src/CUDA/broadphase.cu`, each with
**32 blocks × 4 stream classes × 256 threads**.
The four original Y classes, in order, are aggregate-found,
actor-found, aggregate-lost, and actor-lost. Their X blocks own
contiguous portions of the incoming raw report arrays, with
the original multi-iteration partitioning when the report
count exceeds 8192.

A report is an actor–actor report only when **both** volume
entries are single actors. The source test is
`Bp::VolumeData::isSingleActor()`, equivalent to
`mAggregate == PX_INVALID_U32` in
`physx/source/lowlevelaabb/include/BpVolumeData.h`.
Aggregated actors and aggregate volumes are both nonsingle.
When `aabbMngr_volumeData` is null, the original code
treats **all** reports as actor reports. The portable
descriptor uses a caller-owned array of exactly those
`mAggregate` field values, not a custom actor/aggregate
membership interpretation or a copied pointer-rich
`Bp::VolumeData` layout.

Stage one records 32 block totals per report class, limited
to `min(sharedPairIndex, max_found_lost_pairs)` retained
input records, while preserving the upstream partition based
on the full count. Stage two computes exclusive block
prefixes, classifies records within each block, and writes
stable compacted found/lost actor and aggregate report
buffers. Actor output capacity is the raw report capacity;
aggregate output uses the independent
`max_found_lost_agg_pairs` capacity. The found/lost
aggregate counters reflect **all retained aggregate records**,
even if the smaller aggregate-output allocation clips writes.
The existing `BP_COPY_REPORTS` stage then copies actor
reports to publication staging and sets overflow from the
original raw counters. Original scene publication, CPU
task continuations and buffer swaps are not part of this
standalone source kernel fixture.

The shared HLSL uses 256-thread workgroup reductions and
scans independent of AMD/NVIDIA native wave size. It retains
source X/Y work partitioning and output ordering, and executes
the **three original report kernels consecutively** in one
Vulkan or D3D12 batch. The CPU oracle independently partitions
original found/lost reports and compares all four compacted
outputs, all 128 block histograms, both aggregate counts,
actor publication maps, clipped canaries, and overflow flags.
Cases include zero, single, mixed actor/aggregate and
null-`VolumeData` workloads; differing input/output caps;
raw-count overflows and 8193/16385 pair boundary tests.
The test does not claim a full live-scene broadphase.

### GPU completion and existing PhysX pair-task publication

The upstream source makes the ordering precise:

- `PxgCudaBroadPhaseSap::gpuDMABack` queues a descriptor readback,
  launches `BP_SIGNAL_COMPLETE` against mapped host memory, flushes the
  CUDA stream, and waits (briefly spins, then synchronizes if necessary).
  Only after this boundary does it inspect report counts and overflow.
- `PxgCudaBroadPhaseSap::fetchBroadPhaseResults` calls that fetch, then
  swaps the persistent six-way projection/handle/histogram buffer sets.
- `PxgAABBManager::postBroadPhase` calls `fetchBroadPhaseResults`,
  resizes pair lists, and **then** sets each found/lost `PxLightCpuTask`
  continuation and removes its initial task reference. The existing PhysX
  task manager dispatches those tasks and releases their continuations.

The source-faithful first CPU/GPU boundary is a **blocking fetch on the
actual PhysX post-broadphase path**, not a new GPU callback scheduler.
`src/physx_gpu/broadphase_completion.{h,cpp}` now provides a minimal
`BroadPhaseCompletionGate`: it associates one compute submission with
one frame, waits before caller-provided fetch/resize, installs the
caller's original task continuations **after** successful publication,
and only then invokes `removeReference` on the two original
`PxBaseTask` objects. It covers the
no-GPU-update path, prevents double release, and leaves both tasks blocked
if the GPU wait or the result-publication operation fails; a publication
failure can be retried after the owning PhysX code establishes its abort/
empty-result policy. A **continuation-installation** failure is terminal,
since retrying could install extra task references; the scene's original
abort/teardown path must handle it. A completion readiness query **never**
publishes or releases tasks.

The bridge is **tested in isolation with real PhysX task interfaces and
Vulkan/D3D12 submissions**, but is **not yet wired into
`PxgAABBManager` or `Sc::Scene`**. The caller remains responsible for
providing the original continuation-setup operation, reading the descriptor/pair streams,
checking overflow/abort flags, updating report storage, and swapping state.
It does not make the existing SDK GPU-broadphase-capable or add a worker
thread. Next integration attaches this boundary to actual PhysX broadphase
factory/scene objects as the relevant kernel family becomes runnable.

## Vulkan and D3D12: concurrent portability gates

**Both backends implement the same PhysX GPU source-derived operation and
scene orchestration.** Vulkan is the first executable runtime, not a source
of truth for the simulation architecture. D3D12 implementation begins with the
first translated GPU stage, rather than waiting for the broadphase/solver port
to finish.

For AABB encoding, `src/physx_gpu/kernels/physx_translate_aabbs.hlsl` supplies
one translation compiled by DXC to Vulkan SPIR-V and D3D12 DXIL. Linux CI
executes the upstream-derived encoding reference on Vulkan; Windows CI
executes it on **native D3D12 WARP** and additionally checks
dispatch/copy/dispatch ordering. This is native D3D12 *software-adapter*
execution, **not** demonstrated AMD hardware performance or integrated
PhysX GPU broadphase. Only descriptor and parameter bindings are
backend-conditional. Do not multiply per-API physics algorithms
by adding further independent shader translations.

GPU device-pointer conversion remains a separate, unsolved requirement;
see [memory/addressing experiments](GPU_MEMORY_ADDRESSING_RFC.md).

The portable execution contract needs to specify:

| Responsibility | Vulkan | D3D12 | Shared invariant |
| --- | --- | --- | --- |
| Compiled code | SPIR-V | DXIL | One source-derived algorithm, same parameter/data layout |
| Storage resources | Descriptor sets and storage buffers | SRV/UAV descriptors and heaps | Logical read/write bindings and range/lifetime |
| Small dispatch parameters | Push constants | Root constants/CBV | One declared value schema; backend-specific lowering |
| Dispatch and workgroup | `vkCmdDispatch`, SPIR-V local size | `Dispatch`, HLSL `numthreads` | Logical item indexing, supported group size, bounds checks |
| Intra-GPU dependencies | Compute barriers, queue synchronization | UAV/resource barriers, fences | Explicit producer/consumer and read/write hazards |
| CPU/GPU handoffs | Vulkan submissions/fences | D3D12 queues/fences | PhysX scene task completion and publication ordering |
| Allocation/state | Vulkan buffer/heap/device addressing | D3D12 resources/heaps and addressing | PhysX buffers, aliasing and ownership survive each backend |

D3D12 UAV accesses between dependent dispatches require explicit barriers;
submitting the commands consecutively does not necessarily establish the
correct ordering. A portable batch cannot rely on Vulkan implementation
details for this or expose Vulkan descriptor-set semantics as the GPU scene
interface.

The next cross-backend acceptance gates are:

1. Build this AABB kernel from one source to **both SPIR-V and DXIL**,
   preserving a verified binary parameter/binding contract (**in CI**).
2. Run the same input/reference test on Vulkan and a native D3D12 compute
   backend, including producer/consumer dispatch barriers and result download.
3. Expand the real PhysX broadphase stages with the same dual-backend source,
   memory and task dependencies. Do not treat D3D12 as a final translation
   after all Vulkan kernels and orchestration have already been designed.

## Implementation sequence and acceptance gates

1. **Integrate at the PhysX scene boundary.** Identify the narrowest replacement
   seam in `PxPhysXGpu`, GPU contexts and task completions, preserving
   `Sc::Scene` scheduling. Produce an exact call/dependency map for selected
   GPU rigid-body configuration(s), including CPU/GPU BP combinations.
2. **Trace memory/descriptor ownership end to end.** Follow host-pinned,
   mapped and device allocations; buffer aliasing; contact/patch streams;
   pointer-rich descriptors; resize/lifetime rules; and event synchronization.
   Determine what needs shader layout conversion versus host adaptation.
3. **Trace a real PhysX-derived slice.** Choose a bounded existing subpath
   (e.g., one broadphase stage or a specific narrowphase primitive) with
   identified upstream kernels, launch wrappers, inputs, outputs, dependency
   edges and test fixtures. No new standalone world or invented contact solver.
4. **Build dual-backend execution contracts.** Validate Vulkan/SPIR-V and
   D3D12/DXIL resource and scheduling semantics before treating Vulkan-specific
   bindings as the canonical design.
5. **Connect, verify, and extend.** Execute the PhysX-origin slice at its real
   scene pipeline boundary. Compare behavior with CPU PhysX and original CUDA
   PhysX where available; test ordering, contacts, constraints, sleeping and
   result publication as relevant. Exact floating-point identity is not a gate.

This matrix is a starting inventory. Every implementation PR must refine its
row with concrete source entry points, input/output layouts, synchronization
and behavioral tests. A passing compute smoke test is not PhysX GPU conformance.


### Source new-bounds overlap-work count producer

The next source-owned created-handle branch begins with
`broadphase.cu::writeOutOverlapChecksForInsertedBoundsRegionsHistogram`
(`BP_WRITEOUT_OVERLAPCHECKS_HISTOGRAM_NEWBOUNDS`, 32 × 256).
`SapOverlapWorkArenaStage` lowers this independent per-item producer to the
shared HLSL arena ABI on Vulkan and D3D12. It consumes the existing exclusive
`regionAccum` prefix, current X-sorted handles, X SAP end positions,
region ranges, and prior active/start-region prefix tables. It produces
`overlapChecksRegion` (64-bit counts) and
`overlapChecksHandleRegiones` (handle and region-array offset).

The per-region difference between two histogram entries fits in 32 bits
because each entry counts at most the current endpoint population. The output
count remains 64 bits to match PhysX's subsequent 64-bit prefix and total.
The 64-bit global accumulation belongs to the **next**
`BP_COMPUTE_OVERLAPCHECKS_HISTOGRAM` and
`BP_OUTPUT_OVERLAPCHECKS_HISTOGRAM` stages; it has not been replaced with
32-bit summation. No subgroup or optional device capability is needed in
this producer; capability checks stay in stage construction rather than
dispatch.

The native cross-backend fixture exercises created and quiet frames, source
dispatch geometry, descriptor rejection and an independent two-handle
count/reference comparison including multi-X-region enumeration with
region-distinct histogram values, preserving untouched buffer canaries. This is
a bounded producer test, not the complete overlap-prefix/found-pair pipeline.
The 64-bit prefix kernels and found-pair writer are now translated in
separate stages. `SapNewBoundsSequenceArenaStage` additionally submits all
four in source order on a shared caller-owned arena and is covered by an
integrated GPU oracle. The **live** `PxgCudaBroadPhaseSap` scene task/fetch
and memory-ownership port remain outside this acceptance slice.


### Native 64-bit overlap-comparison prefix

The source sequence after `BP_WRITEOUT_OVERLAPCHECKS_HISTOGRAM_NEWBOUNDS`
consists of `broadphase.cu::computeOverlapChecksForRegionsHistogram`
(`BP_COMPUTE_OVERLAPCHECKS_HISTOGRAM`) and
`outputOverlapChecksForRegionHistogram`
(`BP_OUTPUT_OVERLAPCHECKS_HISTOGRAM`). Both retain **32 × 256** work
geometry, source-ordered dispatch, contiguous 256-element iterations per
block, and the created-handles host gate. The producer's GPU-computed
`regionAccumTotal` drives the count without a CPU readback.

`SapOverlapPrefixArenaStage` takes only caller-owned GPU byte offsets for
the 64-bit per-item overlap counts, 32 64-bit block totals, 64-bit grand
total and 32-bit work-count field. The first shader computes native
`WavePrefixSum(uint64_t)` and `WaveActiveSum(uint64_t)` per physical wave,
followed by a short wave-total prefix and a 64-bit per-block running carry.
The second shader scans exactly 32 block totals using the participating
native waves and adds each preceding-block sum. The implementation uses
three group barriers per 256-item first-pass iteration, requires no
64-bit global atomics, and does **not** lower the original warp scan to
a 256-thread shared Hillis–Steele ladder. Wave-to-logical work assignment
uses the same once-per-group wave reservation already used by other SAP
native-wave shaders, not an assumption that SV_GroupIndex maps onto
hardware lanes.

The context queries and enables Vulkan `shaderInt64` and (on Vulkan 1.2+)
`shaderSubgroupExtendedTypes` when supported. D3D12 uses its existing
`D3D12_FEATURE_D3D12_OPTIONS1` query (`Int64ShaderOps` and `WaveOps`).
The source stage requires these capabilities **when creating the pipelines**;
no capability queries or shader selection occur during dispatch. The
compute context itself stays usable for existing non-64-bit operations on
unsupported devices; there is intentionally no slow 64-bit overlap-scan
variant masquerading as production support. This is an early per-stage
admission contract, not yet a full GPU scene/device shipping gate (#47).

GPU acceptance exercises source 32-block geometry, quiet frames,
1/31/32/256/257/8192/8193/16385-item boundaries, wide synthetic inputs
crossing 32-bit carries, all 32 block counts, 64-bit total, full arena
canaries and invalid host arguments. When the software adapter does not
support the required shader features, the test **reports skipped**, not
passed. Hardware speedup and actual 680M/Arc device admission remain
unmeasured. `BP_GENERATE_FOUNDPAIR_NEWBOUNDS` and original live PhysX
GPU broadphase orchestration still remain ahead.


### Source new-bounds found-pair kernel (standalone acceptance stage)

The original `PxgCudaBroadPhaseSap::generatePairsForNewBounds` issues four
created-handles-gated commands in order: overlap-work producer,
64-bit compute overlap histogram, 64-bit output overlap histogram,
and `BP_GENERATE_FOUNDPAIR_NEWBOUNDS`. The new
`SapNewBoundsPairsArenaStage` lowers this last CUDA
`generateFoundPairsForNewBoundsRegion` command with the **original 64 groups
× 256 threads** (unlike the preceding 32 × 256 overlap histogram passes).
The total candidate domain and work-item index remain 64-bit, and each
block owns ceil(total / 16384) successive 256-item segments. Work-item
lookup uses the source's upper-bound-minus-one search of the producer's
exclusive overlap prefixes, the source `PxgHandleRegion` mapping, current
X SAP positions/new tags, the per-region active/start histograms and ordered
partner lists. Duplicate prevention uses the maximum of both boxes' minimum
region coordinates, not a geometrically different broadphase algorithm.

Filtering preserves group exclusion, optional environment equality / invalid
ID wildcard behavior, and six-coordinate integer AABB intersection. Each
surviving pair enters a native 32-bit `WavePrefixSum` batch, a short
cross-wave prefix, and the source's **single 32-bit global report-counter
reservation per nonempty 256-item batch**. Pairs are written with ascending
handle order, subject to `max_found_lost_pairs`; the shared counter is
not clamped when output capacity is exceeded. No 64-bit global atomics or
whole-workgroup Hillis–Steele scans. GPU/device capability admission occurs
at stage construction; no device queries or frame-time shader selection.

Acceptance includes independently enumerated CPU candidate pairs with
source-style group/environment filters and integer bounds, new and old
histogram selection, 64-region canonical duplicate suppression, report
capacity clipping and persistent prior report counts, and a case spanning
more than 16,384 pair candidates. Unrelated GPU arena memory is canary-
checked; report order between blocks is intentionally unspecified. A
software adapter without native shader int64 and subgroup arithmetic is
explicitly reported as skipped, not verified. The stage intentionally does
not claim real end-to-end PxgAABBManager integration or native GPU speed
measurements; the owning PhysX host must chain the four original commands
and publish their results through its existing task graph.


### Integrated new-bounds GPU dispatch and PhysX completion handoff

The GPU adapter `src/physx_gpu/sap_newbounds_sequence_arena.{h,cpp}`
appends the four source `generatePairsForNewBounds` kernels to the caller's
existing GPU batch in original order (**32 × 256 / 32 × 256 / 32 × 256 /
64 × 256**). It reuses the three translated stage pipelines, a single
shared caller-owned GPU arena, an authoritative set of descriptor aliases
and the created-handle-only gate. No new scene scheduler, buffer-owner,
frame-time device capability query, or intermediate CPU readback.

For `BP_GENERATE_FOUNDPAIR_NEWBOUNDS`, the host passes the actual
`current_boxes` value through a per-dispatch push constant. The GPU arena's
long-lived pair descriptor does not determine the dense region stride after
box additions/removals. The integration test explicitly holds the
descriptor's count stale while exercising a valid report in region 1;
both the overlap producer and the pair reporter must use the same frame
population. This introduces no hot-path hardware queries or readbacks.

The integrated Vulkan and D3D12 tests exercise the existing
`decide_sap_update` host decision, actual producer-to-prefix-to-found-pair
intermediate memory dependencies, per-start projection regions,
64-bit prefixes/block totals and final found reports, with a CPU oracle
checking the complete arena. They also pass the ordered GPU submission
through `BroadPhaseCompletionGate`, verifying that result publication and
validation occur before the existing found/lost task references are
released. This tests a real **four-kernel GPU handoff** rather than four
isolated shaders. It does **not** yet claim a running PhysX GPU scene:
original `PxgCudaBroadPhaseSap` state allocations, upstream producers,
GPU DMA, swaps and full `PxgAABBManager` task integration still need the
larger host orchestration migration, as traced in
[GPU_BROADPHASE_TRACE.md](GPU_BROADPHASE_TRACE.md).
