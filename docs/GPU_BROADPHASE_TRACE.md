<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GPU broadphase source trace and migration ledger

Scope: **actual PhysX 5.11 GPU broadphase and its scene contract**.
Source revision: `6dc53585d7db07346f53d0598a5fd3bfdce47d8d`.
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
   report ordering against the pinned CUDA implementation (if available)
   and CPU PhysX general behavior.
5. **Aggregate regression:** empty aggregate excluded; aggregate membership
   changes, actor–aggregate and aggregate–aggregate pairs, duplicate handling.
6. **Capacity and failure:** saturate found/lost pairs and verify warning,
   counters and omission policy; forced update; aborted submission or device
   loss cannot strand PhysX task references or publish half-complete frames.
7. **Scene-facing fixture:** `PxSceneDesc` configured for GPU broadphase and
   CPU dynamics; actual `simulate` / `fetchResults`; pair notifications
   feeding the original narrowphase. **No second simulation loop.**

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
  including `gpuDMABack` and the resulting PhysX continuation release.
- Validate abort-mode and result-capacity handling before generalizing
  allocator or callback semantics.
- Reconcile original CUDA support flags and CPU fallback behavior with
  proposed GPU broadphase factory and context creation; avoid blind replacement
  of `PxCudaContextManager*` parameter types.

For each question, record the original source function/field, a precise
behavioral claim, evidence, new portable representation, and the acceptance
test before declaring it resolved.
