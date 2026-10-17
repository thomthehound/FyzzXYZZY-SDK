<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native engine ownership and migration plan

## Objective and boundary

FyzzXYZZY replaces PhysX as an independent CPU/GPU physics SDK. PhysX 5.11 is the algorithmic, dataflow, lifetime and behavior reference; it is **not** the portable-GPU production host. FyzzXYZZY must own one authoritative scene, its task graph, simulation state, allocation and frame publication. Its platform backends lower work to CPU, Vulkan and D3D12. Godot is a consumer, not an owner of physics time.

The original PhysX scheduler and GPU host pipeline **must be ported**, not ignored. Source task ordering, continuations, memory hand-offs, collision filtering and frame state remain source-grounded. A native owner may use corresponding source-derived classes and algorithms; it must not allocate a parallel PhysX scene or build an unrelated physics demonstration.

## Existing implementation / disposition

| Area | Current state | Destination |
| --- | --- | --- |
| `World` public API | Core facade; default reference PhysX CPU | Provider-neutral native scene facade, reference adapter removable |
| PhysX CPU scene | External PhysX source built for CPU and current Godot acceptance | Optional reference/test backend (build flag `FYZZXYZZY_ENABLE_PHYSX_SOURCE`); native CPU solver/narrowphase imported or ported into engine |
| Portable Vulkan/D3D12 compute | Native buffer/dispatch/submission backends, stage fixtures | Device-scoped runtime serving native scene jobs |
| PhysX GPU heap and CUDA-style address lowering | Source-derived internal code in SDK | Scene/device-owned allocations and deferred retirement, with no duplicate GPU buffer owner |
| `src/physx_port/gpubroadphase` | Nine source host-stage implementations, many borrowed member views | Consolidated into native SAP/AABB owner method boundaries |
| `src/physx_gpu` | Translated kernels and independent stage tests | Source-ordered compute operations called by SAP/AABB, retained conformance tests |
| PhysX fork portable scene factory | Reverted; prior test contract PR closed | No fork-specific production API, provider or CMake dependency |

## Gate N0 — rollback and contract alignment

- Close the source-factory SDK PR; revert the modified PhysX fork scene.
- Keep CPU reference simulation, ported GPU kernels, memory allocations and CI fixtures.
- Compile the API/world facade without any PhysX headers; place the PhysX-backed CPU implementation in a named reference-only library/source file.
- Use an explicit **execution mode**, not a `PxBroadPhaseType` selector masquerading as backend selection. Native GPU must reject until implemented. Existing CPU behavior and Godot tests must remain green.
- Apply this ownership rule to all documentation and architecture reviews.

**Pass:** fork source restored; isolated reference backend and explicit native selection compiled on Linux/Windows; no production factory ABI depends on the modified PhysX fork. Also require a reference-free CMake build of `FyzzXYZZY::Core` and native Vulkan/D3D12 compute without fetching PhysX. This is a packaging/ownership proof, **not** proof of a native CPU physics implementation.

## Gate N1 — authoritative native scene and source task graph

Implement the source-derived `NpScene`/`Sc::Scene` equivalent *within this repository*, with one native scene lifetime, actor/shape handles, bounds, contact streams and observable simulation state. Migrate or import the required CPU collision, narrowphase, solver and integration components from the available PhysX sources under SDK control. Keep the reference adapter as an independent oracle while native CPU execution is built.

Represent task dependencies explicitly from `ScPipeline.cpp`: first/second AABB pass, bounds update, broadphase, reports, narrowphase, island/contact setup, solver, post-solver and fetch. Dispatch CPU tasks and GPU jobs through one native scheduler. GPU submissions preserve queue dependencies; the task graph, not a shader helper or reference `PxScene`, owns completion and frame progression.

**Pass:** ordinary SDK `World` with native CPU execution can create/move/remove shapes, simulate multiple frames, produce contacts and publish results with comparable PhysX-reference behavior; no PhysX Foundation/PxScene instance is needed. Exercise reinitialization, multiple scenes, resource retirement and Godot consumer behavior.

## N1 implementation status — native scene owner established, solver pending

The SDK-owned `src/native/scene.{h,cpp}` now owns stable actor IDs, static
boxes, dynamic boxes, planes, mutable poses, conservative PhysX-style encoded
integer bounds, source-ordered SAP candidate overlaps, and persistent
found/lost pair history. Pair state is per scene and survives quiet, forced,
create, remove and relocate updates. There is one owner and one pair-report
publication method; no added GPU context, independent dynamics scheduler,
per-kernel controller or replicated actor collection.

The ordinary `World(ExecutionMode::NativeCpu)` now constructs this native
owner and can author, relocate, remove and inspect native bodies even when external
PhysX source is disabled. The same actor lifecycle operations are available
through the separately compiled PhysX CPU reference adapter. Relocating a
reference actor preserves its orientation (including the rotated local plane
of `PxCreatePlane`); native infinite planes translate without changing
their fixed +Y normal. **Its `step()` still throws** because CPU narrowphase, dynamics,
island state, contacts and solver execution are not yet ported. Direct
`advance_broadphase` is an internal integration and conformance seam, **not**
a substitute public physics simulation path. Infinite-plane overlaps are
candidates, not physical contact claims; native narrowphase must reject
noncontacts. This owner will receive the source-derived ScPipeline task
continuations and CPU collision/dynamics port; N1 remains OPEN until it can
actually step and solve multi-frame physics.

Source correspondences: `BpBroadPhaseIntegerAABB.h` integer float encoding,
`BpBroadPhaseSap.cpp` sorted-axis candidate and incremental pair semantics,
`ScPipeline.cpp` CPU ordering first BP pass -> BP update ->
post-broadphase publication. This first path does **not** claim parity with
PhysX's optimized multi-axis incremental SAP, aggregate processing, MBP/PABP,
or CPU contact solver. The shape contact offset is provisionally 0.02m;
move it to the imported per-shape material/contact-distance state during N1.

## Gate N2 — native source-derived SAP and AABB owner

Consolidate `PxgCudaBroadPhaseSap` and `PxgAABBManager` source behavior into two cohesive native owner implementations. Own one persistent buffer/history representation; remove borrowed frame-member scaffolding when its tests have migrated. Drive the original first-pass/second-pass/update/fetch order, including state-changed/forced/idle logic, AABB and aggregate preparation, three-axis projection/region/rank/sort work, pair/report kernels, capacity and abort checks, old/new bounds swap and the seven fetch-history swaps.

Use one device/stream execution service, with reusable compiled pipelines and source-ordered batched submissions; synchronizing at required publication boundaries only. Preserve correct buffers across growth, deferred frees, multi-frame history and failed fetch. Do not create per-kernel controller objects, a second frame state or a separate test-driven scheduler.

**Pass:** native scene selects real GPU SAP on Vulkan and D3D12, found/lost pairs flow into the native CPU narrowphase/solver across multiple frames, and explicit diagnostics prove that neither CPU ABP nor a reference PhysX scene ran instead.

## Gate N3 — correctness, parity and runtime-cost acceptance

A shared standalone/Godot acceptance scene creates, updates and removes actors and aggregates, exercises quiet/forced frames, growth and overlapping/separating bodies, and checks published contact and movement behavior. Compare deterministic observable pair sets and physical invariants against an independently executed reference PhysX CPU scene where semantics permit. Require Linux software Vulkan and Windows D3D12 WARP functional runs before AMD hardware performance profiling.

Record per-frame submissions, GPU pipeline creations, descriptor writes, host staging allocations/bytes, physical heap growth, DtoD copies, readback, GPU waits and kernel timings. Do not require exact byte identity or excessive per-test hardware benchmarking. GPU completion must unblock the native task graph without polling loops or unnecessary device-wide idle.

**Pass:** a real native FyzzXYZZY scene drives GPU broadphase and CPU dynamics and publishes physically relevant found/lost pairs, with zero reference-PhysX execution on that scene. This replaces the old PhysX-hosted G3 criterion.

## Gate N4 — reference retirement and broader engine integration

Make native CPU the default; reference PhysX becomes an opt-in comparison/benchmark target available only to tests or explicit reference builds. Extract upstream source headers still required by the port into SDK-owned, licensed source trees. Remove the mutable PhysX `FetchContent main` dependency from production targets and source-bundle packaging. Run all CPU, GPU, Godot and offline tests. Continue GPU narrowphase, solver, articulations and deformable systems through the same native scene and task DAG.

**Pass:** non-test SDK builds and packages do not fetch, compile or link the reference PhysX repository. A native C++ application and Godot use the same FyzzXYZZY-managed scene and backend state.

## Development/PR rules

1. Each PR identifies a **native production owner**, the PhysX source call/data contract it ports, and the exact frame entry/exit edges connected. Test-only helpers are not milestone completion.
2. Enforce one owner per buffer/state family and exactly one physics-frame scheduler. Prefer absorbing compatible helpers into original source-derived owner methods and deleting superseded representations.
3. Keep source-parity tests, including variant kernels, until an integrated native owner passes equivalent edge cases. Deletion requires traced behavioral coverage.
4. Keep reference PhysX isolated; do not modify its `PxSceneDesc`, scheduler or factories to insert portable SDK code. Source deviations and compatibility tests belong in FyzzXYZZY.
5. Count actual runtime allocations, submissions, copy bytes, fences/waits and pipeline work when integrating; performance tuning follows correctness and coherent source execution.
6. For any change that alters this ownership boundary or introduces a new external production dependency, obtain an explicit architectural decision before implementation.
