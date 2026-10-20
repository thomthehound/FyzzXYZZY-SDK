<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# N1 source-faithful native dynamics intake

**Status: source-derived Gu contact geometry, Dy per-point normal/friction constraints and rigid-body constraint partition ordering active in the native scene. Compliant/multi-shape material correlation, island task execution, reports and full N1 acceptance remain open.**

## Objective: port an execution system, not a collection of formulas

The next native CPU integration must bring the **actual PhysX rigid-body PGS
execution slice** into the *existing* SDK-owned `fyzz::native::Scene` so
`World(ExecutionMode::NativeCpu)` follows the source's real dataflow from
scene changes through contact preparation, island creation, constraint
partitioning, normal/friction solving, integration, reporting and frame
publication. Preserve the source's numerical semantics and state lifetimes.
Do not implement another PGS imitation or a parallel engine to get a
compiling "integration" demonstration.

The PhysX fork remains an **independent reference and source library only**;
no production `PxScene`, `NpScene`, PhysX-owned task graph, GPU service
or hidden reference fallback is permitted.

This source contract complements
[NATIVE_ENGINE_MIGRATION.md](NATIVE_ENGINE_MIGRATION.md) and
[ARCHITECTURE_CONTRACT.md](ARCHITECTURE_CONTRACT.md), and is constrained
by the source-derived GPU/AABB integration in
[GPU_BROADPHASE_TRACE.md](GPU_BROADPHASE_TRACE.md).

## Original code paths and required behavior

| Source family in `thomthehound/PhysX` | Actual role | Required native disposition |
| --- | --- | --- |
| `simulationcontroller/src/ScPipeline.cpp` | `Sc::Scene::simulate`, `collideStep`, `broadPhaseFirstPass`, `broadPhaseSecondPass`, `preIntegrate`, `updateBroadPhase`, `postBroadPhase`, `advanceStep`, `secondPassNarrowPhase`, `postNarrowPhase`, `islandGen`, `solver`, `beforeSolver`, `updateDynamics`, `postSolver`, `afterIntegration`, `finalizationPhase` and task continuations | **One** scene-owned dependency graph, initially synchronously executed only where source dependencies allow; CPU and GPU admissions branch using source-equivalent ordering |
| `simulationcontroller/src/ScScene.cpp`, `simulationcontroller/include/ScScene.h` | Owns or connects shape/actor simulation state, low-level context, island manager, AABB manager, dynamics context, simulation controller and scene lifecycle | Absorb the required state and lifetime contracts into the existing native scene; don't construct or secretly drive another `Sc::Scene` |
| `physx/src/NpSceneFetchResults.cpp` | Public `NpScene::fetchResults`, `fetchResultsStart`, `fetchResultsFinish`, result-completion wait, API stage checks and contact callbacks | Separate simulation completion from public publication; the event/callback result boundary must not be confused with `Sc::Scene::finalizationPhase` |
| `lowlevel/software/src/PxsSimpleIslandManager.cpp`, `PxsIslandSim.cpp` | Speculative/accurate island graphs, node/edge lifecycle, first/second/third island passes, connectivity and sleep/wake | Own one island graph/lifetime contract in native scene, with source pass dependencies; a global contact manifold traversal is not a replacement |
| `lowlevelaabb/src/BpBroadPhaseSap.cpp`, `lowlevelaabb/src/BpAABBManager.cpp` | Source CPU AABB owner and incremental SAP; overlaps and filtering inputs | Retain source pair semantics and broadphase manager ownership; existing rebuild-only X-sweep is temporary scaffolding |
| `gpubroadphase/src/PxgAABBManager.cpp`, `PxgCudaBroadPhaseSap.cpp` | GPU first/second AABB pass, aggregate handling, DMA-up/down, SAP compute and report fetch | Separate N2 source intake: preserve GPU-specific lifetime/descriptor/history semantics; do not substitute CPU SAP or assume GPU dynamics owns GPU broadphase |
| `lowlevel/software/src/PxsNphaseImplementationContext.cpp`, `lowlevel/software/src/PxsContactManager.cpp` | Narrowphase contact manager state, caches, contact streams, touch transitions, manager outputs | Native per-contact-manager records, correlated outputs, contact cache lifetime and publication; no conflation of candidate pairs with physical touch |
| `geomutils/src/contact/GuContactPlaneBox.cpp`, `GuContactBoxBox.cpp` | Geometry-specific source collision/contact generation | Port full admitted geometry behavior and rotations, including edge/corner contacts and contact selection, not only axis-aligned faces |
| `lowleveldynamics/src/DyBodyCoreIntegrator.h`, `DySolverBody.h` | Constrained/unconstrained velocities, mass/inertia, solver-body data and pose integration | Preserve float `PxReal` behavior, damping, speed limits, rotations, angular velocity, inertia, locks and final state writeback |
| `lowleveldynamics/src/DyDynamics.cpp`, `DyDynamics.h` | Source `DynamicsContext`, island solver setup and sequencing, contact/constraint descriptors, task partitioning, integration, per-frame pools | Migrate the required CPU PGS execution as a **coherent owner-owned unit**; avoid feeding source solver with invented descriptors |
| `lowleveldynamics/src/DyContactPrep.cpp`, `DyContactPrep.h` | Contact-patch correlation, contact-to-friction association, normal and friction constraints, restitution, Jacobian/response construction, cached friction writeback | Preserve contact-point and friction-anchor semantics, not a one-normal-per-manifold approximation or custom Coulomb disk |
| `lowleveldynamics/src/DyConstraintPartition.cpp` | Assigns constraints to safe concurrent partitions; prepares batches | Preserve partition dependencies and ordering before parallel CPU/GPU scheduling; one island's bodies/constraints must be owned once |
| `lowleveldynamics/src/DySolverConstraints.cpp`, `DySolverContact.h` | Source normal and friction PGS equations, static/dynamic/ext paths, threshold/writeback and conclude | Port **actual** source constraint representations and solve/update semantics, with source-aware precision and friction limits |
| `lowleveldynamics/src/DySolverControl.cpp` | Position iterations, velocity iterations, special conclude/writeback pass, friction iteration policy and threshold streams | Preserve distinct solver passes, writeback timing and the no-contacts integration case |
| `lowleveldynamics/src/DyTGSContactPrep.cpp`, `DyTGS.h` | Alternate TGS constraint preparation and step solver | **Separate later integration**. Do not claim TGS parity from a PGS port or quietly synthesize TGS behavior |

These paths identify the **initial source intake surface**. They do **not**
authorize compiling all of PhysX into the native SDK. Extract/translate
needed code together with its genuine input, output, allocation and lifetime
contracts, retaining source attribution. Neither rewriting the mathematics
from memory nor a hard-coded equivalent-looking "demo solver" satisfies
this contract.

## Source task ordering: execution and handoffs

Read `ScPipeline.cpp` as a dependency graph, **not** a flat function list.
Important verified source edges:

1. `simulate(dt)` configures scene time and the dynamics-context timestep,
   then chains collision into `advanceStep` through task continuations.
2. Collision dirty-shape/bounds work precedes the AABB **first pass**.
   `setupBroadPhaseFirstAndSecondPassTasks` deliberately differs for CPU
   and the `isUsingGpuDynamicsOrBp()` admission: only the
   **CPU-broadphase + CPU-dynamics** case chains the first pass directly
   to `mBpUpdate`. If **either** GPU broadphase (`mUseGpuBp`) **or**
   GPU dynamics (`mUseGpuDynamics`) is enabled, the scene inserts
   `mBpSecondPass` and `mPreIntegrate` task edges. This includes
   the intended **GPU SAP + CPU dynamics** N2 hybrid. Do not infer
   the continuation branch solely from which backend solves contacts.
3. `updateBroadPhase` calls AABB manager second-pass/update, unblocks
   narrowphase at the prescribed dependency point; `postBroadPhase`
   feeds narrowphase and manager report publication. Filtering,
   interaction/manager creation and island insertions depend on reports.
4. `advanceStep` installs **different explicit continuation chains**.
   Both cases include second-pass narrowphase, `postNarrowPhase`,
   `islandGen`, scene-level `solver`, `updateDynamics`,
   `postSolver`, `afterIntegration` and `finalizationPhase`.
   When `isUsingGpuDynamicsOrBp()` is **true**, the chain also
   includes `updateBodies` before `updateDynamics`, plus
   `updateDynamicsPostPartitioning` and
   `updateSimulationController` after it. This also applies when
   **GPU broadphase is enabled but dynamics remains CPU**.
   `USE_SPLIT_SECOND_PASS_ISLAND_GEN` controls a **separate
   compile-time island dependency**: the split configuration connects
   `mIslandGen` directly to `mSolver` and calls
   `postIslandGen(&mSolver)` from `islandGen`; the unsplit
   configuration inserts `mPostIslandGen` between the two tasks.
   The flag does **not** mean GPU execution, and these are not
   interchangeable dependency graphs. Continuations can join work
   that overlaps other stages.
5. `postNarrowPhase` synchronizes actual contact-manager outputs
   (`fetchUpdateContactManager`), sorts GPU contacts where required and
   releases prior constraints. It does **not** process patches.
   `islandGen` invokes `DynamicsContext::processPatches` using
   narrowphase lost/found patch managers and output counts, then
   calls `processNarrowPhaseTouchEvents(&mUpdateDynamics)`.
   The latter **schedules** `InteractionNewTouchTask` with
   `mUpdateDynamics` as its continuation. The touch tasks can
   overlap scene-level `solver` setup; their required completion
   boundary is **`updateDynamics`**, which consumes the prepared
   contact/island graph state. Do not accidentally create a false
   touch-event-to-`solver` serialization edge.
6. **`Sc::Scene::solver` is not `Dy::DynamicsContext::update`, but
   it is also not *only* force preparation.**
   Under `USE_SPLIT_SECOND_PASS_ISLAND_GEN`, **the first operations**
   in `Sc::Scene::solver` are
   `mSimpleIslandManager->secondPassIslandGenPart2()` and
   `wakeObjectsUp()`. Only then does it call
   `beforeSolver(continuation)` for body forces/threshold state and
   `mLLContext->swapStreams()` for narrowphase. In the **unsplit**
   configuration, `setEdgesConnected` has already completed the
   combined `secondPassIslandGen()` and `wakeObjectsUp()`, so
   `solver` does *not* repeat either operation. In neither case
   does the scene-level `solver` task invoke
   `mDynamicsContext->update`. The **distinct downstream**
   `Sc::Scene::updateDynamics` task acquires constraint memory,
   arranges lost-contact continuations, and calls
   `mDynamicsContext->update(...)` with narrowphase output,
   gravity, timestep, patch count and changed-handle map.
7. **Partitioning is a distinct continuation handoff.**
   Only when `isUsingGpuDynamicsOrBp()` is **false** does
   `updateDynamics` pass `mPostSolver` to the dynamics context
   and then call `updateDynamicsPostPartitioning` **immediately
   in the same thread**. When the predicate is true—including
   GPU-broadphase-only scenes with a CPU solver—the method supplies
   `mUpdateSimulationController` and
   `mUpdateDynamicsPostPartitioning` continuations for the deferred
   path. In both cases
   `updateDynamicsPostPartitioning` calls
   `mDynamicsContext->updatePostPartitioning`, clears destroyed
   partition edges and releases the chained lost-contact task
   references. Do not silently collapse these edges to the earlier
   scene-level `solver` stage.
8. **The underlying CPU `Dy::DynamicsContext::update` queues real
   solver work, not merely a callback.** Its `updateShared` gate
   precedes solver-body pool and descriptor allocation. It zeroes the
   world solver body, obtains accurate active island nodes,
   reserves/resizes solver-body and constraint descriptor pools, and
   starts a continuation task. `updatePostKinematic` enumerates
   active islands, groups work by source solver-batch limits, creates
   solver task chains and schedules force-threshold reporting.
   Per-island setup builds source contact/constraint descriptors,
   partitions batches, performs `solveV_Blocks` (or the parallel
   variant), then integrates bodies; zero-constraint cases still
   integrate. This is the relevant **DynamicsContext execution**
   boundary, not `Sc::Scene::solver`.
9. **`DySolverControl.cpp::solveV_Blocks`** distinguishes position
   iterations, concludes constraints on the last position pass,
   saves motion velocities, then performs velocity iterations with
   a final writeback pass. Friction activation depends on source
   iteration policy; per-contact force thresholds/writeback and
   no-contact integration must not disappear in translation.
10. **Island generation has a third pass tied to lost-contact cleanup,
    sleeping and publication**; the `updateDynamics`-installed
    lost-contact continuation chain eventually executes
    `destroyManagers`, which calls
    `mSimpleIslandManager->thirdPassIslandGen`. That source method
    forks *speculative* and *accurate* third-pass tasks and joins them
    through its post-third-pass continuation. `processLostContacts3`
    removes deferred overlaps and frees AABB manager buffers before
    releasing `mPostThirdPassIslandGenTask`; its
    `postThirdPassIslandGen` callback puts objects/interactions to
    sleep and processes persistent contact events. These dependencies
    must complete before the consuming publication boundary, **not**
    at `secondPassIslandGenPart2`.
11. **`postSolver` is a required state-lifetime barrier**: it merges
    dynamics results, releases solver constraint memory and swaps
    friction streams. When `isUsingGpuDynamicsOrBp()` is true,
    `updateSimulationController` first invokes simulation-controller
    DMA-back and dynamics body-core update. The exact operations
    executed for CPU-dynamics/GPU-broadphase mode must be resolved
    in the controller implementation; do not assume a GPU solver. `afterIntegration` then reconciles
    wake/sleep, bounds and contact-force events; finalization
    handles callbacks, contact visualization, task-pool release and
    report timestamps. The native scene must not publish a
    predicted pose as final source-equivalent post-solver state.

**Current integrated CPU slice and remaining departure:** `src/native/scene.cpp::step` now runs the scene-owned Gu collision stage at the starting pose, source-derived `DyContactPrepShared` per-point rigid-body normal Jacobian/effective-mass/bias preparation, and `DySolverConstraintsShared` normal impulse passes. Following `DySolverControl`, it snapshots motion deltas after biased position iterations and separately executes unbiased velocity iterations; `DyBodyCoreIntegrator` uses saved position-phase motion to advance poses and final velocity-phase deltas to update next-frame body velocity. This retires the predicted-pose/one-normal-per-manifold solver and its angular guard.

The active native contact path now also translates the **noncompliant rigid-material** portion of `PxsMaterialCore.h`, `PxsMaterialCombiner.h`, and `DyContactPrep.cpp`: each currently supported authored shape owns one noncompliant material, with static/dynamic friction, restitution, the source's higher-precedence friction/restitution combine mode (`AVERAGE`, `MIN`, `MULTIPLY`, `MAX`) and friction/strong-friction disable flags. Combined restitution drives per-point normal bounce, while combined static/dynamic friction drives source friction-anchor limits and patch cache admission. The source's `staticFriction >= dynamicFriction` promotion and disable-friction row omission are preserved; default material behavior is unchanged. Public `World::set_body_material` authors material on the existing NativeCpu Scene owner. The optional reference PhysX CPU adapter assigns a real `PxMaterial` to its existing `PxShape` and releases the temporary reference after assignment; neither mode adds a shadow material-table authority. Native and source-enabled reference smoke cases validate this same authoring operation. Native regressions validate the combined coefficients by actual normal/friction body response. Materials with negative restitution (compliant spring/acceleration), damping combinations, multi-material geometry, per-point material assignments and source packed/SIMD constraints are **not** admitted by this subset.

The active PGS preparation now uses the original CPU `DyCpuGpuBiasCoefficient::rigidContact = 0.8f` for both normal penetration correction and friction relaxation. Cached friction normals live in both actors' local frames and survive correlated contact motion; source `DyContactPrepShared::pointsAreClose` uses **absolute** projected anchor separation to retire patches on either excessive compression or positive separation. These numerical and lifetime boundaries are covered by the current draft review, without declaring full source parity.

The same native source reporting path now also implements the low-level **Dy force-threshold stream**: `DySolverConstraints.cpp::writeBackContact` supplies final contact-normal impulses, `DyThresholdTable` aggregates by persistent rigid-body pair, and `DyDynamics.cpp::PxsForceThresholdTask` determines current versus previous threshold exceedance. The current scene has one shape per BodyId but must preserve **two different keys**: the individual `Gu` `BodyId` pair owns geometric contacts, while the low-level `DyThresholdTable` aggregates ALL static-shape contacts on one dynamic body through the shared world-node key. Dynamic/dynamic contacts retain their canonical dynamic body pair. An explicit `ForceThresholdKey` distinguishes these cases without inventing a second island/shape registry. Mapping different static managers to one world key can disturb Gu sort order: the current active threshold candidates are sorted in their reusable arena, combined in place, and previous/current exceeded streams then merge linearly, without per-contact allocations. Dynamic actors opt in with `Scene::set_contact_report_threshold` (source default `PX_MAX_REAL` disables), the pair threshold uses `min(body0, body1)`, and an exceedance requires `sum(finalNormalImpulse) > reportThreshold * dt`. `force_threshold_changes()` emits Found/Persists/Lost states; LOST has zero impulse and PERSISTS carries the prior stream's impulse, matching source stream construction. Collision-only contact progression invalidates visible reports, but cannot erase previous-step exceedance history. Native tests cover these transitions, force-versus-impulse units, dynamic/dynamic minimum thresholds and contact loss. A two-static-support fixture independently measures both Gu support impulses, places a threshold strictly between the largest individual impulse and their sum, and requires a **single world-node Found report** with the combined impulse. Even when both supports exceed a zero threshold, it requires only one event; removal of both supports produces exactly one Lost report.

This is a **bounded low-level source subset**, not the whole `PxSimulationEventCallback` surface. Original `eHAS_FORCE_THRESHOLDS` interaction flags, persistent island `PxNodeIndex` identities, multi-shape shapeInteraction objects, contact-report pair flags, `ScNPhaseCore` event dispatch and complete material-dependent callback publication remain open N1 work. The explicit dynamic-pair/static-world threshold key is a native surrogate for the source island node pair, not a new island or shape registry; wide `double` impulse totals remain an SDK extreme-mass extension where original PhysX uses `PxReal`.

The native scene now also executes the per-contact **normal-impulse writeback** portion of `DySolverConstraints.cpp::writeBackContact` after the final velocity iteration. It copies each source-derived constraint's accumulated normal impulse into one reusable output arena indexed by the original `Gu` contact offset, independent of `DyConstraintPartition` ordering. Geometry-only `advance_collision()` invalidates previous solved values; the next `step()` replaces them. Native regressions check impulses against rigid-body velocity changes, original contact offsets, source partition rollover and contact retirement. Output uses `double` only for the SDK's previously admitted extreme inverse-mass translational cases; original PhysX writes `PxReal`. The cached friction patch's `broken` flag now follows the original `DySolverConstraints.cpp::solveContact` and `solveContact_BStatic` lifecycle: an iteration-local clamp result is collected during each friction solve, but **only the final velocity/writeback iteration** publishes it. A position-pass clamp that subsequently converges does not invalidate the next frame's correlated local anchors. The 600-frame dynamic stack provides a decisive recovered-anchor case: the ground/lower-box patch saturates during a frame-86 position pass but does not clamp at final writeback. Its `broken` report must be false and its cached local anchors must survive through frame 87. A controlled negative test restoring the earlier sticky `broken |= clamped` rule failed this exact assertion in both independent Linux Vulkan and Windows D3D12 suites; the negative mutation was subsequently reverted. A broader low/high slip sweep retains additional Coulomb-limit consistency checks, but cannot alone distinguish final-only break reporting from a stale clamping history. This test does not claim full `DyFrictionCorrelation` parity or heterogeneous materials.

The same Scene-owned final writeback now also reconstructs `DyContactPrep.cpp::writeBackContactFriction`'s two-tangent per-anchor world impulse vectors, preserves correlated friction patch ownership/anchor positions and reports broken state through `Scene::solved_friction_patches()`. Both report streams are invalidated by geometry-only collision advancement and republished after stepping. Distinct-mass/opposite-slip fixture responses validate the signed friction impulses against body velocity changes. A separate deliberately **permuted** three-manager contact graph exercises Gu order `AB, AC, DE` versus Dy partition order `AB, DE, AC` and validates the force writeback against distinct leaf momentum changes, correcting the inadequate sorted-star-only regression. The SDK's extreme-mass reporting extension retains `double` impulse components where the original source writes `PxReal`/`PxVec3`.

Shape-interaction threshold callback/event dispatch, heterogeneous material-dependent writeback, exact source contact/patch mapping, packed descriptor publication and full scene report dispatch remain unported.

The native solver also imports the **rigid-body** ownership/ordering slice of `DyConstraintPartition::partitionContactConstraints`: actor-owned 32-bit dynamic constraint masks, 32-partition overflow waves, per-body static contacts after dynamic constraints, stable counting-prefix ordering of authoritative contact managers and reusable scene-owned arrays. Both normal and friction rows execute in that order within each original PGS pass. A 35-edge dynamic contact star tests partition rollover; twelve unrelated boxes contacting the shared zero-mass static floor verify there is no false partition conflict. This path is serial; it does **not** claim the source island manager, per-island task graph, parallel batch execution, rigid-body static SIMD specialization or articulation partitioning.

The existing scene now also applies a source-derived subset of `DyFrictionCorrelation`: same-normal/material grouping, correlated cached body-local patches, at most two anchors chosen from the source contact geometry, two tangent constraints per anchor, and `DySolverConstraints` patch friction solving under the original iteration policy. This path restores the 600-frame stacked-body regression with the default one velocity pass; it does not add an extra pass or weaken its tolerance.

This is **not** full source solver admission: heterogeneous material combine modes and per-contact target velocities, exact `DyCorrelationBuffer` patch-list ordering and persistence in all cases, packed solver constraint ABI, exact `PxReal`/SIMD rounding, full source island/task scheduling and batch specialization, and contact-manager friction/threshold events and complete force reporting remain unported. The native path is a transitional rigid-body contact/material subset, not an accepted PhysX CPU replacement or N1 completion.

## Stage handoff ledger: do not confuse task labels with solver ownership

These entries are **observed source call sites**, not a proposed
reimplementation of PhysX's `PxBaseTask` machinery. The native
continuation system can be different only if it preserves the same
data readiness, single ownership and completion dependencies.

| Producer (source method) | Handoff and data lifetime | Consumer / permitted next step | CPU / GPU difference |
| --- | --- | --- | --- |
| `Sc::Scene::simulate` / `stepSetupCollide` | Stores `mDt`, configures dynamics `setDt`, queues collision continuation | `collideStep` and later `advanceStep`; no two independent scene clocks | Both paths |
| `preRigidBodyNarrowPhase` / `rigidBodyNarrowPhase` | Dirty shape/contact-distance state and contact-manager work; broadphase/narrowphase dependency refs | AABB first pass, update, post-broadphase/filter/register and narrowphase completion | GPU additionally transfers shapes; first/second pass and pre-integration task edges differ |
| `postBroadPhaseContinuation` / `postBroadPhaseStage2` | Changed-handle bitmap reset, overlap filtering and interaction/contact-manager allocation; island insertion and lost-overlap cleanup | Source contact-manager update / second-pass narrowphase | Registration and island insertion can run as separate tasks |
| `postNarrowPhase` | `fetchUpdateContactManager` makes contact-manager outputs ready; old constraints can be released | `islandGen`, not the dynamics solver | GPU may sort contacts here |
| `islandGen` | `mDynamicsContext->processPatches`, then `processNarrowPhaseTouchEvents(&mUpdateDynamics)` **enqueues** new-touch tasks; split mode also invokes `postIslandGen(&mSolver)` | `InteractionNewTouchTask` signals **`mUpdateDynamics`** when finished; it may overlap scene-level `solver` setup | Unsplit: `mPostIslandGen` is an explicit continuation; split: `postIslandGen` can run from island generation |
| `setEdgesConnected` | Marks contact-manager edges connected; **split** calls `secondPassIslandGenPart1()`; **unsplit** calls complete `secondPassIslandGen()` then `wakeObjectsUp()` | In split mode, accurate-island part 2 is a prerequisite in the scene-level `solver` | Split task can overlap island-generation and partition-related work as source continuations permit |
| `Sc::Scene::solver` / `beforeSolver` | **Split:** first calls `secondPassIslandGenPart2()` (lost/destroyed edges, node retirement) and `wakeObjectsUp()`; **both modes:** `beforeSolver` force/threshold prep, then narrowphase stream swap | Downstream **`Sc::Scene::updateDynamics`**, never `DynamicsContext::update` inside `solver` | Split flag determines part-2 ownership; GPU-dynamics-or-BP predicate separately determines whether `updateBodies` intervenes |
| `Sc::Scene::updateDynamics` | Executes after new-touch tasks signal its continuation; acquires constraint memory, adds **three-stage lost-contact task chain** references, calls `mDynamicsContext->update` | `updateDynamicsPostPartitioning`, dynamics solver work, lost-contact/third-island-pass processing and eventually `postSolver` | Predicate false: immediate post-partition call; predicate true: deferred stage even for GPU-BP/CPU-dynamics |
| `Dy::DynamicsContext::update` and `updatePostKinematic` (CPU) | Owns solver pools, accurate active island lists, constraint descriptors, per-island task creation and partitioning | Source `DySolverControl` PGS iterations, integration, postprocessing, threshold/force reports | GPU dynamics context has a different implementation; do not infer full GPU behavior from CPU `DyDynamics.cpp` |
| `updateDynamicsPostPartitioning` | Calls dynamics-context post-partitioning hook, clears destroyed partition edges and releases staged lost-contact references | `updateSimulationController` if `isUsingGpuDynamicsOrBp()`, else CPU-only post-solver continuation | CPU-dynamics **alone** does not imply the inline path; GPU-BP/CPU-dynamics still uses the deferred scene chain |
| `processLostContacts` / `processLostContacts2` / `destroyManagers` | Lost touch reports and manager retirement; `destroyManagers` invokes `SimpleIslandManager::thirdPassIslandGen` with speculative/accurate continuations | `processLostContacts3` retires overlap/AABB buffers and releases `mPostThirdPassIslandGenTask`; `postThirdPassIslandGen` puts objects/interactions to sleep and emits persistent events | Can overlap other source work; must join the `mPostSolver` dependency, not retire immediately after second island pass |
| `postSolver` / `afterIntegration` / `finalizationPhase` | Merges results, frees constraint blocks, swaps friction streams, completes body-state publication, events and cleanup | Result fetch and next-frame scene mutation | When GPU-dynamics-or-BP is selected, updateSimulationController runs ahead of postSolver; do not infer GPU rigid-body solve was selected |

### Verified source anchors and still-open closure

The following are directly checkable locations in the maintained PhysX
source used for this audit. Line anchors are navigational aids, not
license to assume every transitive dependency was already ported:

- [`ScPipeline.cpp::setEdgesConnected` and `solver`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/src/ScPipeline.cpp#L1718-L1790) — the **compile-time split** of second-pass accurate island generation: part 1 in `setEdgesConnected`, part 2 and waking in `solver` before `beforeSolver` and stream swap. In unsplit mode, `setEdgesConnected` does the complete second pass and waking.
- [`ScPipeline.cpp::beforeSolver`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/src/ScPipeline.cpp#L1929-L2010) — body force and threshold preparation. The scene-level `solver` does **not** call `DynamicsContext::update`.
- [`ScPipeline.cpp::advanceStep`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/src/ScPipeline.cpp#L1396-L1440) — compile-time split bypasses `mPostIslandGen` as a distinct continuation; unsplit mode inserts it between `mIslandGen` and `mSolver`.
- [`ScPipeline.cpp::updateDynamics` / `updateDynamicsPostPartitioning`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/src/ScPipeline.cpp#L2049-L2140) and [`advanceStep`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/src/ScPipeline.cpp#L1371-L1443) — actual dynamics-context entry, the `isUsingGpuDynamicsOrBp()` branch, and CPU-immediate versus GPU-admitted deferred continuations.
- [`ScPipeline.cpp::postNarrowPhase` / `islandGen`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/src/ScPipeline.cpp#L1525-L1560) and [`processNarrowPhaseTouchEvents`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/src/ScPipeline.cpp#L1610-L1672) — patch processing and new-touch tasks, explicitly chained to `mUpdateDynamics`, **not `mSolver`**.
- `DyDynamics.cpp::DynamicsContext::update` and `updatePostKinematic` — inspected their original implementations for world-body reset, pool lifetime, accurate-island batching, and solver-task creation. Extract their exact source ranges and transitive type dependencies before porting; no approximate line anchor is asserted here.
- [`DySolverControl.cpp::solveV_Blocks`](https://github.com/thomthehound/PhysX/blob/main/physx/source/lowleveldynamics/src/DySolverControl.cpp#L160-L313) — position/velocity, conclude, writeback and no-contact branches.
- [`PxsSimpleIslandManager.cpp::firstPassIslandGen`, `secondPassIslandGenPart1/Part2` and `thirdPassIslandGen`](https://github.com/thomthehound/PhysX/blob/main/physx/source/lowlevel/software/src/PxsSimpleIslandManager.cpp#L257-L330) — speculative versus accurate state, lost-edge removal and third-pass deactivation work.
- [`ScPipeline.cpp::destroyManagers`, `processLostContacts3` and `postThirdPassIslandGen`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/src/ScPipeline.cpp#L2368-L2490) — third-pass producer, the AABB-buffer retirement/release, and eventual sleep/persistent-contact publication.
- [`NpSceneFetchResults.cpp::fetchResults` / `fetchResultsStart` / `fetchResultsFinish`](https://github.com/thomthehound/PhysX/blob/main/physx/source/physx/src/NpSceneFetchResults.cpp#L484-L670) — completion readiness and public contact-event publication.
- [`PxgAABBManager.cpp::updateBPFirstPass` / `updateBPSecondPass`](https://github.com/thomthehound/PhysX/blob/main/physx/source/gpubroadphase/src/PxgAABBManager.cpp#L485-L828) and [`PxgCudaBroadPhaseSap.cpp::gpuDMAUp` / `update` / `fetchBroadPhaseResults`](https://github.com/thomthehound/PhysX/blob/main/physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp#L242-L1260) — separate GPU SAP host lifecycle. This document does not certify that N2 is wired.

**Important qualification:** PhysX may process work concurrently where
continuation dependencies permit. An ordered list of task names is not
a proof that each waits on every prior task. Before writing the native
scheduler, extract actual continuation edges, producer-completion
requirements, and buffer retirement from the original task construction
sites into an executable DAG check. The table above describes the
required handoffs, not an invented globally sequential PhysX frame.

### Compile-time second-pass island branch — precise stage ownership

The `USE_SPLIT_SECOND_PASS_ISLAND_GEN` compile-time flag is
**orthogonal** to `isUsingGpuDynamicsOrBp()`. Its two possible
dependency/ownership layouts must be mapped separately:

| Flag | `advanceStep` continuation | `islandGen`/post-island handling | `setEdgesConnected` | `Sc::Scene::solver` entry |
| --- | --- | --- | --- | --- |
| Disabled | `mIslandGen → mPostIslandGen → mSolver` | `postIslandGen` runs as the chained task; its edge/trigger work signals `mSolver` | Connect contact edges; `secondPassIslandGen()` (both parts), then `wakeObjectsUp()` | `beforeSolver`, then `swapStreams` |
| Enabled | `mIslandGen → mSolver` | `islandGen` calls `postIslandGen(&mSolver)` while task continuations permit other work to overlap | Connect contact edges; `secondPassIslandGenPart1()` only | **`secondPassIslandGenPart2()` and `wakeObjectsUp()` first**; then `beforeSolver` and `swapStreams` |

The island manager's `secondPassIslandGenPart1` handles accurate
wake/new-edge work; `secondPassIslandGenPart2` removes destroyed
edges and lost-edge connections, retires destroyed node handles
and clears their bookkeeping. Under the split flag, these side
effects are deliberately moved to the **scene-level `solver`
boundary** rather than dropped or treated as an earlier completed
island pass. Neither this compile-time split nor the third-pass
deactivation tasks imply a GPU physics solver.

Conformance obligations: exercise both compile-time variants where
builds admit them (or document a single supported configuration with
source-grounded justification), verify the same final accurate-island
state, new/destroyed-contact lifecycle, waking and next-frame active
membership, and demonstrate that permitted task overlap does not
change publication results. Treat this as a future test requirement,
**not presently established execution parity**.

### Lost-contact retirement and third-pass island convergence

This is **not** an optional postscript to a two-pass island build.
The original source explicitly connects lost-contact cleanup, active
island state and solver completion, permitting overlap where the task
continuations allow it:

1. `Sc::Scene::updateDynamics` establishes
   `mProcessLostContactsTask` →
   `mProcessLostContactsTask2` →
   `mProcessLostContactsTask3` continuations, with the latter
   connected to `mPostSolver`. `updateDynamicsPostPartitioning`
   releases their staged references when the prerequisites are set.
2. `processLostContacts` identifies destroyed overlaps and schedules
   lost-touch work. `processLostContacts2` clears old island-edge
   connection data while sequencing lost-touch reports, interaction
   unregistration and `mDestroyManagersTask`.
3. `destroyManagers` configures
   `mPostThirdPassIslandGenTask` to join the lost-contact
   continuation chain and starts
   `SimpleIslandManager::thirdPassIslandGen`, which clears
   accurate graph deactivations, schedules speculative and accurate
   third-pass tasks, and joins at `mPostThirdPassTask`.
4. `processLostContacts3` calls `onOverlapRemoved` for destroyed
   shape/trigger overlaps, frees AABB buffers, and only then
   releases `mPostThirdPassIslandGenTask`.
5. `postThirdPassIslandGen` calls `putObjectsToSleep`,
   conditionally deactivates speculative-island interactions
   (unless sleeping is disabled), and processes persistent
   contact events. The **join into `mPostSolver`** must respect
   the third pass and lost-contact cleanup. The exact source
   dependencies may overlap numerical solve work—do **not**
   represent the third pass as an unconditional serial step
   executed after solver arithmetic.
6. In `afterIntegration`, PhysX consults the accurate
   island manager's `getNodesToDeactivate` and can restore the
   previous pose and zero velocities of bodies the solver
   integrated concurrently but the island logic deactivated.
   This reconciliation is relevant to correct next-frame state
   and must not be discarded in a native implementation.

Acceptance regression for this connection: multi-frame dynamic
bodies crossing sleep/wake thresholds, contact loss and
destroyed-manager retirement must preserve active state, event
ordering, and stable next-frame island membership versus PhysX CPU.
This is a **test requirement**, not presently implemented parity.

### Backend admission cases that must not be conflated

`ScPipeline.cpp` does not branch solely on the CPU/GPU **solver**.
The predicate [`Sc::Scene::isUsingGpuDynamicsOrBp()`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/include/ScScene.h#L593-L596) is defined in the original header as `mUseGpuBp || mUseGpuDynamics`; it controls both the
broadphase task route and the `advanceStep` / `updateDynamics`
continuation topology:

| Scene configuration | `mUseGpuBp` | `mUseGpuDynamics` | `isUsingGpuDynamicsOrBp()` | Scene chain |
| --- | --- | --- | --- | --- |
| CPU BP + CPU dynamics | false | false | **false** | CPU-only first-pass/inline post-partition continuation |
| **GPU SAP + CPU dynamics** (planned N2 hybrid) | true | false | **true** | GPU-admitted AABB/pre-integrate and deferred post-partition + simulation-controller edges; do **not** infer GPU solver execution |
| CPU BP + GPU dynamics | false | true | **true** | Same source predicate branch for scene continuations; independent admission/validity of this configuration must be verified |
| GPU BP + GPU dynamics | true | true | **true** | Same predicate branch, with GPU rigid-body dynamics where source admission permits |

**Independent execution detail still to establish:** the GPU-admitted
scene branch calls simulation-controller and dynamics-context methods,
but the exact CPU-dynamics/GPU-SAP device transfers, controller effects
and allowed configurations need source-level verification before the
native hybrid is implemented. The branch predicate alone does not
prove GPU solver work or identical staging cost.

### Readiness and retirement invariants

- **Contact patch freshness and touch-task join:** after
  `postNarrowPhase` has synchronized narrowphase manager outputs,
  `islandGen` consumes *lost/found patch managers and output counts*
  in `DynamicsContext::processPatches`, then schedules
  `InteractionNewTouchTask` with **`mUpdateDynamics`** as
  continuation. The tasks may run alongside scene-level `solver`
  setup. They must join before `updateDynamics`, not before
  `solver`. No invented barrier may serialize permitted overlap.
- **Task ownership:** the scene-level `solver` task is a *setup boundary*,
  not the numerical PGS execution. `updateDynamics` is the source
  `DynamicsContext::update` caller. Its continuation chain owns
  solver work and the output dependency to post-solver.
- **Constraint allocation:** acquire the solver constraint-memory arena
  before `DynamicsContext::update`; retire only in `postSolver` after
  the continuation is done. Friction stream swap likewise belongs to
  post-solver, not to arbitrary manifold creation.
- **Narrowphase streams:** swapped by `Sc::Scene::solver` after the
  second pass completes; a source solver task must not read a stream
  before its producer has completed or after it has been retired.
- **All THREE island passes, including split ownership:** preserve
  separate speculative and accurate graphs from
  `PxsSimpleIslandManager`. First pass handles speculative
  wake/new/lost edges. The accurate second pass is either **combined**
  in `setEdgesConnected` (followed by `wakeObjectsUp`) or **split**:
  `setEdgesConnected` runs part 1, while the *scene-level*
  `solver` runs part 2 and waking **before** force preparation.
  Part 2 removes destroyed edges/lost connections and retires node
  handles. **Neither path replaces the third pass**: it forks
  speculative and accurate deactivation work under a join task.
  The `destroyManagers`/`processLostContacts3`/
  `postThirdPassIslandGen` sequence controls contact retirement,
  AABB buffer release and sleep/event publication.
  Solver/island concurrency means `afterIntegration` may need to
  reconcile solver results for bodies deactivated by island generation.
  Do not invent serialization or publish before third-pass completion.
- **World and solver bodies:** zero the source world solver body, preserve
  the kinematic/dynamic remap and accurate island-node ordering, and
  feed contact constraints with real per-body inertia/Jacobian state.
  Neither a global manifold array nor the public actor ID is a
  substitute for a `PxSolverConstraintDesc` mapping.
- **Branch admission is GPU-dynamics-OR-broadphase:** the exact
  `Sc::Scene::isUsingGpuDynamicsOrBp()` predicate is
  `mUseGpuBp || mUseGpuDynamics`. Its **false** case immediately
  invokes `updateDynamicsPostPartitioning`; its **true** case
  uses GPU-style scene task continuations, **even if dynamics itself
  runs on CPU**. Never wire the planned GPU-SAP/CPU-solver hybrid
  to the false/inline branch solely because of its solver backend.
  Do not insert blocking waits to imitate a flat CPU chain.
- **Fetch/publication boundary:** `Sc::Scene::finalizationPhase` is
  **not** the complete public `NpScene::fetchResults` contract.
  `NpSceneFetchResults.cpp` requires the correct simulation stage,
  observes the physics completion event, handles error checks and
  contact callbacks, and separates pre/post-callback result work.
  A native scene must expose a truthful completion/publication edge
  and must not call user observers on half-published solver state.
- **Failure/retirement:** a solver task failure must not make a
  partially updated actor state appear published; contact-manager,
  constraint, friction, and result buffers retain their owners until
  all consumers complete. This is an **SDK error-policy requirement**,
  not a claim that PhysX itself uses exceptions here.

### Verification status

| Concern | Evidence presently established | Work still required |
| --- | --- | --- |
| `ScPipeline` task boundaries | Verified `solver` versus `updateDynamics`, `islandGen` patch ownership, new-touch task continuation to `mUpdateDynamics`, GPU-dynamics-or-broadphase predicate, **both compile-time second-pass island branches**, third-pass/lost-contact joins and `postSolver` cleanup | Full task refcount DAG across compile-time modes, hybrid controller internals and CCD paths |
| `DyDynamics` CPU integration | Verified CPU `update`, source solver-body pools, `updatePostKinematic` island batching, PGS task creation | Input/output ABI extraction, exact partition and writeback semantics, contact/constraint layout inventory |
| PhysX GPU dynamics | Verified distinct `ScPipeline` GPU continuations and controller handoff | **Not yet independently mapped** GPU-specific `DynamicsContext` implementation, CUDA launch/memory graph and any CPU fallbacks |
| Island manager and public fetch | Inspected original speculative/accurate first, split second, and **third deactivation pass**, its lost-contact/AABB retirement joins, and `NpScene::fetchResults` completion/callback split | Full edge/refcount graph, corner-case sleep policy and native API publication semantics still need closure |
| GPU SAP host lifecycle | Located original `PxgAABBManager` first/second pass and `PxgCudaBroadPhaseSap` DMA/update/fetch methods | Source-stage and buffer-history closure still belongs to N2, not assumed complete by this N1 audit |
| Source geometry and contact correlation | Identified `GuContact*`, `PxsNphaseImplementationContext` and `DyContactPrep` source families | Exact correlation/anchor data structure, cache transitions and geometry edge-case acceptance |
| Native acceptance | Existing source-free scene and oracle smoke tests support an openly bounded subset | Complete rotational PGS, source schedule, GPU SAP and N1/N2 full parity **not admitted** |

## Failed-step publication and performance ledger

A native `Scene::step` is atomic with respect to its authoritative
body state, pending angular impulses, collision/force/threshold reports,
friction cache, island topology and frame counter. A failed attempt
restores the preceding successful publication and permits an unchanged
retry. The published Gu pair/contact arena is restored by reversing
the original `std::vector::swap` handoff, preserving existing
`std::span` storage across failure; transient Dy working arrays are
recomputed rather than duplicated.

The CPU publication boundary now stages published force, friction,
threshold-change, Gu contact and broadphase-pair buffers through reusable
or reversible swaps, rather than copying those result streams each frame.
The capacity-reusing checkpoint saves only *mutable dynamic motion*
(position, orientation, velocities and pending torque) together with
persistent friction history and island/threshold bookkeeping. Immutable
shape geometry and actor materials are not cloned; there is no second
simulation or solver.

**Remaining copy cost:** dynamic motion and persistent friction/island
state are still captured at O(dynamic bodies + retained contact state)
per frame, including successful steps. This is a performance-remediation
item, not the final source publication architecture. As the original
`ScPipeline`/`Dy::DynamicsContext` continuation and fetch boundaries
are integrated, lower the remaining actor/cache checkpoint into staged
writes and source-owned publication. Do not add blocking waits or
re-run the simulation to emulate failure atomicity.

The native regression exercises multiply-combined finite materials that
overflow during contact preparation, repeated failed attempts, preservation
of previously published contact spans and force/threshold reports,
one-shot torque impulse continuity on a retry, and a late pose-overflow
after an earlier body has integrated. The test must fail if any of those
stages leaks state from an unsuccessful step.

## Native ownership and data contracts

The only production scene state owner remains `fyzz::native::Scene`.
Its internal representation must evolve to hold or own the corresponding
source-class responsibilities without adding another world or a
second task scheduler.

| Native responsibility | Source state contract to preserve | Transition rule |
| --- | --- | --- |
| Actor/shape life | Actor IDs, dynamic/static types, shape geometry and transform, local shape offsets, mass/inertia, angular velocity and body-state locks | Grow the existing actor/shape records or source-derived owners; keep external API handles stable; no second actor registry |
| Narrowphase life | Candidate pair vs contact-manager state, persistent cache/correlation, physical touch found/lost, per-contact normals/separations/points/materials | Replace simplified per-pair manifold state as the source manager/correlation is integrated; preserve existing event tests |
| Island life | Connectivity, active/inactive nodes, static touches, per-island solver objects and constraints | Scene-owned island manager, source dependencies; don't solve arbitrary global manifolds as though they were independently islanded |
| Solver data | Per-body linear/angular state, inverse mass/inertia, constraint descriptors, contact-point Jacobians, friction patch anchors, restitution, thresholds | Source-derived data layouts and writeback; no one-Jacobian-per-manifold substitute for multiple contact points |
| Task life | Task continuations/refcounts, CPU batch partition, GPU queue dependency, completion, error/abort and frame publication | One scene task graph and device-scoped services; retain source CPU/GPU split; do not wait for every kernel |
| Allocation and retirement | Persistent scene buffers and frame pools, solver constraint arena, contact/friction cache, scratch lifetimes, per-device pipelines | Allocation owners explicit; reuse existing pools and GPU memory services, retire redundant prototype `*_arena` state after integration |

The native runtime must remain buildable with
`FYZZXYZZY_ENABLE_PHYSX_SOURCE=OFF`; source components
are SDK-owned under preserved notices. The separately linked PhysX CPU
reference retains its existing `PxScene` and serves only as an oracle.

## Admitted awake-rigid island connectivity

The native scene now consumes two distinct source contact-edge streams,
without creating a second physics world or actor registry:

| Source producer and type | Native owner and consumer | Admitted semantics |
| --- | --- | --- |
| `BpBroadPhaseSap` candidate pairs, `PxsSimpleIslandManager::firstPassIslandGen` and `IslandSim::processNewEdges` | `Scene::previous_pairs_` → scene-owned speculative union/find state | Dynamic/dynamic candidate connectivity, including non-touching rotated AABB candidates; static endpoints never merge rigid islands |
| `GuContact*` physical touches, `Sc::Scene::setEdgesConnected`, `PxsSimpleIslandManager::secondPassIslandGenPart1` | `Scene::manifolds_` → scene-owned accurate union/find state | Only touching dynamic/dynamic contact edges connect accurate islands; candidate-only pairs do not |
| `Dy::DynamicsContext::iterateIslandsNodes` and `preIntegrationParallel` | Contiguous accurate-island `BodyId` spans → the existing `SolverBodyRow` arena; same spans drive pose integration | Each live awake dynamic body is admitted once; only bodies with contacts allocate prepared solver rows; the original static-world solver row remains separate |
| `IslandSim::processLostEdges`' connected-component result | On source dynamic pair/touch removal, update the same scene-owned graph state | Accurately split or rejoin components at the next published collision state; static-only contact changes do not rebuild dynamic connectivity |

**Fidelity classification:** semantics-preserving connected-component lowering
**only for the all-awake, rigid dynamic/static, contact-only subset**.
Both source island graphs represent components of their respective
undirected dynamic-body edge sets. The native lowering computes those
components with reusable union/find storage on dynamic edge changes,
rather than importing PhysX's incremental handles and dirty-route repair.
Actor identity remains the original scene `BodyId`; no second actor,
force, contact, or solver world owns results. The source's lifetime and
per-frame interleaving behavior is **not** implied by matching component
membership.

**Cost:** unchanged dynamic connectivity and static-only topology
changes incur no component reconstruction. A newly found
dynamic/dynamic SAP candidate merges the speculative graph in
amortized near-constant union/find time; a new physical touch merges
the accurate graph in the same way. Accurate-node ordering is rebuilt
only when a touch merges two previously separate components, costing
O(dynamic nodes log dynamic nodes). On a lost dynamic edge, the native scene now uses the
`IslandSim::processLostEdges` / `findRoute` connectivity invariant:
if the removed edge's endpoints are still connected in the surviving
published graph, their component has not split. The exact published
speculative and accurate edges are searched separately using
capacity-reused transient adjacency, without changing their ownership.
Redundant lost edges therefore preserve both island components and
solver-node order. A proved disconnection, or dynamic node retirement,
still rebuilds both graphs from surviving edges, costing
O(dynamic nodes log dynamic nodes + candidate edges + touching edges).
A route search costs O(dynamic nodes + relevant surviving edges)
in scratch initialization/traversal, without the component sorting
cost, and is triggered **only on dynamic losses**. This is a graph
invariant-preserving lowering, not a claim to have translated the
original persistent `EdgeInstance`, dirty-node `findRoute`,
`mFastRoute`/hop-count repair, dirty-work budget, stable island
handles, or sleep/activation processing. Those remain N1 requirements.
The full-rebuild instrumentation differentiates redundant edge
removal from a true split; allocations are retained between frames.

**Regression boundary:** dynamic three-node merge, split,
incremental rejoin and removal; redundant edge deletion in a
three-node cycle without rebuild, followed by true bridge-edge
deletion forcing a split; isolated free-flight integration;
multiple dynamic bodies touching one shared static world without an
accidental merge; static-only changes without component reconstruction;
and diagonal thin boxes whose speculative AABB graph is connected while
their accurate Gu contact graph is disconnected. The tests distinguish
an incremental speculative insertion, incremental accurate touch
insertion and a complete lost-edge rebuild through scene-owned
instrumentation. They execute through the production `NativeCpu`
path in source-free Vulkan and D3D12 builds.

**Original source phase boundary (still required):**
`PxsSimpleIslandManager::firstPassIslandGen` processes speculative
new/lost edges with `processLostEdges(..., false, false, ...)`; the
accurate graph's `secondPassIslandGenPart2` also processes losses
without deactivation. The source's split/dirty-route repair is gated
by `allowDeactivation` inside `IslandSim::processLostEdges`, and is
scheduled through the speculative and accurate third-pass tasks.
`ScPipeline::destroyManagers` launches
`thirdPassIslandGen` with a common post-third-pass continuation;
`postThirdPassIslandGen` and `afterIntegration` coordinate
sleep/deactivation and publication. The native all-awake rigid subset now separates two synchronous
stages inside the existing `Scene`: first/second-pass speculative
and accurate **edge insertion** after published Gu results, followed by
lost-edge route checking and component repair once lost-contact
products are ready. In a synchronous frame, that third-pass-compatible
repair can run **before** numerical PGS work; the source does not impose
an unconditional "solve, then third pass" order. Geometry-only
collision publication likewise completes both stages before returning.

This bounded staging does not translate `ScPipeline`'s asynchronous
`destroyManagers`/lost-contact task continuations, deferred edge
handle destruction, speculative/accurate deactivation, or
`afterIntegration` reconciliation. It preserves the completed
published graph state for the admitted all-awake subset, **not** source
task/handle identity. N1 acceptance still requires the actual task
dependencies, persistent edge ownership and dirty-route lifetime,
without a second island scheduler or invented sleep behavior.

**Unported source responsibilities:** stable `PxNodeIndex` and edge
handle generations, speculative/accurate activation flags, sleep/wake,
first/second/third-pass task joins, delayed lost-edge retirement,
`afterIntegration` deactivation reconciliation, source callback
ownership, and complete per-island solve/fetch scheduling. The
union/find lowering must be retired or extended when those source
responsibilities are ported; **N1 is not complete**.

## Initial integrated acceptance boundary: real CPU PGS

A PR may be large. It must connect a **whole source execution slice**
rather than add another isolated "helper." For the initial full-rigid-body
CPU PGS admission, require these jointly:

- The ordinary SDK `NativeCpu World::step` uses source-backed
  rigid-body state and a scene-owned task graph through contact generation,
  islands/partitions, preparation, PGS, integration and publication.
  The original scalar manifold solver is **removed from production** (or
  explicitly isolated outside accepted execution) rather than remaining
  a default fallback.
- Support free and constrained translation **and rotation**, with box
  mass/inertia, angular velocity and quaternion pose; nontrivial contact
  locations must produce torque. Use source contact points, friction anchors,
  material restitution and effective mass; test rotational box collisions,
  stacking, diagonal slide and restitution.
- Source-relevant lifecycle: multiple scenes and bodies, creation/removal
  during ongoing multi-frame simulation, pair/touch found/lost,
  sleeping/wake and reset/reinitialization, no-contacts integration,
  finalization and correct fetch/report ordering. Real callback/CCD
  paths are separately scoped but must be **explicitly unsupported** until
  their source behavior is integrated; no silent approximate admission.
- Compare with independent PhysX CPU for the **same** scene settings and
  inputs, including different creation orders, static/dynamic mass ratios,
  extreme finite velocities, varying timesteps, contact loss and
  frictional/rotational motion. Reject any NaN or infinity **before**
  tolerance comparisons. Document differences and permissible tolerances;
  numerical similarity alone is not proof of source correspondence.
- Check reference-free Linux/Windows compilation, reference-enabled
  Linux/Windows execution, CI architecture contract and independent
  Vulkan/D3D12 builds. Verify memory lifetimes, duplicate owners and
  runtime cost before claiming full integration. Godot acceptance can
  be finalized once the native scene has the full state/API.

**Do not mark N1 complete** solely upon this first rigid-body PGS slice:
exact narrowphase breadth, collision filtering, CCD, articulations,
TGS, GPU scheduling and Godot consumption remain separately auditable
requirements. Conversely, do not call the project "nearly integrated"
when these edges are not connected.

## Integration order by dependency, not by helper count

The source-derived components must be prepared **together** and integrated
on the *actual production call path*:

1. **Source intake and state:** identify the required original source
   headers/types/build dependencies and author an ownership-preserving
   translation of actor/shape/contact/solver-body/island records. Do not
   implement dynamics on an alternate set of classes just to compile.
2. **Execution wiring:** implement the source task dependencies and
   original CPU PGS solve state within the native scene. A synchronous
   CPU dispatcher is permissible when it preserves every relevant
   dependency and publication point. It is **not** permission to invent
   a different physical solve order or bypass islands/constraints.
3. **Replace provisional algorithms:** port full admitted contact
   generation/correlation, normal/friction constraint preparation,
   solver iterations, integration, and per-frame fetch; retire old
   one-normal-per-manifold PGS and incomplete SAT face/box path as their
   actual source replacements become live.
4. **Prove it as one integrated engine:** executable native-vs-reference
   parity, adversarial cases, lifecycle, no-PhysX build and Godot behavior.
   Report missing cases honestly; don't accept a suite of helper-only tests.
5. **Then N2:** wire existing translated GPU SAP/AABB stages into the
   same task graph and scene-owned buffers, source-ordered across Vulkan
   and D3D12. The source GPU stage fixtures remain until true frame-level
   multi-frame pair history can retire them.

This is an **integration boundary**, not a mandated number of PRs.
Any partial PR that cannot cross it must clearly state what production
entry point is wired and which unported source dependencies block
completion; such PRs are not accepted as "the last missing piece."

## Required review packet for source-derived dynamics

For every code PR in this slice, include a source correspondence matrix
identifying the original **methods and data types**, the native production
caller, the exact input/output ownership, numerical precision/ordering,
temporary representation and retirement plan, and the conformance tests
covering the connection. For any implementation difference, classify it as
source-preserving translation, proven semantics-preserving lowering,
or **not yet ported**. The last category cannot be claimed production
equivalent. CI success and a clean Codex review are necessary but never
a substitute for this mapping.

**Stop condition:** If the proposed step appears to demand a second scene,
duplicate contact cache, per-stage scheduler, invented friction model or
reference PhysX runtime dependency, stop and revise the extraction boundary
instead of patching around it.
