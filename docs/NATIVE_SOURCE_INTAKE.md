<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# N1 source-faithful native dynamics intake

**Status: source-derived Gu contact geometry, Dy per-point normal/friction constraints and rigid-body constraint partition ordering active in the native scene. Per-shape compliant spring-damper PGS is connected; heterogeneous per-contact material correlation, island task execution, shape-interaction reports and full N1 acceptance remain open.**

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

**Contact-generator admission.** The current native box-box and plane-box
paths translate original non-PCM `GuContactBoxBox` and `GuContactPlaneBox`.
PhysX enables `PxSceneFlag::eENABLE_PCM` by default, selecting a different
contact-generator family. `ExecutionMode::ReferencePhysXCpu` preserves
that default; `ReferencePhysXCpuLegacyGu` explicitly selects the original
non-PCM reference generator for like-for-like Gu/PGS conformance fixtures.
Passing the latter does not establish default PCM contact/cache/point-selection
parity. PCM support requires its own source translation and independent tests.
Track that near-term N1 follow-on in [issue #114](https://github.com/thomthehound/FyzzXYZZY-SDK/issues/114);
it blocks any assertion of default native contact parity and precedes N2
native GPU scene admission. The reference adapter verifies the *actual*
`PxScene::getFlags()` contact mode on creation, so accidentally converting
the stock oracle into legacy-Gu mode is an immediate runtime error.

These paths identify the **initial source intake surface**. They do **not**
authorize compiling all of PhysX into the native SDK. Extract/translate
needed code together with its genuine input, output, allocation and lifetime
contracts, retaining source attribution. Neither rewriting the mathematics
from memory nor a hard-coded equivalent-looking "demo solver" satisfies
this contract.

## Enhanced-determinism Dy task admission

`PxSceneFlag::eENABLE_ENHANCED_DETERMINISM` is now an opt-in
`WorldOptions` input forwarded to both the original independent PhysX
CPU scene and the native scene. Its specifically source-derived
`DyDynamicsContext::updatePostKinematic` effect is to terminate each
solver task chain after one accurate active island, bypassing the default
128-rigid-body / minimum-one-contact batching rule. The existing
`SolverTask` partition, iteration-count and solver-body offset
boundaries are reused; no second scheduler is created. The original
`createFinalizeConstraints` sets `maxBatchSize=1` for the flag,
which is already the native implementation's scalar per-manager
contact-descriptor treatment. This is **the admitted CPU rigid-subset
scheduling branch**, not a claim that every other PhysX
enhanced-determinism feature or packed SIMD ABI is complete.
Source-free tests separate four differently configured islands,
including a no-contact island, and verify iteration maxima,
partition boundaries and normal-force writeback. Reference-enabled
tests compare the enhanced native and original PhysX CPU scenes over
varying timesteps.

## N1 coupled frictional-rotation source parity gate

A new native-versus-original-PhysX legacy-Gu fixture compares two
frictional rigid boxes over a static support, two inverse mass orderings,
two coupled contact managers (floor/lower and lower/upper), rotated
contact normals, and a friction-disabled
PhysX oracle. The normal-only tests previously missed this coupling.

The **first 16 frames** at 240 Hz are included in the normal reference
CI conformance suite. The original **48-frame** mixed-timestep fixture
remains a reproducible **strict N1 acceptance gate**, enabled by
`-DFYZZXYZZY_STRICT_RIGID_FRICTION_CONFORMANCE=ON`. This longer
test is NOT accepted as passing: on reversed actor admission, the
frame-20 upper body had native/source vertical velocities of
approximately 0.051/0.927 m/s. Native Gu reported 3 upper/lower
contacts at roughly 0.07 normal impulse and 0.00007 m penetration;
the original PhysX legacy-Gu scene reported 3 contacts at roughly
1.02 impulse and 0.0163 m penetration. The lower body's static-floor
contact impulses also differed by roughly 0.21 versus 1.65.
The separate frictionless source control did not reproduce the spike.

Turning off source strong-friction retention in **both** engines
reproduced the same gap, so a cached-anchor explanation has been
disproven. The original Gu box-box generator and native source
translation have identical post-entry algorithm lines, aside from
contact-buffer type adaptation, so the outstanding investigation
must compare contact-manager ordering, source geometry/state
publication, per-iteration PGS integration, and precise floating-point
behavior before assigning root cause. No tolerance widening or
substitute solver makes the extended test pass. N1 cannot close with
this strict gate failing.

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

The active native contact path translates the rigid single-shape `PxsMaterialCore.h`, `PxsMaterialCombiner.h`, `DyContactPrep.cpp` and `DyContactPrepShared.h` material branches: every admitted shape owns its material, whose static/dynamic friction, signed restitution (including compliant stiffness), damping, independently ranked combine modes, acceleration-spring flag and friction/strong-friction disable flags flow into the existing Gu→Dy contact-manager preparation. The `PxsMaterialCombiner` special cases for single-compliant and mixed force/acceleration spring combinations are active. Ordinary restitution and the default material path remain unchanged; compliant contacts retain source spring equations in both position and velocity passes and its `DySolverConstraintsShared.h` impulse-multiplier recurrence on every solve. The `staticFriction >= dynamicFriction` promotion and disable-friction omission are preserved. Public `World::set_body_material` modifies the existing NativeCpu Scene actor; the optional real PhysX CPU backend assigns a `PxMaterial` on the existing `PxShape`. Neither mode owns a parallel material table. Native Dy-row and source-enabled real PhysX parity regressions cover the same inputs. Multi-material shapes, per-contact material assignments, packed SIMD ABI and source event callbacks remain unported.

The active PGS preparation now uses the original CPU `DyCpuGpuBiasCoefficient::rigidContact = 0.8f` for both normal penetration correction and friction relaxation. Cached friction normals live in both actors' local frames and survive correlated contact motion; source `DyContactPrepShared::pointsAreClose` uses **absolute** projected anchor separation to retire patches on either excessive compression or positive separation. These numerical and lifetime boundaries are covered by the current draft review, without declaring full source parity.

The same native source reporting path now also implements the low-level **Dy force-threshold stream**: `DySolverConstraints.cpp::writeBackContact` supplies final contact-normal impulses, `DyThresholdTable` aggregates by persistent rigid-body pair, and `DyDynamics.cpp::PxsForceThresholdTask` determines current versus previous threshold exceedance. The current scene has one shape per BodyId but must preserve **two different keys**: the individual `Gu` `BodyId` pair owns geometric contacts, while the low-level `DyThresholdTable` aggregates ALL static-shape contacts on one dynamic body through the shared world-node key. Dynamic/dynamic contacts retain their canonical dynamic body pair. An explicit `ForceThresholdKey` distinguishes these cases without inventing a second island/shape registry. Mapping different static managers to one world key can disturb Gu sort order: the current active threshold candidates are sorted in their reusable arena, combined in place, and previous/current exceeded streams then merge linearly, without per-contact allocations. Dynamic actors opt in with `Scene::set_contact_report_threshold` (source default `PX_MAX_REAL` disables), the pair threshold uses `min(body0, body1)`, and an exceedance requires `sum(finalNormalImpulse) > reportThreshold * dt`. `force_threshold_changes()` emits Found/Persists/Lost states; LOST has zero impulse and PERSISTS carries the prior stream's impulse, matching source stream construction. Collision-only contact progression invalidates visible reports, but cannot erase previous-step exceedance history. Native tests cover these transitions, force-versus-impulse units, dynamic/dynamic minimum thresholds and contact loss. A two-static-support fixture independently measures both Gu support impulses, places a threshold strictly between the largest individual impulse and their sum, and requires a **single world-node Found report** with the combined impulse. Even when both supports exceed a zero threshold, it requires only one event; removal of both supports produces exactly one Lost report.

This is a **bounded low-level source subset**, not the whole `PxSimulationEventCallback` surface. Original `eHAS_FORCE_THRESHOLDS` interaction flags, typed persistent `PxNodeIndex` identities and their full lifecycle (as distinct from admitted native island handles), multi-shape shapeInteraction objects, contact-report pair flags, `ScNPhaseCore` event dispatch and complete material-dependent callback publication remain open N1 work. The explicit dynamic-pair/static-world threshold key is a native surrogate for the source island node pair, not a new island or shape registry; wide `double` impulse totals remain an SDK extreme-mass extension where original PhysX uses `PxReal`.

The native scene now also executes the per-contact **normal-impulse writeback** portion of `DySolverConstraints.cpp::writeBackContact` after the final velocity iteration. It copies each source-derived constraint's accumulated normal impulse into one reusable output arena indexed by the original `Gu` contact offset, independent of `DyConstraintPartition` ordering. Geometry-only `advance_collision()` invalidates previous solved values; the next `step()` replaces them. Native regressions check impulses against rigid-body velocity changes, original contact offsets, source partition rollover and contact retirement. Output uses `double` only for the SDK's previously admitted extreme inverse-mass translational cases; original PhysX writes `PxReal`. The cached friction patch's `broken` flag now follows the original `DySolverConstraints.cpp::solveContact` and `solveContact_BStatic` lifecycle: an iteration-local clamp result is collected during each friction solve, but **only the final velocity/writeback iteration** publishes it. A position-pass clamp that subsequently converges does not invalidate the next frame's correlated local anchors. The 600-frame dynamic stack provides a decisive recovered-anchor case: the ground/lower-box patch saturates during a frame-86 position pass but does not clamp at final writeback. Its `broken` report must be false and its cached local anchors must survive through frame 87. A controlled negative test restoring the earlier sticky `broken |= clamped` rule failed this exact assertion in both independent Linux Vulkan and Windows D3D12 suites; the negative mutation was subsequently reverted. A broader low/high slip sweep retains additional Coulomb-limit consistency checks, but cannot alone distinguish final-only break reporting from a stale clamping history. This test does not claim full `DyFrictionCorrelation` parity or heterogeneous materials.

The same Scene-owned final writeback now also reconstructs `DyContactPrep.cpp::writeBackContactFriction`'s two-tangent per-anchor world impulse vectors, preserves correlated friction patch ownership/anchor positions and reports broken state through `Scene::solved_friction_patches()`. Both report streams are invalidated by geometry-only collision advancement and republished after stepping. Distinct-mass/opposite-slip fixture responses validate the signed friction impulses against body velocity changes. A separate deliberately **permuted** three-manager contact graph exercises Gu order `AB, AC, DE` versus Dy partition order `AB, DE, AC` and validates the force writeback against distinct leaf momentum changes, correcting the inadequate sorted-star-only regression. The SDK's extreme-mass reporting extension retains `double` impulse components where the original source writes `PxReal`/`PxVec3`.

Shape-interaction threshold callback/event dispatch, heterogeneous material-dependent writeback, exact source contact/patch mapping, packed descriptor publication and full scene report dispatch remain unported.

**Source contact impulse and penetration limits:** The same
`ScBodyCore/PxsBodyCore` authoring now admits
`PxRigidBody::setMaxContactImpulse` and
`PxRigidBody::setMaxDepenetrationVelocity` on the real dynamic body.
`DyContactPrepShared::extractContacts` takes the minimum of both
body-contact impulse limits and applies it independently to each
`SolverContactPoint`; `DySolverConstraints` clamps the accumulated
normal impulse before changing any linear/angular solver velocity.
With a zero source cap, contact extraction contributes **no**
solver contact rows, solver-order entries or friction patches for that
manager, while Gu touch events and the source contact-manager edge
remain alive. Dormant managers likewise need no Dy contact-row placeholders.
The native one-shape-per-actor lowering packs only admitted active solver
rows contiguously, while explicitly zeroing omitted managers' original
Gu-indexed force slots during writeback; later live managers retain their
correct compact Dy offsets and independent Gu offsets. This does not
treat a zero cap as contact separation. `DyContactPrep` separately computes
`max(data0.penBiasClamp,data1.penBiasClamp)`, the negative of the
smaller positive depenetration velocity, and uses it to clamp position
bias before PGS response. The original unrestricted defaults remain
unchanged. Public settings are checked before mutation and also
forwarded to actual reference `PxRigidDynamic` bodies. Regressions
exercise independently saturated multipoint plane contacts
(their total must exceed a single point's cap), the minimum of two
dynamic body limits, zero-impulse response without touch loss, compact
zero/live/zero solver-row transitions with exact Gu force indexing,
restricted penetration recovery and invalid-input rejection. Complete contact
modification, per-point impulse overrides and independent packed
contact stream extraction remain outside the current rigid subset.

**Source rigid body preintegration and per-task iteration control:** The
public `World::set_body_dynamics` now authors `ScBodyCore/PxsBodyCore`
linear/angular damping, the gravity-disable flag, maximum linear/angular
speeds, and the packed 8-bit position/velocity iteration counts. These
inputs travel through the single authoritative actor to the existing
native `DyBodyCoreIntegrator::bodyCoreComputeUnconstrainedVelocity`
translation, including the same damping-before-speed-clamp progression
and per-body velocity-limit squaring. The independent CPU reference
adapter authors the corresponding real `PxRigidDynamic` parameters,
so comparison does not rely on a shadow source model.

`DyDynamics::preIntegrationParallel` accumulates **maximum** position
and velocity iterations from each source task's participating bodies.
The native `DyDynamicsContext::updatePostKinematic` source task chains
now retain that per-task maximum, the exact manager/partition descriptor
range, and the prepared solver-body range. The actual PGS dispatcher
runs each chain independently through its own biased position passes,
position motion snapshot, and unbiased velocity passes, without imposing
an unrelated group's higher iteration count on the entire scene.
The source `DySolverControl::solve` friction policy also scales with the
per-task count: only the final three position iterations (or all of them
when fewer than three) enable friction by default, followed by velocity
solves. This replaces the accidentally 4+1-specific "skip iteration
zero" rule. `NpRigidDynamic::setSolverIterationCounts` permits **zero
requested velocity iterations** (but requires at least one position
iteration). `DySolverControl::solveV_Blocks` still performs one mandatory
final velocity/writeback solve when that request is zero. The native task
descriptor preserves the exact requested maximum, while its PGS execution
uses `max(requestedVelocityIterations, 1)` for the final pass. The
default 4+1 behavior remains unchanged. Native tests exercise 130
independent floor-contact bodies across the 128-body task boundary,
including one task with zero requested velocity iterations. That
task's dedicated test contrasts zero versus one requested pass under
identical closing contact velocities and an intentionally
**non-bouncing penetrated contact**, so `DyContactPrep` retains its
position-phase penetration bias instead of clearing it for restitution.
The final unbiased solver velocity must differ measurably from the
saved position-phase motion velocity, while zero and one requests must
produce the same final velocity and Gu-indexed normal impulse. Merely observing positive
normal-impulse writeback is insufficient, because position iterations
already generate force-buffer values. Additional cases cover authored gravity,
damping, speed-clamp, invalid-input and static-body rejection. The
source-enabled fixture additionally compares native free-body velocities
to a real `PxRigidDynamic` using the same public settings. This is
source **serial task-chain execution**, not a claim that asynchronous
ScPipeline continuation objects or packed SIMD descriptors are complete.

The native solver also imports the **rigid-body** ownership/ordering slice of `DyConstraintPartition::partitionContactConstraints`: actor-owned 32-bit dynamic constraint masks, 32-partition overflow waves, per-body static contacts after dynamic constraints, stable counting-prefix ordering of authoritative contact managers and reusable scene-owned arrays. Both normal and friction rows execute in that order within each original PGS pass. A 35-edge dynamic contact star tests partition rollover; twelve unrelated boxes contacting the shared zero-mass static floor verify there is no false partition conflict. The original `DyConstraintPartition.cpp::initSolverProgress` initializes scheduler progress for every active source partition input, even when `DyContactPrepShared::extractContacts` removes all response rows because either body has a zero `maxContactImpulse`. Native `Scene::prepare_solver_bodies` now resets the three partition fields on such active but unprepared solver bodies during its existing active-island-node traversal. The accurate island and Gu manager remain admitted; only heavy Dy point/body preparation is omitted. A 160-frame dynamic–dynamic regression verifies that their contact descriptor remains scheduled without accumulating partition bits, producing phantom Dy rows, or corrupting Gu force writeback; it also exercises zero-to-live-to-zero cap transitions and mixed one-sided/two-sided caps. No extra dormant-body scan or parallel partition owner is introduced. This path is serial; it does **not** claim the complete source island-manager lifecycle, per-island task graph, parallel batch execution, rigid-body static SIMD specialization or articulation partitioning.

The existing scene now also applies the admitted rigid-body `DyFrictionCorrelation` pipeline: `createContactPatches` retains contiguous same-normal/material Gu runs; `correlatePatches` links those runs in source reverse-chain order to cached or newly created friction patches; `getFrictionPatches` validates unbroken body-local anchors against current relative normals and absolute separation; and `growPatches` retains adequately separated two-anchor caches or rebuilds anchors from the linked contact runs. The `DySolverConstraints` PGS loop consumes each correlated patch's normal rows immediately before its friction rows, while source Gu offset indexing remains authoritative for contact-force writeback. The solver header normal is selected once from the current reversed contact-patch correlation head and used for every correlated normal Jacobian and friction tangent, independently of the historical cached body-local friction normal; the noncontiguous-run regression exercises this distinction with near-aligned normals and reused anchors. Reused per-scene scratch and a mapped row-order vector avoid per-contact allocations and redundant per-patch force rescans. A two-manager fixture checks multi-contact anchor formation, material-specific friction disablement, and cache re-entry; the existing 600-frame stack and velocity-writeback regressions continue to cover solver dynamics.

This is **not** full source solver admission: heterogeneous per-contact material groups and target velocities, strict `DyCorrelationBuffer` SIMD/packed layout and all geometry/mixed-material cache permutations, packed solver constraint ABI, exact `PxReal`/SIMD rounding, full source island/task scheduling and batch specialization, and full shape-interaction event/callback publication remains unported (the opt-in rigid-touch subset is now admitted). The admitted two-material compliant/noncompliant combine modes, force/acceleration spring rows, final per-contact normal/friction writeback and low-level threshold-change stream are implemented; these do not constitute full reporting. The native path is a transitional rigid-body contact/material subset, not an accepted PhysX CPU replacement or N1 completion.

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
   The current serial CPU lowering completes third-pass deactivation
   **before** admitting Dy solver bodies. Those deactivated nodes
   therefore cannot integrate in that frame, satisfying the observable
   state transition without running a redundant pose rollback.
   When solver and island execution overlap, the original
   `afterIntegration` compensation and its dependency join must
   be reinstated and verified against that actual overlap.

The admitted serial CPU regression now verifies the precise
ready-on-prior-frame → island-deactivation → no-Dy-integration
transition, unchanged pose and zero motion, dormant next-frame state,
and an authored wake re-entering Dy. Separate 96-frame native-versus-
original-PhysX CPU free-flight comparisons cover changing timesteps,
unequal mass, rotation and damping; 40-frame Sc contact lifecycles
compare Found/Persists/Lost publication against both original reference
modes. The numerical free-flight test uses explicit positional,
velocity, angular-velocity and orientation tolerances: this is a
bounded, reference-backed conformance subset, **not** proof of
strict multi-frame contact/PGS trajectory parity, concurrent task
refcount equivalence or complete public `fetchResults` semantics.

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

**Completed Dy result publication:** Scene's solver result generation
is distinct from its Gu collision clock. Gu-only advancement invalidates
the current solver-result generation without pretending to have run a
solver/fetch transaction. A normal rigid PGS step publishes the new
generation only after integration, the destroyed-manager consumer
join and Sc contact-event preparation. If any source preparation, solve,
writeback or event stage rejects the step, the existing scene-owned
checkpoint restores the exact previous completed-result generation,
motion and result allocations. This is a truthful synchronous
publication boundary for the admitted native CPU path, not the full
asynchronous `NpScene::fetchResults` callback or GPU task API.

**Destroyed contact-manager completion join:** `ScPipeline.cpp::destroyManagers`
forks speculative and accurate `thirdPassIslandGen` work and
`processLostContacts3` joins it to the scene's solver completion.
The admitted rigid native scene now preserves a destroyed manager's
retiring `EdgeIndex` allocation through accurate/speculative island
repair **and through Dy contact preparation, solve, writeback and
body integration**. Its serial completion edge returns that handle
to the free pool only after all those consumers have finished and
before the successful frame publishes Sc contact events. This is
the source lifetime dependency; it is not a replacement task
scheduler or an assertion of parallel CPU execution. For a standalone
`advance_collision`, no Dy consumer exists, so post-third-pass
retirement completes inside its collision transaction. Both paths
journal the original handle allocation and intrusive edge metadata
for rollback. Tests check the handle before/after the join, failed
transactions, reuse after completion, and combined lost-touch and
force-threshold Lost reports without replay.
**Remaining copy cost:** native steps now journal only mutated
dynamic-body motion, while the source's retained friction anchors
and previous exceeded-threshold streams use reversible scene-owned
buffer rotations instead of O(cache size) checkpoint copies.
Active-node publication and solver-partition metadata still incur
checkpoint bandwidth. Persistent source island handles,
node membership chains, static-touch counts and accurate solver-node
offsets are **now copied only on the first actual graph ownership
mutation in a publication**, never on an unchanged-topology PGS step.
The original topology is retained by a capacity-reusing checkpoint
until the publication succeeds; failed attempts restore it by
allocation-preserving swaps. Source intrusive `EdgeInstance` links
and node-first heads are likewise **not copied per frame**:
an allocation-reusing write-ahead journal records only the
edge slots and heads about to change. On exception, reverse replay
restores original adjacency; newly allocated slots are truncated.
The active/retired/free handle arrays are copied lazily, only when an
overlap is added/removed or a handle is actually retired. Gu-only touch
changes do not copy those arrays. The transient next-edge merge buffer
is never checkpointed; it is cleared and reconstructed on retry.
Journal recording can allocate and throw *before* mutation, while
rollback and publication-buffer restoration cannot allocate.

`Scene::advance_collision` now shares the strong failure boundary:
an AABB overflow or an exception partway through contact/edge
publication restores prior published Gu/contact/impulse spans,
previous force reports, edge ownership and island state. Geometry-only queries deliberately omit dynamic-motion,
friction-history, threshold-history and solver-partition copies
because that operation does not mutate them. For an unchanged scene
without a forced refresh, pending island repair or edge retirement,
`advance_collision` takes the constant-work quiet publication path:
it advances the frame clock and retires transient result/event streams
without checkpointing edge slots, handles or island arrays. Dirty,
forced and outstanding-retirement cases still use failure-atomic
collision/island/edge staging. Regressions verify quiet publication
does not change persistent contact/island state, and preserved
backing-storage identity after a deterministic nonfinite AABB with
successful retry.

The accurate graph's `mActiveNodes` and
`mNodesToPutToSleep` equivalents are also published read-only spans.
Their transaction now follows the same reversible-buffer ownership rule
as solver reports: the checkpoint retains the active-node permutation,
reverse index and deactivation output on each transaction, reserves
full per-actor capacity for the working arrays, and swaps the
existing published arrays into the checkpoint. Persistent island
membership is copied separately and lazily, only before a real
graph owner mutation. Failed frames restore those original allocations,
not copies, so retained spans continue to see the last successful
publication. Preserving the full deactivation capacity across repeated
rollback also keeps the subsequent `noexcept` third-pass from allocating
after the publication boundary. This does not introduce a second
island owner or double-copy the active-node lists.

The failure-atomicity audit also covered two transients that must not
borrow authoritative publication lifetimes. Prepared Dy rows now carry a
separate monotonically increasing preparation generation: a rejected
step restores its frame index but cannot accidentally reuse rows from
that failed attempt if contact geometry changes before retry. The
preparation generation does not roll back; its rare wrap clears stale
row markers. The complete Gu manifold arena continues to be restored
by swaps when the contact epoch advances. When a quiet step never swaps
the arena, a generation-stamped journal records only those Gu managers
whose exposed Dy descriptor fields (solver row offset, friction patch
offset/count and preparation epoch) actually change. The journal stores
pre-step metadata once per touched manager and restores it on rejection
without duplicating contact points or manifold storage. Dormant
managers retire stale solver metadata once, instead of publishing
perpetually changing offsets on every subsequent quiet frame.
The quiet-path snapshot now costs O(mutated Gu managers) rather than
O(all published managers), including after hundreds of dormant
contacts accumulate. Published
Found/Lost event streams likewise rotate reversibly on a quiet step,
even though the pair and contact arena epochs remain unchanged.

An authored wake-work indicator avoids traversing all speculative
island handles on an unchanged collision query with no new edge or
contact events. It is saved in the step checkpoint and restored after
rejected frames. This does not replace the source's explicit activating
node/edge lists and active-index arrays.

A reference-free regression checks that eighty steady free-flight
PGS steps do not trigger any island ownership snapshots, while dynamic
actor admission does. The existing failing-solver local-split
regression additionally requires the snapshot diagnostic to roll
back and a successful retry to match the clean-scene count.
These are structural, not wall-clock, cost assertions: active-index
publication and solver-partition metadata still incur per-step copies.
Motion uses a sparse journal; friction and threshold caches use
reversible buffer ownership, preserving the original published state
when a solve is rejected.
That remaining dynamic-scene checkpoint cost is a
performance-remediation item, not the final source publication
architecture. As the original `ScPipeline`/`Dy::DynamicsContext`
continuation and fetch boundaries are integrated, lower the
remaining actor/cache checkpoint into staged writes and source-owned
publication. Do not add blocking waits or re-run the simulation
to emulate failure atomicity.

**Active-only Dy partition and threshold candidates:** The source task
descriptor stream now uses generation-stamped Gu-indexed partition
assignments rather than writing an unassigned sentinel across the
entire retained manifold arena every PGS frame. Overflow-wave conflict
handling and stable per-partition descriptor ordering are unchanged;
a failed solve advances the scratch generation so a retry cannot
accidentally reuse its assignments. The source force-threshold stage
likewise collects fresh candidates only from managers admitted by
the current active island tasks. It still merges with the retained
previous exceeded-force stream to emit the original Found, Persists
and Lost transitions, including when a manager becomes dormant.
These optimizations remove independent O(all dormant Gu managers)
passes from the source task partition and threshold pipelines.

The native regression exercises multiply-combined finite materials that
overflow during contact preparation, repeated failed attempts, preservation
of previously published contact spans and force/threshold reports,
one-shot torque impulse continuity on a retry, and a late pose-overflow
after an earlier body has integrated. The test must fail if any of those
stages leaks state from an unsuccessful step.

### Rigid-contact solver admission and island-manager ordering

`Scene::add_dynamic_box` may retain extreme finite boxes for the
provisional free-translation path even when their density-scaled source
`PxReal` inverse inertia is not representable. Such a body must **not**
enter `Scene::prepare_solver_bodies` for contact solving: an implicit
zero angular tensor alters the original per-point `DyContactPrepShared`
Jacobians and impulse distribution. The production solver rejects the
unsupported constrained actor, while `StepCheckpoint` restores the
previous frame and its free-flight translation remains usable. The
heavy-density and subnormal-density rejected-contact fixtures exercise
that publication and retry boundary.

**Admitted indexed-manager descriptor order:** The original
`DyDynamicsBase.cpp::iterateIslandsContactEdges` gathers contact
managers by walking accurate `IslandSim` active islands and their
`eCONTACT_MANAGER` edge arrays, not the canonical `BodyId` pair order.
The native accurate `IslandGraph::Island` now owns the corresponding
ordered first/last `EdgeIndex` links; existing persistent contact-manager
slots retain O(1) insertion, source `mergeEdges` concatenation, and
`removeEdgeFromIsland` **swap-last** retirement. No separate contact
registry, copied solver rows or per-manager allocations are introduced.
The actual `partition_constraints` path builds
`indexed_managers_` from active accurate islands and resolves these
handles to the current Gu manifold offsets via generation-stamped
reusable lookup slots. Both dynamic/static partition classification and
stable partition scatter use this source-indexed sequence, while Gu
contact/impulse writeback remains in its original pair order. Rejected
steps recover previously published indexed-manager spans with
allocation-preserving checkpoint swaps. The independent native smoke
fixture creates managers in noncanonical order and verifies the source
swap-last permutation after loss.

**Active-island retirement and rigid solver task grouping:** The
admitted `PxsIslandSim::processLostEdges` deactivation pass now
consumes the accurate graph's `mActiveIslands` in reverse source order,
after wake publication; dormant historical handles are not scanned in
steady-state frames. The graph reconciles sleeping nodes after an actual
deactivation, not through an unconditional second traversal. The rigid
`DyDynamicsContext::updatePostKinematic` descriptor lowering groups
consecutive accurate active islands using the default 128-body
`solverBatchSize` and source minimum of one contact manager, and runs
`DyConstraintPartition` independently within each group. Its
32-partition overflow reprocesses only pending manager endpoints instead
of resetting unrelated solver bodies. The existing single Scene still
executes these disjoint groups serially; asynchronous source task
continuations and heterogeneous constraint groups remain unported.
The native regression checks reverse simultaneous sleep order, quiet
retirement, and 128+2 contact-island groups with full impulse writeback.

**Source-ordered friction preparation:** The active accurate-island
contact-manager stream now drives `DyContactPrep` correlation and
normal/friction-row creation. The original Gu manifold order and
point offsets remain stable for collision history and force writeback.
Retained strong-friction patches are located by the persistent manager
`EdgeIndex`, verified against their original pair identity and the
friction cache epoch. This avoids a global Gu-pair sort when task
descriptor order differs from contact generation order.

Each manager-span index rotates with its friction arena after successful
preparation and reverses with that arena on failed steps. Dormant
managers and recycled handles cannot consume stale anchors. Explicit
shape-pose and offset changes compact the affected patch cache and
reindex surviving managers; ordinary solver frames do not perform
that rebuilding work. Gu-indexed force writeback clears inactive
Dy descriptor metadata without revisiting dormant contacts during
solver preparation. Regressions cover noncanonical task order,
anchor retention after authored edits, and failed publication restore.
**Remaining source ordering breadth:** The admitted non-compound rigid
PGS path consumes the indexed-manager sequence directly, matching
`DyDynamics.cpp::setupDescTask`'s non-compound branch. The
`PGS_SUPPORT_COMPOUND_CONSTRAINTS` alternative performs a separate
stable two-key solver-body remap and contact reduction, and remains an
explicit later source admission; it is not silently approximated by an
unrelated global pair sort. The native source-local lost-edge split
now executes `PxsIslandSim::removeEdgeFromIsland` for each migrating manager
in visited-node adjacency order, then applies `splitEdges`-style
append into its new source-owned island. This preserves the
original/root island's **swap-last-compacted remainder** instead of
replaying managers in the pre-split indexed array order. A linear
five-body split regression distinguishes the resulting source
C-D, D-E order from the previous D-E, C-D fallback.
Local split manager migration seeds each disconnected component's
`mVisitedNodes`-equivalent discovery at the smallest **dirty** rigid
node (the original `mDirtyMap` scan), not at its lowest `BodyId`.
A min-priority heap of the existing hop estimates traverses the
authoritative `EdgeInstance` adjacency; the migration sequence is
recorded **on queue insertion**, as in source `findRoute`, rather than
when elements are popped. An additional split regression makes the
orphan's dirty B node follow its smaller, non-dirty A node in actor
order, and requires B's static manager to precede A's managers in
the new island. New rigid nodes now initialize `mHopCounts` to **zero**, as in
`PxsIslandSim::addNode`, rather than an invented infinite default.
Per-graph hint vectors grow during `addNode`-equivalent island
admission; existing hop estimates are retained across unrelated
actor admissions and backed by the existing journal on rejection.
`PxsIslandSim::processNewEdges` and `mergeIslandsInternal` also
update hop estimates: on merge, the smaller island's member nodes
receive the source `mHopCounts[winnerEndpoint] +
mHopCounts[loserEndpoint] + 1` offset, then the connecting loser
endpoint receives `winnerHop + 1`; a new chord may shorten either
endpoint's route. These writes now occur in the existing loser-node
remapping pass with journaled per-node hints, retaining the original
root-directed connector in the persistent edge slot.

The branched priority regression starts with C-X and B-X already
linked. X has a shorter *merge-derived* root estimate than A, whose
separate A-G island subsequently joins through B-A. B's adjacency
therefore discovers the newer A-B before X-B. Removing the root
bridge B-C and C-X chord in the **same** publication preserves the
source estimates until the split: the hop-priority queue expands X-H
before A-G even though FIFO would do the opposite. The fixture
asserts actual unequal cached values before the loss and H-static
before G-static after migration. It does not rely on an invented
infinite-hop sentinel or on stale values left unadjusted by merges.
For simultaneous splits, dirty nodes are consumed in ascending
`mDirtyMap` seed order. The first failed `findRoute` immediately
splits its component, **reusing the original search's** `mVisitedNodes`
in queue-insertion order. There is no second component traversal and
no deferred list of failed roots. Nodes visited in that split are
marked for the rest of the same graph's dirty-map pass, matching
`processLostEdges`' skip of already-confirmed split members.
Only that component's visited nodes are detached from the original
node chain. The root component keeps
its existing handle and unchanged membership order; the detached
component acquires a free or newly allocated source-style island
handle and a node chain ordered by `mVisitedNodes` insertion. Its
indexed managers are swap-last removed from the original island in
the same visit order, then appended to the new handle. Unrelated
islands are neither traversed nor rebuilt. The full-component scan
remains only for actor admission/retirement and exceptional topology
reconciliation, not ordinary lost-contact splitting. The new
`island_local_split_count` and `island_full_rebuild_count`
diagnostics distinguish those two paths. Shared split witnesses are
now retained within each third pass using generation-stamped scratch.
The third pass now also retains witnesses along validated fast routes,
terminates successful priority searches at an earlier confirmed island
member, unwinds that root/witness route, and propagates hints to the
remaining discovered branches in source `mVisitedNodes` order.
Standalone regressions distinguish split-witness skipping from
successful-search branch repair. The full source `mIslandIds`-based
visited/invalid transition, deferred dirty-node budgets and
kinematic/non-rigid visitation remain fidelity work.
Source `Cm::PriorityQueue` ties are now reproduced by the same binary-heap
push/pop rules: equal-hop insertions do not bubble, and removal selects
the left child when both children tie. The five-way split regression
observes the resulting manager permutation (not FIFO or BodyId order).
Full source witness-driven route reuse, per-pass dirty budgets, typed
`PxNodeIndex` lifetime, and the original continuation graph remain
fidelity gates. Ordinary rigid split-handle allocation is now local.

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

## Admitted rigid island connectivity and sleep lifecycle

The native scene now consumes two distinct source contact-edge streams,
without creating a second physics world or actor registry:

| Source producer and type | Native owner and consumer | Admitted semantics |
| --- | --- | --- |
| `BpBroadPhaseSap` candidate pairs, `PxsSimpleIslandManager::firstPassIslandGen` and `IslandSim::processNewEdges` | `Scene::previous_pairs_` → scene-owned speculative `IslandGraph` with persistent island handles and intrusive node membership | Dynamic/dynamic candidate connectivity, including non-touching rotated AABB candidates; static endpoints never merge rigid islands |
| `GuContact*` physical touches, `Sc::Scene::setEdgesConnected`, `PxsSimpleIslandManager::secondPassIslandGenPart1` | `Scene::manifolds_` → scene-owned accurate `IslandGraph` with persistent island handles and intrusive node membership | Only touching dynamic/dynamic contact edges connect accurate islands; candidate-only pairs do not |
| `Dy::DynamicsContext::iterateIslandsNodes` and `preIntegrationParallel` | Contiguous accurate-island `BodyId` spans → the existing `SolverBodyRow` arena; same spans drive pose integration | Each live awake dynamic body is admitted once; only bodies with contacts allocate prepared solver rows; the original static-world solver row remains separate |
| `IslandSim::processLostEdges`' connected-component result | On source dynamic pair/touch removal, update the same scene-owned graph state | Accurately split or rejoin components at the next published collision state; static-only contact changes do not rebuild dynamic connectivity |

**Fidelity classification:** source-style persistent island IDs, island
handle recycling and intrusive node-membership chains, **only for the
rigid dynamic/static contact-only subset**. The previous
provisional union/find parent/rank arrays have been removed. Both source
island graphs now own independent handles and node-to-island mappings;
the shared contact-manager `EdgeIndex` array owns both adjacency graphs,
not a second actor, force, contact or solver world. Merging components
retains the larger source island's handle and remaps the smaller node
chain. On actor admission/retirement, new rigid nodes are first
admitted into the existing source island ownership and the found edges
are merged in source order, with each merged island's accumulated size
determining its retained handle. Only afterward does the bounded
component traversal reconcile removed edges, retire old actors, and
rebuild intrusive node membership. Thus simultaneous found/lost edges
cannot change retained island identity merely because a new actor was
also admitted. A genuine split still preserves the surviving old-root
component's handle and unrelated islands. The rigid-body active-node
lists and initial-active/activated suffixes are implemented separately
for both graphs; source-local split migration, activated/deactivated
*edge* worklists, non-rigid node types, post-integration scheduling,
and callback interleaving remain open.

The island owner now also maintains the source `Node::mStaticTouchCount`
and `IslandSim::mIslandStaticTouchCount` independently for speculative
SAP candidates and accurate Gu touches. Static-edge found/lost changes
update the owning node and island counters without changing dynamic
connectivity revision; dynamic merges accumulate both islands' static
touch totals, and actual split/node-lifetime rebuilds recount them from
the authoritative surviving intrusive adjacency. A contact with the
SDK infinite floor therefore remains a speculative candidate even
after physical touch is lost, exactly as the underlying edge streams
require. These counters are the source island contact-state inputs for
`Dy::sleepCheck` and whole-island deactivation, now used by the admitted
default-scene non-stabilized sleep path. They remain distinct from the
number of counted Sc contact interactions used by the wake-energy filter.

### Default-scene rigid sleep and wake

Native `Actor` state now retains the original `PxsBodyCore` 0.4-second
creation wake counter, `PxsRigidBody::mSleepLinVelAcc` and
`mSleepAngVelAcc`, and the per-node readiness outcome of
`DySleep.cpp::updateWakeCounter`. The latter consumes each dynamic
body's post-position-pass motion velocity, inverse mass/local inertia,
default sleep threshold (`5e-5 * 10² = 0.005` at the
default `PxTolerancesScale.speed = 10`) and current contact-manager
interaction count. The original creation wake counter is 0.4 seconds;
the source wake filter accumulates motion near expiration rather than
using instantaneous zero-velocity tests. The two speculative/accurate
island owners retain their existing
contacts while a full component becomes inactive; inactive members skip
gravity, constraint submission and pose integration. Explicit velocity,
torque or position authoring reactivates the body, and connected
speculative-island members are woken together **before unconstrained
velocity integration**. New physical touches with static bodies are
additional wake events. Retiring a contact participant or authoring either a static or dynamic
pose also revokes stale readiness in its former Gu contact partners.
This includes the one-frame interval in which the partner is ready to
sleep but not yet deactivated. The explicit authoring path queues
activation before contact removal can split the old island, so a
former sleeping partner does not remain suspended after losing its
support. Quiet queries consume genuine pending wake requests without
scanning the entire island inventory.
These transitions, including rollback of wake counters and readiness,
are compared with the independent `PxRigidDynamic::isSleeping()`
reference for an ordinary settled floor contact.

This is **not** complete `PxsSimpleIslandManager` sleep/wake parity:
the rigid contact-manager `Edge::eACTIVE/eACTIVATING` flags and
activated-edge worklists are now independently owned by the speculative
and accurate graphs. Rigid two-dynamic-endpoint
`IslandSim::Node::mActiveRefCount` bookkeeping is also implemented.
The remaining unsupported non-rigid node-type activation queues,
kinematic consumption of active references, stabilization/freeze state,
deferred deactivation continuations, complete sleep/wake callbacks and
exact `ScPipeline::afterIntegration` task semantics are still not implemented. Independent *rigid* active-index maps, their
prefix/suffix and swap-last behavior are already implemented in both
graphs; this is not a claim of their complete non-rigid lifecycle. The active
subset has no joints or CCD and cannot establish those dependencies.

**Cost:** unchanged dynamic connectivity and static-only topology
changes incur no component reconstruction. A newly found
dynamic/dynamic SAP candidate merges the speculative node chains by
remapping the smaller island's nodes (O(smaller island size)); a new
physical touch does the same independently in the accurate graph.
Source island identities remain stable across quiet frames and redundant
contact changes. Node-lifecycle rebuilds traverse the final live graph
in O(V+E) without a per-incidence search through new-edge streams;
found edges are handled once by the regular source merge operation
before that traversal. Accurate-node ordering is rebuilt only when a touch
merges previously separate islands or a split/node-lifetime rebuild
occurs, costing O(dynamic nodes log dynamic nodes). On a lost dynamic
edge, the native scene uses the
`IslandSim::processLostEdges` / `findRoute` connectivity invariant:
if the removed edge's endpoints are still connected in the surviving
published graph, their component has not split. The scene now retains the original `IslandSim::mEdgeInstances`
per-endpoint intrusive adjacency form directly in each persistent
contact-manager `EdgeIndex` slot: two endpoint links, per speculative
and accurate graph, with per-node first-edge heads. SAP overlap
creation/removal links or unlinks speculative instances; Gu touch
found/lost does the same independently for accurate instances.
Lost-edge route checks traverse this persistent state, using only
visited marks and a reusable hop-priority queue; no temporary adjacency graph or
edge-list reconstruction is needed for each loss. Redundant lost
edges preserve connected components and solver-node order. A proved
dynamic-node admission/retirement still reconstructs island-node
membership from surviving source edge slots, costing
O(dynamic nodes log dynamic nodes + candidate edges + touching edges).
Ordinary contact-loss disconnections instead migrate only locally
disconnected nodes and managers, without a second heap traversal.
The admitted all-awake rigid subset now also follows the source
`IslandSim::removeConnectionFromGraph` → `mDirtyMap` →
`tryFastPath`/`findRoute` sequence: unlinking a dynamic/dynamic
instance queues its rigid endpoints as dirty and clears any direct
`mFastRoute` hint through the removed edge. Third-pass repair walks
those dirty nodes toward their current island root. An advisory
`mFastRoute`/hop-count cache is reused only when **every cached link**
still exists in that graph and hop estimates decrease toward that
root. Otherwise, a graph traversal through the persistent
`EdgeInstance` adjacency finds a surviving root route and refreshes
its path estimates. If root connectivity is lost, the native third pass now detaches
only the failed component into a locally allocated source-style
island handle, leaving unrelated node chains and handles untouched.

When a reconstruction changes an island root, only that root's affected
node hints are invalidated; unrelated actor admissions do not evict
existing cached routes. Failed publications restore route successors,
EdgeIndex handles, hop estimates and stamps through sparse per-slot undo
entries, preserving successful earlier hints on retry without copying
entire arrays. Only dirty dynamic/dynamic edges trigger this work;
static touch loss does not split the admitted awake dynamic components.
Failed fast paths use the source `Cm::PriorityQueue` binary min-heap
ordered solely by estimated hops to the old root. There is **no**
BodyId or discovery-order tie-break: `push` retains equal-priority
parent order, while `pop` prefers the left child on a tie. Outdated
estimates change exploration order, **never** connectivity. In the worst case the traversal visits O(V+E) graph
entries and maintains a O(log V) priority heap per visited node,
versus O(path length) validation for an intact cached route. Visitation
stamps are retained and advanced across publications rather than
clearing all actor marks each frame. Split members are also marked
confirmed for the current source dirty-map sweep, so subsequent dirty
seeds in that same orphan incur no extra route searches. Source
`IslandSim::findRoute` also reuses confirmed witnesses from earlier
successful searches and validates cached paths; those root-witness
and off-path branch operations are now translated for the rigid graph.
The full source typed/invalid island witness-state transitions,
optional dirty-node work budget, and full sleep/activation semantics
are **not yet fully ported**.
The current queue implements the source's hop-priority traversal for
the admitted all-awake rigid graph, not the entire island algorithm.
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
products are ready. In the admitted synchronous CPU path, the complete
pair/touch publication and island connectivity handoff now precedes
`Dy::bodyCoreComputeUnconstrainedVelocity`: newly connected sleeping
nodes must receive their own gravity/damping preparation in the same
frame in which `IslandSim::wakeIslands` activates them. This removes
the earlier split wake sequence (old-island wake before gravity, new-touch
wake after gravity) without a second actor or solver owner.

The source-derived active-island handoff is now explicit:
after third-pass activation/deactivation the synchronous CPU scene
maintains the accurate graph's `IslandSim::mActiveNodes` and
`mActiveNodeIndex` rigid-node lowerings, using source-style swap-last
retirement, append activation and a dense reverse map. The third pass
also emits `mNodesToPutToSleep`-equivalent accurate rigid deactivations
only when a previously ready island actually sleeps. These lists reside
in the same persistent accurate island owner and participate in its
rollback transaction. The dispatch then builds indices into the
existing accurate-island node batches, mirroring
`IslandSim::mActiveIslands`. Unconstrained velocities, prepared-contact
solver-body admission, and pose integration consume those active batches,
not the entire actor array or dormant graph islands. Index storage is
reused, and no copied actor registry or second island owner is introduced.
A regression verifies the dormant/awake split, free-flight integration
and subsequent wake admission. Rigid per-node active-index permutations
are now retained through append activation, swap-last deactivation,
and rollback in both graphs. Source rigid contact-manager
`mDeactivatingEdges` output is now lowered for both graphs as a
separate sleep-lifetime event: walking each newly inactive node's
intrusive links emits the edge only when its other endpoint is already
inactive or static. The first sleeping endpoint of a dynamic/dynamic
accurate contact does not emit it; the second emits once. The speculative
graph can contain AABB-only links between independently sleeping
accurate islands: it emits **no edges** while its own island retains an
awake member. Both newly sleeping accurate members **and** roots of
components detached by speculative third-pass lost-edge repair are
candidate triggers: a SAP-only bridge can retire its final awake member
without any new accurate sleep event. Only those locally touched island
roots are examined, deduplicated with generation stamps; there is no
global island scan. On the resulting speculative island deactivation,
its original island-node and EdgeInstance order emits each static/dynamic
or dynamic/dynamic edge once. Destroyed/lost
contact edges are *not* sleep events, and manager handles remain live.
The one-frame outputs are graph-owned, geometrically capacity-reserved
before the no-throw deactivation stage, and restored on failed
publication. The distinct-sleep regression first asserts that accurate
floor deactivation does not prematurely emit speculative floor edges,
then removes the SAP-only bridge and requires the formerly sleeping
component's speculative floor edge to deactivate in that new publication,
without a new accurate sleep event or a repeated quiet-frame event.
The source rigid contact-manager active/activating flags, active-edge
counts and activated-edge worklists are now preserved independently by
both graphs. Non-rigid node and constraint edge types, callback consumers
and concurrent task continuations remain unported.
The graph-owned accurate active-node index also reconciles
on successful geometry-only publications (including new admission,
retirement and quiet authored wakes). Its reusable active-ID capacity
and reverse map are prepared by the topology rebuild, so that no-throw
handoff cannot allocate after the editor collision transaction commits.
The last solver active-batch view remains a separate successful-step
publication. Each successful quiet collision publication also clears
the previous one-frame accurate rigid deactivation output without
performing a third-pass island traversal; otherwise the old sleep event
would be observable again against a newer frame index.

The accurate rigid graph also retains
`IslandSim::mInitialActiveNodeCount[eRIGID_BODY_TYPE]`, captured before
the wake/publication pass. Its `markInactive` lowering relocates
removed initial members to the prefix tail before swap-last removal,
keeping `getNodesToActivate` as a genuine suffix. The surrounding
index traversal rechecks the original compacted slot after every
removal; multiple initial rigid nodes can deactivate in the same pass
without skipping a relocated dormant member. A two-island simultaneous
sleep/new-admission regression tests both numerical and collision-only
publication and subsequent authored reactivation. Topology rebuilds
retain the active-node permutation and its prefix rather than
reconstructing active nodes from actor sleep flags. The count is
checkpointed alongside the source active-index map.

The speculative IslandSim now independently maintains
`mActiveNodes[eRIGID_BODY_TYPE]`, `mActiveNodeIndex`, and
`mInitialActiveNodeCount` with the same source append/swap-last
mechanics, rather than leaving its active-node list empty. These are
**separate graph-owned permutations** over the one Sc actor sleep state:
one scene-owned no-throw synchronization walks both graphs; each
maintains its own reverse indices, originally active prefix, and
newly activated suffix. Quiet authored wakes and collision-only
publications update both. Before a potentially failing publication,
the checkpoint reserves full actor capacity for both working active
lists and swaps the prior published allocations into its undo state;
a failed topology rebuild restores both original list pointers and
index maps. A standalone regression exercises two ready-to-sleep
islands, a quiet wake and non-replayed suffix, dynamic actor admission,
rejected rebuild, successful retry, and retirement. The existing Dy
body admission continues to use **accurate** active island batches,
not speculative SAP-only contact candidates.

This translates the admitted *rigid active-node index* contract for
both IslandSim owners, alongside their independent rigid contact-manager
**active/activating edge flags, activated-edge worklists, and active counts**.
Complete kinematic/non-rigid deactivation, constraint-edge activation,
active-reference accounting, original per-type partitions and task
continuations remain explicit acceptance requirements. The rigid
Sc sleep state is not duplicated as an independent speculative scheduler.

The admitted serial dependency chain is **contact publication → first/second
island connectivity → third-pass lost-edge repair → whole-island wake
→ third-pass deactivation from previously published readiness →
unconstrained velocities → solver preparation/PGS → integration and
readiness for the next third pass**. The third-pass-compatible
repair can run before numerical PGS work; the original source permits
speculative/accurate third-pass tasks to overlap the solver and join
at post-third-pass. The admitted serial ordering is a valid
dependency-ready lowering, not an implementation of its actual
continuations or concurrent task graph. Geometry-only collision
publication also completes the shared source-island wake stage before
returning: an editor contact query may introduce a new Gu static touch
while the body is asleep, and the resulting activation must not be
dropped by a later quiet query. The activation operation is no-throw
and runs after collision-only checkpoint commit, preserving the
existing low-cost geometry-only failure boundary. Full `step()`
reuses the same operation before gravity, damping and Dy submission.

The current synchronous CPU scene consumes prior-frame
`Dy::sleepCheck` readiness at the third-pass boundary instead of putting a
body to sleep in the same frame its PGS solve first marks it ready.
`PxsIslandSim::processLostEdges(allowDeactivation)` owns the original
per-island test. The admitted native scene now applies rigid-body
deactivation to accurate island membership, leaving speculative-only
connections to govern broadphase/contact interaction activity; it no
longer treats every awake speculative member as a wake event. Explicit
sleep-to-wake authoring queues an actor-local activating flag, and
new speculative/accurate edges activate their affected islands during
the collision-publication handoff. Quiet collision queries also consume
pending authoring activation. That flag is part of rollback state.
`ScPipeline::afterIntegration` restores prior poses
when solver and island tasks overlap. Serial CPU deactivation before PGS
avoids the concurrent integration that requires restoration in the original
schedule. Regressions require a visible ready-but-awake frame followed
by an inactive frame with an unchanged pose, and verify that a newly
published Gu touch clears the readiness of an **already awake** node
before third-pass deactivation. Quiet island propagation must not clear
readiness every frame, or no island would ever fall asleep. This implements the admitted rigid active-node index, activated suffix,
and accurate deactivating-node output, alongside the admitted
rigid contact-manager deactivating-edge output in both graphs,
and now the source rigid contact-manager activated-edge queues and
active/activating bits, but **not** kinematic rules, active-reference
bookkeeping, callback lists, or the parallel
third-pass/post-integration join.

**Persistent contact-manager EdgeIndex (admitted source subset):**
`PxsSimpleIslandManager::addContactManager` allocates one `EdgeIndex`
for each created SAP contact-manager overlap. `native::Scene` now owns
the corresponding indexed edge slots and reusable handle pool alongside
the original sorted pair stream; it does **not** allocate a separate
collision graph or duplicate the physics actors. A single linear
old/new SAP merge publishes found/lost events, carries forward
identical handles for surviving pairs, and allocates handles only for
new overlaps. Source `setEdgeConnected`/`setEdgeDisconnected`
semantics are represented by an accurate-connection bit on that slot,
updated from Gu found/lost touch events. Both speculative and accurate
island connectivity now consume the same persistent edge slots.
Loss of physical touch does **not** destroy its contact manager as long
as the candidate overlap remains alive.

`removeConnection` queues removed-overlap handles for retirement.
They remain unavailable for reuse until the source-corresponding
third-pass join has completed graph repair; only then are they returned
to the scene-owned free-handle pool. New overlaps created in the same
publication cannot take a pending-retirement handle. Source handle
indices may subsequently be reused on later overlaps. A failed
`Scene::step` restores the slot pool, accurate connection bits, active
handle ordering and retirement queue alongside actor/contact history.
The native smoke tests exercise touch loss/reentry, overlap replacement,
no premature same-frame reuse, subsequent handle reuse, and rejected
step retry identity. An additional graph regression retires a link in
a dynamic cycle, reuses the same handle for a different contact, then
tests redundant lost-edge routing and a real split against the
relinked adjacency. The same source edge slots, not a second graph
owner, supply both graphs and their traversal.

**Remaining source and performance differences:** persistent
`EdgeInstance` next/previous links and per-node first-edge indices
are now represented and maintained by the existing scene's contact
manager handles. Source dirty-node endpoint marking, hop-priority exploration and validated
fast-root/hop-route hints are now active for all-awake rigid links,
with speculative and accurate state kept separate. Static-touch
counters now follow physical touch and candidate transitions, including
dynamic component merge/split and static actor retirement. Stable island
handles and per-node island membership replace the provisional
union/find roots. Source hop-priority queue visitation and its
equal-priority `Cm::PriorityQueue` heap order are now implemented
for the admitted rigid graph, including split migration visitation
order. Split-local `mVisitedState` witnesses now survive across dirty
seeds in the same graph pass, and failed `findRoute`'s discovery
list feeds `splitEdges` without a second component traversal.
The admitted rigid graph now reuses successful-search and fast-path
witnesses as well, including source-style off-path branch hint
propagation; **what remains** is the full invalid-island state
transition, deferred multi-frame dirty-work budgets, typed node
lifetime, non-contact-manager edge activity and
the remaining non-rigid lifecycle. The rigid
`mDeactivatingEdges[eCONTACT_MANAGER]` sleep output is implemented
independently of manager retirement.
Ordinary lost-contact splitting now migrates only the failed
component's nodes and indexed-manager edges using source visit order,
swap-last removal and local handle allocation. The full-component
rebuild remains for actor admission/retirement. Neither path replays
managers in canonical Gu pair order. Failure rollback journals **only the modified** source edge
slots, route hints and intrusive node-head links, including splices into
neighboring edge instances. It does not traverse or copy old EdgeIndex
allocations at the start of successful steps. At most O(changed edge
writes + changed node-head writes) adjacency copies occur in a frame;
the persistent island-node/handle arrays are now snapshotted only
on structural mutations; unchanged-topology PGS steps retain only
active-node publication and motion/friction rollback state.
Incremental ownership undo for exceptionally large changing islands
remains a future performance investigation.
Tests verify a stable multi-contact solve journals zero edge records,
and an exception after removing an existing accurate edge restores
both published contact storage and the original intrusive connectivity.
`ScPipeline`'s asynchronous `destroyManagers`/lost-contact
continuations, deactivation for unsupported constraint/non-rigid edge
types, kinematic/non-rigid active-reference paths and `afterIntegration`
reconciliation also remain unported. Rigid contact-manager
`IslandSim::Node::mActiveRefCount` bookkeeping is implemented. Rigid contact-manager deactivation
is already published independently by the accurate/speculative graphs. The admitted
rigid-node active lists and persistent island handles are real source
lowerings, but a correctly published graph does **not** establish full
source island lifetime or task semantics.

**Unported source responsibilities:** complete source `PxNodeIndex`
node lifecycle and destruction; non-rigid, kinematic and articulation
variants of the shared visited-island witness and `mVisitedNodes`
split-migration paths; unsupported constraint/kinematic edge partitions
and their activation/deactivation lists; complete sleep/wake callback
semantics; first/second/third-pass task joins; post-third-pass interaction
retirement; `afterIntegration` deactivation reconciliation; source
callback ownership; and complete per-island solve/fetch scheduling.
The rigid contact-manager graph already implements shared route
witnesses, local split migration and both activating/deactivating
edge lists; these are **not** still-open rigid lifecycle tasks. The provisional union/find
parent/rank representation **has already been retired** in favor of
source-style graph-owned island handles and node chains. These
remaining responsibilities still prevent N1 completion.

## Shape offsets and contact-slop source correspondence

The admitted one-shape-per-actor native Scene now owns the shape's
`contact_offset` and `rest_offset`, initially 0.02 m and 0 m.
`World::set_body_shape_offsets` validates the same finite, nonnegative
contact and strictly smaller rest constraints as `NpShape::setContactOffset`
and `NpShape::setRestOffset`. The optional real PhysX CPU adapter edits
the existing `PxShape` with a setter order valid for either transition;
it does not create an alternate shape or material registry.

The contact offset follows `ScShapeCore::setContactOffset` through
the native `Scene::update_bounds` Gu AABB enlargement and the pair-summed
contact distance in original `GuContactPlaneBox` and translated
`GuContactBoxBox`. Raw Gu contact separation stays a geometric surface
separation; `DyContactPrepShared::constructContactConstraint` subtracts
the pair's **rest distance** when computing penetration and bias, and
`DyContactPrep::growPatches` uses that same distance in its friction
anchor admission threshold. Authoring an offset retires only affected
friction patches and schedules a fresh Gu publication. Geometry-only
queries retain the same failure-atomic publication boundary.

`RigidBodyDynamics::contact_slop_coefficient` follows
`NpRigidBodyTemplate::setContactSlopCoefficient`, its original
`PxsBodyCore::offsetSlop`, and `ScShapeInteraction`'s maximum of
the two rigid bodies. The production `Scene::prepare_constraints`
suppresses small normal angular Jacobian components with the
`DyContactPrepShared` relative-angular/linear velocity scale, and
independently suppresses small friction angular Jacobians and tangent
anchor-position errors following `DyContactPrep.cpp`. The independent
reference backend forwards the coefficient to real `PxRigidDynamic`.
Both parameters remain per-source actor/shape fields, not scene-global
solver tunables.

Production-path regressions exercise separated box-box broadphase and
Gu touch found/lost, plane-box offset admission, nonzero rest-distance
bias through native and optional original PhysX CPU World, invalid and
reversed offset edits, and normal/friction prepared Jacobians under
nonzero versus default contact slop. Multi-shape actors, CCD and per-
contact modification remain separately unported admission boundaries.

## Source rigid-pose authoring and shape alignment

`World::set_body_pose` now lowers full `PxRigidActor::setGlobalPose`
semantics into the authoritative native `Scene::set_pose` and independent
reference PhysX CPU actor. `PxQuat::isSane` magnitude tolerance
(1e-2), `PxTransform::getNormalized` quaternion publication, native
actor wake, authoritative orientation, friction-anchor retirement,
and Gu/SAP geometry invalidation share the existing position-edit
transaction; translation-only edits retain the stored quaternion
without roundoff-only discontinuities. The scene has no second
transform registry. `Scene::prepare_solver_bodies` subsequently derives
rotated world inertia from that same published actor quaternion.

The currently admitted logical horizontal floor remains a +Y-normal
primitive with identity user orientation; the reference adapter
preserves `PxCreatePlane`'s private +X-to-+Y alignment on a pose edit.
Arbitrary plane-normal rotation remains unsupported, rather than
silently claiming general rotated plane geometry. Tests exercise
quaternion rejection and normalization, translation preserving spin,
rotated static-box SAP and original Gu SAT contact transitions, public
World free-flight integration, and rejection of unsupported floor
rotation across native and optional reference CPU backends.

## Compliant spring-damper PGS correspondence and ownership

`PxsMaterialCombiner.h::PxsCombineMaterials` consumes both actor-owned
`Material` values when preparing the EXISTING `Scene::prepare_constraints`
stream. A single compliant material (negative `restitution`) takes
precedence over any rigid material regardless of ordinary restitution
combine mode. Two compliant materials honor the higher-precedence
restitution and damping combine modes; `Multiply` restores the
negative stiffness sign. When exactly one of two compliant materials
has `eCOMPLIANT_ACCELERATION_SPRING`, that material's stiffness and
damping take precedence. The combined acceleration flag follows the
original OR of both material flag sets.

`DyContactPrepShared.h::computeCompliantDamping` zeroes damping only
for a separated contact that is not closing within the timestep.
`computeCompliantContactCoefficients` then computes `nrdt`, `a`,
`b`, `x`, `velocityMultiplier`, `impulseMultiplier` and the
equal compliant biased/unbiased errors; force-based springs use the
unit response, while acceleration-based springs use its reciprocal
for the mass factor. `Scene::solve_constraints` consumes that
`impulseMultiplier` at every original PGS normal row iteration before
clamping each source Gu contact's impulse. Ordinary noncompliant
contacts retain an exact multiplier of one and their distinct
position/velocity-phase bias.

The same `Scene::FrictionPatch` cache stores the combined spring
properties and checks their equivalence alongside friction/normal
before correlating previous-frame anchors. Contact point order,
indexed contact-manager scheduling, Gu-indexed force writeback,
force-threshold changes and failed-step rollback retain their
existing ownership. The reference adapter authors
`PxMaterial::setDamping`, `setDampingCombineMode` and
`eCOMPLIANT_ACCELERATION_SPRING` on the original material shape.

Native regressions inspect the actual prepared normal constraints
for force- and acceleration-spring equations, mixed compliant and
rigid pairs in either body order, negative-stiffness multiply,
precontact damping suppression and public World motion. The
reference-enabled regression compares native and independent real
PhysX CPU free-motion/contact results for the same authored material
and timestep. Heterogeneous shape/triangle material IDs, packed
`PxSolverConstraintDesc` layout, contact modification callbacks
and complete island task continuations remain N1 requirements.

## Signed compliant friction limits and actual production force

The source `DySolverConstraintsShared.h::solveDynamicContacts`
performs its one-sided normal-force cap after applying
`impulseMultiplier`, leaving a possible negative compliant
force-buffer value. Source `DySolverConstraints.cpp::writeBackContact`
retains it unmodified, and the normal-force threshold stream does
not treat it as an invalid numeric state. The signed result also
feeds `DySolverConstraints.cpp::solveContact` and
`solveContact_BStatic`'s per-patch friction bounds.

Those original routines evaluate
`FMin(maxDynamic, FMax(-maxDynamic, totalImpulse))` when
friction saturates. A compliant patch can have negative
`maxDynamic`: C++ `std::clamp` with reversed bounds is undefined
behavior and does not reproduce the original nested SIMD expression.
The native solver now uses the exact min/max ordering instead of
inventing a nonnegative total or taking the absolute value.
An 80-frame public-World reproduction, a source force-buffer
writeback fixture, and a two-floor signed spring/damper regression
exercise the real multiframe solver and require at least one
negative Gu-indexed force generated by dynamics rather than
injected by a test. The source `FMin/FMax` expression is also the actual inline
operation used by production PGS, and its numerical regression
exercises deterministic negative and positive friction bounds that
distinguish the source result from an absolute-value clamp.
The multiframe scene regression independently requires negative
Gu-indexed normal forces generated by the actual solver. Since an
individual negative contact does not imply the *sum* of a correlated
friction patch is negative, final tangential rows are additionally
checked whenever that separate case occurs, rather than making
the test depend on a geometric condition the fixture cannot guarantee. This remains separate from strict packed-SIMD
rounding parity and does not claim negative force is a normal
noncompliant-contact condition.

## Source contact-touch reporting publication

`WorldOptions::contact_notifications` opts the admitted one-shape
rigid-pair scene into `PxPairFlag::eNOTIFY_TOUCH_FOUND`,
`eNOTIFY_TOUCH_PERSISTS`, `eNOTIFY_TOUCH_LOST` and
`eNOTIFY_CONTACT_POINTS`. The original default no-notification
filter remains unchanged. The public `World::contact_events()`
returns a read-only span of `ContactEvent` outputs from the latest
completed `World::step`: canonical actor IDs, event kind, current
contact count, range into `World::contact_event_points()`, and summed
signed per-Gu-contact normal impulses. The second published span
holds each source Gu point's world-space position, normal, signed
separation and associated signed normal force-buffer slot. Distinct
contact managers share one contiguous arena but own nonoverlapping
point ranges. The native source and original PhysX
`PxContactPair::extractContacts` oracle use the same data contract;
the reference callback gets its force slots directly from
`PxContactPair::contactImpulses`. The SDK does not fabricate
per-point tangential friction: PhysX's separate friction-anchor
writeback remains distinct until its full report stream is admitted.

Native `Scene::publish_contact_events` consumes the **existing**
sorted Gu found/lost pairs, current manifold stream, per-Gu-point
Dy force-writeback arena and actor activity; it does not invent an
independent collision registry or a second source of touch state.
Persistent touch events are emitted only when an involved rigid body
is active, matching
`ScNPhaseCore::processPersistentContactEvents`'s active-actor
check. Zero maxContactImpulse does not suppress Sc touch reporting,
because its Gu/accurate-island contact manager still exists. Output
is prepared in a reusable temporary arena and published only after
body integration and all exception-capable stages succeed. Failed
frames preserve the prior event span's backing storage and content.

The **independent** original PhysX CPU oracle now registers a real
`PxSimulationEventCallback` at scene creation when opted in; a
filter shader preserves `PxDefaultSimulationFilterShader` group
behavior while requesting only the above additional contact flags.
`onContact` consumes PhysX's actual pair-event and per-contact
force data from `fetchResults`, with canonical actor IDs carried
through the existing actor `userData` slot rather than an
O(all-actors)-per-event reverse lookup. It reports no synthesized
events from pose overlap and no events by default. The source
callback's `eREMOVED_ACTOR_0/1` flags also enter the existing
per-fetch retirement boundary: on explicit actor release, only
the original `PxActor*` address and SDK BodyId token are retained,
before destruction. A removal callback matches the pointer without
ever dereferencing it and translates it to the same canonical
actor IDs used in live reports. These tokens are discarded after
`fetchResults`, not retained as another actor registry. The native
scene consumes the same Gu lost-touch history on the next step.

Regression coverage exercises found→persists→lost→found transitions,
zero-response Gu touches, absent default notifications, independent
legacy-Gu and default-PCM callbacks, and failure-atomic native event
and per-point arena publication with a successful retry. A
two-interaction regression places contact pairs eight metres apart,
so an exchanged point window fails spatial and canonical-normal
checks even when both impulse sums remain internally consistent.
Retiring two dynamic actors before a source fetch, then retiring
a static support actor, independently exercises the lifetime-safe
removed-actor `Lost` notifications in native, original legacy Gu
and default PCM reference modes. This is **not** complete
`Sc::ShapeInteraction` callback parity: trigger reports,
threshold-callback joining, CCD notifications, interaction flag
mutations, filter edits and batched per-shape/multi-shape callbacks
remain later N1 admission gates. The new stream is an admitted rigid contact-touch
subset, not a replacement for the entire PhysX callback machinery.

## Source ScBodySim spatial force and impulse lifecycle

The single production native Scene now owns all four PhysX
`PxForceMode` forms through the actor-owned VelocityMod state:
per-step linear/angular velocity deltas, and per-second
linear/angular accelerations. `NpRigidBodyTemplate::addSpatialForce`
converts authored forces and impulses using the actor's inverse mass
or rotated inverse-inertia tensor; acceleration and velocity-change
modes already express the corresponding solver-space changes.
`ScBodySim::updateForces` lowers the per-second accumulation by
the current timestep at the `ScPipeline::beforeSolver` dependency,
before damping/clamping and Dy PGS. Default per-frame clearing and
the existing failure-atomic sparse motion checkpoint cover all four
velocity-mod channels, without a second force integrator. Native tests
distinguish per-second and per-step consumption, accumulated/cleared
categories, rejected-frame replay and coupled contact PGS. Independent
PhysX CPU tests compare both masses, angular inertia, mixed force modes
and clear operations over 90 changing-timestep frames. The source
`eRETAIN_ACCELERATIONS` scene flag is not admitted in this slice.

## Source gyroscopic-force velocity update

The existing actor-owned RigidBodyDynamics now forwards
PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES into the independent PhysX
reference scene and applies the original
DyRigidBodyToSolverBody::copyToSolverBodyData angular-momentum update
in the native unconstrained/PGS input path. The local inverse-inertia
tensor and body quaternion remain authoritative; no second solver-body
or gyroscopic simulation is introduced. Both contact and no-contact
islands consume one velocity correction per frame, after unconstrained
velocity clamping and before axis locks. Source-free tests distinguish
enabled and disabled anisotropic-body motion; reference-enabled tests
compare the original CPU result and quaternion response across
changing timesteps.

## Source rigid dynamic world-axis locks

The existing native Scene accepts the six PxRigidDynamicLockFlag axes
through actor-owned RigidBodyDynamics. Following
DyRigidBodyToSolverBody::copyToSolverBodyData, locked angular baseline
velocity is suppressed without changing the world inverse-inertia
tensor. DyBodyCoreIntegrator::integrateCore suppresses locked linear
motion and solver-space angular components at integration and final-velocity
publication, without discarding the pre-solver linear contact-relative
velocity. Original contact Jacobians retain their inertia response.
Native smoke covers no-contact motion, production PGS, flag rejection
and lock release; reference-enabled smoke compares original PhysX CPU
under changing timesteps, torque impulses and lock transitions.

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


## Rigid contact-manager eACTIVE/eACTIVATING source lifecycle

The retained scene-owned EdgeIndex slots now hold independent speculative
and accurate `IG::Edge::eACTIVE` and `eACTIVATING` state. Source
`markEdgeActive` publishes the same EdgeIndex into that graph's
`mActivatedEdges[eCONTACT_MANAGER]` on its inactive-to-active transition,
updates `mActiveEdgeCount`, and does not re-emit an already active edge.
The following `wakeIslands` publication clears prior `eACTIVATING`
flags and worklists without clearing persistent `eACTIVE` state.

Source `deactivateNodeInternal` emits a deactivating contact manager
**only** when its edge was active; an already inactive edge produces no
new event. Canceling a still pending activation uses source replace-with-
last list removal. A retired SAP manager or lost Gu touch cancels its
activity without manufacturing an island-sleep event. Accurate physical
contacts and speculative SAP overlaps retain distinct histories across
accurate-island sleep and subsequent speculative-only bridge splitting.
Only touched source edge slots are journaled on failed publication;
per-graph activity counts and activated-edge worklists are restored.
Both graph owners now also carry `IslandSim::Node::mActiveRefCount` for
the source's two-valid-endpoint rigid contact managers. Static-world
contacts never increment a dynamic node's references. Activation,
true deactivation, touch loss, retirement, retry and the checkpoint
restore both graphs' independent reference counts without a parallel
contact registry. A deduplicated per-node undo journal, rather than
dense reference-array copies on every step, handles the failure boundary. The counters are source-owned prerequisites for
later kinematic activation, not a claim that kinematics are admitted.
Activity flags replay **after** the intrusive edge-undo records, which
may contain post-deactivation bits. Quiet nontransactional publications
clear expired activation flags without acquiring rollback records.
Both local lost-edge splits and a retiring dynamic edge that disconnects
an already sleeping rigid peer record the affected source node for the
third-pass decision. The latter survives the first-pass ownership rebuild;
there is no heuristic component-size inference or global island rescan. Separate regressions require exactly one
retained-manager speculative sleep event after retiring an awake SAP-only
peer, and failure-atomic eACTIVE/count restoration after Gu touch loss.

The accurate and speculative island owners also retain separate
source `mActivatingNodes` wake-event sequences, including authored
reverse-order wakes and graph-specific SAP/Gu events. Generation-stamped
node membership suppresses duplicate event insertion without quadratic
wake-list scans or allocations in the no-throw publication. The rollback checkpoint pre-reserves active-handle and pending
wake capacity in both owners before entering any no-throw publication.
`mActiveIslands` dense handle lists and their `Island::mActiveIndex`
reverse indices. Source swap-last removal and append preserve active
handle order across sleep/wake, actor admission, split and merge;
the serial `Dy::DynamicsContext::iterateIslandsNodes` handoff now
resolves only active handles against the existing contiguous accurate
island-node ranges, instead of scanning every dormant node range.
These are not a second island registry or a separate task scheduler.

ScPipeline.cpp::activateEdgesInternal(eCONTACT_MANAGER) and
postThirdPassIslandGen now consume the speculative island owner's
activated/deactivating edge streams into persistent
`Sc::InteractionFlag::eIS_ACTIVE` on the **same** source EdgeIndex
contact-manager record. Source accurate sleep leaves the interaction
active while a speculative SAP-only bridge remains awake. True
speculative deactivation retires the interaction exactly once; SAP
overlap destruction clears the flag via overlap retirement rather than
fabricating a sleep-deactivation event. The aggregate count and the
per-edge flag are failure-atomic under the existing sparse activity
journal, including rejected Dy frames. This is the admitted rigid
interaction-activity subset, not shape-interaction callback registration
or persistent-contact event processing.

The implemented scope is rigid `eCONTACT_MANAGER`. Kinematic,
articulation and constraint edges, kinematic consumption of the now
implemented rigid contact-manager active references, per-type bitmap
partitions, full shape-interaction activation/notification
semantics, persistent contact events, callback publication and full
island task continuations still require source-derived N1 integration.
The rigid speculative contact-manager interaction-activity consumer
is already implemented; it is not a remaining general interaction
callback pipeline. Packed solver/contact admission and CCD
remain separate gates; the new activity lifecycle alone does not close N1.
