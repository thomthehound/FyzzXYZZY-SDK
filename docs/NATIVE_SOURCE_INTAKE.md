<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# N1 source-faithful native dynamics intake

**Status: architecture and source audit; no solver implementation or N1 acceptance in this document.**

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
   and GPU: CPU chains the first pass directly to `mBpUpdate`, while GPU
   runs `mBpSecondPass` and `mPreIntegrate` in the GPU-specific route.
   Do not manufacture an identical CPU/GPU path.
3. `updateBroadPhase` calls AABB manager second-pass/update, unblocks
   narrowphase at the prescribed dependency point; `postBroadPhase`
   feeds narrowphase and manager report publication. Filtering,
   interaction/manager creation and island insertions depend on reports.
4. `advanceStep` installs **different explicit continuation chains**.
   The CPU chain passes through second-pass narrowphase,
   `postNarrowPhase`, `islandGen`, the **scene-level** `solver` task,
   `updateDynamics`, `postSolver`, `afterIntegration` and
   `finalizationPhase`. The GPU branch additionally chains
   `updateBodies` before `updateDynamics`, then
   `updateDynamicsPostPartitioning` and
   `updateSimulationController` after it, before `postSolver`.
   Some island-gen tasks differ under
   `USE_SPLIT_SECOND_PASS_ISLAND_GEN`; preserve that source branch
   rather than assuming a single fixed list. GPU completion must
   satisfy the corresponding continuation, without per-kernel polling
   or forcing unrelated device-wide idle.
5. `postNarrowPhase` synchronizes actual contact-manager outputs
   (`fetchUpdateContactManager`), sorts GPU contacts where required and
   releases prior constraints. It does **not** process patches.
   `islandGen` invokes `DynamicsContext::processPatches` using
   narrowphase lost/found patch managers and output counts, then runs
   `processNarrowPhaseTouchEvents` to update contact/island graph
   state before the solver consumes it. This ordering is a source
   dependency, not an interchangeable stage label.
6. **`Sc::Scene::solver` is not `Dy::DynamicsContext::update`.**
   `Sc::Scene::solver` calls `beforeSolver(continuation)` to prepare
   body forces/threshold state, and swaps narrowphase streams once
   narrowphase is complete. It **does not invoke**
   `mDynamicsContext->update`. The distinct, downstream
   `Sc::Scene::updateDynamics` task acquires constraint memory,
   arranges the lost-contact task continuations, and calls
   `mDynamicsContext->update(...)` with the narrowphase output,
   gravity, timestep, patch count and changed-handle map.
7. **Partitioning is a distinct continuation handoff.**
   For CPU, `updateDynamics` passes `mPostSolver` to the dynamics
   context and then calls `updateDynamicsPostPartitioning`
   **immediately in the same thread**. For the GPU admission,
   `updateDynamics` supplies `mUpdateSimulationController` and
   `mUpdateDynamicsPostPartitioning` continuations, allowing the
   source's distinct post-partition task. In both cases
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
10. **`postSolver` is a required state-lifetime barrier**: it merges
    dynamics results, releases solver constraint memory and swaps
    friction streams; on the GPU branch,
    `updateSimulationController` first transfers simulation data
    back and updates body cores. `afterIntegration` then reconciles
    wake/sleep, bounds and contact-force events; finalization
    handles callbacks, contact visualization, task-pool release and
    report timestamps. The native scene must not publish a
    predicted pose as final source-equivalent post-solver state.

**Explicit existing departure:** `src/native/scene.cpp::step` currently
predicts poses, rebuilds an X-only SAP, publishes face-only contacts, and
runs a single normal constraint for an entire manifold after position
prediction. This does not implement the above graph or source rigid-body
solver. It remains an admitted **limited transitional execution mode**,
not a certified PhysX CPU replacement.

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
| `islandGen` | `mDynamicsContext->processPatches` **then** `processNarrowPhaseTouchEvents`; patch/touch data and island adjacency | Accurate active island state for `solver`, later `updateDynamics` | Split-island-gen compile-time paths affect edges |
| `Sc::Scene::solver` / `beforeSolver` | Updates per-body forces and threshold buffers, swaps narrowphase streams after second pass | Downstream **`Sc::Scene::updateDynamics`**, not `DynamicsContext::update` from within `solver` | GPU update-bodies task intervenes |
| `Sc::Scene::updateDynamics` | Acquires constraint memory, adds lost-contact task references, calls `mDynamicsContext->update` with actual narrowphase and island inputs | `updateDynamicsPostPartitioning` and solver task continuations; eventually `postSolver` | CPU calls post-partitioning immediately, GPU schedules distinct post-partition/update-simulation-controller edges |
| `Dy::DynamicsContext::update` and `updatePostKinematic` (CPU) | Owns solver pools, accurate active island lists, constraint descriptors, per-island task creation and partitioning | Source `DySolverControl` PGS iterations, integration, postprocessing, threshold/force reports | GPU dynamics context has a different implementation; do not infer full GPU behavior from CPU `DyDynamics.cpp` |
| `updateDynamicsPostPartitioning` | Calls dynamics-context post-partitioning hook, clears destroyed partition edges and releases staged lost-contact references | `updateSimulationController` on GPU, or source CPU post-solver continuation | The call is source-visible even when CPU dynamics executes post-partition work inline |
| `postSolver` / `afterIntegration` / `finalizationPhase` | Merges results, frees constraint blocks, swaps friction streams, completes body-state publication, events and cleanup | Result fetch and next-frame scene mutation | GPU first performs DMA-back and body-core update |

### Verified source anchors and still-open closure

The following are directly checkable locations in the maintained PhysX
source used for this audit. Line anchors are navigational aids, not
license to assume every transitive dependency was already ported:

- [`ScPipeline.cpp::solver` and `beforeSolver`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/src/ScPipeline.cpp#L1763-L1945) — scene-level solver setup and stream swap, **not** the `DynamicsContext::update` call.
- [`ScPipeline.cpp::updateDynamics` / `updateDynamicsPostPartitioning`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/src/ScPipeline.cpp#L2049-L2140) — actual dynamics-context entry, CPU/GPU continuations and lost-contact task ownership.
- [`ScPipeline.cpp::postNarrowPhase` / `islandGen`](https://github.com/thomthehound/PhysX/blob/main/physx/source/simulationcontroller/src/ScPipeline.cpp#L1525-L1560) — contact-manager sync **before** patch processing in island generation.
- `DyDynamics.cpp::DynamicsContext::update` and `updatePostKinematic` — inspected their original implementations for world-body reset, pool lifetime, accurate-island batching, and solver-task creation. Extract their exact source ranges and transitive type dependencies before porting; no approximate line anchor is asserted here.
- [`DySolverControl.cpp::solveV_Blocks`](https://github.com/thomthehound/PhysX/blob/main/physx/source/lowleveldynamics/src/DySolverControl.cpp#L160-L313) — position/velocity, conclude, writeback and no-contact branches.
- [`PxsSimpleIslandManager.cpp::firstPassIslandGen`, `secondPassIslandGenPart1/Part2`](https://github.com/thomthehound/PhysX/blob/main/physx/source/lowlevel/software/src/PxsSimpleIslandManager.cpp#L257-L310) — speculative versus accurate state, lost-edge removal and deferred cleanup.
- [`NpSceneFetchResults.cpp::fetchResults` / `fetchResultsStart` / `fetchResultsFinish`](https://github.com/thomthehound/PhysX/blob/main/physx/source/physx/src/NpSceneFetchResults.cpp#L484-L670) — completion readiness and public contact-event publication.
- [`PxgAABBManager.cpp::updateBPFirstPass` / `updateBPSecondPass`](https://github.com/thomthehound/PhysX/blob/main/physx/source/gpubroadphase/src/PxgAABBManager.cpp#L485-L828) and [`PxgCudaBroadPhaseSap.cpp::gpuDMAUp` / `update` / `fetchBroadPhaseResults`](https://github.com/thomthehound/PhysX/blob/main/physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp#L242-L1260) — separate GPU SAP host lifecycle. This document does not certify that N2 is wired.

**Important qualification:** PhysX may process work concurrently where
continuation dependencies permit. An ordered list of task names is not
a proof that each waits on every prior task. Before writing the native
scheduler, extract actual continuation edges, producer-completion
requirements, and buffer retirement from the original task construction
sites into an executable DAG check. The table above describes the
required handoffs, not an invented globally sequential PhysX frame.

### Readiness and retirement invariants

- **Contact patch freshness:** after `postNarrowPhase` has
  synchronized narrowphase manager outputs, `islandGen` consumes
  *lost/found patch managers and output counts* in
  `DynamicsContext::processPatches` before touch-event processing.
  A task named "patch prep" in another subsystem cannot shift this edge.
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
- **Island passes:** preserve distinct speculative and accurate island
  graphs from `PxsSimpleIslandManager`. Its first pass handles
  speculative wake/new/lost edges; second pass part 1 handles accurate
  wake/new edges, while part 2 removes destroyed edges and lost
  connections and retires destroyed node handles. The source explicitly
  splits parts for permitted overlap with GPU patch work, so don't
  serialize everything solely because a text ledger shows one line.
- **World and solver bodies:** zero the source world solver body, preserve
  the kinematic/dynamic remap and accurate island-node ordering, and
  feed contact constraints with real per-body inertia/Jacobian state.
  Neither a global manifold array nor the public actor ID is a
  substitute for a `PxSolverConstraintDesc` mapping.
- **CPU/GPU split:** on a CPU run, `updateDynamicsPostPartitioning`
  executes immediately on the caller; on a GPU run, different
  continuations connect post-partitioning and simulation-controller
  body-core publication. Never insert blocking GPU waits to simulate
  an incorrect flat CPU chain.
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
| `ScPipeline` task boundaries | Verified `solver` versus `updateDynamics`, `islandGen` patch ownership, CPU/GPU continuations and `postSolver` cleanup in actual source | Full task refcount/dependency DAG, including compile-time branch and CCD paths |
| `DyDynamics` CPU integration | Verified CPU `update`, source solver-body pools, `updatePostKinematic` island batching, PGS task creation | Input/output ABI extraction, exact partition and writeback semantics, contact/constraint layout inventory |
| PhysX GPU dynamics | Verified distinct `ScPipeline` GPU continuations and controller handoff | **Not yet independently mapped** GPU-specific `DynamicsContext` implementation, CUDA launch/memory graph and any CPU fallbacks |
| Island manager and public fetch | Inspected original speculative/accurate island passes and `NpScene::fetchResults` completion/callback split | Full edge/retirement graph and native API publication semantics still need closure |
| GPU SAP host lifecycle | Located original `PxgAABBManager` first/second pass and `PxgCudaBroadPhaseSap` DMA/update/fetch methods | Source-stage and buffer-history closure still belongs to N2, not assumed complete by this N1 audit |
| Source geometry and contact correlation | Identified `GuContact*`, `PxsNphaseImplementationContext` and `DyContactPrep` source families | Exact correlation/anchor data structure, cache transitions and geometry edge-case acceptance |
| Native acceptance | Existing source-free scene and oracle smoke tests support an openly bounded subset | Complete rotational PGS, source schedule, GPU SAP and N1/N2 full parity **not admitted** |

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
