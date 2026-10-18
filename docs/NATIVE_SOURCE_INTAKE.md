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
| `lowlevelaabb/src/BpBroadPhaseSap.cpp`, `lowlevelaabb/src/BpAABBManager.cpp` | Source CPU AABB owner and incremental SAP; overlaps and filtering inputs | Retain source pair semantics and broadphase manager ownership; existing rebuild-only X-sweep is temporary scaffolding |
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
4. `advanceStep` chains second-pass narrowphase to post-narrowphase,
   island generation, solver, dynamics and post-solver. GPU execution
   specifically adds update-bodies, partitioning and simulation-controller
   continuation edges. Retain independent GPU workload completion signals;
   no CPU polling loop or per-kernel wait.
5. `postNarrowPhase` synchronizes actual contact-manager outputs
   (`fetchUpdateContactManager`), sorts GPU contacts where required and
   releases prior constraints. It does **not** process patches.
   `islandGen` invokes `DynamicsContext::processPatches` using
   narrowphase lost/found patch managers and output counts, then runs
   `processNarrowPhaseTouchEvents` to update contact/island graph
   state before the solver consumes it. This ordering is a source
   dependency, not an interchangeable stage label.
6. `solver` delegates to `beforeSolver` and the dynamics context, swaps
   narrowphase streams only at the source-defined boundary. In
   `DyDynamics.cpp`, constraint setup/partitioning produces the batch
   descriptors consumed by `DySolverControl.cpp`.
7. `solveV_Blocks` distinguishes position iterations, velocity iterations,
   conclude and final writeback; it saves motion velocities for integration
   and applies the source's friction iteration policy. The zero-constraint
   branch still integrates bodies.
8. `afterIntegration` and `finalizationPhase` reconcile activation,
   bounds, CCD/callback state and publication. The native scene must not
   publish an intermediate predicted pose as final source-equivalent
   post-solver state.

**Explicit existing departure:** `src/native/scene.cpp::step` currently
predicts poses, rebuilds an X-only SAP, publishes face-only contacts, and
runs a single normal constraint for an entire manifold after position
prediction. This does not implement the above graph or source rigid-body
solver. It remains an admitted **limited transitional execution mode**,
not a certified PhysX CPU replacement.

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
