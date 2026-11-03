<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FyzzXYZZY capability roadmap

Status: November 3, 2026. This document owns the active implementation order
and capability acceptance criteria. The objective is complete PhysX-derived
SDK capability coverage, independently executable on native CPU, Vulkan and
D3D12 where the original capability admits GPU execution. Completion is
measured by real scene behavior, not kernel count, PR count or historical
milestone labels.

## Engine contract

One scene owns actors, shapes, shape elements, interactions, island membership,
contact managers, articulation and nonrigid state, simulation tasks and result
publication. CPU and GPU are execution modes of those owners, never distinct
physics engines. The algorithms, numerical contracts, data structures,
reporting and lifetimes follow PhysX source behavior; GPU addressing,
scheduling and storage may differ where equivalence and performance are
demonstrated.

GPU execution is the primary target. CPU is the independently built
conformance implementation and supported-device fallback. The separate
reference CPU backend is only a comparison oracle. Neither Godot nor an
external PhysX runtime is required by the native SDK.

The full target includes simulation, geometry, cooking, scene queries,
constraints, articulations, deformables, particle systems, controller and
vehicle extensions, serialization, callbacks and diagnostics. A supported
feature must not silently route to the reference implementation or to an
independently invented physics model.

## Current production coverage

| Subsystem | Native implementation |
| --- | --- |
| Scene, lifecycle and tasks | Source-derived Sc/island/Dy scene and PGS task chains for a one-shape-per-actor rigid subset; full public scene semantics absent |
| Geometry and cooking | Box and infinite plane; no general shape/actor graph, mesh cooking or full geometry dispatch |
| Broadphase | CPU SAP and persistent GPU AABB/SAP/aggregates on Vulkan and D3D12; scene admission remains bounded |
| Narrowphase | CPU Gu PCM for box/box and plane/box; optional GPU Gu PCM box/box with GJK/EPA continuation and Sc contact publication |
| Rigid solving | CPU PGS and explicit GPU Dy contact/distance preparation, partitions, correlated friction, solve and integration for the admitted subset; Sc still prepares and fetches host-visible streams |
| Constraints | Limited center- and local-anchor distance constraints; general joints, TGS and CCD absent |
| Articulations | No integrated articulation solver |
| Nonrigid | No integrated deformable surface/volume or particle/fluid solver |
| SDK entry | Native CPU explicit; native GPU via explicit factory/shader bundles; ordinary World default remains reference CPU where enabled |

This is a functioning but narrow physics slice. The current hybrid readbacks,
one-shape actor model and contact-manager admission are implementation
boundaries, not acceptable general-engine architecture.

## Immediate execution sequence

These are coherent capability deliveries, not one-PR-per-kernel tasks. Closely
related implementation changes stay together until a runnable acceptance
workload exists. Correctness, source semantics and performance are designed
into each capability; micro-optimizations do not substitute for expansion.

### 1. Actor, shape and element ownership

Replace the one-shape actor contract with original actor/shape/ElementSim
identities and independent shape lifetimes. Preserve stable body identities
through attachment/removal, distinct shape-local transforms, per-shape
material/filter/contact offsets, trigger and simulation flags, and
post-publication retirement. Broadphase pairs and contact managers key on
shapes; island nodes, mass/inertia and solver velocities key on bodies.
Do not encode extra shapes as fake rigid bodies or use aggregate grouping
as a substitute for compound actors.

**Acceptance:** an independently constructed native actor carries multiple
shape elements with different local transforms. Shape additions, removals,
filter changes and ID reuse preserve source Sc/Gu history and event identities.
A compound dynamic actor collides against multiple neighbors and solves as
one body, with source-correct combined mass/inertia and angular response.
The same live scene runs CPU and Vulkan/D3D12 GPU broadphase, narrowphase
and Dy paths across several frames, including rollback and retirement.

### 2. General geometry, cooking and collision dispatch

Admit sphere and capsule, then convex geometry and cooked triangle meshes,
heightfields and applicable signed-distance representations through the real
geometry and collision pipeline. Use source-defined convex/support,
GJK/EPA/SAT, PCM cache and mesh-contact algorithms, their pair dispatch,
material assignment and contact stream rules. Geometry representation and
cooking artifacts are shared between CPU and GPU execution without
box-shaped stand-ins.

**Acceptance:** interacting sphere, capsule, convex and cooked-mesh shapes
produce correct contact manifolds, normals, separation, friction, restitution
and persistent touch transitions. Mixed geometry contacts solve and report
in compound and independent rigid scenes on both native GPU APIs; each
supported geometry pair has an explicit backend admission contract.

### 3. Complete rigid simulation and queries

Complete discrete/contact-modified and continuous collision detection,
speculative contacts, triggers and filtering, raycasts, sweeps, overlaps,
contact reports, scene origin shifting and other public scene semantics.
Port the full supported rigid constraint and joint families, body flags,
locked axes, breakage, drives, PGS and TGS solver modes, partitioning and
writeback. Retain correct source island, sleep/wake and result boundaries.

**Acceptance:** broad and narrowphase, CCD, joint/contact solving and scene
queries operate in one continuously simulated world, with correct actor and
shape identities and frame-to-frame reports. Both native APIs pass scene-level
parity against the independent CPU source semantics; unsupported feature
combinations reject explicitly rather than changing simulation behavior.

### 4. Fully device-resident task and data dependencies

Remove mandatory host publication between device-produced bounds, contacts,
constraint preparation, solving, integration and subsequent bounds updates.
Keep Sc's authoritative ownership and its externally visible event/fetch
boundary without forcing every intermediate result through the CPU. Preserve
the original task continuation graph, GPU completion, failed-frame rollback,
allocation and ID lifetimes. Choose physical address lowerings appropriate
to Vulkan and D3D12; wave32 must work and wave64 is permitted where valid.

**Acceptance:** an integrated heterogeneous rigid scene executes successive
frames of shape updates, broadphase, mixed-geometry PCM, constraints, solve
and integration on the GPU without an unconditional intermediate contact or
body readback. Instrument the actual DMA and submission graph. CPU-visible
results are fetched only at source-defined consumer boundaries.

### 5. Articulations and coupled constraints

Port articulation link topology, reduced-coordinate dynamics, joint limits,
drives, tendons, internal and external constraints, and contact feedback.
Preserve source solver ordering, state and wake/activation rules; articulation
links must share broadphase, narrowphase, islands and reporting with rigid
bodies, not run in a separate simulator.

**Acceptance:** an articulated mechanism driven through contacts and joints
interacts reciprocally with ordinary rigid bodies, with reproducible motion,
breakage/limits where supported, correct lifecycle and end-to-end GPU task
completion on Vulkan and D3D12.

### 6. Deformables, particles and mixed physics

Port deformable surfaces, deformable volumes, PBD particle/fluid systems,
their source materials, constraints, collision detection, self-contact,
solver scheduling and interactions. Admit the source-defined GPU data
structures and runtime behavior rather than inserting independent cloth,
fluid or soft-body implementations.

**Acceptance:** mixed rigid/articulation/deformable/particle scenes exchange
contacts and forces in both directions under one scene clock, with coherent
state publication, retirement and failure handling across native backends.

### 7. SDK surface, extensions and completion

Fill the remaining public PhysX-derived SDK functions and extension modules:
geometry creation/cooking, controllers, vehicles, queries, aggregates,
serialization, scene callbacks, filtering, diagnostics and host integration.
Build and run the independent SDK without the reference backend or an
application-specific host. Maintain a tested capability/backend matrix for
every advertised feature rather than using a single completion percentage.

**Acceptance:** each supported feature has the source contract, standalone
SDK behavior, device admission where applicable, failure semantics and
consumer integration required for a complete release.

## Workstream overlap and dependency rules

- The shape/element model precedes broad geometry admission; adding more
  actor kinds is not an acceptable geometry implementation strategy.
- Mixed geometry, cooking and rigid solver dataflow advance together.
  Articulation and nonrigid source intake can proceed in parallel once
  their interactions have authoritative scene and island owners.
- GPU kernels enter production only with the real source task caller,
  buffer lifetime, result publication and a multiframe SDK witness.
- A performance change must enable a capability, remove a demonstrated
  architectural bottleneck, or improve a measured integrated workload.
  Avoid additional transfer, buffer or scheduler layers without necessity.
- Portability means actual Vulkan and D3D12 execution and tests; WARP and
  software Vulkan are correctness infrastructure, not physical GPU speed
  evidence. Profile integrated workloads on native AMD hardware.

## Capability acceptance evidence

For each delivered capability record:

1. The source operation, state owner, numerical behavior, allocation
   and lifetime contracts preserved by the native implementation.
2. A production API and single-scene caller that executes the feature
   through creation, motion, interaction, retirement and failed publication.
3. A representative mixed-scene conformance workload, not just one kernel
   or isolated arithmetic fixture. Compare CPU/native GPU behavior and
   explain intentional numerical differences.
4. Explicit API/backend support and failure behavior; no hidden
   reference execution, shadow physics state or invented alternate solver.
5. GPU submissions, bytes copied, waits, residency, allocations and
   integrated frame timings at the capability's real performance checkpoint.

The first acceptance workload for the new program is a **compound actor
with mixed geometry**, exercising original shape identities, GPU contact
generation, Dy body response and correct multi-frame retirement. Do not
declare the architecture complete until that workload runs through
the real SDK on both GPU backends.

## Beyond the source capability baseline

Full source capability remains the first deliverable. The following
additional systems are future independent engineering work, not current
PhysX parity claims or prerequisites for rigid, articulation or nonrigid
source admission. Preserve their integration requirements in current
architecture decisions; implement and benchmark them after the relevant
source subsystem is functional.

| Future capability | Physical scope and integration requirement |
| --- | --- |
| Nonlinear continuum materials | Large-strain, anisotropic, viscoelastic, hyperelastic and near-incompressible solid response; material parameters and constitutive state belong to deformable elements, not rigid shape flags |
| Slender bodies and strands | Cosserat/elastic-rod or equivalent bending, twist, extension and contact for hair, fibers, cables and tendons, with attachment to moving deformable or articulated surfaces |
| Thin shells and membranes | Bending, stretching, wrinkling, frictional sliding, enclosed-volume and pressure constraints, including contact with solids and liquids |
| Wet contact and interfacial flow | Thin-film transport, wetting, adhesion, lubrication and drainage coupled to surface geometry, material state and friction rather than a visual-only wetness parameter |
| Fluid/structure interaction | Internal and external pressure, viscous flow, surface tension, fluid containment, leakage and bidirectional force exchange with rigid, articulated, shell and volumetric bodies |
| Thermal and surface state | Temperature, moisture and other transported surface quantities where they affect physical material behavior; renderers consume simulation state without becoming physics authorities |

The solver architecture must support heterogeneous objects and contacts,
typed constraints, distinct material/field states, bidirectional force
and momentum exchange, and appropriate substepping. A single monolithic
numerical solver is not mandatory. Partitioned solvers are acceptable when
their coupling, iteration, stability and convergence have been validated.
Every interacting component must participate in the same scene clock,
topology lifetime, completion dependencies and result visibility rules.

### Future multiphysics acceptance

Representative capability-level validation includes:

- Two contacting nonlinear deformable volumes with sustained sliding,
  self-contact and calibrated near-incompressible response.
- A pressurized thin membrane containing fluid and compressed between
  moving deformable surfaces, with pressure and contact forces returned to
  every participating object.
- Flexible strands attached to moving soft or articulated surfaces,
  exchanging collision forces while bending, twisting and contacting
  neighboring strands or surfaces.
- A wet sliding interface whose fluid film and friction state evolve under
  pressure, motion and drainage instead of using prescribed animation.
- Mixed rigid, articulated, deformable, fluid and strand bodies running
  over many frames with explicit energy/momentum error budgets, convergence
  behavior and failure/retirement handling on both GPU backends.

Numerical accuracy requires constitutive laws, discretization and measured
material parameters appropriate to each workload. Visual plausibility alone
is not an equivalence test, and real-time performance is an independently
measured requirement rather than a claim of universal continuum fidelity.

## Technical references

- [Architecture and fidelity contract](ARCHITECTURE_CONTRACT.md)
- [Native scene source correspondence](NATIVE_SOURCE_INTAKE.md)
- [GPU task/dataflow trace](PHYSX_GPU_PORT.md)
- [GPU memory addressing](GPU_MEMORY_ADDRESSING_RFC.md)
- [GPU performance policy](GPU_PORT_PERFORMANCE.md)
- [SDK build and standalone integration](BUILDING.md)

Historical numbered plans and kernel audit records are implementation
references, not active completion gates.
