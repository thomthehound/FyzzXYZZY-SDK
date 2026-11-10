<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FyzzXYZZY implementation program

Status: November 3, 2026. This is the active capability inventory, dependency
order and acceptance contract. The target is a complete, independent,
modern CPU/Vulkan/D3D12 physics SDK with PhysX 5.11 capability and behavior
coverage, followed by additional coupled physics. No full-engine, complete
GPU-pipeline or hardware-performance acceptance has been established.

## Scope and completion

The source baseline includes the public simulation, geometry, cooking,
query, immediate-mode, extension and diagnostic facilities, not only rigid
GPU kernels. Reference updates require an explicit inventory and conformance
review. Record the resolved reference version and build configuration with
comparison results; a moving reference branch is not a reproducible oracle.
Consumers do not need a reference runtime or a particular host engine.

Native CPU, Vulkan and D3D12 are execution choices within one engine. CPU
coverage follows the source's CPU capabilities; GPU-only features need GPU
source comparisons and independent numerical checks. A CPU fallback cannot
claim to implement an absent deformable or particle solver. Each capability
records its actual execution stages and supported combinations.

Completion has five independent conditions: a public SDK operation works;
its source semantics are established; lifecycle and failure behavior hold;
representative workloads meet numerical quality requirements; and physical
hardware meets an explicit performance budget. A passing helper, translated
kernel, copied header, populated enum or PR review form satisfies none of
these conditions by itself.

## Audited starting point

This table describes merged native production code. Reference-backend
features do not count as native coverage. The compound rigid scene now supports independent box/sphere attachments,
shape-keyed contacts, mass-frame dynamics and compound GPU broadphase/Dy.
GPU sphere contact generation, general shape authoring, and hardware
performance acceptance remain open.

| Capability | Merged native behavior and evidence | Remaining coverage |
| --- | --- | --- |
| Public construction | Explicit NativeCpu and shader-bundle GPU factories; `src/world.cpp`, `native_gpu_world.h`, `native_boundary_smoke.cpp` | Ordinary World defaults to reference CPU; direct NativePortableGpu construction rejects; full scene configuration/API absent |
| Actors, shapes, mass | Independent box/sphere attachments with local poses and explicit mass-center/principal-inertia recomputation in the native CPU/GPU scene | Shared/exclusive geometry resources, full shape flags and broad geometry dispatch |
| Broadphase/aggregates | Persistent CPU SAP and Vulkan/D3D12 AABB/SAP/aggregate reports with separate shape elements and actor-owned filtering identity | Other source broadphase modes, general shape/node categories and full public controls |
| Contacts | CPU box/box, sphere/sphere, sphere/box and plane Gu/PCM; GPU box PCM/GJK/EPA and shape-keyed contact histories connected to Sc | Device sphere/capsule contacts, mesh midphase, general dispatch and full contact/material forms |
| PGS/dynamics | CPU and explicit GPU contact/distance prep, friction, solve, integration, body controls and sleep for admitted rigid scenes | Full source solver forms, batching, mixed nodes and scalable GPU execution |
| GPU frame | Shared source heap/stream, persistent SAP/PCM history and reuse of device contact/pose data | Host contact publication, CPU task/partition construction and solver fetch still constrain device progress |
| Events/filtering | Bounded touch, threshold, material, simulation-flag and island lifecycle coverage | Shape-level callbacks, triggers, general filtering, reporting and scene publication API |
| General rigid physics | Limited distance constraints and discrete collision | Full joint families, TGS, CCD, contact modification and remaining rigid controls |
| Queries/cooking | No general native production API | Geometry queries, scene queries, acceleration structures and all cooked geometry |
| Articulations | Island node categories reserved | No integrated articulation topology, dynamics, drives, tendons or inverse dynamics |
| FEM surfaces/volumes | Island node categories reserved | No integrated deformable solver, material, attachment, self-contact or coupling |
| Particles/fluids | Island node category reserved | No integrated PBD system, buffers, phases, neighborhood/collision or fluid behavior |
| Extensions/tools | Independent builds, source foundation/heap, narrow Godot consumer | Controllers, vehicles, serialization, immediate mode, direct GPU access and general diagnostics |

`tests/headless_smoke.cpp` contains independent PhysX CPU comparisons.
`tests/physx_gpu_source_sc_pcm_scene_native.cpp` principally compares native
CPU and native GPU scenes. That is valuable lowering coverage, but a defect
shared by both implementations can pass it. Source-enabled comparisons and
independent physical checks remain necessary.

### Architectural qualification still open

| Evidence in current code | Required resolution | Delivery |
| --- | --- | --- |
| Compound shape-element and body identities now separate collision history from rigid Dy response | Shared geometry ownership, per-shape filters/triggers and complete public scene semantics | Scene and geometry |
| Native CPU/GPU rigid rows use source PxReal; compiled SPIR-V/DXIL checks reject FP64 requirements | Extend error/conditioning and cost validation with each subsystem; numerical representation follows the supported physical domain | Every capability |
| Wide Dy conflict partitions run across multiple workgroups in one ordered submission. Compact tasks retain one fused workgroup. | Measure the dispatch crossover and connected-island scaling on physical GPUs | Device foundation |
| Scene builds host task/constraint/contact arrays; Dy solve downloads before returning | Preserve efficient CPU metadata ownership where useful; remove bulk readbacks and mandatory host joins between GPU numerical stages, separating public fetch from internal dependencies | Device foundation |
| Direct buffer views/32-bit offsets cover the current bounded stage set | Qualify local offsets plus cross-allocation references before pointer-rich solver/nonrigid intake | Device foundation |
| Existing memory timing harness compares two offset-loading forms of one SAP operation | Add actual nested/multi-allocation source accesses; current results cannot select a general ABI | Device foundation |
| Software GPU CI and case wall-time metrics exist | Physical GPU frame/quality/scaling measurements; case wall time is not GPU kernel time | Every capability |

These are open engineering decisions and qualification tasks, not measured
hardware regressions. Fix each at its existing owner; do not introduce a
second scene, solver state store or per-stage scheduler.

## Numerical quality and implementation freedom

Define fidelity through contact, constraint, material and coupled-scene
behavior, supported error, stability and runtime cost. Select precision per
operation against those requirements. Source arithmetic provides the first
conformance baseline; physical accuracy is assessed through scene outcomes,
conditioning and convergence. Validated lower-cost representations remain
available without reducing the supported physics or hiding unstable cases.

## Architecture decisions

| Concern | Required design |
| --- | --- |
| Ownership | One scene owns topology and publication. Each rigid body, articulation, FEM body and particle system owns its actual state and solver data; typed interactions connect these owners. |
| Identity | Distinguish actor, body/link, shape attachment, collision element, contact manager and constraint identities. Recycling follows outstanding GPU work, caches and observable reports. Geometry and source nonexclusive shapes may be shared; each attachment has its own collision element and derived world transform. Shared shape edits reach every attachment. |
| Frames | Separate actor, body center-of-mass/principal-inertia, shape-local and world frames. Convert contact lever arms and attachments at the owning boundary; compound geometry cannot assume actor origin equals mass center. |
| Geometry | Geometry is independent of motion. Shape dispatch and cooked resources serve collision, queries and CPU/GPU execution; extend pair dispatch rather than actor-kind switches. |
| Solver composition | Keep source PGS, TGS, reduced-coordinate articulation, FEM and PBD algorithms with their numerical state. Typed coupling must return reaction to every dynamic endpoint; no generic rigid-row surrogate for a deformable body. |
| Time and visibility | One scene clock and dependency graph; subsystem substeps and coupling iterations have explicit joins. API fetch/callback visibility does not force every internal stage onto the host. |
| Storage | Source-owned persistent pools, local 32-bit offsets for hot known-arena accesses, 64-bit segmented logical references for cross-allocation traversal. Physical lowering may differ by backend. Resolve stable references outside iterative solve loops. |
| Execution | CPU and GPU placement follows measured end-to-end latency and throughput. Keep sparse structural work on the CPU when cheaper, while device numerical streams remain resident across dependent stages. CPU/compute backends do not own a second scene. Vulkan BDA and D3D12 descriptor views may specialize the same logical operation. |
| Extension | New materials, fields, contacts and solvers attach to existing topology/time/lifetime contracts. Add concrete state when a working capability needs it; avoid a speculative universal framework. |
| Fidelity | PhysX establishes initial algorithms and observable behavior. Preserve required numerical/dependency contracts; source class names, CUDA launch geometry and physical ABI are not permanent architectural constraints. Prove equivalent lowerings and measure improvements. |

## Delivery order and exit workloads

Scene and device foundations are the immediate critical path. Geometry,
rigid completion, articulations, FEM and particles then advance against those
shared contracts. SDK operations and coupling ship with their owning
capabilities; they are not deferred to a final integration project.

### Scene foundation — finish actual compound simulation

Replace actor-keyed collision state throughout World, Sc, SAP, Gu and reports
with shape attachment/element identities. Retain actor/body-keyed velocity,
island response and Dy integration. Include shape create/attach/detach,
local pose, geometry/material/filter edits and safe retirement. Preserve
shared/exclusive shape semantics and simulation/query/trigger flags;
unimplemented operations still reject explicitly. Geometry resources and
shared shape authoring state have explicit owners.

Implement the source mass-properties operation: density-weighted shape
contributions, rotated inertia tensors, parallel-axis shifts, aggregate
center of mass and principal axes. Shape attachment and mass recomputation
are distinct operations, with the source's authored update semantics.

The migration stays in the existing production path:

| Boundary | Required change |
| --- | --- |
| `world.h`, `world.cpp`, reference adapter | Actor/shape attachment operations, shape-aware events and explicit mass update; reference comparisons select their backend explicitly |
| `native::Scene::Actor`, Pair and contact/cache records | Move authored shape state out of Actor; key collision/cache records by attachment; retain body identity and mass-frame state |
| `source_sc_scene_elements_native.h` and GPU producer | Map each attached element/volume to its rigid owner and shape pose; retire elements independently; retain sibling filtering |
| Sc interaction/island edges and Gu managers | Keep one manager per shape pair; map responding endpoints to body/link nodes without collapsing multiple contacts between the same bodies |
| Dy preparation/integration and reporting | Use body mass-center lever arms and one body update; publish shape identity alongside actor identity |

Keep stable topology mappings at admission/edit boundaries; do not rebuild a
second shape-to-body registry or all shape descriptors each frame.

**Exit workload:** a rotating dynamic compound has unequal, offset shapes
and a displaced mass center. Multiple shape pairs between the same two
actors retain separate contact histories and reports while producing one
linear/angular body response. Mutate and remove one shape, reuse its slot,
change motion/filter state, reject a frame, and continue simulation without
reviving contacts or losing the body. Compare the independent CPU reference
and execute compound box contact solving on both GPU APIs. Mixed sphere
contacts also run end to end; CPU sphere generation in a GPU scene is recorded
as hybrid until the Geometry delivery supplies its device producer.

Compound rigid bodies and independent shape lifetimes are implemented in
the native scene. The complete workload remains open for shared shape
authoring, filter/trigger semantics, device-generated sphere contacts and
cross-API physical hardware qualification.

### Accelerated execution foundation — hybrid CPU/GPU scene

The native Scene owns actor, shape, contact-manager and island identity, rigid
state and result publication. Vulkan and D3D12 construct selected SAP,
aggregate, PCM, and Dy stages in one compute context, allocator and ordered
queue. Source numerical execution can run on the CPU or GPU between completed
frames without creating a second scene or discarding contact history.

- **Placement:** choose CPU or GPU stages from whole-frame cost, including CPU
  preparation, kernel time, dispatch, synchronization, transfer traffic and
  publication. Sparse structural work and bounded analytic Gu contacts may
  remain on the CPU. Broadphase, batched contact work and numerical Dy use GPU
  acceleration where their total cost justifies it. Executor changes invalidate
  affected device pose caches without rebuilding unrelated shape state.
- **Dependencies:** keep GPU-produced records resident when their consumers
  can use them directly. A PCM witness fetch is permitted when Sc genuinely
  consumes contact geometry or friction history and the resulting CPU/GPU
  handoff is more efficient than moving that work to the device. Overlap
  independent host preparation with GPU submissions and readback. Record
  unavoidable joins; do not require zero host synchronization as a design goal.
- **Scheduling:** preserve accurate contact/joint connectivity, task ordering,
  sleep/wake, rollback and generation-safe retirement in one Sc graph. Reuse
  CPU-prepared Dy task boundaries when they expose the available independent
  islands; device admission can split groups that otherwise conceal GPU
  parallelism. Tile large connected islands without serializing them into one
  workgroup. Do not introduce separate persistent CPU and GPU island owners.
- **State and lifetime:** preserve the same scene-owned contact/response
  identities across mixed GPU PCM, CPU Gu, and CPU/GPU Dy execution. CPU Gu
  touch transitions update the shared device edge history independently of
  solver selection. Device-produced numerical state must not be re-uploaded
  merely to satisfy a host-facing abstraction. Keep hazard tracking,
  completion-delayed recycling, failure-atomic publication and explicit
  rollback/rebuild behavior.
- **Addressing and precision:** qualify local 32-bit offsets and segmented
  cross-allocation references under growth, relocation, aliasing and delayed
  reuse. Compare appropriate Vulkan BDA and D3D12 descriptor lowerings.
  Select precision by supported input range, conditioning, numerical quality
  and hardware cost rather than imposing FP64 on all operations.
- **Extension:** articulations, deformables, particles and coupled constraints
  join this existing scene clock and execution graph through their real state
  and response contracts. Their algorithms determine the necessary compute
  placement; no rigid-body surrogate or independent scene scheduler substitutes
  for their numerical models.

**Acceleration-candidate exit:** one production World can compose the available
GPU stages, run real coupled contacts and rigid constraints, switch CPU/GPU
numerical execution across completed frames, preserve consistent shape/contact
history and publication, and reject failed work atomically. Vulkan and D3D12
must compile and pass their available functional gates. A complete SDK feature
inventory and physical-GPU performance acceptance are separate milestones.

**Hardware qualification:** profile quiet and changing mixed-primitive scenes
at 1K, 10K and 100K active bodies where supported, wide connected networks,
many independent islands and authored topology churn. Record numerical quality,
CPU and GPU critical paths, peak allocations, dispatches, transferred bytes and
waits. Measure CPU/GPU solver crossovers and switching costs before installing
an automatic selection policy. WARP and software Vulkan validate functional
behavior, not native GPU throughput.

### Geometry — primitives, convex, meshes and queries

Deliver sphere/capsule device contacts, general convex/support mapping,
convex-core/custom geometry where source-supported, triangle-mesh midphase,
heightfields and SDF geometry. Port cooking, BVH construction/refit, mesh
scaling, face/material identity and applicable geometry query operations
with the collision users. Reject unsupported pairs explicitly.

**Exit workloads:** mixed compound primitives and convex bodies on cooked
terrain, moving concave/SDF geometry where supported, and authored mesh
updates. Verify witness positions, normals, separation, feature/material IDs,
cache retirement and dynamics. Raycast, sweep and overlap results consume
the same authored geometry. Device-resident admission is tracked per pair;
CPU custom callbacks are explicit host boundaries.

### Rigid completion — constraints, TGS, CCD and scene semantics

Complete source contact/material forms, general constraint prep and joint
families (D6, fixed, revolute, spherical, prismatic, distance, gear and
rack-and-pinion), limits/drives/breakage/projection, PGS/TGS, discrete and
speculative CCD, sweep CCD, triggers, filtering/contact modification and
reports. Include scene queries/pruning, origin shifts, split simulate/fetch,
scene access rules, sleep/wake and capacity/failure behavior.

**Exit workloads:** a jointed rigid mechanism with limits,
drives and breakage; fast thin-feature impacts; conveyor/contact-modified
pairs; and a sleeping mixed-geometry world under insertion/removal, queries
and origin shift. Compare source scene behavior and validate long-running
stability. TGS is independently admitted; PGS success cannot close it.

### Articulations — start after shared topology/device contracts

Port reduced-coordinate links, spatial inertia, forward/inverse dynamics,
cache/Jacobian/mass-matrix operations, joint coordinates/limits/drives,
tendons/mimic constraints, loop constraints and source-supported GPU access.
Rigid and articulation contacts participate in the same interaction and
completion graph. This work need not wait for every rigid extension.

**Exit workload:** a driven multi-link mechanism manipulates and supports
ordinary rigid objects, with reciprocal contact response, limits, sleeping,
link/shape reports and lifecycle. Compare source articulation outputs plus
independent momentum and joint-residual checks. Add closed-loop and tendon
cases as those operations are admitted.

### FEM — surfaces and volumes with real coupling

Port simulation/collision mesh preparation, material models, FEM state and
integration, attachments/filters, self-contact and source-supported mixed
interactions. Volume and surface algorithms retain their own degrees of
freedom; this is not rigid motion with a deformed render mesh.

**Exit workloads:** a volumetric object deforms under gravity/contact and
returns reaction to a moving rigid or articulated participant; a surface
drapes and slides with thickness/self-contact. Check source stage results,
load/displacement response, mesh/time-step convergence, conditioning and
material limits. Attach, edit and retire objects during a multiframe scene.

### Particles and fluids — source PBD systems

Port particle buffers/phases/materials, neighborhood construction, collision,
PBD iteration, density/pressure/viscosity and source-supported boundaries
and feedback. Include particle creation/removal and applicable source
smoothing, anisotropy, neighborhood and isosurface utilities. Rendering
output does not replace fluid dynamics or force exchange.

**Exit workload:** a fluid basin with a moving dynamic obstacle and a
source-supported rigid/articulation coupling, showing reciprocal forces,
density/volume error and controlled particle loss at stated resolution and
step size. Compare source stages and hydrostatic/momentum checks. Verify
which interactions across separate particle systems are source-defined.

### SDK completion — carried with each subsystem

Complete native scene/configuration APIs, immediate mode, direct GPU state
access, standalone queries/broadphase, controllers, vehicle components,
serialization/collections, deformable/mesh/particle extensions, statistics,
visualization and debugging. Include independent packaging and consumer
integration. The vehicle program includes suspension, tire, wheel,
drivetrain, steering, braking, road-query and rigid-body/constraint coupling.

**Exit workloads:** a controller and vehicle operate on the shared queried
world; serialized scenes restore all supported topology/material/state and
declared cache behavior; a GPU consumer reads/applies batched state without
a CPU pose round trip. A reference-free standalone SDK and host consumer
exercise each advertised feature and reject unsupported operations clearly.

## Interaction coverage

For each pairing of rigid, articulation, surface, volume and particle
systems, record collision, attachment, filtering, self-contact, material,
reaction, event/query availability and backend execution. Distinguish
source-supported behavior, a source limitation and a planned extension.
Unsupported source pairings remain explicit extension work; they do not
quietly count as implemented multiphysics.

Each admitted mixed interaction must identify its typed endpoints, geometric
witnesses, response mapping, substep clock, coupling iteration/order,
completion dependency and retirement owner. Test both directions of force
exchange. Prescribed kinematic motion is identified separately from a
participant capable of receiving reaction.

Use source-supported rigid/articulation, rigid/deformable and rigid/particle
couplings as early witnesses. Audit surface/volume, articulation/nonrigid,
particle/deformable and cross-system particle combinations before promising
parity. Deeper coupling beyond source capability remains a required extension
program, with its own algorithm and validation work.

## Evidence and performance gates

Every delivered capability updates its inventory entry with the production
API, state owner, source operation, supported pairs/options/backends,
conformance workload, numerical limits, failure behavior and measurements.
Track functional, fidelity, robustness and performance acceptance separately.
No percentage based on files, kernels or subsystem names represents parity.

Use four complementary checks: independent source comparisons; exact
stage-level invariants where ordering permits; analytical/metamorphic
physics checks; and multiframe stress/lifecycle workloads. Native CPU/GPU
agreement alone is insufficient. For GPU-only features, source-stage
comparisons and reference device captures supplement physical checks.
CPU and GPU source algorithms may differ; compare equivalent configurations
and document legitimate numerical differences rather than demanding false
bitwise parity or widening tolerances after a failure.

Declare quality budgets before comparisons: contact/constraint residuals,
penetration, trajectory and angular error, density/volume error, and
energy/momentum balance after accounting for external work and intended
dissipation. Run time-step/iteration/mesh convergence where applicable.
Specify units, tolerances scale and supported parameter ranges.

Performance measurements cover a quiet scene, authored churn, many small
islands, one large island, dense contacts, mixed geometry and eventually
mixed solvers. Scale workloads by actual contacts/constraints/vertices/
particles as well as actor count. Start with 1K/10K/100K rigid checkpoints;
report resource-limit failures rather than silently simplifying them.

Record warmed Release-build median/p95/p99 host frame time, device critical
path/stage timestamps, host blocking, bytes transferred each direction,
submissions, descriptor/pipeline work, allocations/growth and peak memory.
Record adapter, driver, API, compiler, scene seed, step, solver settings,
quality and actual CPU/GPU stage selection. Construction and warm-up are
reported separately. Extend the existing scene harness and compute counters
rather than creating another benchmark scheduler or simulation path.

Set an absolute target frame budget and relative regression limit for each
representative hardware workload before accepting it. Establish these from
measured baselines; no hardware speed acceptance is currently closed. Compare
against the same-quality native baseline and source implementation where
hardware permits. An intentionally more expensive feature needs an explicit
quality/cost tradeoff, not a hidden change to the benchmark workload.

Vulkan 1.4 and D3D12 SM6.6 remain the baseline, with actual feature probes.
Wave32 must work; width-independent algorithms may use wave64. AMD is the
primary performance target, including both UMA and discrete-memory costs.
Software Vulkan/WARP remain functional infrastructure. Optional BDA, matrix
instructions or newer shader facilities must earn their packing, occupancy
and synchronization costs in integrated workloads.

## Beyond the source baseline

The extension backlog includes nonlinear/anisotropic/viscoelastic and
near-incompressible materials; rods/strands with bending and twist; thin
shells/membranes and pressure; wetting, adhesion, lubrication and film
transport; enclosed fluid/structure coupling; and material/surface fields
such as moisture and temperature. These are additional physics to implement,
not effects assumed to emerge from source parity.

Acceptance includes sliding nonlinear volumes, a compressed fluid-filled
membrane, strands attached to deforming surfaces, and pressure-dependent wet
sliding, eventually combined in one scene. Validate constitutive models,
resolution, coupling convergence and energy/momentum budgets. Partitioned
solvers are valid when their coupling is demonstrated. Measured fidelity and
frame cost determine supported use; visual plausibility alone does not.

## Execution discipline

Finish the current Scene delivery and Device decisions before another
unrelated optimization sequence. Start articulation, FEM and particle
production work as their shared prerequisites close; do not wait for complete
rigid SDK parity. Each implementation slice includes public use, real solver
execution, lifecycle and evidence. Optimize when it enables the next
capability or addresses a measured integrated bottleneck.

Keep this inventory current in the capability's own change. Historical
source-intake documents retain technical mappings, but do not define a
second milestone/status system. Review text and CI status cannot establish
semantic correctness; qualification must cite executable behavior.

Technical contracts: [architecture](ARCHITECTURE_CONTRACT.md),
[source intake](NATIVE_SOURCE_INTAKE.md),
[GPU trace](PHYSX_GPU_PORT.md),
[address lowering](GPU_MEMORY_ADDRESSING_RFC.md),
[performance](GPU_PORT_PERFORMANCE.md), and [build/integration](BUILDING.md).
