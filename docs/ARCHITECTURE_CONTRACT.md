<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SDK-wide architecture contract

**Applies to every production subsystem and supported backend:** native CPU, reference PhysX,
portable GPU, Vulkan, D3D12, allocators, scheduling, kernel translations,
Godot adapter, future engine consumers, and packaging. Source-code tests
are required to validate invariants, but they do not define the production
architecture. Reuse the narrowest existing owner rather than introducing
a new controller simply to make another increment independently testable.

## 1. One authoritative owner and state representation

- FyzzXYZZY's native scene and task graph are the production authorities for
  simulation and collision state. PhysX source defines the port's behavioral
  and algorithmic reference, not its runtime ownership. GPU broadphase,
  narrowphase, solver, articulations and buffers participate in FyzzXYZZY's
  single native scene lifetime.
- A component must have one clearly identified allocation owner, one
  authoritative state representation, and an explicit lifetime/retirement
  contract. Borrowed views must not quietly become shadow ownership.
- Portable GPU code may lower CUDA mechanisms (addresses, kernel arguments,
  buffers, descriptors, stream/event/synchronization). Port the source scene
  dependency graph into the one FyzzXYZZY scheduler; do not let reference
  PhysX, a kernel fixture, or a second owner drive production frames.
  Preserve required source dependencies and numerical contracts. Physical
  storage, task batching and hardware work decomposition may change when
  equivalence and measured cost justify the lowering.
- Keep cohesive ownership and responsibility boundaries. Source classes
  and methods provide correspondence, not a permanent physical ABI or
  CUDA-shaped scheduling requirement. Helpers serve a real production
  operation; do not accumulate an abstraction per fixture, launch or buffer.
- Actor motion, body/link dynamics, shape attachments, collision elements
  and contact managers have distinct identities. Collision keys identify
  shapes; solver endpoints identify the actual responding bodies or nodes.
  Geometry sharing does not merge collision elements. Source nonexclusive
  shape edits propagate to every attachment without sharing their contact
  histories. Actor, shape and center-of-mass/principal-inertia frames remain
  explicit.
- Articulations, FEM bodies and particles retain their numerical state and
  degrees of freedom under the scene lifetime. Typed interactions connect
  them with reciprocal response and explicit substep/completion contracts.
  A reserved node enum or rigid-body substitute does not implement a solver.
- GPU compute is the primary execution target for dense, repeatedly updated
  physics data and numerical work. Persist device-produced contacts,
  constraint state, velocities and poses across dependent GPU stages. The
  CPU may own sparse topology, contact-manager handles, island/task metadata,
  authoring, callbacks and retirement when that placement minimizes total
  cost and preserves scene semantics. A host-owned decision is not a reason
  to fetch its entire device input or interrupt the GPU queue each frame.
  Device submissions and CPU work remain in one scene task graph.
- Native CPU execution is an explicit fallback for unsupported devices and
  a correctness baseline. Reference CPU execution is only an independent
  comparison mode. Production behavior must not silently switch backends
  on failed GPU admission: fail explicitly and make stage execution
  observable. Explicit hybrid configurations identify their host-produced
  contacts, callbacks and solver stages. CPU fallback covers only features
  actually implemented there; source GPU-only features have separate
  numerical/reference validation requirements.

## 2. Native runtime cost is part of correctness

Every integration change must account for CPU and GPU work introduced by
the port: actual GPU submissions, staging allocations and bytes, physical
heap growth and copy bytes, descriptor writes, pipeline creations, resource
retirement, synchronization/waits and kernel work. Eliminate redundant
work where source dependencies allow it. Preserve correctness and lifetime
before reducing submissions or barriers. Do not insert blocking per-kernel
waits, unconditional polling, hot-path shader/pipeline compilation, or
per-frame re-creation of device-owned pipelines.

**Optimize the real frame path, not isolated microtests.** CI software Vulkan
and D3D12 WARP prove functionality, not relative native GPU throughput.
Use the [active implementation program](ROADMAP.md) for hardware workloads,
quality budgets and delivery dependencies. Source-matched precision, large
connected-island parallelism, cross-allocation addressing and residency must
be qualified before pointer-rich articulation/nonrigid intake scales up.
A blanket FP64 requirement, one-workgroup task mapping or bounded arena ABI
is not accepted as the general engine architecture without that evidence.

### Heterogeneous execution and synchronization

Choose each operation's processor from **end-to-end frame cost**, not an
all-CPU or all-GPU ownership target. Sparse, branch-heavy or low-volume
operations can remain on the CPU even when their consumers are GPU kernels.
Large regular scans, contact generation, numerical preparation, solving and
integration belong on the GPU when supported and performant. Ownership,
execution location and data residency are separate decisions.

- A CPU operation is acceptable on the simulation critical path when its
  useful work and data handoff cost less than the equivalent GPU operation.
  Do not migrate allocators, contact-manager identity or island bookkeeping
  solely to obtain a device-resident label.
- Preserve GPU-produced numerical data for later GPU work. Avoid a mandatory
  GPU completion → CPU readback/decision → GPU resubmission between stages
  that can consume the same device records. Transfer compact events,
  identities and authored changes rather than bulk contact/body snapshots.
- A host-side view may be maintained for filtering, public events or task
  authoring when its delta update is cheaper than reconstructing GPU state.
  Identify the authoritative owner, supported divergence window and
  retirement/rollback rule; never create an untracked second simulation.
- Distinguish quiet steady-state frames, topology churn, explicit callbacks
  and public fetch. Synchronization justified by a topology change or API
  visibility does not authorize an unconditional per-frame wait.
- Compare alternatives using host and GPU critical-path time, queue stalls,
  transfer bytes, submissions, memory growth and physical adapter results.
  UMA coherence does not itself eliminate synchronization; separate
  device-memory bandwidth from CPU/GPU completion dependencies.

A placement change without a demonstrated critical-path benefit is not
progress toward the GPU-performance gate. Correctness may justify a
temporary hybrid boundary; record its cost and the consumer that requires it.

### Stage placement and dataflow qualification

An accelerator is a consumer of packed, parallel work, not a replacement
for Sc command processing or a second runtime authority. Each source stage
specifies its input residency, produced data, CPU consumers, device consumers,
queue dependencies, persistent state, and retirement boundary. Placement is
chosen for the complete producer-to-consumer chain, not isolated kernel time.

- **Broadphase:** incremental CPU SAP is a candidate for sparse endpoint
  crossings and low pair churn; GPU SAP is a candidate for large dynamic
  sets and GPU-resident bounds/contact consumers. Neither is required for
  every scene. Keep the existing single authoritative BP pair history in
  each configuration. Selection is explicit until workload crossover and
  state-transition costs justify runtime switching.
- **Narrowphase:** CPU analytic/contact-modification work may contribute
  bounded records to the device stream. Dense GPU PCM must feed GPU
  numerical preparation without a routine full witness round trip.
  Stable contact-manager identity does not require CPU observation of
  every contact point. New identities, callbacks, and CPU friction
  consumers have explicit reconciliation edges.
- **Topology and solver:** CPU Sc owns persistent manager and island
  identity. Device-connected components and response worklists are
  transient numerical products only when they expose more parallelism
  than CPU-prepared task spans. Never build both task sets merely to choose
  between them after the work is done.
- **Storage:** compact CPU-authored inputs may use direct coherent UMA
  allocations when supported; hot numerical arrays use GPU-preferred
  storage. Shared addressing does not eliminate cache-coherence costs,
  producer/consumer barriers or fences. Discrete GPU paths stage only
  the necessary deltas and outputs.
- **Scheduling:** batch dependent dispatches on the existing ordered queue;
  submit once per useful command segment, not once per translated source
  kernel. CPU work independent of a pending GPU segment may overlap it.
  Do not insert a host wait to relay device-produced records between
  device consumers. A host join names its actual CPU consumer.

Compare placements on the same production Scene with matching geometry,
contact density, moving fraction and numerical quality. Capture quiet,
sparse-motion, dense-motion and high-churn frames at 1K, 10K and 100K
bodies where supported. Record full-frame p50/p95, CPU enqueue/prepare/join,
actual GPU work and queue idle time, submissions, barriers, uploads,
readbacks, peak residency and device/host state reconstruction. An isolated
CPU or GPU SAP microbenchmark cannot establish a frame-level winner; a
configuration lacking an equivalent production path remains unqualified.

### Binding scene execution graph

The memory placement, topology reconciliation and publication contract is
specified in [Scene-owned compute frame](SCENE_COMPUTE_FRAME.md).

The scene owns **one simulation transaction** and one dependency graph across
CPU and device work. A function call is not a completion edge. Every stage
declares its producer, consumers, residency, possible host observation and
rollback lifetime; backends lower those dependencies without introducing
another simulation scheduler.

| Stage | Authority and work placement | Required successor |
| --- | --- | --- |
| Admit | CPU validates authored changes, sparse handles, filters and topology edits; device mirrors only the inputs needed for execution | Sparse changes and device uploads become visible before their consumers |
| Broadphase | CPU or GPU executes the selected overlap algorithm; Sc owns the stable shape and contact-manager identities | Manager admission precedes newly identified contact work |
| Narrowphase | Each manager has one numerical Gu producer, CPU or GPU. GPU-owned witnesses and contact history remain resident where they have GPU consumers | The actual Gu touch, not speculative overlap, determines accurate connectivity; Sc may join witnesses needed for CPU correlation |
| Frame topology | CPU Sc owns persistent identity, active island routing and public graph. Transient GPU admission splits compact numerical groups where it exposes useful independent tasks | Sc task ranges or device-admitted ranges supply Dy without a second persistent graph |
| Dynamics | CPU or GPU Dy consumes the prepared rigid contact, joint and body streams. Execution can change between completed frames in the same Scene | Each selected numerical stage completes before authoritative Sc writeback and publication |
| Publish | One scene completion boundary validates compact topology/event deltas and requested results, reconciles public CPU views and commits all persistent histories | Failed transactions retire or restore submitted device work before state reuse |

**Ownership is distinct from observation.** The CPU can own an EdgeIndex and
its published lifecycle without owning the numerical witnesses occupying its
device slot. The GPU can construct a transient numerical contact graph while
the CPU maintains stable identifiers and authoritative active islands. Neither
may independently invent touch transitions. Cached contact/friction history
has one owner per epoch, with explicit invalidation when its producer changes.

**Preserve useful producer-to-consumer locality.** GPU contact generation
can feed device correlation and numerical solve directly when supported and
less expensive. CPU Sc friction correlation or accurate-island work may
consume GPU witnesses at an explicit earlier join when that division
minimizes whole-frame cost. A host join must have a real consumer; a GPU
launch must expose useful parallelism or computation. Measure transfer,
submission and waiting costs before moving these dependencies. Neither
processor duplicates numerical operations for implicit verification.

**Ordered source submissions.** CPU-authored device inputs are snapshotted
when queued. Uploads, device copies and dependent shader dispatches retain
their source order in one native batch; a new host upload does not itself
submit or wait on the GPU queue. Vulkan and D3D12 pack the owned upload
bytes into fence-owned transfer staging. Physical heap-root retirement
submits outstanding source commands before releasing the root; rollback
discards only commands that never entered the queue.

**Host report admission.** Source SAP schedules its descriptor readback
asynchronously and prepares bitmap retirement while the transfer is in
flight. Sc publishes found/lost pairs and rotates source endpoint history
only after the descriptor fence and source-lifetime validation complete.
GPU PCM witness joins remain mandatory where CPU Gu correlation needs
those witnesses; no unchanged-topology assumption may bypass them.

**No false connectivity shortcut.** Broadphase overlap can feed speculative
waking and admission; it cannot stand in for accurate contact connectivity.
Mixed rigid, articulated and nonrigid interactions must use typed responding
nodes, ordered constraints and explicit cross-system dependencies within
this same graph. A new feature must not require a second independent frame
loop, contact registry, island manager or solver-state truth.

**Connectivity and solver response are separate facts.** An accurate
contact or authored joint can connect responding nodes without producing
constraint rows. Device island admission preserves that connectivity while
excluding response-disabled edges from numerical task counts. An active seed
can admit its entire connected component; its position/velocity iteration
limits are the maxima of all admitted component nodes, not merely the seed.
Device component roots are transient execution IDs, while EdgeIndex and
source-ordered constraint identity remain stable. Runnable component tasks
are compacted by a device-resident prefix in canonical root order rather
than racing to claim atomic task ordinals. The same prefix reserves exclusive body and response-descriptor spans
for runnable components, without another host count or a per-component
allocation. A single stable device root/type sort fills responding-body spans in
source-node order and response spans in original Sc encounter order.
Small streams use one direct GPU phase; larger streams share global
radix passes across both input types. Within each canonical island the
device orders responding bodies first, then source-sorted 1D constraints,
then contact managers. The authored joint input preserves descending
constraint identity within its type; the device does not invent a
second joint ordering. The compact root prefix now emits the exact 36-byte native GPU Dy task ABI directly: descriptor and partition capacity ranges, body ranges, iteration maxima, and the error word. Root identity remains in the per-root task-index map, avoiding a second task stream or conversion dispatch. The same scatter writes typed
contact and joint source tokens without another body-sort pipeline,
dispatch or upload. Direct joint-row tokens are permitted only when
the source ordinal addresses device-authored 1D geometry; other joints
refer to the full Dy descriptor table. Numerical table ordinals are
not interchangeable with graph-edge ordinals. Device admission receives
the explicit capacities of the contact-manager, device-joint-row and full
descriptor tables; an out-of-range numerical token on a touching contact or live joint
invalidates the resident task launch before Dy may consume it. A
non-touching broadphase overlap requires no numerical source record.
Dormant and response-disabled edges do not occupy
numerical descriptor slots. Independent components can
execute concurrently, but their task identity is stable across runs;
constraints sharing a responding body retain the required solver ordering.
Nonresponding static and kinematic endpoint nodes may participate in
source-edge admission but never enter the responding-body prefix or
connect otherwise independent dynamic components through a shared
nonresponding endpoint. Their zero iteration words are valid, and Dy
initializes only the actual emitted solver-body ordinals rather than
assuming a dense responding prefix.
The generated indirect launch triple is copied into the source queue's
existing physical indirect resource after island compaction; it requires no
additional host upload, allocation or synchronization boundary. Native Dy
copies the task, body-ordinal and response-token streams between resident
source allocations. Task and body-ordinal destinations are outside the Dy
host-upload extent; raw descriptors, colors and partition-state scratch are
also omitted from the source packet. The remaining input prefix and
numerical source-table suffix share one staging submission. Its compact-task
numerical launch consumes the device indirect triple rather than a CPU
workgroup count. Sc supplies separate numerical task boundaries for independent active
islands. One-constraint tasks with one or two responding bodies execute
as GPU lanes; each lane owns its PGS sequence and body integration, with
up to 128 tasks per workgroup. Other compact tasks execute through the
ordinary fused workgroup path. Both use the same command stream and
numerical storage; one dependent island does not disable packing for
unrelated islands. Large tiled tasks retain their partition dependencies. The host provides bounded numerical allocations, not an
exact prediction of runnable roots or response counts. The queued Dy shader
rejects admission errors and any task/body/descriptor count exceeding its
own buffer capacity before initializing numerical state. Unused capacity
does not launch numerical work or publish body integration. Dy fetches
the actual compact task records and body ordinals as additional spans in
its existing result batch, without extra GPU mirror copies or an additional
host synchronization. Writeback uses those GPU-authored ranges.
Rejected epochs retain their original task-error reporting before rollback.
Large tiled tasks retain
their explicit partition launch dependencies until the device-owned wide
scheduler is complete.

For compact PCM-owned production frames, Scene consumes the accurate
island boundaries it has already prepared. It does not reconstruct those
islands in a second device graph. Explicit device-authored numerical
admission remains available at the producer interface, where a resident
source graph rather than CPU Sc must provide the task stream. Wide tiled
tasks retain their existing numerical partition scheduler. The Gu witness
join and Sc friction/constraint preparation precede both execution choices
when those operations consume contact geometry. Zero authored velocity
iterations retain Dy's mandatory final solve/writeback iteration.

CPU Gu and GPU PCM publish physical-touch transitions into the same device
EdgeIndex history. Accurate device connectivity and new-touch wake admission
read that resident history for both Gu producers. Sc-authored edge metadata
cannot override a missing physical touch or invent a new-touch transition.
Only authored joints connect independently of the contact history.
Host-only Gu epochs journal only the prior/current touch words from the
existing resident 84-byte contact records; mixed epochs retain
the full GPU manifold journal but snapshot only touch words for CPU-owned
managers. Ordinary CPU Gu contacts excluded from the PCM candidate
list publish eight-byte EdgeIndex/current-and-prior-touch records on the same
ordered source queue. These transitions snapshot the old two-word device
history and restore it ahead of the PCM journal on rollback. The PCM owner
reserves the complete Sc EdgeIndex high-watermark, so an ordinary CPU Gu
contact cannot silently fall outside the resident history allocation.
A rejected transaction restores the original device touch epoch
before another producer can update that EdgeIndex. In mixed frames, CPU Gu
uploads enter the ordered source queue immediately after their witnesses are
ready, independently of the preceding GPU PCM witness readback. Sc's host
join retires only producer tokens preceding that readback; later uploaded
CPU Gu and device touch publication stay in flight for GPU consumers.
For compact GPU tasks, Scene now emits contact-manager identities without
resolving their numerical solver-row endpoints on the CPU. Native Dy
reconstructs responding contacts from its resident contact-preparation
table. Only zero-response managers, which have no contact-preparation
record, resolve their endpoint ordinals during the final source-manifold
publication. Wide tasks still supply full endpoints for their host launch
bounds; the production coupled-scene regression verifies that compact
contact descriptors never carry redundant CPU-resolved endpoints.

Dy's GPU-owned compact tasks now bypass host endpoint-degree
classification entirely. The host planner scans source descriptors only for
tasks with at least 512 body ordinals and 512 response descriptors, where
wide GPU tiling can change the launch schedule. Smaller tasks retain their
existing device-colored fused path without dense degree-array clearing or
redundant host numerical inspection.

Native GPU Dy no longer allocates CPU numerical contact-Jacobian rows
during Sc friction correlation. Sc maintains the source Gu row ordinal
and packs CPU Gu's 28-byte contact input directly while the exact correlated
header normal is in scope; PCM contact geometry stays resident. Device
normal impulses return directly into the original Gu-indexed Sc writeback
without first copying through a CPU solver-row mirror. GPU Dy's friction path also packs source-ordered 24-byte body-local
anchors directly during Sc correlation, retaining exact tangent-row
offsets without allocating CPU tangent-row placeholders. Device Dy
constructs both tangents per anchor and returns compact world impulses.
CPU PGS retains its full contact and friction rows unchanged.

The source Dy numerical stage submits its existing solver command graph
independently of the CPU result fetch. Its queued execution consumes the
device-ready PCM/CPU Gu stream after the input producers, and the normal
single Dy readback, task-error checks and Sc writeback occur at the result
join. A pending Dy epoch cannot be overwritten or published on rollback.
The shared Context retains one outstanding readback batch, so the PCM
witness DMA must be retired before Dy requests its final result transfer.
Native Scene now enqueues GPU Dy numerical work before initializing
its Gu-indexed result buffers and consuming source force modifications.
The existing Dy completion join still validates numerical results before
Sc publishes contact impulses, integrated poses and sleep state. CPU
accurate-island/friction/Dy descriptor construction still happens after
the PCM witness join; this scheduling overlap does not retire that
earlier CPU admission dependency.

**Current hybrid execution boundary:** general GPU PCM publishes its
contact witnesses for CPU Sc correlation and accurate-island routing before
Dy's numerical execution. Independent CPU Gu work overlaps the queued GPU
PCM generation and witness transfer. GPU and CPU Dy share the authoritative
Sc scene, contact history and completion contract; changing executor only
invalidates affected device pose caches. A transient device admission pass
is selected only when it exposes numerical parallelism absent from the
existing CPU task groups. The witness join is a measurable cost, not an
unconditional migration requirement.

Qualify CPU and GPU critical paths and scene-consumer synchronization before
moving this boundary. Verify failure recovery across queued witness fetch,
mixed contact producers, topology splits/merges, anchor persistence,
sleep transitions, and executor changes on both APIs. A performance
improvement requires physical-device evidence of the critical path and
queue occupancy, not a device-residency count.

## 3. Source-faithful production integration

For each change identify its native FyzzXYZZY production caller, authoritative
state owner, and the corresponding original PhysX source operation. A dedicated
fixture may validate an operation in isolation, but **passing that fixture
does not mean a native FyzzXYZZY scene is executing it**. Keep the reference
PhysX CPU baseline independent; experimental `world_vulkan` and `*_arena`
demonstrations are not substitutes for integrated production physics.

Track the exact production boundary and absent connections against the
[active implementation program](ROADMAP.md). The native GPU broadphase
publishes source-derived multi-frame found/lost pairs into native Sc/PCM/Dy
on Vulkan and D3D12. Explicit GPU PCM and Dy shader bundles also execute
box contacts, contact/distance preparation, PGS, friction and integration
for the admitted rigid subset. Host contact publication, task construction
and result fetch remain intermediate boundaries. These are integrated
capabilities with limited scope, not full GPU PhysX or residency acceptance.
A reference scene or independent fixture scheduler cannot substitute for
production execution.

### Numerical quality

Choose numerical representations per operation against physical error,
stability and measured cost. Source arithmetic establishes the initial
conformance baseline. Contact, constraint, material and coupled-scene
behavior define the capability; precision choices require evidence across
the supported parameter domain and must preserve that capability.

### Source fidelity is a release criterion, not just an intent

For every physics algorithm, shader, and orchestration change:

- Identify the **specific upstream source methods**, input/output invariants,
  state lifetime, ordering, numerical assumptions and output semantics.
  Similar mathematical vocabulary (SAP, SAT, PGS, Coulomb friction) is not
  sufficient evidence of an equivalent PhysX implementation.
- Classify the change as **source-translation** or
  **semantics-preserving-lowering**; demonstrate the correspondence with
  source-level mapping and parity tests. A lower-cost implementation is
  acceptable when it preserves the relevant invariants, edge cases and
  observable outputs; measurements are needed for performance claims.
- A prototype, toy substitute, missing angular term, collapsed manifold,
  fabricated contact, invented scheduling edge or simplified friction
  equation **must not silently become production physics**. If essential
  source behavior is missing, mark the operation unsupported or isolate it
  behind an explicit non-production experiment; keep the acceptance gate
  open. Never advance a fidelity gate merely because a fixture passes.
- Review differences in memory/ownership *and* actual algorithmic
  behavior: contact generation, Jacobians, effective mass, friction
  correlation, restitution, warm-start invalidation, island partitioning,
  substep clocks, task dependencies and publication. Classify N/A with a
  concrete reason rather than substituting a plausible approximation.
- Tests include adversarial numeric limits, many-body behavior, cross-frame
  lifecycle and independent source comparisons. Native CPU/GPU agreement
  cannot detect shared mistakes. GPU-only source capabilities require
  appropriate source-stage/device evidence plus independent analytical,
  conservation and convergence checks. Numerical budgets precede results;
  legitimate CPU/GPU algorithm differences require equivalent configurations.
- Record capability, fidelity, robustness and performance acceptance
  separately. Source limitations and planned extensions remain explicit.
  Full source coverage includes public SDK operations and supported mixed
  interactions, not only the individually named solver subsystems.
- For an intentional departure, document why it is equivalent, how it
  reduces cost, what it changes, and which executable conformance checks
  would detect a regression. Anything unproven stays non-production.

## 4. Consolidation and deletion

- Reuse existing mechanisms rather than adding overlapping representations.
  Migrate consumers of obsolete scaffolding, retain semantic regression
  coverage, and delete obsolete production paths once proven redundant.
- Preserve genuinely necessary CPU fallbacks, GPU specialization and
  backend lowerings. Similar names alone do not prove equivalent purpose.
- Favor fixing interfaces and ownership at the existing call site over
  adding compatibility facades that permanently retain the old problem.
- Every meaningful code PR must identify any duplicate implementations,
  lifetime/ABI impacts, and a retirement plan for transitional abstractions.
  It is acceptable to state **"none; checked"**, with a rationale, but
  not to omit the analysis.

## 5. Enforced review contract

Pull requests with production/test code changes must supply five review
sections, also instantiated by `.github/pull_request_template.md`:

1. **Owner and source contract:** production entry point, authoritative
   state and source correspondence
2. **Runtime cost and scheduling:** allocations, submissions, waits,
   pipeline/descriptor lifetimes and relevant performance implications
3. **Integration evidence:** executable tests on the relevant platforms;
   explicit difference between unit, subsystem, and live-scene evidence
4. **Redundancy and retirement:** what was reused, consolidated or retired,
   or why an additional representation is indispensable
5. **Source fidelity and deviations:** one explicit fidelity classification;
   PhysX source method correspondence, departures and equivalence evidence.
   Native physics code—including translated kernels, historical
   contact and GPU-demo paths—cannot claim `not-applicable` or
   `non-production-only` when it changes production algorithms. Independent
   backend mechanism lowerings still require explicit correspondence and
   justification, even when PhysX source arithmetic is untouched.

The lightweight `SDK architecture contract` workflow checks that a
non-draft code PR supplies substantive sections and reruns on PR-body
edits/ready-for-review. This is a review gate, **not** a claim that text
alone proves runtime correctness. Reviewers must reject ungrounded claims
and require executable evidence for changes that affect behavior.

**GitHub merge enforcement requires branch protection:** configure a
ruleset covering `main` to require the `Architecture ownership and runtime
review` status check, as well as Linux and Windows SDK CI. A check that
runs but is not required by branch rules is informative, not sufficient to
prevent a bad merge. The SDK cannot configure repository-wide rulesets from
its source tree; this is a repository-owner setting and must be verified
separately.
