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

### Binding scene execution graph

The scene owns **one simulation transaction** and one dependency graph across
CPU and device work. A function call is not a completion edge. Every stage
declares its producer, consumers, residency, possible host observation and
rollback lifetime; backends lower those dependencies without introducing
another simulation scheduler.

| Stage | Authority and work placement | Required successor |
| --- | --- | --- |
| Admit | CPU validates authored changes, sparse handles, filters and topology edits; device mirrors only the inputs needed for execution | Sparse changes and device uploads become visible before their consumers |
| Broadphase | GPU processes dense bounds, persistent overlap history and reports; CPU admits or retires stable contact-manager identities from required report deltas | Manager admission precedes new contact work |
| Narrowphase | Each manager has exactly one numerical Gu producer, CPU or GPU. GPU-owned witnesses and persistent device contact history remain resident | Device touch state drives device constraint admission without full witness readback |
| Frame topology | CPU owns stable scene handles, callbacks and persistent public graph. GPU owns same-frame accurate contact connectivity, affected island routing and numerical task admission when those depend on GPU touch | Actual touch, not speculative overlap, determines active constraint connectivity |
| Dynamics | GPU consumes resident contact, friction, joint and body streams for preparation, partitioning, solving, integration and sleep decisions; sparse CPU-only work enters through explicit dependencies | Device-dependent stages run in ordered submissions without a default host fence between them |
| Publish | One scene completion boundary validates compact topology/event deltas and requested results, reconciles public CPU views and commits all persistent histories | Failed transactions retire or restore submitted device work before state reuse |

**Ownership is distinct from observation.** The CPU can own an EdgeIndex and
its published lifecycle without owning the numerical witnesses occupying its
device slot. The GPU can own the current frame's accurate contact graph while
the CPU maintains stable identifiers and a reconciled public graph. Neither
may independently invent touch transitions. Cached contact/friction history
has one owner per epoch, with explicit invalidation when its producer changes.

**Normal-frame dependencies must remain on-device** from general GPU
contact generation through contact/friction correlation, accurate task
admission, constraint preparation and solving. Sc may consume compact
transitions or topology changes at the final publication boundary.
Callbacks, CPU-only geometry and sparse topology edits may require earlier
host joins, but must identify the actual consuming task and affect only the
necessary workload; they are not permission for a global per-frame witness
download. A numerical operation is never duplicated on both processors as
an implicit verification or scheduling strategy.

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
second joint ordering. The same scatter writes typed
contact and joint source tokens without another body-sort pipeline,
dispatch or upload. Direct joint-row tokens are permitted only when
the source ordinal addresses device-authored 1D geometry; other joints
refer to the full Dy descriptor table. Numerical table ordinals are
not interchangeable with graph-edge ordinals. Dormant and response-disabled edges do not occupy
numerical descriptor slots. Independent components can
execute concurrently, but their task identity is stable across runs;
constraints sharing a responding body retain the required solver ordering.
The generated indirect launch triple is copied into the source queue's
existing physical indirect resource after island compaction; it requires no
additional host upload, allocation or synchronization boundary.

CPU Gu and GPU PCM publish physical-touch transitions into the same device
EdgeIndex history. Host-only Gu epochs journal only the prior/current touch
words from the existing resident 84-byte contact records; mixed epochs retain
the full GPU manifold journal but snapshot only touch words for CPU-owned
managers. A rejected transaction restores the original device touch epoch
before another producer can update that EdgeIndex. In mixed frames, CPU Gu
uploads enter the ordered source queue immediately after their witnesses are
ready, independently of the preceding GPU PCM witness readback. Sc's host
join retires only producer tokens preceding that readback; later uploaded
CPU Gu and device touch publication stay in flight for GPU consumers.
Dy's GPU-owned compact tasks now bypass host endpoint-degree
classification entirely. The host planner scans source descriptors only for
tasks with at least 512 body ordinals and 512 response descriptors, where
wide GPU tiling can change the launch schedule. Smaller tasks retain their
existing device-colored fused path without dense degree-array clearing or
redundant host numerical inspection.

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

**Current migration boundary:** general GPU PCM still downloads its 84-byte
contact records before CPU Sc friction correlation and accurate-island
scheduling. The ordered PCM enqueue/join starts general GPU PCM before
independent CPU primitive witness generation and speculative IslandSim
first/second-pass topology admission. Both numerical producers and SAP-owned
speculative graph admission can run before Sc observes GPU contact witnesses.
The packed witness DMA is submitted immediately after the final PCM
kernel and remains Context-owned until Sc reaches its actual-touch join.
Both API backends publish the borrowed Gu arena only on completion;
a rejected transaction discards its DMA without exposing stale contacts.
Only actual-touch accurate topology waits for Gu. Lost-edge route repair,
wake/sleep decisions and Dy task admission retain their source dependencies.
This overlap does not remove the final same-frame Sc synchronization
dependency. The next integration gate is GPU-consumed touch and friction/task
admission with correct same-frame connectivity, followed by compact public
reconciliation. Only then can normal GPU PCM -> Dy be considered free of an
intermediate host witness fence. Test rollback between enqueue and join,
mixed producer ownership, topology splits/merges, anchor persistence, sleep
transitions and both API backends. Physical-device traces must show the
actual critical path, transfer counts and queue idle intervals before
claiming a performance improvement.

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
