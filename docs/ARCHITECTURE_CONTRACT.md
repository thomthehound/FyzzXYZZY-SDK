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
- GPU compute is the primary full-physics implementation target. The
  selected GPU scene must retain shape/transform, broadphase, contact,
  constraint and integrated body state on the device across dependent stages.
  The host owns scene admission and required public result publication,
  not a mandatory intermediate CPU copy of each device result. Device
  stage submissions remain in the one scene task graph and resource owner.
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
