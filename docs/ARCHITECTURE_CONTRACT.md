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
  Algorithm departures require source-grounded justification and tests.
- Keep original source classes and method boundaries where they correspond
  to actual ownership and responsibilities. Helpers are welcome for a
  cohesive private operation; **do not accumulate a new abstraction for
  every test, CUDA launch, buffer family, or PR**.
- Production behavior must not silently switch to a different backend
  on failed GPU admission. If a path is unsupported, fail explicitly
  and make the execution mode observable.

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
Hardware measurements can follow coherent subsystem integration, as set out
in [GPU_PORT_PROGRAM.md](GPU_PORT_PROGRAM.md).

## 3. Source-faithful production integration

For each change identify its native FyzzXYZZY production caller, authoritative
state owner, and the corresponding original PhysX source operation. A dedicated
fixture may validate an operation in isolation, but **passing that fixture
does not mean a native FyzzXYZZY scene is executing it**. Keep the reference
PhysX CPU baseline independent; experimental `world_vulkan` and `*_arena`
demonstrations are not substitutes for integrated production physics.

Track the exact production boundary and absent connections against the
[active capability roadmap](ROADMAP.md). The admitted native GPU broadphase
now publishes source-derived multi-frame found/lost pairs into native
Sc/PCM/Dy on Vulkan and D3D12; GPU narrowphase and rigid dynamics are not
yet integrated. A separate reference scene, test-only GPU fixture, or
independent simulation scheduler cannot substitute for either capability.

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
- Tests must include adversarial numeric limits, many-body behavior,
  cross-frame lifecycle and independent PhysX comparisons when relevant.
  No test alone proves complete source parity; review the source mapping.
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
