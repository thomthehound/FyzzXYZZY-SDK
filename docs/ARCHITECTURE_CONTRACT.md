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

For each change identify the concrete production caller and where it
connects to its original source owner. A dedicated fixture may validate
an operation in isolation, but **passing that fixture does not mean a
live PhysX scene is using it**. Keep the CPU baseline; do not turn
experimental `world_vulkan` or `*_arena` demonstrations into a
replacement production PhysX pipeline.

Track exact implementation gates and known absent connections. In particular,
[GPU broadphase G3](https://github.com/thomthehound/FyzzXYZZY-SDK/issues/101)
is **not complete** until the source-derived GPU SAP executes from a real
`PxScene` and publishes correct found/lost pair state into the scene task
graph over multiple frames.

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

Pull requests with production/test code changes must supply four review
sections, also instantiated by `.github/pull_request_template.md`:

1. **Owner and source contract:** production entry point, authoritative
   state and source correspondence
2. **Runtime cost and scheduling:** allocations, submissions, waits,
   pipeline/descriptor lifetimes and relevant performance implications
3. **Integration evidence:** executable tests on the relevant platforms;
   explicit difference between unit, subsystem, and live-scene evidence
4. **Redundancy and retirement:** what was reused, consolidated or retired,
   or why an additional representation is indispensable

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
