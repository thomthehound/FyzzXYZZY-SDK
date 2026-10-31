<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FyzzXYZZY capability roadmap

Status: October 31, 2026. This is the authoritative **active** roadmap.
Implementation and executable acceptance, not the count of kernels or commits,
determine completion. Historical milestone names in source-intake and trace
documents do not define additional active release gates.

## Goal and boundaries

Deliver an independent, source-faithful PhysX-derived physics SDK with one
FyzzXYZZY-owned scene, task graph and simulation state. CPU, Vulkan and D3D12
are execution choices, not separate physics engines. Lower CUDA execution
mechanisms to the native platform while preserving source algorithm, state,
lifetime and observable behavior. Change scheduling, memory placement and
kernel implementation when the new compute model permits a demonstrably
better solution without changing those semantics.

Godot is a consumer and validation path; the SDK must also build and operate
without Godot or a reference PhysX runtime. **GPU compute is the primary
execution target.** Native CPU execution remains the supported-hardware
fallback and conformance mode; the separate reference CPU backend is an
independent comparison oracle, never a hidden production substitute.

## Delivered foundation and current limits

| Capability | Current production boundary |
| --- | --- |
| Native CPU rigid scene | Source-derived Sc/island/task ownership, CPU SAP, PCM contacts and Dy PGS for an admitted **box/plane and one-shape-per-actor subset**; limited source-derived distance constraints, materials, contact events and body lifecycle |
| Portable compute | Real Vulkan and D3D12 execution, compilation, resource ownership, submissions and synchronization; independent SDK tests without reference PhysX |
| GPU broadphase | Explicit native GPU World factory, source-derived persistent AABB/SAP and aggregate ownership, multi-frame found/quiet/lost report publication into the same Sc/PCM/Dy scene |
| GPU box/box contacts | Explicit `NativeGpuPcmShaders` selection runs GuPCM and sparse GJK/EPA on the device for live Sc managers, with source world-space contact publication; other admitted contacts use CPU narrowphase |
| GPU rigid dynamics | No integrated GPU constraint preparation, rigid solve or body integration |
| Rigid feature breadth | General geometry, multishape actors, CCD, triggers, complete filtering/reporting, joints, articulations and other PhysX modes are not fully admitted |
| Nonrigid simulation | GPU deformables, particles and their reciprocal interactions are not integrated |
| SDK selection and consumers | `WorldOptions` defaults to the reference CPU backend when enabled; `NativeCpu` is explicit. `NativePortableGpu` through the ordinary `World` constructor rejects; `create_native_gpu_world(...)` is the explicit supported GPU-broadphase entry point. Godot acceptance uses native CPU |

The GPU broadphase and optional GPU box/box PCM form a working **hybrid scene
slice**, not a completed GPU physics pipeline or general PhysX compatibility. There is no
meaningful single percentage for feature coverage: source breadth,
execution-device coverage, and end-to-end scene behavior must be tracked
separately.

## Active implementation priorities

These are **capability gates**, not numbered projects. Closely related
dependencies may overlap. A gate closes only when the original algorithm and
scene handoffs execute through the real SDK owner, with supported inputs and
failure paths tested on the relevant backends.

### GPU body and shape state

Extend the source-derived simulation-controller and scene/task integration
beyond CPU-authored broadphase inputs. Admit the required source body and
shape representations, dirty/update streams, device-produced state, GPU
AABB preparation and pre/post-collision dependencies. Carry identities and
ownership across frame boundaries without creating a second scene, CPU
projection oracle or separate scheduler.

**Acceptance:** a live native scene evolves device-produced body/shape state
through the corresponding GPU broadphase work across multiple frames,
including growth, retirement, sleep/kinematic changes and failed publication.
The source task order and host/device transfer contracts are observable.
The GPU shape/transform and bounds data must remain resident for subsequent
narrowphase and dynamics stages; the CPU-owned N2 hybrid input is transitional.

### GPU narrowphase and contact streams

Port the actual supported source GPU collision pipeline: shape-pair
admission, persistent contacts, manifolds, materials, patch/contact streams,
touch/report transitions and original CPU/GPU eligibility rules. Grow geometry
support only as the corresponding source path is brought into production;
do not replace unsupported cases with invented collision rules.

**Acceptance:** native GPU-produced contacts, not just broadphase candidate
pairs, feed the same scene's GPU constraint preparation across creation,
motion, separation and retirement. CPU Sc contact-manager publication remains
the transitional N2 hybrid path, not a required host readback between GPU
broadphase, narrowphase and solving. Validate exact stream contracts and
numerical contact behavior on Vulkan and D3D12.

### GPU rigid solving and integration

The first acceptance workload is a coupled rigid box stack with persistent
contacts, friction, angular response and multiple active bodies, retaining
the same Sc EdgeIndex, Gu PCM and Dy task owners. This is the initial
performance-comparable subset, not a general rigid-geometry promise.

Bring source-derived GPU constraint preparation, partitioning, solving,
integration and writeback into the same native task graph. Preserve PGS/TGS
distinctions and the source-supported contact, friction, joint and island
behaviors. Exercise GPU completion, rollback and wake/sleep transitions
without shadow CPU solver state or unconditional host round trips.

**Acceptance:** GPU-produced body motion and contact/constraint results
survive multi-frame standalone scenes on both APIs, including collisions,
coupled rotation/friction, supported joints, varied island topology and
correct publication. CPU/reference comparisons check behavioral contracts,
not arbitrary bit identity.

### Rigid scene breadth and native-default readiness

Expand scene shape/actor semantics, geometry and cooking, multishape
interaction, CCD, triggers/filtering, contact modification/reporting, scene
queries, constraints and other source-defined runtime behavior. Close
independent native CPU omissions in parallel with GPU work. Default-world
reference retirement is a separate packaging/API acceptance decision, not
a proxy for complete GPU dynamics.

**Acceptance:** each advertised capability has source correspondence and
standalone regression coverage. Ordinary non-test SDK construction and
packaging do not require a reference scene; unsupported features reject
explicitly rather than quietly substituting a different engine.

### Articulations and advanced rigid interactions

Port source articulation dataflow, constraints, feedback, joints/tendons where
supported and their interactions with contact/island solving. Validate mixed
articulated and ordinary rigid scenes, not isolated articulation arithmetic.

### Deformables, particles and mixed scenes

Port the source deformable/particle components and their contacts as coherent
subsystems, including reciprocal rigid interactions and shared scene stepping.
No isolated cloth or particle simulator may stand in for these contracts.

### Consumer and release maturity

Grow standalone packaging, diagnostics, supported-device capability
negotiation, Godot integration, engine-facing APIs and cross-vendor coverage
from the same SDK engine. Maintain a feature/backend support matrix that
identifies tested hardware and unsupported combinations.

## Evidence required for any capability gate

- **Source fidelity:** mapped source operations, numerical semantics,
  ownership, buffer layout, task ordering, and deliberate portable lowerings.
- **Real integration:** production callers and outputs, not only isolated
  shader fixtures; tests through the scene across multiple frames.
- **Failure and retirement:** ID reuse, quiet frames, growth, abort/recovery,
  GPU fences, no silent reference or CPU substitute.
- **Portability:** Vulkan and D3D12 correctness/API execution as applicable;
  software Vulkan and D3D12 WARP are not hardware throughput evidence.
- **Runtime design:** retain efficient algorithmic complexity, useful
  parallelism, batched work, allocation reuse, minimal transfers and necessary
  synchronization. Do not add a slow safety path solely because it is easier
  to port.

## Performance: design now, measure at coherent checkpoints

Performance discipline applies to every implementation change: reject obvious
serialization, excessive readback, per-kernel waits, redundant copies,
avoidable allocations and scalarized source parallelism before they spread.
Record submissions, copies, waits and allocator activity where useful, and
use targeted measurements to resolve a concrete design decision.

The first performance checkpoint is a coupled rigid box scene executing
broadphase, PCM, constraint preparation, PGS and integration on the device.
The current hybrid GPU-contact/CPU-solver path can diagnose transfer and
scheduling costs, but cannot predict complete GPU-rigid throughput. Profile
the integrated rigid slice on native AMD Vulkan and D3D12 before extending
the solver to pointer-rich or broader geometry workloads. Optimize the measured bottleneck, using
subgroup, wave and device-specific primitives when semantically equivalent.
Do not confuse CI wall time or software-device performance with hardware
physics throughput.

## Technical references

- [Architecture and fidelity contract](ARCHITECTURE_CONTRACT.md)
- [Native scene implementation and historical migration gates](NATIVE_ENGINE_MIGRATION.md)
- [CPU source correspondence](NATIVE_SOURCE_INTAKE.md)
- [GPU host execution and stage trace](PHYSX_GPU_PORT.md)
- [GPU source implementation program and historical gates](GPU_PORT_PROGRAM.md)
- [GPU performance engineering policy](GPU_PORT_PERFORMANCE.md)
- [Building and SDK integration](BUILDING.md)

Previous M-, N- and G-numbered schedules are **retired as planning schemes**.
Their source maps, tests and recorded decisions remain useful; the active
capability gates above govern subsequent implementation and acceptance.
