<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PhysX-to-portable-GPU engineering program

Status: source audit and native-engine migration, October 2026. This is the program-level
companion to [the execution trace](PHYSX_GPU_PORT.md) and
[the detailed broadphase migration ledger](GPU_BROADPHASE_TRACE.md), not a declaration of
completed PhysX GPU functionality. Source audit follows the PhysX family in `thomthehound/PhysX` and
the SDK's own ported sources; no product build should require an exact
external PhysX commit. Refresh this audit when upstream internals change.
Proposals below are explicitly distinguished from behavior observed upstream.

## 0. Objective, scope, and truthfulness

The deliverable is an **independent FyzzXYZZY port of PhysX's GPU execution path**, including the
source-derived scene lifecycle, host task orchestration, algorithms, data
representations, synchronization, and reporting. CUDA driver entry points,
CUDA device pointers, CUDA streams/events, warp intrinsics and compiled kernels
must be translated to portable mechanisms. PhysX's domain algorithms and
public simulation behavior remain authoritative unless a departure is recorded
and justified.

Vulkan is first for bring-up; D3D12 is a **parallel supported backend**. Avoid
building a Vulkan-specific scene pipeline and retrofitting D3D12 months later.
A single source for translated kernels is the preferred default; independently
maintained Vulkan and D3D physics algorithms are not.

Important distinction: an isolated numerical kernel is **not** integrated GPU
broadphase; GPU broadphase is **not** GPU narrowphase/solver; GPU rigid bodies
are **not** full PhysX GPU coverage (articulations, particles, deformables,
and features such as direct-GPU access require separate work).

Do not turn `src/world_vulkan.cpp` or the experimental floor/contact kernels
into an alternative PhysX implementation. Their purpose is compute
infrastructure experimentation. They stay opt-in and outside production.

## 1. What the upstream source actually requires

### 1.1 Scene and task graph, not a second simulation loop

- `physx/source/physx/src/NpScene.cpp`:
  `simulate` / `collide` / `advance`, `fetchResults`, user-visible scene
  lifecycle, callbacks, and state visibility.
- `physx/source/simulationcontroller/src/ScScene.cpp`:
  scene flags select GPU broadphase and GPU dynamics independently. Both
  currently require a usable CUDA context manager, GPU allocator/wrangler,
  and factory products. GPU dynamics also supplies the contact, patch,
  force/index and friction streams to low-level context and narrowphase.
- `physx/source/simulationcontroller/src/ScPipeline.cpp`:
  `broadPhaseFirstPass`, `broadPhaseSecondPass`, `preIntegrate`,
  `updateBroadPhase`, `postBroadPhase`, touch/interaction processing,
  island generation, `updateDynamics`, `updateDynamicsPostPartitioning`,
  `updateSimulationController`, `postSolver` and `afterIntegration`.
  These are **tasks with continuation dependencies**, not a simple list of
  successive CPU function calls.
- `physx/source/gpusolver/src/PxgContext.cpp`:
  `PxgGpuContext::update` schedules GPU preintegration, prep and solving
  against CPU incremental island partitioning. Post-partition work cannot
  consume partition arrays before the continuation has completed.
- `physx/source/gpusolver/src/PxgCudaSolverCore.cpp` and
  `PxgTGSCudaSolverCore.cpp`:
  distinct PGS/TGS paths, multiple streams, events and mapped completion
  flags; copying only one solver inner loop is insufficient.

**Required first architectural deliverable:** a source-to-task DAG with
submission, dependency, completion and result-publication edges for the
targeted scene mode. The portable GPU queues provide execution; **FyzzXYZZY owns the scene scheduler and authoritative state**. Port the
source task continuations and dependencies into that scheduler instead of
running a parallel PhysX scene or duplicating actor state.

### 1.2 GPU broadphase is persistent, incremental, and observable

Primary source: `physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp`,
`PxgAABBManager.cpp` and `CUDA/broadphase.cu` / `aggregate.cu`.

The meaningful frame path is:

1. `PxgAABBManager::updateBPFirstPass` gathers created/removed handles,
   detects changes, stages bitmaps and aggregate-related data.
2. `PxgCudaBroadPhaseSap::preBroadPhase` stages float bounds, groups,
   contact distances and optional environment IDs; updated GPU-produced
   bounds can take a different route.
3. `PxgAABBManager::updateBPSecondPass` delegates to
   `PxgCudaBroadPhaseSap::update`. The latter conditionally performs
   bounds encoding; marks changed/removed/created pairs; sorts 3-axis
   projections; handles incremental SAP; generates histograms and
   regions; generates new/lost pairs; swaps previous-frame bounds;
   clears new flags and copies reports.
4. `PxgCudaBroadPhaseSap::gpuDMABack` retrieves result counters,
   synchronizes completion and handles capacity/abort conditions.
   `fetchBroadPhaseResults` swaps the six double-buffered handle/histogram
   arrays.
5. `PxgAABBManager::postBroadPhase` / `processFoundPairs` /
   `processLostPairs` deduplicate, resolve aggregate pairs, update overlap
   lists, and release scene continuations into narrowphase/interaction work.

Cases requiring separate tests: initial population; unchanged frames;
forced one-time update after sleep; created/removed/updated handles;
CPU-produced versus GPU-produced bounds; aggregate insert/remove/empty;
environment IDs; origin shift; overflow; abort; buffer growth; capacity
compaction; consistent repeated simulation/fetch.

**Specific upstream hazards:** the AABB manager checks that empty aggregates
cannot enter GPU BP (see `PxgAABBManager::updateBPFirstPass`, OMPE-92788).
The broadphase checks found/lost capacity overflow and warns that omitted
pairs change simulation behavior. Those are portability acceptance cases,
not mere performance considerations.

### 1.3 GPU descriptors and allocators expose a harder ABI problem

- `gpucommon/include/PxgDevicePointer.h` wraps `CUdeviceptr` in
  pointer-sized objects; the host often casts device addresses directly to
  typed pointers.
- `gpubroadphase/include/PxgBroadPhaseDesc.h` contains dozens of **typed
  pointers and nested arrays of pointers** copied into GPU-visible memory.
- `gpucommon/include/PxgCudaBuffer.h` separates raw allocation pointers
  from 128-byte-aligned device pointers. Buffer swap must preserve the raw
  allocator key, not merely the aligned address.
- `gpusolver/include/PxgSolverCoreDesc.h` and
  `gpusimulationcontroller/include/PxgSimulationCoreDesc.h` contain further
  pointer-rich descriptors and data pools.
- `gpucommon/src/PxgMemoryManager.cpp` and `PxgCudaMemoryAllocator.cpp`,
  plus `PxgHeapMemoryAllocator`, own host, mapped, device and heap/pool
  semantics. GPU solver/narrowphase share some streams and memory pools.

An opaque Vulkan buffer ID is **not** interchangeable with a `CUdeviceptr`
embedded in a kernel descriptor. A naive struct memcpy, pointer cast,
or shader register remapping will not preserve the ABI.

See [GPU memory/addressing RFC and benchmark contract](GPU_MEMORY_ADDRESSING_RFC.md).

**Addressing decision gate:** evaluate, on real PhysX descriptor graphs,
(a) explicit descriptor/offset indirection, (b) a GPU address scheme on Vulkan
where supported, and (c) D3D12 shader-model-compatible descriptor indexing.
D3D12 GPU virtual addresses cannot simply be dereferenced as arbitrary HLSL
pointers. Do not choose a Vulkan-only buffer-device-address model as the
canonical shared descriptor format. Whichever representation wins must
demonstrate nested traversal, safe reallocations, aliasing, pointer lifetimes,
32/64-bit sizes and alignment on **both** backends.

### 1.4 Warp and device execution semantics are a high-risk port

Samples from the original CUDA source:

- `gpubroadphase/src/CUDA/broadphase.cu`: ballot and barriers in sorting,
  incremental SAP, and region work.
- `gpubroadphase/src/CUDA/aggregate.cu`: shuffle, ballot, syncwarp, atomics.
- `gpucommon/src/CUDA/RadixSort.cuh`: shuffle/ballot/barrier primitives.
- `gpunarrowphase/src/CUDA/cudaGJKEPA.cu`: extensive shuffle/ballot usage,
  convergence/control-flow sensitivity.
- `gpusolver/src/CUDA/solverMultiBlock.cu` and
  `solverMultiBlockTGS.cu`: multiblock synchronization and atomic operations.
- `gpusolver/src/CUDA/constraintBlockPrePrep.cu`: includes
  threadfence operations and warp-wide work.
- `gpuarticulation/src/CUDA/forwardDynamic2.cu`: warp-sensitive kernels.

A 32-thread CUDA warp is **not** automatically a D3D wave or Vulkan
subgroup. Hardware with 32- and 64-lane subgroup choices must produce the
same logical algorithm (through subgroup-size control when supported, or
redesigned lane grouping that preserves semantics). Verify masks, active
lanes, divergence, barriers, atomics, reductions, shared memory, and
inter-workgroup ordering. Never substitute an apparently analogous shuffle
without proving the original active-lane contract.

Relevant external feature mechanisms are
[Vulkan subgroup-size control](https://docs.vulkan.org/guide/latest/subgroups.html)
and [D3D12/HLSL Shader Model 6.6 WaveSize](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html).
Neither mechanism should be assumed available without querying device
capabilities. SM 6.6 also offers
[dynamic descriptor-heap indexing](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html),
which is a *candidate* for descriptor indirection, not an automatic
replacement for CUDA's arbitrary typed pointer graph.

### 1.5 Other PhysX GPU obligations remain visible

- `gpunarrowphase/src/PxgNphaseImplementationContext.cpp` chooses a GPU
  bucket or the original CPU fallback by shape pair and compatibility.
  Preserve that classification; do not silently drop unsupported pairs.
- `gpusolver`: both PGS and TGS, contact/joint constraint prep, island
  partitioning, integration, force threshold reporting and writeback.
- `gpuarticulation`: reduced-coordinate articulation operations and the
  synchronization required by constraints and solver.
- `gpusimulationcontroller`: rigid body/shape state, DMA back, sleeping,
  joint manager and direct GPU API, plus particle, soft-body and FEM-cloth
  paths. The audited source contains all these modules; parity claims must
  identify which are supported, falling back, or still unavailable.
- CPU-side CCD, triggers and callbacks still exist in GPU-enabled scenes.
  Their CPU execution is not a deficiency to "fix"; their ordering and
  interaction with the GPU pipeline must be preserved.

## 2. Target architecture (proposal, subject to verification)

### 2.1 Keep PhysX as the host implementation

Do **not** fabricate a new `World` scheduler. Evolve a portable implementation
of the GPU products presently created by `PxPhysXGpu`
(`physxgpu/src/PxgPhysXGpu.cpp`) and selected by `Sc::Scene`. Treat
`Bp::BroadPhase`, `Bp::AABBManagerBase`, the simulation controller,
narrowphase implementation and dynamics context as real integration contracts.
GPU broadphase-only is the first coherent integration target; GPU rigid
dynamics follows, not the reverse.

Do not assume every factory argument currently typed
`PxCudaContextManager*` can be replaced with a typedef. Inventory
downstream casts, lifetime assumptions, memory allocators and capability
checks before changing PhysX interface types.

### 2.2 Portable GPU executor with explicit dependencies

Requirements for the internal compute substrate, independent of Vulkan/D3D12:

- Device/capability negotiation and backend selection; DXC compiles common
  HLSL to SPIR-V and DXIL; record module, entry, ABI and feature requirements.
- Owned, typed buffer allocations, suballocations, mapped/staging transfers,
  alignment, aliasing, address/descriptor views and reallocation contracts.
- Commands: copies, fill/clear, dispatch, explicit read/write resource
  dependencies, barriers, event/signal/wait, readback and completion tokens.
- Multi-queue and CPU task completion semantics when demanded by PhysX.
  A single queue may initially *serialize* an identified subgraph, but must
  not silently erase required task edges, publication or CPU concurrency.
- GPU failures and abort propagation into PhysX scene continuation cleanup;
  task references must be released on every success/failure/early-out.
- Low allocation churn: transient command/descriptors, cached pipelines
  and rings/pools rather than one device submission per kernel.
- CPU trace/diagnostics showing source task, buffer version, submission,
  completion and result publication; do not copy entire GPU state every frame.

**Important existing limitation:** `src/compute_vulkan.cpp` currently
implements one queue, a `CopyCommand | DispatchCommand` batch, coarse
barriers between every batch operation, fence-backed completion, simple
buffers and per-dispatch descriptor allocation. This is valuable
infrastructure, but **not** an adequate PhysX execution scheduler. Refine
it from actual source dependencies, not by guessing a generic task graph.

### 2.3 Same source, backend-specific lowering

Keep a **single behavioral kernel source** where feasible. Shared HLSL/DXC
is the current baseline. Permit narrowly scoped Vulkan or D3D12 annotations
for resource representation. Reassess Slang if the pointer-rich descriptors,
subgroups or resource lowering create concrete limitations. A compiler switch
is not a license to change the physics algorithm.

Acceptable platform differences: pipeline creation, descriptors,
root-signature/push-constant setup, resource states, barriers, fences,
queue submission and capability-based kernel specialization.

Unacceptable without written justification: independent contact solving,
different pair semantics, a separate simulation loop, different sleeping
state machine or a backend-specific body ownership model.

## 3. Critical path: gates, not calendar promises

| Gate | Deliverable | Must be proven before proceeding |
| --- | --- | --- |
| G0 - baseline | Source/scene map; build modes; CPU test oracle; trace/profiling format | Source-grounded PhysX call/dependency map, CPU regression fixtures, reproducible builds |
| G1 - two execution backends | Native Vulkan and D3D12 compute, same AABB shader and identical reference fixtures | SPIR-V and DXIL compilation, dispatch, copy, resource barriers, readback, fence and failure tests; native Windows D3D12 execution |
| G2 - addressing and tasks | Portable device descriptor schema, allocation/lifetime policy and GPU completion bridge into PhysX continuations | Real pointer-rich descriptor traversal on both backends; task continuation exactly once; resize, alias, abort, sequencing cases |
| G3 - coherent GPU broadphase | Source-derived SAP/sort/region/aggregate/pair pipeline entered from a real PhysX scene | Found/lost-pair parity, frame-to-frame state, forced update, aggregation, overflow, scene publication, CPU dynamics with GPU BP |
| G4 - body/shape pipeline | Portable simulation-controller body/shape updates, contact stream ownership and GPU narrowphase foundation | Original GPU/CPU fallback matrix, contact/patch output, touch/CCD/trigger interaction and island input |
| G5 - rigid solver | Source-derived PGS/TGS constraint preparation, partitioning, rigid solver and integration | Stable bodies, contacts, joints, restitution/friction, sleep/wake, callback/report semantics, multi-frame regression |
| G6 - articulation | GPU articulation streams, data prep, solve/feedback consistent with PhysX | Chains/loops where supported; joints/tendons; edge cases and CPU fallbacks clearly classified |
| G7 - GPU breadth | Deformable surface/volume, PBD particles and remaining PhysX GPU services | Per-feature source trace, capacity constraints, mixed-feature interactions and coverage report |
| G8 - production | Direct GPU API policy, interop, tuning, hardware matrices, packaging and engine consumers | Honest public support matrix, profiling, robust recovery and repeatable release acceptance |

**Implementation priority:** G0/G1/G2 *before* adding lots of CUDA kernels.
G3 is the first scene-integrated PhysX GPU milestone. G4/G5 are the main
rigid-body product milestone. G6/G7 are significant projects in themselves;
do not imply full PhysX 5.11 GPU parity merely because G5 works. Gates can
overlap where dependencies allow, but no acceptance gate is waived.

## 4. Tests that distinguish a port from a demo

### 4.1 Oracle hierarchy

1. Source-derived **exact contracts**: layouts, handles, buffer contents
   where integer encoding/ordering is defined; reference kernel outputs.
2. PhysX **behavioral contracts**: generated/lost overlap sets and event
   order, stable actor identities, contact stream shape, scene states and
   callbacks, buffer capacity/overflow/abort behavior.
3. CPU PhysX for general behavior and original CUDA PhysX for GPU-specific
   reference traces when compatible NVIDIA hardware is available.
4. Numerical comparisons use appropriate tolerance and physical invariants;
   CPU and GPU algorithms do not promise identical floating-point bits.

Do not treat an independently invented demo physics world as an oracle for
the PhysX GPU pipeline. Do not loosen an incorrect test to make a GPU
implementation pass.

### 4.2 Required fixtures by stage

- Broadphase: 0/1/16/17/513 objects; hot creation/destruction; repeated
  no-change and sleep/wake; updated bounds and GPU-updated bounds; three-axis
  degeneracies; aggregates and empty aggregates; environment IDs; origin
  shifts; capped found/lost buffers and overflow warning; abort/recovery;
  producer/consumer order and double-buffer swap.
- Narrowphase: original shape-pair GPU bucket/fallback matrix, contact
  material/patch/force streams; PCM persistence; contact modification;
  unsupported cooked geometry; mixed CPU/GPU contact pairs.
- Solver: PGS and TGS independently; static/kinematic/dynamic mixes;
  joints, stacks, friction, restitution, varying time step; partitioned
  islands; velocity/position iterations; sleep, force thresholds and contact
  reports; mixed articulation constraints.
- GPU state: host upload / device-produced bounds, mapping, resizes, pool
  turnover, aliasing, staged download, command-buffer reuse and interrupted
  work. Check for stale descriptors and early frees.
- Compatibility: CPU backend remains available; tests run when CUDA is
  absent; Godot is an optional consumer and must not be the only harness.

### 4.3 Platform acceptance and hardware test cadence

**Routine development is CI-first, not user-hardware-first.** FyzzXYZZY
does not require an AMD workstation run for every kernel, buffer, or
dispatch change. Do not make real-hardware performance results a condition
for continuing the shader port. We can establish source correspondence,
portable execution and software correctness without that burden:

- Linux CI: compile Vulkan SPIR-V, run reference fixtures with an available
  Vulkan ICD, and retain SDK integration tests. Label software ICD results
  as software **API/behavior** acceptance, not representative GPU speed.
- Windows `windows-latest`: compile DXIL, run the same source-derived
  reference fixtures on D3D12 WARP, and retain Windows integration. Label
  WARP as **correctness/API** execution, not native GPU profiling.
- Both: test descriptor layouts, dispatch shape, queue/barrier dependencies,
  lifetime, error cleanup, state across submissions, and data publication.
  Do not silently treat unexercised device features as passed.
- CI should run focused compute tests before long PhysX/Godot builds;
  documentation-only changes do not require full compilation. Keep
  CMake/CTest invocable independently of GitHub Actions.
- Later: expand cross-vendor 32/64-lane, atomic and device-limit coverage
  with NVIDIA/Intel hardware when practical; do not block early porting.

**Hardware checkpoints are coherent subsystem events, not PR events:**

1. **First AMD profiling session:** GPU broadphase entered through an
   actual PhysX scene with CPU dynamics, spanning multiple frames, persistent
   SAP state, pair publication, realistic creation/removal traffic, and
   representative batching. Exercise native AMD Vulkan and D3D12 where
   practical. This is the first point when end-to-end timing can meaningfully
   guide memory and dispatch choices.
2. **Second major session:** native GPU narrowphase and a meaningful
   rigid-body solver slice (contacts/joints, islands, state writeback and
   original task completions) execute together. Profile frame latency,
   CPU/GPU overlap, memory traffic, solver occupancy and adverse workloads.
3. **Later:** expand device matrix, direct GPU API, scene mixing, renderer
   interference, feature breadth, and production performance/reliability.

These sessions inform **optimization**, not whether source-derived porting
is allowed to continue. If CI cannot expose a particular hardware-only
contract, record that limitation and design a targeted device test at the
next checkpoint rather than fabricating evidence of coverage.

A PR's lightweight test ledger records: original source function,
corresponding portable implementation, input/output and allocation
owners, predecessor/successor tasks, validation fixture, backend actually
tested, unsupported cases and deliberate deviations.

## 5. Risks and decisions to resolve deliberately

| Priority | Risk / evidence | Decision or experiment |
| --- | --- | --- |
| Critical | Device-pointer-rich PhysX descriptors (BP, solver, simulation core) | Prototype representative nested descriptors and dynamic reallocation on both APIs before widespread kernel translation |
| Critical | CUDA streams/events and mapped signal completion crossed with CPU task continuations | Trace the DAG; implement explicit submission/completion tokens and failure-safe continuation bridge |
| Critical | CUDA 32-lane warp assumptions, ballots/shuffles/atomics | Cross-vendor subgroup/atomic conformance suite; decide shader variants or safe logical-warp lowering |
| High | CPU/GPU fallback routing in GPU narrowphase | Preserve upstream bucket selection and CPU fallback; verify cross-boundary contact stream ABI |
| High | Broadphase incremental state and aggregate reports | Port actual SAP stages with per-frame oracle fixtures; never replace with fresh all-pairs |
| High | GPU buffers cannot always grow dynamically; overflow can drop pairs or contacts | Expose original capacities, warnings and statistics; keep failure behavior explicit |
| High | PGS/TGS and partitioning have different continuation/solver paths | Separate source traces, tests and performance gates |
| High | Particles, FEM and articulation substantially expand scope | Declare rigid-only milestone versus complete GPU feature parity; prioritize by product demand |
| High | D3D12 could be left behind | Maintain G1 as critical-path gate, shared kernels and parity test fixtures |
| Medium | Shader compiler/reflection/layout drift | Manifest and validate bindings/strides/parameter sizes against both binaries; pin compilers in CI |
| Medium | CI or unavailable hardware gives false confidence | Local scripts, software smoke versus hardware labels, no silent skipping |

Open decisions (not pre-decided): descriptor/physical-address strategy;
subgroup specialization; initial scope of direct GPU API; PGS/TGS order;
when to expand beyond rigid-body behavior; whether DXC alone handles the
complex kernels or Slang earns its integration cost. Resolve from
actual source-derived prototypes, not preference.

## 6. Working protocol for the next months

1. **Read a coherent source slice first:** entry task, factory, host launch
   wrapper, shader(s), data structures, allocated memory and consumers.
   Sketch the DAG and identify every observable output.
2. **Record a dataflow ledger:** who allocates each buffer, who mutates it,
   stream/queue ownership, when copies happen, and which event publishes it.
3. **Implement one incremental, testable path** in the native FyzzXYZZY architecture.
   Compile the shared shader for both platforms as soon as that operation
   exists; reserve Vulkan-first runtime bring-up without indefinitely
   deferring D3D12.
4. **Review for semantic divergence:** active masks, atomics, ordering,
   overflow/abort, persistent data, fallbacks and CPU task continuation.
5. **Close evidence gaps:** use Linux Vulkan and Windows D3D12/WARP CI
   routinely, compare against source PhysX behavior and CPU oracles, and
   flag missing hardware coverage. Reserve AMD profiling for coherent scene
   milestones described in §4.3; do not demand it per PR.
6. **Update this program and [execution trace](PHYSX_GPU_PORT.md)** as each
   gate closes. Treat unresolved deviations as actionable engineering work,
   not "legacy" or "good enough because a test passed."

This program is a **roadmap and audit**, not an instruction to build a
new architecture on speculation. Source traces determine what the portability
layer needs; working code and acceptance tests determine when a gate is done.

### 6.1 What to tighten before scaling up shader translation

The correct target is a **minimum viable faithful port substrate**, not a
fully generalized compute framework completed before any more kernels.

1. **Make memory access intent explicit.** Dispatch metadata should
   distinguish read, write and read/write views (including offsets/lengths
   and aliases). PhysX's launch-side descriptors and workgroup dependencies
   provide this information; do not conservatively label every storage
   binding read/write forever. Maintain a safe conservative fallback for
   genuinely unknown access. Keep backend-specific lowering isolated from
   source-derived kernel logic.
2. **Handle completion and ownership without device-wide idle.** A
   submission's command/descriptors/staging buffers and overwritten or
   relocated PhysX arenas must survive until the actual completing fence/
   timeline value. Asynchronous upload, copy, download and deferred free
   need to correspond to PhysX's existing copy manager and stream/event
   semantics. Implement completion-to-native-task continuation
   signaling; do not invent a second physics scheduler.
3. **Compile kernels from one behavioral source.** Establish a repeatable
   mapping from CUDA source function and host launch configuration to
   common shader, parameter/descriptor layout, backend capability and
   test fixture. Deal with warp/subgroup assumptions explicitly when they
   first occur, not by guessing universal defaults.
4. **Maintain bounded, honest compute backends.** Both APIs must execute
   each accepted source-derived stage with equivalent state/pair semantics;
   avoid building elaborate backend runtime selection, descriptor heaps,
   multi-queue policy or address-encoding generalization without an
   upstream task requiring it. The current single-queue executor can
   serialize a coherent subgraph as a correctness baseline, provided its
   edges and publication are retained and its serialization is marked.
5. **Then port source-coherent kernel families.** Continue from AABB
   encoding and removed-pair marking into created/updated/projection
   bookkeeping and SAP initialization; bring in the actual PhysX radix
   sort/rank/handle stages as a *shared dependency*; follow with endpoint/
   region histograms, incremental pair generation and aggregate processing;
   finally wire report retrieval and original AABB-manager/scene
   continuations. Test successive *frames*, not only independent dispatches.
   Order adjacent kernels by their **upstream launches and dependency
   graph**, not by apparent shader complexity or novelty.
6. **After integrated broadphase:** extend original CPU/GPU shape-pair
   routing and narrowphase, then constraint preparation/partitioning,
   PGS/TGS integration and state publication. Solver and articulation
   data/warp requirements may justify alternate physical address lowering,
   without changing the authoritative PhysX algorithms.

**Defer until evidence exists:** speculative shader fusion, unconditional
BDA/SM6.6 dependence, complete multi-queue scheduling, NPU tiling,
universal descriptor graphs, hardware-specific subgroup specialization,
and parameter micro-optimizations. Carry candidate designs in the RFC,
but do not block shader-family translation waiting for an AMD benchmark.

## 7. Primary source index

Source reference paths in the PhysX family (not a dependency pin):

- `physx/source/physx/src/NpScene.cpp`
- `physx/source/simulationcontroller/src/ScScene.cpp`
- `physx/source/simulationcontroller/src/ScPipeline.cpp`
- `physx/source/physxgpu/src/PxgPhysXGpu.cpp`
- `physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp`
- `physx/source/gpubroadphase/src/PxgAABBManager.cpp`
- `physx/source/gpubroadphase/src/CUDA/broadphase.cu`
- `physx/source/gpubroadphase/src/CUDA/aggregate.cu`
- `physx/source/gpubroadphase/include/PxgBroadPhaseDesc.h`
- `physx/source/gpucommon/include/PxgDevicePointer.h`
- `physx/source/gpucommon/include/PxgCudaBuffer.h`
- `physx/source/gpunarrowphase/src/PxgNphaseImplementationContext.cpp`
- `physx/source/gpusolver/src/PxgContext.cpp`
- `physx/source/gpusolver/src/PxgCudaSolverCore.cpp`
- `physx/source/gpusolver/src/PxgTGSCudaSolverCore.cpp`
- `physx/source/gpusolver/src/CUDA/constraintBlockPrePrep.cu`
- `physx/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu`
- `physx/source/gpuarticulation/src/CUDA/forwardDynamic2.cu`
- `physx/source/gpusimulationcontroller/src/PxgSimulationController.cpp`
- `physx/source/gpusimulationcontroller/include/PxgSimulationCoreDesc.h`

Upstream interpretation is authoritative for what **PhysX does**;
this document's suggested architecture and gate ordering are decisions for
FyzzXYZZY, not claims made by NVIDIA.

## Native scene ownership and G3 acceptance

The default `fyzz::World` presently uses an isolated PhysX reference CPU backend.
The production API explicitly rejects `NativePortableGpu` until the SDK owns a
native scene and source-derived GPU SAP/AABB manager, native task lifecycle,
GPU-completion publication, and a corresponding narrowphase/dynamics path.
The short-lived source-level factory experiment in the PhysX fork was reverted;
no production GPU integration shall depend on patching the reference PhysX
`PxSceneDesc` or `Sc::Scene`.

Implementation dependencies, measurable gates, runtime ownership and code
retirement are specified in [NATIVE_ENGINE_MIGRATION.md](NATIVE_ENGINE_MIGRATION.md)
and tracked in [issue #101](https://github.com/thomthehound/FyzzXYZZY-SDK/issues/101).

