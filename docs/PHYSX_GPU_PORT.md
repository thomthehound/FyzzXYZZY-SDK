<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PhysX GPU port: execution architecture and traceability

See also [the multi-month GPU port program, blockers and acceptance gates](GPU_PORT_PROGRAM.md).

Status: initial source-grounded execution-path audit. This is a port dependency
map, not a claim that any PhysX GPU subsystem has been implemented on Vulkan or
D3D12. Source reference: the PhysX revision pinned in the root `CMakeLists.txt`.

## Port boundary

FyzzXYZZY ports PhysX GPU physics **within the PhysX simulation pipeline**.
Kernel translation and host-side orchestration are equally part of the port.
The portable GPU implementation must retain the causal dependencies between
scene updates, collision detection, contact management, constraint preparation,
solving, integration, sleeping, and publication of scene state.

CUDA streams, events, modules, and device pointers are implementation mechanisms
to translate. The stages they connect, ownership and lifetime invariants,
task continuations, contact/constraint representations, and ordering constraints
must first be accounted for. Changes to PhysX's algorithms or execution
architecture require an identified source counterpart and a technical rationale.

## Verified scene-to-GPU path

These entry points and dependencies are present in the pinned PhysX source:

1. `physx/source/physx/src/NpScene.cpp`:
   `NpScene::simulate` enters `simulateOrCollide`; `executeScene` invokes
   `mScene.simulate`. The scene state machine also supports separated
   `collide`, `fetchCollision`, `advance`, and result fetching.
2. `physx/source/simulationcontroller/src/ScScene.cpp`:
   `Sc::Scene` selects GPU broadphase and GPU dynamics independently from
   `PxSceneDesc`. When GPU dynamics is enabled, it creates the GPU dynamics
   context, narrowphase implementation (with CPU fallback context), and
   simulation controller using `PxPhysXGpu`; it joins their contact, patch,
   force/index, and friction-patch stream pools with `PxsContext`.
   The PhysX GPU factory is implemented in
   `physx/source/physxgpu/src/PxgPhysXGpu.cpp`.
3. `physx/source/simulationcontroller/src/ScPipeline.cpp`:
   `Sc::Scene::simulate` schedules task continuations; `collideStep` and
   `rigidBodyNarrowPhase` coordinate shape upload, broadphase, and narrowphase
   with task dependency edges. `broadPhaseSecondPass` schedules
   `preIntegrate` before the later broadphase update. Contact manager updates
   and lost/found touches feed island generation and dynamics.
4. In the same `ScPipeline.cpp`, `updateBodies` invokes the simulation
   controller; `updateDynamics` calls `mDynamicsContext->update`, followed
   by `updateDynamicsPostPartitioning` and
   `mDynamicsContext->updatePostPartitioning`. The GPU branch uses different
   task continuations from the CPU-only path. The upstream comments explicitly
   identify partitioning/continuation scheduling as an architectural concern.
5. `physx/source/gpusolver/src/PxgContext.cpp`:
   `PxgGpuContext::update` consumes narrowphase outputs and GPU contact data,
   accesses the island manager, synchronizes with the simulation controller,
   and prepares solver state. `updatePostPartitioning` consumes partition
   arrays, stages solver memory, and synchronizes joint data before solving.
   `PxgCudaSolverCore.cpp` and `PxgTGSCudaSolverCore.cpp` own kernel-launch
   sequences and stream/event dependencies for the respective solver paths.
6. `physx/source/simulationcontroller/src/ScPipeline.cpp`:
   `updateSimulationController` initiates GPU DMA-back and
   `mDynamicsContext->updateBodyCore`; `postSolver` merges results and swaps
   friction streams; `afterIntegration` updates scene body/shape simulation
   state. These steps are part of physics correctness and state visibility,
   not optional presentation or rendering code.

The GPU source also contains distinct CPU fallback paths for unsupported
contact managers and configurations. Port decisions must classify these paths
explicitly, rather than silently substituting simplified physics.

## Port traceability matrix

| Responsibility | PhysX host-side source | GPU algorithm/implementation | Portable work required |
| --- | --- | --- | --- |
| Scene selection, initialization, ownership | `simulationcontroller/src/ScScene.cpp`, `physxgpu/src/PxgPhysXGpu.cpp` | Selected GPU factory products | Preserve scene/controller/context topology; replace CUDA context provision and capability checks |
| Scene task ordering | `physx/src/NpScene.cpp`, `simulationcontroller/src/ScPipeline.cpp` | Task continuations and GPU callbacks | Preserve task graph and stage ordering; bridge to portable queue completion |
| Device memory and kernel registration | `gpucommon/src/PxgMemoryManager.cpp`, `gpucommon/src/PxgCudaMemoryAllocator.cpp`, `gpucommon/src/PxgKernelWrangler.cpp` | CUDA allocations, modules, device pointers | Design typed resource/descriptor indirection, lifetime and allocation contracts |
| Broadphase and pair management | `gpubroadphase/src/PxgCudaBroadPhaseSap.cpp`, `gpubroadphase/src/PxgAABBManager.cpp` | `gpubroadphase/src/CUDA/broadphase.cu` and related kernels | Port GPU SAP/pair pipeline with original pair and bounds contracts |
| Narrowphase and persistent contacts | `gpunarrowphase/src/PxgNphaseImplementationContext.cpp`, `gpunarrowphase/src/PxgNarrowphaseCore.cpp` | `gpunarrowphase/src/CUDA/` | Preserve contact-manager mapping, contact/patch outputs and fallback decisions |
| Scene GPU state and transfers | `gpusimulationcontroller/src/PxgSimulationController.cpp`, `gpusimulationcontroller/src/PxgSimulationCore.cpp` | `gpusimulationcontroller/src/CUDA/` | Preserve shape/body update and DMA-back boundaries; translate stream/event joins |
| Islands, partitioning and constraint state | `gpusolver/src/PxgContext.cpp`, `gpusolver/src/PxgSolverCore.cpp` | `gpusolver/src/CUDA/constraintBlockPrePrep.cu` and constraint-prep kernels | Preserve island and partition representation, contact/joint batches and host continuations |
| PGS/TGS solve and integration | `gpusolver/src/PxgCudaSolverCore.cpp`, `gpusolver/src/PxgTGSCudaSolverCore.cpp` | `gpusolver/src/CUDA/solver.cu`, `integration.cu`, TGS equivalents | Reproduce algorithm and iteration ordering, solver buffers, synchronization and writeback |
| Articulation dependencies | `gpuarticulation/`, `gpusimulationcontroller/src/PxgSimulationController.cpp` | `gpuarticulation/src/CUDA/` | Account for articulation streams and joint/contact preparation before adopting a slice |

Paths are relative to the `physx/source/` prefix unless they begin with
`physx/source/`; CUDA directories are source anchors, not independently
complete implementations.

## Execution requirements discovered so far

- **GPU narrowphase and solver cannot be replaced independently of their data
  interface.** `Sc::Scene` obtains contact, patch and force/index streams from
  `mDynamicsContext`, supplies them to the GPU narrowphase, then exposes the
  same streams through `mLLContext`.
- **Scene flags are not one GPU switch.** GPU broadphase and GPU dynamics have
  separate selection paths. Unsupported primitive handling may go through
  PhysX's own fallback, and the intended portable policy must be explicit.
- **GPU solving is not an independent dispatch.** The scene pipeline prepares
  contacts, islands, bodies and constraints, then schedules dynamics and
  post-partitioning continuations.
- **GPU work uses multiple dependency domains.** Solver, simulation core,
  broadphase, narrowphase and articulation components create and synchronize
  streams/events. One ordered Vulkan command batch is a useful primitive,
  but is not yet a substitute for those inter-stage dependencies.
- **State publication is part of the simulation.** DMA-back, body-core update,
  friction stream swapping, contact reporting and body/shape state updates
  must preserve PhysX's timing and consistency guarantees.
- **PGS and TGS are distinct GPU paths.** Porting a generic replacement solver
  would bypass real PhysX iteration, contact and partitioning behavior.

## Current FyzzXYZZY implementation classification

| Existing code | Classification | Disposition |
| --- | --- | --- |
| `src/world.cpp` | Actual PhysX CPU scene | Keep; CPU conformance baseline |
| `src/compute_vulkan.cpp`, `include/fyzzxyzzy/compute.h` | Portable execution infrastructure | Keep; extend only against observed PhysX GPU requirements |
| `src/kernels/rigid_floor.comp`, `src/floor_contact.cpp` | Independent demonstration, not a PhysX GPU kernel port | Build only with experimental opt-in |
| `src/world_vulkan.cpp` | Independent floor-simulation backend, not a PhysX GPU scene | Build only with experimental opt-in; do not extend as the production solver |
| PR #9 vertical-contact work | Experimental, non-PhysX contact pipeline | Leave unmerged; evaluate separately from the production port |

The current Vulkan compute `Context` and batch API do not yet expose a
PhysX-compatible task/synchronization contract. Do not attach a new physics
algorithm directly to the public `WorldBackend` as a substitute.

## First translated PhysX GPU stage: broadphase bounds encoding

The first executable translation is maintained under `src/physx_gpu/`:

- Source kernel: `physx/source/gpubroadphase/src/CUDA/broadphase.cu`,
  `translateAABBsLaunch` and `encodeFloat`.
- Upstream host caller: `physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp`,
  `translateAABBsKernel`, invoked by `PxgCudaBroadPhaseSap::update`.
- Upstream binary layouts: `PxBounds3` (six contiguous 32-bit floats),
  `PxgIntegerAABB::mMinMax[6]` (six contiguous 32-bit unsigned integers),
  and per-box `PxReal` contact distances / optional `PxU32` environment IDs.
- Translated kernel: `src/physx_gpu/kernels/physx_translate_aabbs.comp`.
  It preserves the per-component float-bit ordering, inflated bounds, per-axis
  shifts, min-even/max-odd encoding and shared-environment sentinel behavior.
- Host dispatch: `AabbEncodingStage::append` in
  `src/physx_gpu/aabb_encoding.cpp`. A 128-thread workgroup encodes
  16 boxes (eight logical threads per box). Upstream launches 256 threads;
  the reduced physical workgroup size is required for Vulkan core portability
  and preserves the original per-box lane mapping and encoding semantics.
  The stage appends work to the caller's command batch and never owns a
  scene or overrides collision behavior.
- Test: `fyzzxyzzy_physx_aabb_encoding` compares the encoded integer output
  with an upstream-formula CPU reference for default and custom axis shifts,
  environment IDs, absent environment arrays, capacity boundaries, and invalid
  configuration rejection. The Vulkan test is required in Linux CI.

**Current limit:** this stage is not yet called by a PhysX scene. It does not
produce collision pairs or provide GPU broadphase on its own. The GLSL Vulkan
source is a first translated implementation; shared HLSL/Slang and D3D12
execution remain to be established. The data and algorithm contract belongs
to PhysX rather than a new physics subsystem.

### Broadphase host orchestration required to integrate it

The pinned `PxgCudaBroadPhaseSap.cpp` provides a specific call sequence:

1. `preBroadPhase` stages changed floating-point bounds, per-object contact
   distances, filter groups and optional environment IDs.
2. `update` calls `gpuDMAUp` to update GPU buffers and descriptors, tracks
   created/removed/updated handles and sleeping/forced-update state, then
   dispatches `translateAABBsKernel`.
3. It marks removed/updated pairs, maintains incremental SAP projections,
   executes sorts, histogram/scans and comparison passes, and generates found
   pairs. Operations depend on previous-frame state and cannot all be
   indiscriminately recomputed without changing behavior.
4. `runCopyResultsKernel` and `fetchBroadPhaseResults` publish the found/lost
   pair data for scene narrowphase and update double-buffered projections,
   handles and histograms.
5. `PxgAABBManager` owns additional GPU aggregate and dirty-bounds handling
   and its coordination with the broadphase; its origin and synchronization
   dependencies must be preserved.

PhysX's GPU broadphase does **not** require GPU dynamics, but the current
factory and `Sc::Scene` initialization still require the CUDA context manager
and GPU-specific memory allocators. Simply setting `PxBroadPhaseType::eGPU`
in the existing CPU build would not enable this Vulkan stage. The next
integration work must adapt that factory/controller boundary and its resource
lifetime, without introducing a separate simulation loop.

## Vulkan and D3D12: concurrent portability gates

**Both backends implement the same PhysX GPU source-derived operation and
scene orchestration.** Vulkan is the first executable runtime, not a source
of truth for the simulation architecture. D3D12 implementation begins with the
first translated GPU stage, rather than waiting for the broadphase/solver port
to finish.

For AABB encoding, `src/physx_gpu/kernels/physx_translate_aabbs.hlsl` supplies
a DXIL-target source translation compiled in Windows CI using Windows SDK DXC.
This is currently a **compile-only D3D12 proof**. The Vulkan stage uses a
GLSL implementation; both follow the same PhysX kernel, and they must be
consolidated into a single HLSL/Slang-compatible source once SPIR-V compilation
and runtime behavior from that source are validated. Do not multiply per-API
physics algorithms by adding further independent shader translations.

The portable execution contract needs to specify:

| Responsibility | Vulkan | D3D12 | Shared invariant |
| --- | --- | --- | --- |
| Compiled code | SPIR-V | DXIL | One source-derived algorithm, same parameter/data layout |
| Storage resources | Descriptor sets and storage buffers | SRV/UAV descriptors and heaps | Logical read/write bindings and range/lifetime |
| Small dispatch parameters | Push constants | Root constants/CBV | One declared value schema; backend-specific lowering |
| Dispatch and workgroup | `vkCmdDispatch`, SPIR-V local size | `Dispatch`, HLSL `numthreads` | Logical item indexing, supported group size, bounds checks |
| Intra-GPU dependencies | Compute barriers, queue synchronization | UAV/resource barriers, fences | Explicit producer/consumer and read/write hazards |
| CPU/GPU handoffs | Vulkan submissions/fences | D3D12 queues/fences | PhysX scene task completion and publication ordering |
| Allocation/state | Vulkan buffer/heap/device addressing | D3D12 resources/heaps and addressing | PhysX buffers, aliasing and ownership survive each backend |

D3D12 UAV accesses between dependent dispatches require explicit barriers;
submitting the commands consecutively does not necessarily establish the
correct ordering. A portable batch cannot rely on Vulkan implementation
details for this or expose Vulkan descriptor-set semantics as the GPU scene
interface.

The next cross-backend acceptance gates are:

1. Build this AABB kernel from one source to **both SPIR-V and DXIL**,
   preserving a verified binary parameter/binding contract.
2. Run the same input/reference test on Vulkan and a native D3D12 compute
   backend, including producer/consumer dispatch barriers and result download.
3. Expand the real PhysX broadphase stages with the same dual-backend source,
   memory and task dependencies. Do not treat D3D12 as a final translation
   after all Vulkan kernels and orchestration have already been designed.

## Implementation sequence and acceptance gates

1. **Integrate at the PhysX scene boundary.** Identify the narrowest replacement
   seam in `PxPhysXGpu`, GPU contexts and task completions, preserving
   `Sc::Scene` scheduling. Produce an exact call/dependency map for selected
   GPU rigid-body configuration(s), including CPU/GPU BP combinations.
2. **Trace memory/descriptor ownership end to end.** Follow host-pinned,
   mapped and device allocations; buffer aliasing; contact/patch streams;
   pointer-rich descriptors; resize/lifetime rules; and event synchronization.
   Determine what needs shader layout conversion versus host adaptation.
3. **Trace a real PhysX-derived slice.** Choose a bounded existing subpath
   (e.g., one broadphase stage or a specific narrowphase primitive) with
   identified upstream kernels, launch wrappers, inputs, outputs, dependency
   edges and test fixtures. No new standalone world or invented contact solver.
4. **Build dual-backend execution contracts.** Validate Vulkan/SPIR-V and
   D3D12/DXIL resource and scheduling semantics before treating Vulkan-specific
   bindings as the canonical design.
5. **Connect, verify, and extend.** Execute the PhysX-origin slice at its real
   scene pipeline boundary. Compare behavior with CPU PhysX and original CUDA
   PhysX where available; test ordering, contacts, constraints, sleeping and
   result publication as relevant. Exact floating-point identity is not a gate.

This matrix is a starting inventory. Every implementation PR must refine its
row with concrete source entry points, input/output layouts, synchronization
and behavioral tests. A passing compute smoke test is not PhysX GPU conformance.
