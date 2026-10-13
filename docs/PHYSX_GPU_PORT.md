<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PhysX GPU port: execution architecture and traceability

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
