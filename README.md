<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FyzzXYZZY

Burping new life into real-time physical simulations.

FyzzXYZZY is a cross-platform, independent physics SDK derived from PhysX. The
objective is hardware-neutral CPU/GPU simulation (Vulkan and D3D12 initially),
not an SDK that depends on any particular game engine.

Godot 4.7 is the first target consumer and acceptance harness, **not a core
dependency**. The SDK is intended to support Linux, Windows, native C++
applications and other host engines.

The native CPU scene implements an admitted box/plane rigid-body subset
with source-derived SAP, PCM narrowphase, island/task ownership, Dy PGS
dynamics, materials, constraints and contact reporting. This is not complete
PhysX CPU feature coverage.

The native GPU World integrates source-derived AABB/SAP and aggregate
broadphase on **Vulkan and D3D12**, with persistent frame histories and
found/lost pair publication into the same native Sc/PCM/Dy scene. Supplying
the optional `NativeGpuPcmShaders` bundle also runs source GPU box/box PCM,
sparse GJK/EPA continuation and world-space Gu contact generation for live
Sc contact managers. Without that bundle, narrowphase remains on the CPU.

This is still a **hybrid** pipeline: Sc publishes GPU contacts for CPU Dy
constraint preparation and rigid solving. Device-resident constraint
preparation, GPU PGS and body integration, broader geometry, articulations
and deformables are not yet production features.

Use `ExecutionMode::NativeCpu` for the independent CPU scene. The ordinary
`World` default still uses the optional reference CPU backend, and direct
`WorldOptions::NativePortableGpu` construction is not admitted. The explicit
`create_native_gpu_world(...)` factory in `fyzzxyzzy/native_gpu_world.h`
constructs the source-derived GPU-broadphase hybrid with compiled shader
bundles. There is no implicit reference-physics or CPU-SAP fallback in that
mode. The Godot adapter currently exercises native CPU execution.

Both native GPU backends have reference-free SDK builds and device-executed
conformance coverage. CI software Vulkan and Windows D3D12 WARP establish
functional/API behavior, not representative hardware performance.

**[Current capabilities and implementation priorities](docs/ROADMAP.md)**
is the authoritative roadmap. Older numbered migration and GPU execution
gates are historical source-intake records, not parallel milestone programs.

- [Native engine ownership and migration plan](docs/NATIVE_ENGINE_MIGRATION.md)
- [SDK-wide enforced architecture contract](docs/ARCHITECTURE_CONTRACT.md)
- [CPU SDK build and testing](docs/BUILDING.md)
- [Portable compute backend](docs/COMPUTE.md)
- [PhysX GPU port architecture and traceability](docs/PHYSX_GPU_PORT.md)
- [Source-grounded GPU port implementation program](docs/GPU_PORT_PROGRAM.md)
- [PhysX CUDA-host compatibility feasibility and source integration](docs/PHYSX_CUDA_HOST_COMPATIBILITY.md)
- [Source-grounded GPU broadphase trace](docs/GPU_BROADPHASE_TRACE.md)
- [GPU memory addressing and performance experiment RFC](docs/GPU_MEMORY_ADDRESSING_RFC.md)
- [**GPU performance policy and shipping tiers (SM6.6, Vulkan 1.4, optional WMMA/SM6.8)**](docs/GPU_PORT_PERFORMANCE.md)
- [Retroactive CUDA-to-HLSL performance audit, corrections, and closure decision](docs/GPU_PERFORMANCE_AUDIT.md)
- [Apache 2.0 license](LICENSE.md)

New FyzzXYZZY code uses SPDX Apache-2.0 headers; imported upstream/third-party
code retains its original license and attribution.

The C++ API uses the `fyzz` namespace; the public headers are under `fyzzxyzzy/`.
FyzzXYZZY native targets and consumers require **C++23**; see [build requirements](docs/BUILDING.md).
