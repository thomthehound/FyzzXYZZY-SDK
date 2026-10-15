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

The CPU SDK provides a world facade and headless smoke test using PhysX-derived
solver source. Linux and Windows builds are covered by GitHub Actions.
An optional Godot 4.7 GDExtension consumer exercises the same core.

The optional standalone Vulkan compute backend supports device and buffer
management, transfers, SPIR-V dispatch, ordered batches and submission
synchronization. The GPU translation now includes independently tested PhysX SAP broadphase
kernels and source-ordered stage chains, with known high-priority initial
CUDA-to-HLSL performance regressions remediated. Integration into the live
PhysX GPU scene/task graph, production GPU capability admission, and
hardware performance acceptance remain pending.
Its independent floor/contact demonstration is opt-in only
(`FYZZXYZZY_ENABLE_GPU_PHYSICS_PROTOTYPES=ON`) and does not constitute a
PhysX GPU port. Production GPU physics requires porting PhysX's original
algorithms, GPU contexts, scene orchestration and dataflow.

- [Development roadmap](docs/ROADMAP.md)
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
