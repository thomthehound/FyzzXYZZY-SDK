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

The CPU SDK provides a world facade and headless smoke test using pinned
PhysX 5.11 source. Linux and Windows builds are covered by GitHub Actions.
An optional Godot 4.7 GDExtension consumer exercises the same core.

The optional standalone Vulkan compute backend currently supports device
enumeration, storage buffer allocation, host transfers, and GPU-side
buffer copies. Shader execution and GPU physics remain future work.

- [Development roadmap](docs/ROADMAP.md)
- [CPU SDK build and testing](docs/BUILDING.md)
- [Portable compute backend](docs/COMPUTE.md)
- [Apache 2.0 license](LICENSE.md)

New FyzzXYZZY code uses SPDX Apache-2.0 headers; imported upstream/third-party
code retains its original license and attribution.

The C++ API uses the `fyzz` namespace; the public headers are under `fyzzxyzzy/`.
