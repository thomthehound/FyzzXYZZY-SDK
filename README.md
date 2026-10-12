<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FizzX-SDK

Burping new life into real-time physical simulations.

FizzX is a cross-platform, independent physics SDK derived from PhysX. The
objective is hardware-neutral CPU/GPU simulation (Vulkan and D3D12 initially),
not an SDK that depends on any particular game engine.

Godot 4.7 is the first target consumer and acceptance harness, **not a core
dependency**. The SDK is intended to support Linux, Windows, native C++
applications and other host engines.

The current M0 development branch provides a **provisional** CPU-backed world
facade and headless smoke test using pinned PhysX 5.11 source. It has **not yet
been verified on both platforms** or integrated into Godot. Do not mistake this
bootstrap for a released engine.

- [Development roadmap](docs/ROADMAP.md)
- [M0 build and testing](docs/BUILDING.md)
- [Apache 2.0 license](LICENSE.md)

New FizzX code uses SPDX Apache-2.0 headers; imported upstream/third-party
code retains its original license and attribution.
