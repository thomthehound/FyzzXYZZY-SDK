<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PhysX memory-access GPU timing experiment

This is the first **measurement instrument**, not a selection of the final
device-pointer ABI. The workload is the original PhysX CUDA
`broadphase.cu::markRemovedPairsLaunch` semantics, ported to one arena
with exact whole-arena CPU oracle comparisons.

## Controlled comparison

The same HLSL is compiled twice:

- **GPU descriptor path:** load the removed-handle, old-bounds, and three
  SAP-array byte offsets from the GPU-packed `MarkRemovedPairsArenaDesc`.
- **Host-resolved subset:** supply those **five** offsets as uniform root /
  push constants, leaving all three previous-frame handle-array offsets
  resolved from the same GPU descriptor. This is a *partial* comparison, not
  a true buffer-descriptor-per-allocation or Vulkan-BDA baseline.

Both kernels execute the same indexed removed-handle loop, clear all three
axes' previous-frame SAP handles, and write the same six empty-bound values.
The exact-result CTest runs both variants and checks all arena words.
The optional benchmark repeats 128 idempotent operations in one ordered
batch, warms up two pairs, alternates the order of seven comparison pairs,
and reports medians. Input memory and arithmetic are held constant.

## How to run

Linux Vulkan, with DXC including SPIR-V support on PATH:

```sh
cmake -S . -B build-perf -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DFYZZXYZZY_BUILD_VULKAN=ON \
  -DFYZZXYZZY_BUILD_PHYSX_GPU_AABB_TEST=ON
cmake --build build-perf --target fyzzxyzzy_physx_mark_removed_arena
ctest --test-dir build-perf -R '^fyzzxyzzy_physx_mark_removed_arena$' --output-on-failure
./build-perf/fyzzxyzzy_physx_mark_removed_arena \
  build-perf/kernels/physx_mark_removed_pairs_arena.spv \
  build-perf/kernels/physx_mark_removed_pairs_resolved.spv --benchmark
```

Windows D3D12, with `dxc` on PATH or `-DFYZZXYZZY_DXC=...`:

```powershell
cmake -S . -B build-perf -A x64 -DBUILD_TESTING=ON `
  -DFYZZXYZZY_BUILD_D3D12=ON -DFYZZXYZZY_BUILD_D3D12_AABB_TEST=ON
cmake --build build-perf --config Release --target fyzzxyzzy_physx_mark_removed_arena_d3d12
ctest --test-dir build-perf -C Release -R '^fyzzxyzzy_physx_mark_removed_arena_d3d12$' --output-on-failure
./build-perf/Release/fyzzxyzzy_physx_mark_removed_arena_d3d12.exe `
  build-perf/kernels/physx_mark_removed_pairs_arena.dxil `
  build-perf/kernels/physx_mark_removed_pairs_resolved.dxil --benchmark
```

For hardware D3D12 profiling, ensure
`FYZZXYZZY_D3D12_FORCE_WARP` is **not** set to `1`.
For Vulkan, enumerate devices and choose a real GPU in your local
environment before interpreting reported timings.

## What the numbers mean

`Context::profile_batch_gpu_nanoseconds` is an opt-in blocking operation.
Vulkan records timestamps around one command batch via a timestamp query
pool and converts valid timestamp ticks using device `timestampPeriod`.
D3D12 inserts two timestamp queries, resolves them to a readback buffer,
and divides the ticks by the compute command queue frequency. Device
timestamp unavailability returns `std::nullopt`, **never CPU time
mislabeled as GPU time**.

The GPU duration includes all **in-batch copies and synchronization
barriers**, not isolated kernel execution. These executors currently
allocate/record command lists and descriptors per batch. The reported
host submit+wait wall time includes those costs and OS scheduling;
it is *not* a precise CPU-only encoding metric. There is no production
overhead when unprofiled submissions use the normal path.

**Interpretation rules:** do not use WARP or llvmpipe/lavapipe relative
speeds as evidence of AMD hardware throughput. Do not assume a 5-offset
specialization wins on other PhysX kernels, represents a general
pointer-decode cost, or addresses the 104-slot broadphase descriptor.
Run repeated hardware measurements across representative scene sizes,
then add direct binding, segmented logical references and Vulkan BDA to
a common harness before selecting an architecture.

The benchmark reports hardware observations only when you actually run it
on hardware. The repository's CI establishes that the two variants produce
the same PhysX result; it does **not** choose the faster representation.
