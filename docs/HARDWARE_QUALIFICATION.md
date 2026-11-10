<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Physical-GPU execution qualification

The Windows PowerShell 7 appliance in `tools/hardware/qualify.ps1` runs the
existing native Sc/PCM/Dy scene cases through their production Vulkan and
D3D12 compute paths. It does not introduce a benchmark scene scheduler,
separate solver, or alternative runtime ownership.

## Preparation

Build the source-GPU scene fixture in **Release** configuration with the
independent native core:

```powershell
cmake -S . -B build-native -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DFYZZXYZZY_BUILD_VULKAN=ON -DFYZZXYZZY_BUILD_D3D12=OFF `
  -DFYZZXYZZY_ENABLE_PHYSX_SOURCE=OFF `
  -DFYZZXYZZY_BUILD_SOURCE_GPU_TESTS=ON
cmake --build build-native --target fyzzxyzzy_source_sc_gpu_pcm_scene --parallel 8

cmake -S . -B build-native-d3d12 -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DFYZZXYZZY_BUILD_VULKAN=OFF -DFYZZXYZZY_BUILD_D3D12=ON `
  -DFYZZXYZZY_ENABLE_PHYSX_SOURCE=OFF `
  -DFYZZXYZZY_BUILD_SOURCE_GPU_TESTS=ON
cmake --build build-native-d3d12 --target fyzzxyzzy_source_sc_gpu_pcm_scene_d3d12 --parallel 8
```

Use a development shell with MSVC and DXC accessible, and a Vulkan SDK for
the Vulkan build. Existing configured Release directories can be reused.

## Run

```powershell
pwsh -NoProfile -File tools/hardware/qualify.ps1 -Backend Both -Scale
```

The default uses `build-native` and `build-native-d3d12`. To override:

```powershell
pwsh -NoProfile -File tools/hardware/qualify.ps1 `
  -VulkanBuild build-native -D3D12Build build-native-d3d12 `
  -OutputDirectory build-hardware-evidence -Samples 7 -WarmupRuns 2 -Scale
```

Use `-Backend Vulkan` or `-Backend D3D12` for isolated diagnosis. Omit
`-Scale` for the core suite only. Each backend executes correctness cases
once, then separate warm-up and sampled repetitions of selected execution
patterns. A failure stops execution immediately and retains the failure log.

The appliance enforces physical-device creation in both backends.
`FYZZXYZZY_D3D12_REQUIRE_HARDWARE=1` rejects D3D12 software adapters.
`FYZZXYZZY_VULKAN_REQUIRE_HARDWARE=1` rejects Vulkan CPU, virtual and
other non-integrated/non-discrete devices. It also rejects an inherited
`FYZZXYZZY_D3D12_FORCE_WARP=1`. Every scene invocation reports the
backend-selected adapter, numerical device limits and UMA classification.

## Workload and evidence

The correctness run covers production-World 1K independently contacting bodies,
integrated Scene contact and device admission, the indirect API, coupled
PCM/Gu/1D rigid solving, CPU/GPU Dy switching,
plane-only PCM, solver recovery, free integration, finite and convex
geometry, aggregate coupling, mixed producers, wide islands and
nonresponding constraints. With `-Scale`, it adds a production-World 10K contacting-body
comparison and 10K star-island and wide-island cases. These stress tests retain the fixture's assertions;
they do not downsize the simulation to achieve a pass.

Results are written beneath `build-hardware-evidence/<UTC timestamp>/`:

- `metadata.json`: host OS, graphics drivers, source revision, build paths,
  run parameters.
- `runs.jsonl`: backend, scenario, phase, process-wall duration, selected
  adapter, structured `FYZZXYZZY_METRIC` fields and corresponding log.
- `summary.json`: sampled process-wall p50/p95/p99 for each profiled case.
- Individual logs: full correctness diagnostics and native scene metrics.

The hardware-scene fixture creates two equivalent production Worlds
sequentially, with the same GPU broadphase and PCM stages but different
explicit numerical executors. Each World owns only its own context. It
measures 48 completed frames at 1K bodies or 16 at 10K, after four warm-up
frames, and checks all final body positions and velocities. Its p50/p95/p99
are **whole-scene step** durations inclusive of host work, transfers and
synchronization. It uses one static platform and isolated dynamic contacts;
actor count and contact complexity are reported separately from the wide
constraint workloads.

The coupled-solver fixture additionally measures eleven in-process CPU/GPU
steps per solver mode and reports p50/p90, contact/friction/pose errors,
upload/download bytes, dispatch counts and saved preparation. The script's
p50/p95/p99 are **whole-process** timings that include startup and pipeline
setup. None of these values is a device-stage timestamp.

## Acceptance and next measurements

A passing appliance establishes real-device execution of its covered
correctness scenarios, including the coupled numerical paths and switching.
The remaining hardware gate requires device stage timestamps, host wait
and overlap attribution, buffer-allocation peaks, descriptor/pipeline
churn, 100K whole-scene throughput, and workload contact/constraint counts
beyond the rigid-body totals. Compare GPU and CPU Dy on equivalent configurations
and solver quality before setting a crossover policy. In particular,
measure the PCM-to-Sc witness join against its actual CPU consumer rather
than assuming its removal improves the frame.

Do not merge a throughput regression on the basis of process-wall times
alone. Record the absolute frame budget and relative regression threshold
for each representative workload before accepting an optimization.
