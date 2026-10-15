# Retroactive PhysX-to-GPU performance-departure audit

**2026-10-14; current scope:** the 24 common broadphase HLSL stage
sources on `main` immediately after PR #44. Comparison is with pinned
PhysX GPU CUDA sources:
- `physx/source/gpucommon/src/CUDA/RadixSort.cuh`
- `physx/source/gpubroadphase/src/CUDA/broadphase.cu`
- `physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp`.

**Purpose:** find and correct *our* large, avoidable speed losses
relative to source PhysX; not invent a faster broadphase, and not
micro-optimize every instruction before the GPU scene exists.
See [the binding policy](GPU_PORT_PERFORMANCE.md).

## Audit method and limitations

Reviewed kernel source for workgroups, per-thread work, scratch arrays,
barrier count/loop nesting, atomics, memory patterns, source dispatch
bounds, and whether a CUDA warp primitive was replaced by a large
workgroup-level operation. The scan is **static**, not a benchmark.
Barrier counts in comments refer to *sites or upper/lower bounds*;
loop multiplicity matters. Numeric GPU-time changes must be measured
on actual target devices; native CI correctness is not throughput
evidence.

**Priority definitions:** P0 = obvious, potentially large regression
introduced by our CUDA lowering; P1 = repeated serialization/sync
overhead deserving a source-aware follow-up; P2 = audit, measure
before touching. Separate *upstream-native* costs from port costs.

| Original PhysX stage / port | Static finding | Priority / action |
| --- | --- | --- |
| `RadixSort.cuh::radixSortCalculateRanks` → `physx_radix_scatter_arena.hlsl` | **16 digit-by-digit 1024-thread shared scans**, each with ten group barriers, plus carries. Original uses packed counters and warp scans. **At least 160 full-group scan barriers** per radix iteration were introduced by us | **P0 corrective PR #46:** native SM6.6 packed two-digit `WavePrefixSum` lowering, eight pairs and native reductions, retaining all source stable rank/order semantics; preserve fallback; GPU validation and real timing outstanding |
| `RadixSort.cuh::radixSortSingleBlock` → `physx_radix_count_arena.hlsl` | 4 shared `InterlockedAdd` calls per uint4 input, contending on just **16** workgroup bins; source accumulates register-local packed counts then warp-reduces | **P0 corrective PR #46:** register-packed two-bin counts with native `WaveActiveSum`, a bounded number of atomic group updates; validate collision-heavy digit distributions and register pressure |
| `broadphase.cu::computeStartAndActiveRegionHistogram` → `physx_sap_active_region_compute_arena.hlsl` | Original packed warp scan was initially lowered to two separate 512-entry LDS scans (8 KiB and nine rounds each) | **Resolved functionally in PR #44**: original packed 32-bit arithmetic restored, wave fast path (2 KiB scratch, 3 barriers/chunk), portable packed fallback. Bench still missing |
| `broadphase.cu::computeEndPtsHistogram` → `physx_sap_endpoint_histogram_first_arena.hlsl` | 256-wide `uint3` Hillis–Steele scan, eight sync rounds in an inner block loop; source uses warp scans / per-warp totals | **P1:** introduce native wave scan + compact warp-total prefix while retaining exact triple histogram and multi-frame behavior |
| `broadphase.cu::computeHistogram` → `physx_sap_region_histogram_first_arena.hlsl` | 256-wide two-bank shared prefix scan, eight rounds per iteration; source CUDA warp scan strategy | **P1:** native subgroup fast path; source partition, 32 block totals and four-dispatch scan order unchanged |
| `broadphase.cu::computeIncrementalComparisonHistograms_Stage1` → `physx_sap_incremental_comparisons_first_arena.hlsl` | Per-iteration three-axis `uint3` shared prefix, eight rounds; source CUDA warp primitives | **P1:** wave prefix plus 32-block totals, retain source comparison selection and rank pairing |
| `broadphase.cu::performIncrementalSAP` → `physx_sap_incremental_pairs_arena.hlsl` | Two-output `uint2` shared prefix with eight rounds per batch; global report-count reservation still uses atomic increments | **P1:** wave-based packed/dual report scan, preserve original global reservations and report order; do not change candidate correctness rules |
| `broadphase.cu::accumulateReportsStage_2` → `physx_broadphase_report_classification_second_arena.hlsl` | 256-thread shared scan and repeated per-block carry barriers | **P1:** examine wave prefix / report-label propagation, maintain original actor/aggregate found/lost classification |
| Other copied/report producers and simple PhysX kernels (AABB encoding, initialized rank/index lists, removed projections/pairs, clear new flag, update handles) | No obvious **high-order new workgroup-wide scan** in static inspection; typically direct per-entry loops and/or source-sized workgroups. This is **not** a measured performance endorsement | **P2:** collect timestamps and source diff as these become integration-relevant; no speculative redesign |
| 64-region dense histograms, large ordered arrays, multi-pass source radix and PhysX fixed 32-block SAP partitions | **Inherited from original PhysX**, regardless of whether modern hardware could do better | **Defer structural redesign** until source-faithful end-to-end GPU scene and representative benchmarks exist |
| Full-arena storage bindings and GPU task graph | Current `ReadWrite` descriptor intent is often conservative; synchronization and whole-arena spans may exceed true resource dependencies. Host orchestration is not yet a live PhysX scene | **Cross-cutting P1/P2:** preserve PhysX GPU ordering and ownership, measure barriers/copies/descriptors when real GPU submission pipeline is integrated |

## SIMD implementation guardrails

- **Wave subgroup size is a runtime capability**, not a fixed CUDA 32.
  An HLSL `WavePrefixSum` must work on the actual contiguous logical
  keys; if physical subgroup assignment is unspecified, give each
  subgroup a contiguous logical range and prefix subgroup totals in
  that logical order. Do not presume physical `SV_GroupIndex`
  maps predictably to subgroup indices.
- Packed **16-bit** halfword arithmetic is safe only when the
  maximum *pre-unpack* low-half sum is bounded by 65535.
  For the radix-count kernel, per-thread counts share a
  source-style packed representation; validate size bounds before
  relying on them, especially if arena addressing is enlarged later.
  For the radix-scatter kernel each 1024-vector chunk includes at
  most 4096 scalar keys, so a packed pair can be scanned before
  unpacking without cross-half carry.
- Avoid invoking fast shaders on feature-unsupported devices.
  SM6.6 device support is queried for D3D12; Vulkan requires actual
  compute subgroup arithmetic. WARP/lavapipe may exercise fallback
  and therefore **cannot alone validate native-wave runtime behavior**.
- Compare meaningful **complete multi-pass radix stable order**,
  histogram block totals, untouched padding, active/removed/new tags,
  and original report publication against independent CPU oracles.
- WMMA does **not** fit bitwise radix binning, sparse prefix scans
  or simple index compaction. Evaluate it aggressively later on
  actual dense constraint/contact computation shapes, without
  making it a mandatory GPU capability.

## Next targeted corrective steps

1. Finish PR #46's **source-style packed-wave radix count + scatter**
   after Linux and Windows CI, Codex review, and integrity checks.
   Record whether each native runner exercised the hardware wave
   variant or only the portable fallback. Obtain actual GPU timings
   on RX 5000 and/or Radeon 680M and a representative NVIDIA GPU
   when that becomes practical; WARP timing is not a performance result.
2. Audit the **three 256-thread prefix families** (endpoint histograms,
   region prefix, incremental comparisons) as one source-aware SIMD
   lowering effort; avoid three subtly different hand-rolled scans.
   Add shared tested utilities if it improves execution, but never
   assume fixed wave width.
3. Examine incremental-report compaction and radix host dispatch
   batches for introduced serialization. Preserve the original
   CPU/GPU task graph, async transfers, resource lifetimes and
   publication before declaring full-pipeline parity.
4. Only after the faithful pipeline works: profile the inherited
   dense 64-region arrays, radix strategy and potential matrix/
   WMMA stages as optimization research, without silently changing
   PhysX algorithms.

This audit is maintained alongside `GPU_PORT_PERFORMANCE.md`
and must be updated with source-based evidence, correction PRs,
newly discovered risks and eventually measured GPU timings.
