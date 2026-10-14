# GPU port performance contract

**Status:** binding project engineering policy for the FyzzXYZZY GPU backend
(October 14, 2026). Applies to all new work and retroactive audits of the
existing port. See `GPU_BROADPHASE_TRACE.md` for pinned upstream source
mapping and `PHYSX_GPU_PORT.md` for progress.

## Objective and reference

- We are porting **NVIDIA PhysX GPU physics**, including its existing CPU
  orchestration, dispatch dependencies, data ownership, synchronization,
  publication, and kernel computations, to Vulkan and D3D12. It is **not**
  an independently invented replacement simulator.
- **Performance is a first-class acceptance criterion, not a cleanup
  after correctness.** An unusably slow port is not successful merely
  because its small fixtures match the CPU version.
- Original NVIDIA **source behavior AND performance strategy** form
  the first reference: preserve asymptotic cost, work decomposition,
  locality, parallelism, workgroup/subgroup usage, memory layout and
  transfers unless a departure is justified and documented.
- The upstream source is not automatically optimal on AMD or other
  hardware. Preserve its *semantics* and host sequencing; deliberately
  change hardware-specific execution when evidence indicates a better
  portable or vendor-tailored implementation. **Never insert a slow
  workaround merely to avoid using a modern GPU primitive.**
- Prefer native GPU measurements to speculation. Correctness tests
  remain necessary but are *not* a performance substitute. Do not
  claim a speedup or parity without a reproducible benchmark.

## Minimum GPU capability target

The intended shipping floor is **DirectX Shader Model 6.6 capable hardware
or better**. Do not optimize for GPUs below that floor. This does **not**
mean Vulkan exposes a single `SM6.6` bit: declare and query Vulkan
equivalent capabilities separately (including subgroup arithmetic,
supported subgroup stages, 64-bit ops, and any optional matrix or
atomic operations used). Backend support must be explicit, not silently
assumed. The product hardware floor may increase by user decision.

The Vulkan and D3D12 CI runners—including software adapters—serve as
**functional test environments**, not proof that a GPU meets the intended
shipping floor or achieves acceptable throughput. Keep development CI
usable independently of the release capability policy. Explicitly
track D3D12 `D3D12_FEATURE_SHADER_MODEL >= 6_6` and relevant WaveOps
availability before enabling an optimized native fast path; do not
quietly equate shader compilation to device support. For Vulkan, query
`VkPhysicalDeviceSubgroupProperties` including
`VK_SUBGROUP_FEATURE_ARITHMETIC_BIT` and compute-stage support.
For compatibility with CI software implementations, a portable
**correctness fallback** may exist but should not dictate the
released fast path. Fallbacks are not a reason to retain a slow
implementation on supported hardware.

## Required performance impact record for every nontrivial port

Whenever the implementation differs from pinned PhysX, document in
the PR description and appropriate source-trace document:

1. **Upstream reference:** exact source path/function and pinned commit.
2. **Delta and reason:** CUDA primitive → Vulkan/D3D12 lowering,
   workgroup/wave shape, state/memory/API difference, or deliberate
   algorithmic optimization, clearly labeled.
3. **Cost hypothesis:** likely effects on dispatch count, subgroup ops,
   barriers, shared memory, register pressure, occupancy, divergence,
   allocations, transfers, memory bandwidth, and asymptotic behavior.
   Quantify analytic bounds when possible; say *unmeasured* when true.
4. **Alternative approaches:** at least plausible native or
   hardware-specific options and why the selected design is preferable
   given current evidence. Performance regressions require explicit
   justification; "easier to implement" alone is insufficient.
5. **Evidence:** correctness oracle and useful performance measurements.
   When the integrated benchmark cannot yet run, name the missing
   harness, record a performance-risk follow-up, and use a low-overhead
   strategy in the meantime.
6. **Hardware considerations:** AMD wave32/wave64 where relevant,
   NVIDIA/Intel behavior, subgroup capabilities, and why WMMA/matrix
   hardware is or is not useful for this dispatch.

Do not inflate every change with speculative precision. A compact
per-stage row can satisfy the record, but **no unreviewed fallback
to high-barrier scalar/shared-memory implementations**.

## Practical acceptance and review order

- Confirm a faithful source dataflow/original host owner.
- Audit reference performance strategy and available SM6.6-era GPU
  primitives **before coding**.
- Implement the best justified native lowering, retaining the original
  behavioral contract and CPU oracle; test at native API boundaries.
- Profile representative scene sizes, distributions, and multi-frame
  workloads when the relevant integrated pipeline exists. Compare
  **GPU time**, not CI wall-clock build/test time; include dispatch
  overhead, synchronization, memory bandwidth, and downstream latency.
- Optimize the measured bottleneck, not the easiest kernel. Maintain
  correctness against pinned PhysX and its CPU reference.
- Keep review and CI gates. Functional green CI does **not** imply
  performance acceptance.

## AMD and WMMA optimization direction

AMD is a primary target, **not** a mandatory execution dependency.
Use AMD-specific optimizations opportunistically with independent
paths for other compliant devices. Consider vendor-specific wave
sizes, LDS traffic, register pressure, occupancy, and dispatch
launch overhead. Use wave/subgroup arithmetic for scan, reduction
and compaction where appropriate rather than emulating CUDA warp
intrinsics with large shared-memory scans.

**Explicitly evaluate WMMA and other matrix/tensor instructions**
when identifying compute-heavy kernels. Candidate areas include
batched dense constraint blocks, transform operations, Jacobian
products and small matrix solves *only if the real PhysX data
shapes, accuracy requirements, packing/conversion cost, and
scheduling make them suitable*. Do not force WMMA into SAP scans,
radix sorting, sparse collision filtering, or other algorithms
without a measurable matrix-shaped workload. Record negative
findings as well as wins. WMMA is an optimization opportunity,
not license to replace PhysX's simulation mathematics.

## Performance-risk register (initial retroactive review)

| ID | Origin | Concern | Current disposition |
| --- | --- | --- | --- |
| P-001 | **Our lowering** | PR #44 initially replaced CUDA packed warp scan of `BP_COMPUTE_ACTIVE_HISTOGRAM` with **two** 512-wide shared-memory Hillis–Steele scans: 8 KiB scratch, nine rounds, many workgroup barriers per chunk; unnecessary synchronization on AMD/NVIDIA | **Fixed in PR #44, acceptance pending**: packed `WavePrefixSum` fast path and packed shared fallback, 2 KiB wave-total scratch, 3 workgroup barriers per chunk (plus initial sync), original dataflow and independent native CPU oracle. Compare on real AMD/NVIDIA hardware when bench available |
| P-002 | **Upstream-native** | Fixed 64-region dense start/active histograms and two ordered lists scale linearly with 64×number of endpoints; substantial bytes cleared/scanned | **Document, defer architectural rework** until faithful end-to-end broadphase exists. Do not invent a sparse replacement prematurely |
| P-003 | **Our lowering, audit pending** | Existing `sap_region_histogram_first` shared-memory prefix scans use multi-round workgroup barriers instead of CUDA warp scans | Audit/benchmark wave-op strategy next; source behavior already verified |
| P-004 | **Our lowering, confirmed significant synchronization exposure** | `physx_radix_scatter_arena.hlsl` processes all 16 digits with ten 1,024-thread Hillis–Steele barriers per digit, **at least 160 full-group scan barriers per radix iteration**, versus packed CUDA warp scans. `physx_radix_count_arena.hlsl` does up to four highly contended LDS atomics per thread, unlike reference register-packed counts | **P0 remediation [issue #45](https://github.com/thomthehound/FyzzXYZZY-SDK/issues/45)**, before new GPU kernel work; benchmark clustered as well as uniform digits, restore packed wave strategy without violating source stable ordering |
| P-005 | **Backend integration, audit pending** | Whole-arena `ReadWrite` bindings can conservatively synchronize independent work and enlarge descriptor/memory traffic | Validate precise access intent and Vulkan/D3D12 hazard/ownership semantics; optimize only where source ordering allows |
| P-006 | **GPU orchestration, future** | CPU/GPU synchronization, staging copies, frame-role rotations and dispatch granularity can dominate small workloads even with fast kernels | Trace upstream task graph, async transfers, batched dispatch and publication; profile in original scene owner |
| P-007 | **Hardware opportunity** | WMMA/matrix acceleration not yet evaluated against actual constraint/contact kernel geometry | Build candidate list when approaching solver kernels; only enable where arithmetic intensity and shapes justify it |

Estimates or analytic size illustrations in trace documents are *not*
measured regressions. Distinguish upstream-native costs from newly
introduced port costs, triage ours immediately, and defer structural
changes to original PhysX costs until a working faithful simulacrum
can be profiled and compared. Update this register whenever material
risks or wins become known.

## PR #44: source-to-port performance record (unmeasured)

- **Original:** `broadphase.cu::computeStartAndActiveRegionHistogram`: one packed start/active 32-bit prefix per 512-element chunk, warp-local arithmetic, subgroup totals, separate cross-chunk carries. Geometry `64×512`; 64 independent region arrays.
- **Rejected port adaptation:** two full 512-element 32-bit shared-memory prefix scans in parallel, 4×512 u32 scratch = **8 KiB**, nine shared prefix rounds and roughly 12 group barriers per chunk. Excessive synchronization independent of PhysX semantics.
- **Selected native:** same packed arithmetic as upstream; one WavePrefixSum/WaveActiveSum per lane, one per-wave shared subtotal and serial short prefix by workgroup leader, **three group barriers per chunk**, 512-word subtotal allocation = **2 KiB**. One InterlockedAdd per hardware wave once at kernel start assigns logical wave order safely on varying wave sizes. Vulkan selects on queried arithmetic subgroup feature, D3D12 on queried WaveOps+shader model. This design is expected to reduce shared-memory traffic and barriers versus the rejected adaptation, but **speedup is not measured**. Check subgroup occupancy and wave atomic overhead with real hardware.
- **Fallback for CI/software:** one packed 32-bit workgroup Hillis–Steele scan with two 512-word scratch arrays = **4 KiB** and nine shared scan rounds. No longer the doubly scanned workaround; not selected on hardware offering the wave fast path.
- **Alternatives considered:** (a) infer subgroup ID directly from `SV_GroupIndex / WaveGetLaneCount` (fewer atomics but not guaranteed thread-to-lane mapping); (b) separate wave scans of unpacked start/active counters (extra subgroup instructions); (c) 64-bit packed scan (unnecessary integer cost and optional feature constraints); (d) change upstream 64-region dense layout (out of scope until faithful scene integration); (e) mixed two-pass global scan (additional dispatch/ownership complexity). Fast path avoids fixed wave32 assumptions.
- **Validation:** both versions feed the same whole-arena independent CPU oracle, source 0/1 tag arithmetic and separate wide carries, multi-512 projections. Both CI native GPU backends must pass. Benchmark hardware latency and bandwidth, frame impact and register/LDS occupancy when GPU frame integration/bench is practical; WARP and llvmpipe measurements are only CI diagnostics.
- **WMMA:** this scan contains prefix/reduction and compaction, no dense matrix multiplication; no defensible WMMA use here. Revisit on batched constraint solves and other eligible kernels with actual shape and conversion-cost evidence.
