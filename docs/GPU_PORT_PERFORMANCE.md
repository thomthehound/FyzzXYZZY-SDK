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
| P-001 | **Our lowering** | PR #44 initially replaced CUDA packed warp scan of `BP_COMPUTE_ACTIVE_HISTOGRAM` with **two** 512-wide shared-memory Hillis–Steele scans: 8 KiB scratch, nine rounds, many workgroup barriers per chunk; unnecessary synchronization on AMD/NVIDIA | **Fix now**: source-like packed representation and subgroup-aware fast path, preserve isolated CPU oracle, measure when GPU bench available |
| P-002 | **Upstream-native** | Fixed 64-region dense start/active histograms and two ordered lists scale linearly with 64×number of endpoints; substantial bytes cleared/scanned | **Document, defer architectural rework** until faithful end-to-end broadphase exists. Do not invent a sparse replacement prematurely |
| P-003 | **Our lowering, audit pending** | Existing `sap_region_histogram_first` shared-memory prefix scans use multi-round workgroup barriers instead of CUDA warp scans | Audit/benchmark wave-op strategy next; source behavior already verified |
| P-004 | **Our lowering, audit pending** | Endpoint histograms, incremental comparison/report prefix kernels and radix sort may have serialized steps or unhelpful per-stage barriers versus source | Source-to-source perf inventory and representative GPU timings when usable |
| P-005 | **Backend integration, audit pending** | Whole-arena `ReadWrite` bindings can conservatively synchronize independent work and enlarge descriptor/memory traffic | Validate precise access intent and Vulkan/D3D12 hazard/ownership semantics; optimize only where source ordering allows |
| P-006 | **GPU orchestration, future** | CPU/GPU synchronization, staging copies, frame-role rotations and dispatch granularity can dominate small workloads even with fast kernels | Trace upstream task graph, async transfers, batched dispatch and publication; profile in original scene owner |
| P-007 | **Hardware opportunity** | WMMA/matrix acceleration not yet evaluated against actual constraint/contact kernel geometry | Build candidate list when approaching solver kernels; only enable where arithmetic intensity and shapes justify it |

Estimates or analytic size illustrations in trace documents are *not*
measured regressions. Distinguish upstream-native costs from newly
introduced port costs, triage ours immediately, and defer structural
changes to original PhysX costs until a working faithful simulacrum
can be profiled and compared. Update this register whenever material
risks or wins become known.
