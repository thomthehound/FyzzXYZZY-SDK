# Retroactive PhysX-to-GPU performance-departure audit

**2026-10-14 baseline; closure reviewed 2026-10-14 (UTC merges through 2026-10-15):** the 24 common broadphase HLSL stage
sources initially assessed immediately after PR #44. The original findings are preserved below; remediation status is updated separately. Comparison is with pinned
PhysX GPU CUDA sources:
- `physx/source/gpucommon/src/CUDA/RadixSort.cuh`
- `physx/source/gpubroadphase/src/CUDA/broadphase.cu`
- `physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp`.

**Purpose:** find and correct *our* large, avoidable speed losses
relative to source PhysX; not invent a faster broadphase, and not
micro-optimize every instruction before the GPU scene exists.
See [the binding policy](GPU_PORT_PERFORMANCE.md).

## Active cross-backend memory optimization ledger (October 16, 2026)

This ledger records our **own design and translation departures** that impose
avoidable synchronization, bandwidth, or architectural costs. A source-faithful
kernel port is not sufficient unless the original PhysX GPU/CPU allocation and
publication mechanisms have been audited as well. Keep entries open until
validated on a real target workload, not merely green software-renderer CI.

| Debt / root cause | Severity and consequence | Corrective work / remaining acceptance |
| --- | --- | --- |
| **D3D12 ignored `MemoryPlacement::HostVisible` entirely.** Every buffer was placed in a device-only DEFAULT heap, despite the API claiming that `HostVisible` supported mapping. A kernel-centric, overly literal CUDA lowering failed to account for PhysX's CPU-readable, GPU-written mapped broadphase pair reports and completion flags. | **P1 correctness/architecture; P0 candidate performance** on bandwidth-constrained UMA (e.g. Radeon 680M). Forced GPU→CPU staging and extra memory traffic could erase the benefits of GPU simulation; the actual speed penalty is not yet measured. | **PR #85 merged:** strict GPU-writable CPU-mappable `HostVisible`, D3D12 UMA WRITE_BACK custom-heap selection and capability probing, fence-controlled direct mapping and cross-backend tests. **PRs #86–#88 merged:** original mapped-host heap, native source fetch and all three GPU report kernels now share actual PhysX buddy-allocated host-pair storage, with direct mapping or staged fallback. **Still open:** real native scene linkage, GPU/CPU cache behavior and zero-copy-versus-staged measurement on representative hardware (intentionally deferred until architecture complete). |
| **Busy-poll inheritance from upstream `PxgCudaBroadPhaseSap::gpuDMABack`** (`spinWait` on `mEventMapped` before stream synchronization). A literal port would waste CPU cycles and potentially read poorly cached mapped memory. | **P1 host-side scheduling and synchronization.** Source task graph must remain intact, but a polled flag need not be emulated as a CPU spin. | Existing `BroadPhaseCompletionGate` uses source submission fences: Vulkan `vkWaitForFences`, D3D12 `ID3D12Fence::SetEventOnCompletion` + OS event. **Still open**: wire into the real GPU scene and remove the original mapped-flag polling call site; assert no busy-poll loop is introduced. |
| **Source actor-pair mapped-host writes initially implemented as staged report copies** in PR #84. Correctly preserves publication order but duplicates bytes even on devices with fast coherent UMA. | **P1/P2 bandwidth and latency.** Staging is a required compatibility fallback, not the desired default for integrated GPUs. | **PRs #86–#88 merged:** original host-root callback supplies the same CPU/GPU backing on capable adapters or a staged fallback; the original report kernels write to that PhysX-owned heap and source publication observes the GPU fence. Real GPU-scene integration and subsequent hardware comparison remain open; no speedup claimed. |

| **Original `PxgCudaBuffer` growth and pointer ownership previously existed only as fixture-local field models.** The runtime could not construct an actual source buffer over the ported buddy heap, making source allocation/fetch integration incomplete. | **P1 source-host integration.** A duplicate buffer/lifetime policy risks extra allocation, premature reuse, and GPU/CPU stalls. | **PR #89 merged:** original PhysX GPU buffer class uses native source stream and buddy-allocator lifetime semantics; both platform tests passed. **PR #90 merged:** original `preBroadPhase` input buffer uploads now consume `Bp::BroadPhaseUpdateData`. **Still open:** attach the source method to the real broadphase owner, integrate `gpuDMAUp` and live-scene paths, and measure GPU host overhead on representative hardware. |

**Guardrail:** every future source integration review must inspect the CUDA
host allocations, GPU aliasing, cache-policy requirements, CPU/GPU completion
signals and allocation/retirement contracts **before** choosing native
storage and dispatch designs. Any deliberate departure gets an entry here
with functional coverage and eventual performance evidence.

## C++23-enabled native-host optimization candidates (October 16, 2026; PR #91)

**Status: investigation ledger, not achieved speedups.** PR #91 makes ISO
C++23 a required SDK language/library baseline (including `std::expected`);
it does not itself eliminate host allocations, change GPU dispatches, or
alter original PhysX orchestration. Some useful primitives below
(`std::span`, `std::array`, `std::atomic::wait`) predate C++23; the newer
baseline merely guarantees they are usable. Keep source/runtime semantics
and actual compiler/library support distinct from language-standard labels.

| Candidate / observed call site | Priority and plausible benefit | Experiment, constraints and acceptance |
| --- | --- | --- |
| **Per-frame upload-list allocations in the source SAP boundary:** `PxgSapPreBroadPhaseNative.cpp::pre_broadphase_native` constructs and reserves `std::vector<SourceSapUpload>` (at most four items) every invocation, while `SourceSapDmaNative::queue_uploads` builds/reserves a second `std::vector<compute::UploadSpan>`. `Context::submit_upload_batch` currently takes a vector reference. | **P1 host-path candidate:** avoidable small heap allocations while constructing GPU upload descriptors; improvement is plausible but **not measured**. C++20 `std::array` + `std::span` are available under C++23. | Carry fixed-size/non-owning views through source DMA and backend upload APIs (add overloads rather than break downstream consumers where appropriate); only allocate if backend staging truly requires it. Preserve order, zero-byte behavior, validation-before-commit, exact host-byte snapshot, and GPU-lifetime retention. Test 0 transient descriptor allocations on repeated frames, then compare per-frame CPU setup and GPU-visible bytes on **both** backends. |
| **Temporary metadata in native submission:** `SourceSapDmaNative` retains `pending_` command and `tokens_` vectors; Vulkan/D3D12 `submit_uploads` and `submit_batch` build validation, copy and resource-tracking collections. Some are retained/reused and others rebuilt for each call. | **P2 measure first:** fewer per-frame CPU allocations, vector growths and descriptor rebuilds; not a proven bottleneck. Typed span-based read-only input and reusable frame-owned scratch are candidate tools, *not* an excuse to remove essential lifetime storage. | Instrument allocation count, peak capacity, CPU submission latency and shader descriptor binding counts in a connected source-frame workload. Separate required GPU staging/command-list allocations from avoidable host temporaries. Keep queue/fence ordering, command validation and failure atomicity; change only measured hot cases. |
| **Recoverable backend admission versus exceptional failures:** `compute_vulkan.cpp` and `compute_d3d12.cpp` can throw `UnsupportedMemoryPlacement` for unavailable GPU-writable host-visible memory; the source-root path has an intentional staged fallback. | **P2 conditional C++23 `std::expected` candidate:** represent **ordinary unsupported-capability outcomes** without exception unwinding *if* they are encountered repeatedly. No assumed gain on the successful path; unexpected OOM/device loss still must fail distinctly. | First measure how often admission runs (per adapter, heap root or frame). Consider `std::expected<..., UnsupportedReason>` only at a well-bounded internal admission interface. Preserve explicit no-silent-fallback policy, keep ABI changes contained, and verify all host-visible/staged cases plus allocation failure and device-loss behavior. |
| **Typed non-owning views of original PhysX descriptors:** `PxgBroadPhaseDesc` has indexed projection, histogram and handle arrays; host lowering and dispatch setup currently manipulate fixed C arrays and explicit byte counts. | **P3 correctness/locality investigation, not an established speed win:** C++23 `std::mdspan` or `std::span` could reduce indexing/marshaling helpers and *possibly* redundant host copies. `mdspan` standard-library availability is **not** implied by passing the current C++23 probe. | Use typed views only after verifying compiler support and identical byte layout, padding, alignment, serialization and HLSL descriptor ABI. Do **not** replace PhysX device pointer fields with view objects or alter GPU-facing structs. Add byte-for-byte descriptor tests; require observed avoided work before calling it a performance optimization. |

**Guardrails:** Track C++23 cleanups separately from **measured** optimizations.
Do not substitute `std::atomic::wait` or host polling for native GPU fences
(`std::atomic::wait` only works with correctly synchronized CPU-visible
atomic state); do not use `[[assume]]`/`std::unreachable` to skip validation
of external GPU addresses or dynamic sizes. Neither `std::ranges` nor
`std::expected` nor a newer compiler mode guarantees faster code.
The source-ordered PhysX GPU task graph, original buddy allocator, deferred
retirement and mapped-host report ownership remain the governing design.

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
| `broadphase.cu::firstPassEndPtsHistogram` → `physx_sap_endpoint_histogram_first_arena.hlsl` | Triple-`uint3` full-group shared Hillis–Steele prefix discarded the source's *already packed* three 10-bit counters and warp scan | **Resolved functionally in PR #49:** restore exact CUDA 10+10+10-bit packing, one native wave scan, 3 barriers/256-entry chunk, source second pass unchanged, CPU oracle both variants/frames |
| `broadphase.cu::computeHistogram` → `physx_sap_region_histogram_first_arena.hlsl` | Original port used 256-wide two-bank shared prefix, eight full-group rounds per iteration vs upstream per-warp scan and short warp-total prefix | **Resolved functionally in merged PR #48:** one `WavePrefixSum`/`WaveActiveSum` per lane, short wave-total prefix and **three barriers per 256-entry iteration**, 512 bytes shared scratch versus 2 KiB; preserve 32 block totals and original four dispatches; portable oracle remains |
| `broadphase.cu::computeIncrementalComparisonHistograms_Stage1` → `physx_sap_incremental_comparisons_first_arena.hlsl` | Three-axis `uint3` 256-entry shared prefix uses eight group-wide scan rounds, unlike CUDA's three warp scans and short warp-total prefix | **Resolved functionally in merged PR #50:** three independent native 32-bit wave prefixes, 3 workgroup barriers/chunk vs ~11, 1536 B scratch vs 6144 B, original stage2/host dispatch and source oracle retained |
| `broadphase.cu::performIncrementalSAP` → `physx_sap_incremental_pairs_arena.hlsl` | Original HLSL added eight rounds of full 256-thread shared `uint2` scan per found/lost batch where CUDA uses two warp scans plus small warp totals; original global reservation is correctly atomic | **Resolved functionally in merged PR #52:** one packed 16+16 native wave prefix per 256-candidate batch, three workgroup barriers, two original 32-bit global counter reservations and all original pair/filter rules retained |
| `broadphase.cu::accumulateReportsStage_{1,2}` → `physx_broadphase_report_classification_{first,second}_arena.hlsl` | Original port introduced an eight-barrier 256-thread shared reduction for stage1 and eight-round shared-memory scans for stage2, unlike CUDA's warp ballots/reductions and warp-first stable compaction | **Resolved functionally in merged PR #51:** stage1 native per-wave reduction with one atomic update per wave; stage2 wave-prefix stable compaction with three group barriers/chunk. Original 32×4×256 dispatches, actor/aggregate found/lost output and copy-report publication unchanged |
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


## Remediation closure decision (October 14, 2026)

**Disposition: known P0 CUDA-to-HLSL broadphase translation regressions corrected; return to source-faithful translation.** This decision covers the specific high-synchronization and contended-counter patterns identified in the static 24-stage audit. It does not mean that the port is integrated with a GPU-enabled PhysX scene or that native hardware throughput is proven.

- PR #44 restored packed active-region arithmetic; PR #46 restored packed radix count/scatter; PRs #48–#52 restored native-wave region, endpoint, comparison, pair-report and report-classification operations. All six final remediation PRs (#46, #48, #49, #50, #51, #52) are merged. Their PR-head GitHub Actions SDK CI completed successfully and the final Codex reviews reported no major issues.
- Functional verification covers both portable and native-wave-selected variants on Vulkan and D3D12 test runners, independent CPU references, original dispatch geometry, multi-stage ordering, full output/canary checking, and stress cases described in individual PRs. Wave selection is feature-driven rather than WMMA-driven; physically exercised wave32/wave64 behavior on production-class GPUs is not yet established.
- Review-raised integrated-caller omissions in #46 and #49, and the post-merge CMake argument regression in #52, were corrected before merge. The combined source-order fixtures now require both the wave and fallback shader binaries, rather than silently making the slow-only path successful.
- Remaining risks are **not additional confirmed P0 translation regressions**: overly broad descriptor access and barriers, dispatch overhead, inherited dense 64-region data structures and radix passes, register/occupancy tradeoffs, possible serial short-wave-total prefixes, physical-device feature admission (issue #47), and native workload profiling (issue #34).
- The old portable full-group shaders remain explicit development/correctness fallbacks. No production performance or parity is claimed for them. Neither successful software GPU CI nor analytical barrier reductions establishes measured speedup on actual hardware.

**Restart gate:** New translation stages may proceed now, using the performance contract before implementation: compare pinned upstream CUDA arithmetic and work decomposition, preserve the original host scheduling and ownership, select hardware-native subgroup operations without assuming 32 lanes, run source-order CPU oracles on both backends, and record introduced costs. Benchmark at an integrated GPU-scene milestone; only reopen remediation as a blocker if evidence identifies a new material translation-induced regression.

The original chronological next-step list below is retained as an audit history rather than an open task list; the actions concerning #46 and #48–#52 are complete. The performance measurement and host-integration items remain open.

## Next targeted corrective steps (historical plan; superseded by closure above)

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

## Capability distinction: wave arithmetic is NOT WMMA

**Wave/subgroup arithmetic is the baseline execution strategy for the
ported broadphase**, not a matrix accelerator. The native wave paths
use `WavePrefixSum`, `WaveActiveSum`, `WaveGetLaneCount` and
`WaveReadLaneFirst`. These are HLSL Shader Model 6.0 wave operations;
our compiled D3D12 fast binaries use SM6.6 to match the project floor.
**None requires WMMA or cooperative matrix hardware.** Radeon RX5000/
680M reference-class devices should be assessed for WaveOps/subgroup
support independently of matrix features.

- D3D12: host checks `D3D12_FEATURE_SHADER_MODEL >= 6_6`
  **and** `D3D12_FEATURE_D3D12_OPTIONS1::WaveOps` before selecting
  the native shader.
- Vulkan: host queries `VkPhysicalDeviceSubgroupProperties` for
  the **compute stage** plus
  `VK_SUBGROUP_FEATURE_ARITHMETIC_BIT`; Vulkan 1.4 alone
  guarantees *basic* subgroup support, **not every arithmetic
  operation**. The production Vulkan 1.4 admission gate is still
  outstanding (issue #47).
- WMMA/cooperative matrices: a *separate optional acceleration tier*
  applicable only where a real matrix-shaped workload warrants it.
  No broadphase scan requires or probes WMMA.

The retained high-barrier HLSL variants currently serve as
independent correctness fallbacks on software/feature-poor development
adapters; they are **not a satisfactory high-performance production
path**. Before claiming a production device supported, require
sufficient subgroup arithmetic (and all other necessary features)
**or** demonstrate a measured adequately fast alternative. Do not
quietly label a device production-supported and switch its critical
kernels to our expensive correctness fallback. A lack of WMMA must
never cause that fallback selection by itself.

## PR #48: region-histogram source-performance restoration (unmeasured)

- **Source:** `broadphase.cu::computeHistogram<WARP_PERBLOCK_SIZE>`
  invoked separately for `BP_COMPUTE_START_REGION_HISTOGRAM` and
  `BP_COMPUTE_REGION_HISTOGRAM`. Source uses a 32-lane warp exclusive
  scan followed by a short warp-total prefix and a block carry.
- **Rejected lowering:** two 256-word shared arrays for a full
  Hillis–Steele prefix, eight synchronization rounds, roughly eleven
  group barriers per iteration (including the group carry), 2 KiB
  scratch. Source CUDA warp primitives do not require this.
- **Corrective native fast path:** a wave-sized exclusive prefix plus
  wave total per physical subgroup, a short prefix across those
  totals, three workgroup barriers per iteration and two 64-word
  scratch arrays (512 bytes). One workgroup-scope wave-ID allocation
  per physical subgroup at kernel entry establishes deterministic
  *logical* projection order independently of physical scheduling.
  Cross-iteration carry remains 32-bit and source-derived.
- **Unchanged:** 32×256 workgroups, 32 per-block totals, start/region
  input arrays, four source-ordered compute/output dispatches,
  created-handles gate, backend ownership and global offsets.
- **Fallback:** original two-bank shared scan remains strictly for
  development environments lacking native compute subgroup arithmetic.
  Both variants run the same entire-arena CPU oracle after the source
  `BP_CREATE_REGIONS` producer on both native backends. Device
  selection must not silently bypass the wave shader.
- **Alternative designs:** fixed warp32 shuffle (not portable);
  one large full-group shared scan (rejected slow fallback);
  subgroup-prefix followed by parallel prefix of subgroup totals
  (additional barriers versus serial prefix of only 4–64 wave totals,
  consider if measured leader bottleneck); decomposing original
  blocks into more dispatches (breaks host/dispatch contract).
- **Performance hypothesis:** reduced global workgroup synchronization
  and LDS traffic, but possible serial leader bottleneck for devices
  running many small subgroups. This is an analytical expectation;
  **no measured speedup** yet. Later profile 680M/RX5000 and
  representative NVIDIA hardware with clustered region counts.
- **WMMA:** sparse scans/prefixes have no compelling dense matrix
  structure; not applicable to this kernel.

## PR #49: source CUDA triple-packed endpoint histograms (unmeasured)

- **Reference:** `broadphase.cu::firstPassEndPtsHistogram` uses a
  32×3 grid of 256 threads, scans the three contributions
  (`start`, `undeleted end`, `all ends`) **as one packed 32-bit
  integer** (`start<<20 | end<<10 | allEnd`). CUDA's comment
  explicitly establishes ten bits per counter: each 256-projection
  iteration produces at most 256, so none can overflow its field.
  Warp `__ballot_sync`/`__popc` counts and a packed warp-total scan
  determine prefixes, with three *independent 32-bit carried totals*
  between iterations.
- **Rejected port:** three-component `uint3` Hillis–Steele workgroup
  scan over two 256-entry shared banks (6 KiB of uint3 scratch
  ignoring padding), eight rounds of full-group barriers per iteration,
  plus carry barriers. The increased synchronization was *our*
  translation, not imposed by CUDA.
- **Corrective native:** identical 10/10/10 bit packing, one
  `WavePrefixSum` and `WaveActiveSum` per projection, two 64-word
  shared arrays (**512 B**), three workgroup barriers per chunk.
  Each hardware wave is given a contiguous logical input slice
  once on entry; per-wave totals are prefixed in their logical order.
  The group leader unpacks the total into three **32-bit**
  carried counters, preserving cross-iteration semantics and preventing
  halfword leakage. Works for physical wave32/wave64 without WMMA.
- **Unchanged:** 32×3×256 original grid, both incremental/current and
  removed/previous frame paths, same start/end/total-end dense arrays,
  original block histogram positions, two compute/output passes,
  original per-axis and per-frame inputs and outputs, host dispatch
  and CPU-GPU ordering.
- **Fallback and admission:** older portable shader remains a
  functional oracle/soft GPU fallback, not an acceptable silent
  production performance substitution. Actual wave feature checks
  select the SM6.6/Vulkan subgroup-arithmetic path. WMMA is not used.
- **Alternatives:** triple independent wave scans waste two wave
  primitives relative to source packed arithmetic; an all-workgroup
  scan adds needless barriers; fixed warp32 ballot would force vendor
  assumptions; modern ranged wave-size declarations might reduce
  specialization but aren't part of our SM6.6 floor.
- **Evidence:** full-arena independent CPU reference (source sorted
  handle + three-axis, all frames, removed/new flags) exercises
  both variants, 0–8201 previous boxes and extra mixed-created
  workloads. CI validates functional results; **no hardware latency
  claim** until RX5000/680M and comparable vendor benchmarks exist.

## PR #50: native three-axis incremental comparison prefixes (unmeasured)

- **Pinned source:** `broadphase.cu::computeIncrementalComparisonHistograms_Stage1`;
  original CUDA runs **32 × 256** and performs *three separate*
  32-bit `warpScanAdd` passes (X, Y and Z). It collects each
  warp's subtotal, prefixes those eight warp subtotals and
  accumulates independent 32-bit carries across per-block
  iterations. Stage 2 adds the 32 block totals and publishes
  three final comparison counts. All original rank/direction,
  deleted-projection and source current/previous endpoint
  selection semantics remain unchanged.
- **Rejected port adaptation:** two arrays of 256 `uint3` values
  (6144 B shared scan scratch) with eight workgroup-wide
  Hillis–Steele scan rounds, roughly 11 full-group barriers
  per iteration. This synchronization is an *introduced*
  cost absent from CUDA's warp arithmetic.
- **Corrective fast path:** three native `WavePrefixSum` and three
  `WaveActiveSum` operations per projection, 64 `uint3`
  wave totals plus 64 wave bases (1536 B shared), three full
  workgroup barriers per 256-entry iteration. Each physical
  wave claims a contiguous **logical** projection range once
  before processing the group, avoiding ungrounded assumptions
  about wave32 or group-index-to-lane mapping. A workgroup
  leader prefixes the small number of wave totals, then
  advances **three separate 32-bit carries**.
- **Why no packing:** Unlike endpoint 0/1 values, a projection's
  comparison count can be arbitrarily large within the source
  numeric contract. Packing three values into narrow fields
  would cause erroneous cross-field carries. Three native
  wave arithmetic operations are faithful to the original
  CUDA strategy; no WMMA acceleration is applicable.
- **Preserved:** `BP_COMPUTE_INCREMENTAL_CMP_COUNTS1/COUNTS2`
  source order, exact 32×256 geometry, original per-axis
  arena offsets, full comparison prefix outputs, original
  32-block totals, host ownership and no additional dispatch.
  Existing endpoint-histogram and downstream incremental-pair
  source-order fixtures now require the comparison wave
  shader and assert actual capability-based selection.
- **Validation:** independent CPU source comparator traverses
  sorted/ranked previous and current endpoints, then checks
  the full GPU arena including canaries, on both wave-selected
  and forced portable paths; incremental-pair integration
  also checks the wave-selected predecessor. Host validation
  now rejects `2*boxes+255` overflow, matching the original
  projection-rounding arithmetic.
- **Alternatives:** fixed CUDA warp32 translation (not portable),
  packing variable-width comparison counts (unsafe), a single
  256-lane shared scan (reintroduces barriers), extra global
  reduction dispatch (alters original host/dispatch graph).
- **Performance:** barrier/LDS improvements are *analytic*,
  not a measured latency win; many tiny waves could introduce
  a serial leader prefix cost, so benchmark native wave32/64
  devices when a meaningful GPU timeline is available.


## PR #51: report classification and stable compaction (unmeasured)

- **Pinned CUDA:** `broadphase.cu::accumulateReportsStage_1/2`.
  Source stage1 counts actor/aggregate eligibility for found/lost
  streams with warp ballot and per-warp reduction; stage2 prefixes
  the 32 block counts and uses warp ballot/popcount plus a small
  warp-total prefix to compact reports stably. It publishes
  aggregate found/lost totals and preserves each stream's
  original report order for the subsequent `copyReports` kernel.
- **Introduced cost:** the original HLSL stage1 used a 256-value
  shared reduction requiring eight group barriers after staging,
  rather than CUDA's native warp reduction; stage2 used two
  256-value shared scan arrays (**2 KiB**) with eight
  workgroup-wide scan rounds per report iteration.
- **Corrective stage1:** each thread counts eligible reports over
  its original source-defined iterations; `WaveActiveSum`
  reduces per hardware wave, and one wave leader atomically
  adds the subtotal to a single shared `uint`. Two
  workgroup barriers total. Compared with CUDA's warp ballot,
  the reduction semantics are identical for integer counts;
  integer commutativity makes native wave scheduling irrelevant.
  This does introduce up to 64 wave-leader LDS atomics per
  workgroup on hypothetical wave4 hardware, so check actual
  GPU timings, wave width and contention before claiming parity.
- **Corrective stage2:** `WavePrefixSum`/`WaveActiveSum` of
  per-report 0/1 eligibility, using two 64-word wave-total
  arrays (**512 B**) and three group barriers per 256
  reports. Each physical subgroup claims a contiguous
  **logical report range** once at kernel entry; its prefix
  rank and previous-iteration carry are added to the source
  32-block exclusive base. This preserves **stable** order
  for all four streams. The original 32 report block totals are
  loaded in parallel by logical GPU lanes and prefixed with native
  wave arithmetic rather than 32 serialized device loads from the
  workgroup leader; a short cross-wave carry preserves the source
  block base. Aggregate totals are still published exactly once by
  the source-designated workgroups.
- **Kept unchanged:** the 32×4×256 stage1/stage2 launches,
  single caller-owned arena, source pair buffers and volume
  membership semantics, original found/lost/actor/aggregate
  partitions, overflow/clamp capacities, downstream
  `BP_COPY_REPORTS`, original CPU/GPU batch and publication
  owner. There are no new source-dispatch boundaries.
- **Alternatives:** retain the full-group Hillis–Steele scan
  (known structural cost), fixed 32-lane CUDA ballot
  translation (not portable to wave64), prefix the small
  wave totals in parallel (extra barriers; assess only if
  serial leader becomes measurable), subdivide into more
  GPU kernels (unnecessary orchestration rewrite). Matrix
  WMMA operations have no useful role in sparse report scans.
- **Validation:** independent complete-arena GPU→CPU oracle
  compares both native wave and forced-portable variants,
  including aggregate/actor classification, cross-workgroup
  stable found/lost report order, empty inputs, short-tail,
  >8192 reports, overflow capacities, three-dispatch
  classification + copy pipeline and canaries. Linux Vulkan,
  Windows D3D12 and Codex remain PR acceptance gates.
- **Performance:** all quoted barrier/LDS differences are
  analytical, *not measured GPU-time improvements*. Source
  structural costs and real-hardware profiling are separate.


## PR #52: source incremental SAP found/lost report compaction (unmeasured)

- **Source:** `broadphase.cu::performIncrementalSAP`,
  `BP_INCREMENTAL_SAP` in the pinned PhysX CUDA implementation,
  maintains **256×3×256** original launch. Per 256 comparison batch
  it computes a 0/1 found flag and 0/1 lost flag, scans them
  separately within warps and prefixes the warp totals, then uses
  at most two global 32-bit atomic reservations for the shared
  found/lost pair streams. The output ordering **between**
  unrelated GPU workgroups is not deterministic in the original,
  but each reserved batch is internally stable.
- **Introduced slow path:** two 256-element `uint2` shared scan
  banks (**4 KiB**) and eight workgroup-wide Hillis–Steele
  rounds per batch, plus reservation/sync overhead. CUDA uses
  warp-local scans and at most a short prefix over eight
  warp totals; this extra synchronization is ours.
- **Native correction:** pack 0/1 found/lost flags into one
  32-bit `(found << 16) | lost` sum. Each whole 256-entry
  batch can produce at most 256 of each, so there can be **no**
  16-bit cross-field carry. One `WavePrefixSum` +
  `WaveActiveSum` per candidate and two 64-word wave-total
  shared arrays (**512 B**) produce stable within-batch
  found/lost ranks after three full group barriers.
  Retain exactly the two original independent 32-bit global
  atomic reservations per group/batch, skipping zero-count
  reservations, and all output clipping/normalization rules.
- **Workgroup mapping:** each hardware subgroup reserves
  a contiguous logical projection slice once on entry,
  preserving the original binary-search candidate pairing
  and all logical rank order without assuming physical
  subgroup indices equal `SV_GroupIndex / 32`. Actual
  WaveOps/compute subgroup arithmetic is probed; this is
  **not** a WMMA path. The old shared implementation remains
  as an explicitly selected correctness fallback, not
  the supported-hardware default.
- **No architectural departure:** no new dispatches, CPU
  staging, buffer ownership, counters, found/lost output
  publication or filters. This change affects the *scan*
  only; all original ranks, sorted handles, group/env
  rules, old/new AABB overlap checks and shared-pair
  allocation semantics are retained verbatim.
- **Validation:** full source-order endpoint histogram +
  incremental comparison + pair producer native GPU oracle
  checks each optimized and forced-portable pair stage
  against an independent CPU found/lost-pair model under
  empty/short/overflow/movement/cross-block scenarios;
  also tests portable comparison-prefix interoperability.
  Both Vulkan and D3D12 targets compile and supply the
  optimized shader explicitly.
- **Alternatives:** two independent wave scans would also
  preserve source dataflow, but double per-wave arithmetic;
  16-bit packed scan saves one native prefix/reduction
  without risking low-half overflow at 256 entries; a
  full-workgroup shared scan recreates the regression,
  extra parallel dispatches alter PhysX orchestration.
  WMMA has no useful matrix-shaped workload here.
- **Performance:** reduction in workgroup barriers and
  shared scratch is analytic, **not** a hardware-benchmarked
  win. Real AMD/NVIDIA timings and full-scene scheduling
  remain separate acceptance work.

