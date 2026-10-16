<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Portable PhysX GPU source-integration audit — 2026-10-15

Status: manual architecture/source review accompanying the proposed connected
ten-operation native GPU region-chain test (PR #73). **Do not infer production
scene acceptance from this test.** The specific regression must still pass
Linux Vulkan and Windows D3D12 and undergo review before merging.

This review uses the upstream PhysX source family as the semantic authority,
without pinning consumers to a particular PhysX revision. Source anchors:
`gpubroadphase/src/PxgCudaBroadPhaseSap.cpp`,
`gpubroadphase/src/CUDA/broadphase.cu`, `gpubroadphase/src/PxgAABBManager.cpp`,
`gpucommon/src/PxgCudaBuffer.cpp`, `gpucommon/src/PxgHeapMemoryAllocator.cpp`,
`simulationcontroller/src/ScPipeline.cpp`, `ScScene.cpp`,
`gpunarrowphase/src/PxgNphaseImplementationContext.cpp`, and
`gpusolver/src/PxgContext.cpp`, `PxgCudaSolverCore.cpp`,
`PxgTGSCudaSolverCore.cpp`. See `GPU_BROADPHASE_TRACE.md` and
`GPU_PORT_PROGRAM.md` for the full host task/dependency map.

## 1. Exact scope of the connected run

The original host calls are consecutive on `mStream` only under the
*created-handles* gate:

| Source host operation | Device action | Native source implementation |
| --- | --- | --- |
| `computeRegionHistogramKernel` | clear active histogram D32 | `SourceFillD32NativeStage` |
| same | clear start histogram D32 | same |
| same | `BP_CREATE_REGIONS`, 32 × 256 | `SourceRegionCreateNativeStage` |
| same | `BP_COMPUTE_START_REGION_HISTOGRAM`, 32 × 256 | `SourceRegionHistogramsNativeStage` |
| same | `BP_OUTPUT_START_REGION_HISTOGRAM`, 32 × 256 | same |
| same | `BP_COMPUTE_REGION_HISTOGRAM`, 32 × 256 | same |
| same | `BP_OUTPUT_REGION_HISTOGRAM`, 32 × 256 | same |
| `computeStartAndActiveHistogramKernel` | `BP_WRITEOUT_ACTIVE_HISTOGRAM`, 32 × 256 | `SourceActiveRegionListsNativeStage` |
| same | `BP_COMPUTE_ACTIVE_HISTOGRAM`, 64 × 512 | same |
| same | `BP_OUTPUT_ACTIVE_HISTOGRAM`, 64 × 256 | same |

All commands use a single caller-provided `std::vector<compute::BatchCommand>`
and **one GPU submission** in the fixture. The two source histogram totals
are written into the original GPU-resident `PxgBroadPhaseDesc` and consumed
by later GPU kernels without readback. The source-created gate skips all
ten operations. Independent virtual CUDA allocations and a *physically
shared root resource* exercise the resource-binding alternatives. Neither
path requires a separate simulator, frame graph or CPU/GPU transfer
between source kernels.

### Acceptance gates for the connected run

1. Check exact 2 fills + 1 producer + 4 histogram scans + 3 active/list
   commands, with the source's 32/64 block counts and shader thread geometry.
2. Independently derive region ranges, source start prefix, block histograms,
   two GPU descriptor totals, 64 dense region masks, per-region exclusive
   active/start histograms and output handle lists. The reference iterates
   logical X-sorted *start* projections, not GPU binary-search work indices.
3. Verify every byte of all allocation roots, not only populated results;
   untouched padding/canaries must remain unchanged. Check mixed created/
   removed source populations and the zero-created gate.
4. Verify physically shared CUDA buddy-heap suballocations without mixed
   SRV/UAV resource state in D3D12; use RAW UAV views and track read/write
   intent separately.
5. Test optimized/wave-capable selection and portable fallback. CI pass is
   correctness evidence for the variants actually *executed*, not proof of
   physical wave32/wave64 performance or occupancy.
6. Fail closed on bad addresses, insufficient source capacities or an
   unsupported 64×512 device dispatch before queueing any partial sequence.
   The fixture's readback/wait comes **after** the entire source batch.

## 2. Host-execution audit: what is implemented vs not

| Original PhysX behavior | Evidence currently available | Remaining acceptance |
| --- | --- | --- |
| Native GPU allocations and interior `CUdeviceptr` mapping | Address registry and per-kernel native views; multi-root and shared-root GPU fixtures | Original raw-vs-128-byte-aligned ownership, exceptional buddy blocks, growth/retire and deferred frees in real allocator |
| Per-source-kernel shader identities/descriptor fields | Native broadphase regions and four-kernel new-bounds chain; GPU scalar writes on original descriptor | Resolve the remaining native SAP stages and descriptor layouts without repacking to arena fixtures |
| Source conditional gates and history rotations | `SapHostFrameRoles` / `BroadPhaseCompletionGate` isolated source-contract tests | Drive them from the **actual** `PxgCudaBroadPhaseSap::update` and fetch paths, including idle/forced update and all swaps |
| CUDA stream-ordered dispatch, D32 clear, HtoD/DtoD | Native batch, fill, copy and retirement fixtures | True source `mStream` adapter, nonblocking HtoD/DtoH, events, multi-stream waits and queue completion mapping |
| `PxgAABBManager::postBroadPhase` task publishing | Both continuation and inline/null-continuation branch tested | Production PhysX task continuation wiring, abort/error propagation and exactly-once fetch/report release |
| Found/lost pairs, aggregates and reports | Many source-derived standalone broadphase kernel oracles; source-native four-kernel new-bounds sequence | Aggregate CUDA kernels, complete source-order handling for created/removed/updated/idle frames, report overflow and multi-frame scene parity |
| `PxScene::simulate/fetchResults` with GPU broadphase | Source trace, existing CPU scene tests | **Not connected**; production GPU broadphase must run through original PhysX scene and update/fetch continuation graph |
| Narrowphase, solver, articulations, particle/deformable GPU features | Upstream source map and CPU PhysX baseline | **Not ported as live native GPU PhysX subsystems**; each has its own source algorithms and host/stream dependencies |

**Audit conclusion:** A successful ten-operation test closes a valuable
*internal broadphase GPU-dataflow* gap. It does **not** close any of the
missing live-scene obligations in the bottom half of this table.

## 3. Specific risks and priorities after the connected run

### P1 — correctness/architecture dependencies before GPU scene acceptance

- **Missing live ownership and task integration.** The per-stage `append`
  APIs deliberately borrow source data and append to a caller's batch.
  Production work must occur under PhysX's existing `PxgCudaBroadPhaseSap`
  and `PxgAABBManager`/scene task ownership, rather than making this fixture
  the runtime scheduler. Preserve original update gates, buffer swaps,
  publication and continuation reference counts.
- **Original descriptor remains pointer-rich.** The native HLSL kernels
  bind the resolved allocations explicitly. This can preserve semantics
  without making embedded CUDA pointer values dereferenceable in HLSL,
  but every source kernel requires a correct resource manifest; nested
  descriptors and cross-allocation pointer traversal need separate audits.
- **Allocator and copy hazards.** Virtual-address registry tests do not
  replace GPU allocator integration. Validate raw allocation/free identity,
  128-byte aligned device pointers, live overlap after growth, deferred
  retirement, and copy/dependency edges between source heap versions.
- **Output path and lost pairs.** The generated region lists are consumed
  by later new-pair and incremental pair/report kernels. The coherent GPU
  broadphase gate still requires existing found/lost report publication,
  overflow, abort, duplicate/aggregate semantics and historical rotations.
- **Device range and capacity.** Source views have explicit native storage
  range limits and shader-relative 32-bit indexing. Although D32 fills
  can be split at descriptor limits, other kernels' large arrays must
  either prove a sufficiently bounded source allocation or get a
  source-faithful segmented/indirect lowering. Do not silently reject
  valid real-world scenes while describing the GPU path as complete.

### P2 — performance and validation

- **No physical GPU throughput claim.** CI Vulkan implementations and
  D3D12 WARP are functional conformance targets. Benchmark real AMD/Intel
  devices before comparing performance with CUDA PhysX, especially
  dispatch/descriptor updates, coarse barriers, 64-region dense arrays,
  inherited sorting passes and 32/64-lane wave paths.
- **Over-synchronization risk.** A single sequential native command batch
  is semantically appropriate for this ten-operation original stream
  segment; it does not justify global per-kernel host waits or
  unconditionally serializing unrelated PhysX GPU streams.
- **Source fills may merit backend-native lowerings.** The existing D32
  shader is portable and correct in its tested envelope. Benchmark
  `vkCmdFillBuffer` and suitable D3D12 native alternatives using source
  interior-pointer, shared-heap and ordering requirements before optimizing.

### Explicit non-findings

Passing CI does **not** certify hardware wave32/wave64 speed, GPU-memory
scalability, full typed-device-pointer ABI, correct end-to-end frame
transitions, or scene-level parity. Conversely, none of these absent
measurements is evidence that the translated ten source operations are
incorrect. Distinguish confirmed bugs from uncompleted integration work.

## 4. Implementation estimate (scope-labeled)

There is no meaningful deterministic line-count percentage: upstream CUDA
translation effort, pointer graphs, and host integration have very different
complexities. The following is an engineering judgment, not a completion
measurement or schedule:

- Portable compute infrastructure: substantially implemented and exercised
  on both APIs, but production stream/resource scheduling needs work.
- Standalone PhysX broadphase algorithm coverage: **substantial**, with
  many source-derived tests and multiple contiguous native source chains.
  Do not confuse arena prototypes with fully adopted native bindings.
- **GPU broadphase through a real PhysX scene (G3): not yet achieved.**
  G2 source allocator/task bridging is incomplete despite successful
  intermediate GPU fixtures.
- Complete portable GPU PhysX feature parity: **roughly 10–20%** based
  on current breadth and remaining live-scene, narrowphase, PGS/TGS solver,
  articulation and particle/deformable requirements. The range is a
  consciously rough planning indicator; changing the promised feature
  tier (broadphase-only, rigid bodies, complete parity) changes it.

## 5. Next source-faithful integration gates

1. Merge the ten-op test only after both backend CI jobs and code review
   pass. If it fails, fix actual shader/dataflow causes rather than
   changing the independent reference to match erroneous GPU output.
2. Connect the region chain to the upstream new-bounds/source-native
   pair-generation chain, preserving `mBPDescBuf` contents and original
   `PxgCudaBroadPhaseSap::update` condition gates.
3. Finish the source `gpuDMAUp` / buffer allocator ownership, descriptor
   refresh, asynchronous stream/event/fetch and lifetime boundaries.
4. Bring the **actual GPU broadphase-only PhysX scene** online with CPU
   dynamics, using original task continuations and found/lost pair
   publication. Exercise multiple frames and actual created/removed/
   updated/idle state transitions.
5. Audit performance on representative physical Vulkan/D3D12 devices.
   Then use the same source-to-host/algorithm method to tackle the
   narrowphase and solver without inventing a parallel simulation engine.
