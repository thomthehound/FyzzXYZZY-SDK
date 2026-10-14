<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GPU memory/addressing: decision RFC and benchmark contract

**Status:** evidence-gathering RFC; **NO representation decision has been made**.
**Source baseline:** pinned PhysX fork revision
`6dc53585d7db07346f53d0598a5fd3bfdce47d8d`.
**Goal:** preserve PhysX GPU algorithm, allocation ownership, CPU/GPU task
dependencies and semantic data layouts while making access fast on Vulkan
and D3D12. Any backend-specific lowering must be demonstrated equivalent,
not silently change physics behavior.

See [GPU port program](GPU_PORT_PROGRAM.md) and
[broadphase trace](GPU_BROADPHASE_TRACE.md).

## 1. Verified source constraints (not design preferences)

1. `gpucommon/include/PxgDevicePointer.h` uses an eight-byte CUDA device
   pointer wrapper. `PxgBroadPhaseDesc` has **51 pointer declarations
   occupying 104 pointer slots** when its fixed-size arrays expand (with
   `SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU` enabled). These are not 104
   independent resources; many are views into persistent allocations.
2. `gpubroadphase/src/CUDA/broadphase.cu::markRemovedPairsLaunch` loads
   multiple previous-frame arrays from `PxgBroadPhaseDesc` and performs
   indexed updates. `translateAABBsLaunch`, by contrast, accesses simple
   contiguous arrays through its launch arguments. Access locality and
   pointer-resolution frequency vary enormously between operations.
3. `gpucommon/src/PxgHeapMemoryAllocator.cpp` implements a buddy-style
   device heap (minimum 128-byte block) with larger exceptional allocations,
   mutex-protected bookkeeping, allocation group statistics, an address map
   and deferred deallocations. It explicitly guards against duplicate live
   addresses (OMPE-104415).
4. `gpucommon/src/PxgCudaBuffer.cpp` grows capacity geometrically,
   provides a distinct asynchronous copy-and-grow variant, and retires the
   old allocation *after* queued copy completion. Ordinary grow need not
   preserve contents; the two paths have different contracts.
5. `gpucommon/include/PxgCudaBuffer.h` exposes a 128-byte-aligned device
   view while preserving the original pointer for correct freeing and
   swapping. A portable allocation may use different physical alignment,
   but the caller's required alignment and swap/ownership contract survive.
6. `gpucommon/src/PxgMemoryManager.cpp` distinguishes device,
   pinned-host, and device-mapped host allocators; there are CPU/GPU
   shared-address and completion assumptions to audit.
7. `gpucommon/src/PxgCopyManager.cpp` accumulates a mapped-host array of
   `PxgCopyDesc` entries, launches a GPU balanced-copy kernel, and
   signals a CUDA event. Merely replacing it with synchronous CPU
   `memcpy` would remove the original batching and CPU/GPU overlap.
8. Broadphase reports, radix-sort buffers and solver pools persist across
   frames. The pointer values in descriptors may need to be rebuilt on
   grow, swap or relocation. Absence of immediate deallocation is important
   for work already submitted on another stream.

**Implication:** a representation that merely maps each `T*` to a resource
ID is not enough. We need byte-exact typed access, cheap per-thread
dereferences, nested descriptor traversal, allocation generation/lifetime
rules, alias-safe swaps and the existing task graph's visibility guarantees.

## 2. What the APIs guarantee (and do not)

**Vulkan:** buffer device address / SPIR-V `PhysicalStorageBuffer` permits
actual shader dereference of GPU addresses when supported and enabled. It
requires correct `VkBuffer` usage, allocation flags, alignment declarations
and lifetimes. This is an attractive *Vulkan-specific* fast path.

**D3D12:** `ID3D12Resource::GetGPUVirtualAddress()` does return a 64-bit
buffer GPU VA, but **ordinary HLSL cannot arbitrarily cast this number and
dereference it as a CUDA `T*`**. GPU VAs are used by certain root/indirect
interfaces; resources in HLSL are generally accessed through SRV/UAV/CBV
bindings. Shader Model 6.6 permits `ResourceDescriptorHeap[index]` after
checking device support and root flags. The DXC Vulkan-SPIR-V compatibility
of identical SM6.6 heap syntax is an **open compiler-validation question**,
not a foundation we may assume.

**Consequence:** "8 bytes" in a shared descriptor can preserve *packed ABI
size* without preserving native CUDA pointer semantics. It may carry a
logical reference that is lowered differently on the two GPUs. A single
behavioral HLSL source can use carefully isolated access helpers or
backend-specific representation lowering. It need not enforce literally
identical machine instructions.

References:
- [Vulkan buffer device address](https://docs.vulkan.org/guide/latest/buffer_device_address.html)
- [Physical storage buffer alignment](https://docs.vulkan.org/guide/latest/buffer_device_address_alignment.html)
- [D3D12 SM6.6 dynamic resources](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html)
- [D3D12 GPU virtual address](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12resource-getgpuvirtualaddress)
- [D3D12 residency and heap allocation](https://learn.microsoft.com/en-us/windows/win32/direct3d12/residency)
- [Vulkan descriptor buffer proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_descriptor_buffer.html)

## 3. Candidates to prototype rather than prematurely select

| Candidate | Representation/access | Likely strength | Failure mode / required proof |
| --- | --- | --- | --- |
| A. Scoped arena-relative reference | Known arena at dispatch, `uint32` byte offset for references local to it; one bound `ByteAddressBuffer`/RW buffer or a small arena table | Few bindings, inexpensive arithmetic, compact references, potential to reuse PhysX heap allocation policy | A scene or one allocation may exceed a 4 GiB address window; not all pointers remain inside one arena; reallocation/mapping still requires lifetime discipline |
| B. Segmented logical pointer | Packed `{arena-or-view index, byte offset}` in 64 bits, decoded through a bounded descriptor array or per-dispatch table | Supports multiple pools; preserves pointer-slot size, explicit null sentinel and arena ownership | Per-dereference dispatch/descriptor selection, dynamic nonuniform indexing, descriptor limits; offset range cannot cross a segment |
| C. Descriptor-per-allocation | Each pointer resolves to descriptor index + offset; D3D12 SM6.6 heap indexing, Vulkan descriptor indexing | Flexible resource identity, natural allocation isolation | 100+ pointer slots is only the beginning; many fine-grained allocations/descriptors, divergent nonuniform indexing, descriptor update cost, shader-version feature limits |
| D. Backend-native lowering | Vulkan BDA physical pointer path; D3D12 arena or descriptor path, shared semantic IR and tests | Potential top throughput per API without a lowest-common-denominator bottleneck | Two physical ABIs and compiler transformations to validate; extra implementation burden and migration complexity |

Candidate D is **not** necessarily a standalone choice: it is the mechanism
by which A/B can have specialized fast paths. Do not abandon same-source
physics semantics just because address lowering differs. Conversely, do
not require Vulkan to emulate a D3D resource heap if BDA is substantially
faster, unless we explicitly accept that cost.

**Working hypothesis, not selection:** A + B can form an efficient logical
model: local `uint32` offsets for known-arena hot paths, and a sparse
64-bit cross-arena reference only when needed. A D3D backend can resolve
references to `ByteAddressBuffer`/RW views; Vulkan may use BDA after
capability checks. The model must prove support for typed and atomic
operations before acceptance.

Precise encoding is **not decided**. Reserve a null representation, avoid
ambiguous arena-zero/address-zero cases, specify byte units and alignment,
and handle segments and offsets outside 32-bit ranges. CPU-only handle
generation must not be mistaken for GPU lifetime safety: deferred reuse
is governed by submitted work completion. Do not cache references across
a physical move without explicit repack/versioning.

The first [GPU timestamp and host-resolved-offset comparison harness](GPU_MEMORY_BENCHMARK.md)
tests a deliberately narrow source-derived access-path hypothesis. It is
not a selection among A/B/C/D or a native AMD performance result.

## 4. Two independent costs: addressing and memory management

**Access-path cost** is paid by GPU threads:
- global reads/writes, pointer chaining, cache misses, descriptor divergence,
  32-bit versus 64-bit arithmetic, register pressure and occupancy;
- typed/nested layouts and atomics, including 64-bit operations if used;
- branch and lookup overhead when a shader crosses arenas.

**Management cost** is paid by host and GPU orchestration:
- creating/releasing buffers and heap pages, descriptor and root bindings,
  staging, extra copies, waits, barriers and submissions;
- growth and compaction, reconstruction of packed GPU descriptors,
  per-frame uploads and publication latency;
- residency and memory fragmentation across persistent, transient,
  mapped/pinned, device-local and exceptional allocations.

Avoid optimizing one dimension by ignoring the other. An extra dependent
global load inside a solver kernel can be more expensive than several CPU
instructions once per dispatch. Conversely, a fast GPU load path is no
victory if every kernel forces a heap allocation or CPU/GPU fence.

### Current bring-up executors are **not** performance baselines

`src/compute_vulkan.cpp` currently uses per-buffer allocation, one compute
queue, and coarse barriers between operations. Its host transfer code calls
`wait_all()` and creates/destroys staging buffers for device-local
transfers. `src/compute_d3d12.cpp` currently creates committed D3D12
resources per logical buffer, staging resources per upload/download,
an allocator/command list per submission, descriptor heap per dispatch batch,
and `wait_all()` before buffer destruction. Its `HostVisible` option
currently maps to staging through device-local memory.

These choices are acceptable for an **initial correctness proof**, but are
not adequate evidence of GPU memory architecture efficiency. Do not
benchmark candidate pointer encodings *through* this host path and
mistake submission/allocator overhead for shader access cost.

Plan a reusable command/descriptor/staging allocator first or measure the
GPU stage separately with timestamps on a warmed, prepared command stream.

## 5. Required implementation experiment (next coding PR)

### 5.1 Real PhysX layouts as fixtures

Start with:
- `PxgBroadPhaseDesc`: pointer tables, nested arrays, old/new bounds
  and 2×3 handles across frame swaps;
- `PxgRadixSortDesc`: six pointer fields, input/output aliases and
  rank ordering;
- a representative portion of `PxgSolverCoreDesc`: rigid-body,
  constraint and writeback pools, not a fabricated `Vec3` array;
- `PxgCopyDesc`: batched device copies and completion.

Produce *separate* host-side source layouts and packed GPU-descriptor
layouts where needed, with compile-time `sizeof`/`alignof`/`offsetof`
checks and a generated or explicitly verified field map. Do not upload
host-native raw pointers into GPU data and claim it is portable.

Identify per field: type, access direction, allocator, producer, consumer,
lifetime, alignment, aliasing relationship and whether its address escapes
into another allocation. Preserve little-endian scalar layouts and
handle nontrivial nested structures deliberately.

### 5.2 Correctness probes

- Compile and dispatch the same PhysX-source-derived accessor kernels on
  Vulkan and D3D12, not two unrelated demos.
- Sequential, random and divergent indexed accesses; nested descriptor
  lookup; aliasing and null; alignment 4/8/16/128/256 bytes.
- Persistent state and old/new swaps; reallocate-and-copy;
  generational recycling only *after* completion; deferred free on an
  outstanding submission; error/abort without publishing partial results.
- Sparse and densely packed heaps, >4 GiB **logical aggregate** address
  space, and an explicit oversize-allocation policy. Do not allocate
  4 GiB on hardware just to validate address arithmetic.
- Buffer access race and bounds tests under Vulkan validation and D3D12
  debug/GPU-based validation when available.
- CPU reference reproduces source field semantics; GPU results compare
  identically for integer descriptors and within declared tolerances only
  where floating point necessitates them.

### 5.3 Performance probes

Run each candidate with common **input data and computation**:
- `translateAABBsLaunch`-like contiguous streams (measure bandwidth),
- `markRemovedPairsLaunch`-like scattered updates (gather/scatter),
- radix-sort descriptor swapping and repeated passes (many stages),
- a solver-like pointer-rich indirect read/write path derived from
  `PxgSolverCoreDesc`, once its real access pattern is traced,
- copy batches with per-frame changing descriptor fields.

Report:
1. GPU duration with timestamp queries (warm cache and cold/cache-thrashing),
   including separate memory bandwidth and useful operations/second.
2. GPU occupancy/register usage where tool support allows; descriptor
   divergence and cache utilization where profilers expose them.
3. CPU command encoding, descriptor updates, allocation, staging and queue
   submission times, independently of GPU kernel duration.
4. Peak committed/resident bytes, internal fragmentation and number of
   descriptors/resources; reallocations and copied bytes per frame.
5. Time to publish PhysX-visible results, and what CPU/GPU work overlaps.

Test 0/1/16/513/10k/100k+ elements, resource counts from small to many,
cold startup and warmed steady state; steady-state must distinguish
no-change frames from creation/removal storms. Use repeats, median, p95
and variance. No absolute performance claims until real measurements exist.

**Test matrix:** Vulkan on AMD hardware, D3D12 on AMD hardware, Linux
software Vulkan for correctness and Windows WARP for correctness only.
Use the same shader work and representative scene-state datasets.
WARP throughput is **not** evidence of hardware efficiency.
Capability checks (BDA, SM6.6, descriptor indexing, memory limits) must
be recorded per device, not assumed from operating system.

### 5.4 Acceptance criteria (conditional, not invented targets)

A candidate advances only when it satisfies **all** semantic and lifetime
tests on Vulkan and D3D12, and measurements show it is acceptable on
target hardware. Compare against the following *relative* baselines:

- CPU/GPU reference algorithm correctness (non-negotiable).
- Directly bound resource with no pointer decode (shader lower bound).
- Current executor CPU overhead **and** an amortized command-buffer baseline.
- Competing candidate encodings on the same input and hardware.

First optimize the expensive path; do not impose a universal 5% or 10%
budget without seeing the source workload. A feature that is accessed once
per scene tick and a lookup inside every solver iteration have different
economic constraints. Document precisely which path loses and why.

## 6. Allocation/synchronization requirements to preserve

1. Allocation identity vs view address: retain allocator-owned handle and
   aligned subrange separately. Swapping views cannot corrupt free maps.
2. Persistent arenas separate from transients and staging rings; reuse
   allocations at safe completed timeline/fence values, not global GPU idle.
3. Growth: respect preserving and nonpreserving variants; repack only
   descriptor fields affected by a move; avoid full repacks every frame.
4. Small arena/descriptor table updates may be batched and uploaded once
   per source-defined stage. Avoid per-kernel CPU uploads unless PhysX
   changes the data between those kernels.
5. Host-pinned/mapped allocations do not imply CPU/device cache coherence.
   Explicit flush/invalidate and synchronization rules must be mapped
   per backend. Original CUDA mapped completion flags should become
   GPU completion tokens with PhysX task-continuation release, not polling
   in a spin loop on an arbitrary host pointer.
6. Use the PhysX graph to place barriers; don't insert global all-stage
   barriers between every shader dispatch in production.
7. On OOM/device-lost, report and abort through the original PhysX
   error/task contract; don't silently fall back to fabricated simulation.
8. Shared D3D/Vulkan logical buffer ownership belongs to one Context/device;
   underlying GPU memory is not automatically shared between APIs.
9. Reserve support for direct GPU API and renderer interop as distinct
   future requirements: do not promise zero-copy with every engine/device.

## 7. Future NPU / MLIR-AIE compatibility: a useful design constraint

**Longer-term option, not a current delivery commitment:** execute selected
source-derived PhysX work units on an AMD Ryzen AI NPU, with MLIR-AIE/IRON
as a possible compiler and runtime path. This is **not** a proposal to
recompile arbitrary Vulkan SPIR-V/D3D12 DXIL or pointer-heavy HLSL
automatically onto the NPU, nor to replace PhysX's scene scheduler.

Unlike a GPU, the AIE array is built around local scratchpads, stream/DMA
routing, placement, and producer/consumer dataflow. IRON exposes Workers,
ObjectFifos, DMA and runtime sequences; on npu2 a compute tile has around
64 KiB of local data memory shared by stack, statics, and FIFO buffers.
A globally dereferenceable GPU virtual pointer is not a portable NPU
programming primitive.

**Constrain the logical interface, not the fastest physical lowering:**

- Describe each work unit's data as typed views: allocation identity,
  byte offset/size, shape/stride, access mode, lifetime, producer,
  consumer and required precision. Buffer generations and publication
  events remain part of the scene-to-device execution dependency graph.
- Permit GPU-specialized realizations: a known-arena hot path can use
  32-bit offsets; Vulkan may use buffer device addresses; D3D12 may
  use bound SRV/UAV views. An NPU lowering can instead pack/stream a
  *bounded tile* from the same semantic view. Do **not** force a costly
  "universal NPU pointer" into every GPU solver loop.
- Avoid dependence on globally visible mutable pointer graphs *within*
  a work unit when source semantics allow input staging, resolved views
  and local scratchpad access. Preserve true cross-work-unit aliases and
  ordering; do not indiscriminately copy mutable state.
- Expose coarse-enough work-unit boundaries to amortize NPU command
  configuration, transfers, repacking and synchronization. Prefer
  persistent NPU-side input/state and pipelining when the original
  PhysX dependencies permit it. A per-contact CPU→NPU→CPU round trip
  is unlikely to be worthwhile.
- Candidate NPU work belongs to **measured cohesive computational
  islands** (batched constraint/contact transforms, reductions, certain
  articulated-body substeps), not automatically to irregular global
  broadphase sorting or every solver synchronization point. Partition
  using observed source access patterns, not "NPU supports BF16".

**Arithmetic policy:** BF16 has much less significand precision than FP32,
though it retains a similarly wide exponent range. It must **not** be
applied indiscriminately to integer handles, float-bit AABB encoding,
sorting/order keys, exact masks, or collision/topology predicates.
For eligible math, test BF16 inputs with suitable wider accumulation
and/or refinement, retaining FP32 or integer representation for sensitive
state. The required output standard is stable, useful PhysX simulation,
not identical FP32 intermediate bits. Error amplification across iterative
contacts, small separation distances, stacks and long horizons is a
performance/quality experiment, not an assumption.

**Hardware matters:** AIE2, AIE2P and newer variants differ in BF16
multiply/accumulate and block-float capabilities. In particular,
efficient BFP16 versus native BF16 paths are not interchangeable across
generations. Feature-test the target and measure the emitted code;
do not assume the label "Ryzen AI NPU" implies native peak BF16.

**Selection gate for a future NPU slice:** compare total
frame-critical-path time, including packaging/layout conversion,
GPU/NPU/host transfers, NPU compute, synchronization and PhysX result
publication. Include concurrent GPU render/physics traffic and limited
local-memory tiling. An NPU implementation belongs in the product only
when it wins on a realistic workload at acceptable physical behavior.

Upstream background:
[IRON programming model](https://xilinx.github.io/mlir-aie/1.4.4/programming_guide/section-1/),
[core memory](https://xilinx.github.io/mlir-aie/1.4.4/programming_guide/core_data_memory/),
[ObjectFIFO lowering](https://xilinx.github.io/mlir-aie/1.4.4/ObjectFifoLowering/),
[AIE capability differences](https://github.com/Xilinx/mlir-aie/discussions/3390).

## 8. Immediate work order

1. Fix pre-existing stale "D3D12 compile-only" docs following PR #15.
2. Build measurement infrastructure: GPU timestamps, CPU counters, reused
   command/staging allocations, device capability report; preserve current
   conformance tests.
3. Implement two real-source-derived accessor kernels and representative
   `PxgBroadPhaseDesc`/radix descriptor host packing (both APIs).
4. Run A/B/C (and native Vulkan BDA) comparisons on the same fixtures.
5. Decide physical addressing only after correct runtime results and
   actual AMD measurements. Record measured trade-offs in this RFC.
6. Refactor the executors behind a single runtime-selectable `Context`
   without introducing an unrelated PhysX scene scheduler.
7. Integrate a slice of source-authentic broadphase host orchestration;
   immediately test lifetime, swap and task-continuation paths.

**Non-goals for this RFC:** selecting a winner from intuition, replacing
the PhysX heap policy with a general-purpose game-engine allocator,
duplicating the solver, making WARP performance predictions, or claiming
a complete portable GPU broadphase.
