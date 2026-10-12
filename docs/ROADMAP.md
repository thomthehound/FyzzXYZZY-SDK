# FizzX-SDK: initial engineering roadmap

Status: **proposed working plan**, 2026-10-12. Update milestones when evidence changes. This is not a promise to preserve prototype designs forever.

## Mission and constraints

FizzX is an **independent, PhysX-derived, general-purpose physics SDK** with hardware-neutral GPU acceleration. Godot 4.7.2 is the first consumer and a required early integration test, **not** FizzX's runtime or architectural center. Other native applications and game engines must be able to embed the same SDK without any Godot dependencies.

- License FizzX-authored code under Apache-2.0; preserve upstream and third-party copyright, license, and notice requirements. Do not adopt the Wild Ox Godot reference implementation by default.
- Prefer working upstream PhysX algorithms, data, API concepts, and tests where they remain sound. Replace CUDA-dependent contracts or other architecture when they obstruct portability.
- CPU operation is a supported path, not a CUDA fallback masquerading as vendor neutrality.
- Vulkan is the first non-CUDA GPU runtime; Direct3D 12 is the second. AMD is the first validation target, but portability includes Intel and NVIDIA hardware.
- HLSL + DXC is the low-risk default **candidate**, not a permanent decision. Compare Slang on the same real GPU workloads before choosing.
- Public SDK APIs must not expose Godot types, CUDA types, Vulkan/D3D12 resource objects, or shader compiler details.
- Functional simulation and observed plausibility matter more than bit-for-bit correspondence with CUDA. Test stability, conservation behavior where appropriate, collisions, contacts, and performance independently.

## Repository and dependency posture

- Keep `thomthehound/PhysX` as the upstream reference/tracking fork.
- Import a clearly identified PhysX SDK 5.11 baseline into FizzX (only required components; do not take on Blast, Flow, Omniverse, or the third-party Godot module merely because they exist).
- Preserve source provenance and SPDX notices. A root Apache-2.0 license does not erase differing third-party requirements.
- Favor reproducible normal CMake builds and an SDK that downstream consumers can package without requiring CUDA, NVIDIA GPU binaries, or Godot.
- Keep reference/prototype tools separable from distributed runtime dependencies.
- Organize public headers, runtime backends, kernels, tests, samples, and consumers as separate targets; their final directory names can be settled in M0.

## Proposed layers

```text
  Standalone C++ app      Godot 4.7 adapter      Other consumers
             \                 |                /
              +-------- FizzX public API -------+
                         Scene / bodies / constraints
                         contacts / particles / deformables
                                  |
                         Physics implementations
                                  |
                        Compute execution contract
                                  |
                  CPU       Vulkan        D3D12
```

The compute contract should cover: device discovery/capabilities, buffer ownership and views, buffer+offset addressing, upload/download, descriptors or equivalent bindings, kernel pipeline metadata, dispatch, dependency scheduling, barriers, fences, diagnostics, and optional host-resource interoperation. It must **not** emulate arbitrary CUDA device pointers with unportable integer casts.

Maintain **one authoritative FizzX simulation world** for rigid/deformable/contact interactions; do not let the Godot adapter create a separate cloth physics universe or silently delegate reciprocal collisions to an unrelated engine.

GPU/renderer sharing is an optional *later* optimization with explicit lifetime/synchronization contracts. Initial Godot tests may use staged CPU readback to draw results, so long as that limitation is measured and not mistaken for production performance.

## Milestones and gates

### M0 — Source and CPU foundation
- Import the needed PhysX 5.11 SDK source and preserve all legal notices; retain `thomthehound/PhysX` as tracking/reference.
- Make CPU build and basic scene tests pass on Windows and Linux using an ordinary CMake workflow.
- Establish an independent library target, tiny host application, smoke tests, and a Godot adapter target that consumes the same library (it may initially be CPU only).
- Record feature/build baseline and the initial GPU dependency map; do not restructure unrelated parts of PhysX as a prerequisite.

**Gate:** CPU FizzX scene works independently; a Godot 4.7.2 sample can create and advance a FizzX-managed scene without invoking CUDA or using the reference Godot fork.

### M1 — Portable GPU execution vertical slice
- Implement the minimum **native Vulkan** device, buffer, command, and synchronization runtime, independent of Godot's RenderingDevice.
- Define an explicit CPU/GPU buffer data contract and use portable pipeline reflection/build metadata.
- Run representative kernels (simple stream operation plus a difficult one featuring subgroup operations, atomics, or structured data) through HLSL/DXC and Slang. Compare correctness, translation complexity, portable bindings, toolchain friction and measured performance.
- Use vendor-neutral fallback implementations for features absent on some devices.

**Gate:** the same FizzX workload runs on AMD Vulkan *outside Godot* and via Godot's FizzX adapter, with checked output and no CUDA installation or NVIDIA driver dependency. Record and resolve the HLSL-vs-Slang decision.

### M2 — First complete GPU physics pathway
- Port a small coherent sequence, not a disconnected collection of compiled kernels: e.g. bounds update → broadphase candidates → contact handling for limited primitives → integration/constraint response.
- Reuse or translate scan, sort, compaction and dispatch utilities as required by the pathway; leave nonessential kernel families unported.
- Support a fixed-step interactive test scene with rigid bodies and visibly correct collision/response.
- Document GPU/CPU state transfer semantics; measure synchronization and readback overhead.

**Gate:** nontrivial, interactive *GPU-computed* rigid-body behavior on AMD Vulkan through FizzX in both consumers; collisions are not secretly computed by Godot/Jolt.

### M3 — First reciprocal deformable/rigid pathway
- Add enough existing PhysX particle/PBD/FEM/contact functionality to support an interacting deformable body or garment against FizzX rigid bodies.
- Establish a common collision/contact ownership model, consistent simulation stepping and GPU buffer update path.
- Benchmark first with staging transfers; then investigate zero-copy/host-interop and asynchronous frame scheduling if the measurements justify it.

**Gate:** a repeatable Godot scene with a deformable object reacting to moving rigid geometry, plus the equivalent standalone test. Verify contact stability, penetration and plausible behavior.

### M4 — SDK and Godot integration maturity
- Expand the adapter to ordinary Godot physics nodes and queries through a selectable PhysicsServer3D implementation where supported; investigate GDExtension first, use an engine module only if evidence requires it.
- Provide first-class FizzX extension interfaces for functionality beyond the stock PhysicsServer3D contract.
- Add packaging, consumer samples, diagnostics and clear CPU-only operation. Do not require users to fork Godot to embed basic FizzX features if avoidable.

**Gate:** a Godot project loads/selects FizzX and uses everyday 3D physics behavior; advanced simulation uses the same FizzX world. Native C++ sample still requires no Godot.

### M5 — D3D12 and cross-vendor validation
- Implement the D3D12 compute runtime using the shared kernel sources (DXIL), with backend capability negotiation.
- Exercise representative workloads on AMD and at least one independent additional vendor/device when accessible.
- Expand Windows/Linux regression and packaging matrix.

**Gate:** one physics scene produces comparable acceptable physical behavior through Vulkan and D3D12 on supported devices, with verified diagnostics when a capability is unavailable.

### M6 — Broader PhysX GPU feature coverage and optimization
- Port remaining narrowphase, articulated dynamics, solver modes, particles, soft bodies, cloth, and other PhysX GPU capabilities by priority and demonstrated reuse.
- Benchmark realistic contact density, small and large scenes, dispatch overhead, GPU occupancy, transfer cost, and host/editor latency.
- Optimize implementations per backend where needed; no requirement for identical instruction streams or bit-exact numerical behavior.
- Evaluate Blast and Flow separately if/when they add value; neither is currently an SDK dependency.

**Gate:** publish a feature coverage matrix and performance comparisons against CPU PhysX and, where available, upstream CUDA PhysX.

## Testing discipline

Each GPU milestone uses a paired standalone-and-Godot scenario from the **same FizzX implementation**. Minimum recorded outcomes:

- Correct scene progression: contacts, constraints, integration and state transitions.
- Physics quality: penetration, jitter, energy/momentum drift where applicable, solver stability and numerical robustness.
- CPU/GPU work accounting: GPU simulation time, transfer time, synchronization and frame latency separately.
- Portable assumptions: subgroup sizes, atomic support, buffer alignment/layout, memory lifetime and compiler backend differences.
- Error reporting: unsupported device or backend is explicit, never silent fallback to an unrelated physics engine.

No hash/checksum ceremony is needed for normal verification; use ordinary builds, tests, source history, and behavioral validation.

## Immediate work package

1. Import and build the minimal PhysX 5.11 SDK CPU baseline in FizzX; preserve provenance and legal files.
2. Establish separate `fizzx-core`, `fizzx-vulkan` (initially stub), `fizzx-godot` and standalone test targets, without gratuitously replacing working PhysX types.
3. Produce a *focused* CUDA-to-portable dependency inventory of actual launch sites, `CUdeviceptr` representation, streams/events, allocation, device-side pointer graphs, kernel registries, GPU shared data structures, subgroup assumptions, and build/binary dependencies.
4. Deliver the M0 CPU scene in a standalone harness and Godot. Treat the Wild Ox module as a reference for testing/documentation, not imported source.
5. Start M1 with the smallest functioning Vulkan physics-related compute slice; run the same representative HLSL and Slang kernels through it.

## Open decisions (resolve with experiments, not ideology)

- HLSL/DXC versus Slang as the primary shader authoring/compiler path — decide in M1.
- Preserve PhysX public API as-is versus a thin FizzX API/compatibility layer — keep minimal changes through M0, document any unavoidable breaks.
- Native Vulkan allocation only, Godot RenderingDevice-hosted execution, or explicit resource sharing for fast rendering interop — prototype after the native backend exists and choose based on measured costs.
- GDExtension versus compiled Godot module for full `PhysicsServer3D` coverage — choose on implementation and packaging evidence.
- Additional platform/vendor requirements and shader feature tiers — expand the test matrix as hardware and demand justify it.
