# D3D12 WARP GJK Execution Diagnostics

## Scope

PR #328 extends the GPU PCM continuation with bounded GJK/EPA for convex rigid shapes. The D3D12/WARP failure is reproducible in the independent Windows GPU suite, using the sparse PCM fixture with one pending GJK pair. The preceding candidate-generation and sparse-worklist stages finish normally. Failure occurs when the GJK dispatch executes or is prepared by WARP.

The shader is `src/physx_gpu/kernels/physx_source_pcm_box_gjk_native.hlsl`; its simplex implementation is in `physx_source_pcm_box_gjk_native.hlsli`. The D3D12 shader target is `cs_6_0`, with 32 threads per group and one pair per lane. The production solver has a 32-iteration limit.

## Reproduction

Configure the independent D3D12 GPU test suite with `FYZZXYZZY_ENABLE_PHYSX_SOURCE=OFF`, `FYZZXYZZY_BUILD_D3D12=ON`, `FYZZXYZZY_BUILD_SOURCE_GPU_TESTS=ON`, and a Windows SDK `dxc.exe`. Then run:

```powershell
cmake --build build-native --config Release --target fyzzxyzzy_source_pcm_box_candidates_d3d12 --parallel 4
ctest --test-dir build-native -C Release -R '^fyzzxyzzy_source_pcm_gjk_two_step_probe_d3d12$' --output-on-failure
ctest --test-dir build-native -C Release -R '^fyzzxyzzy_source_pcm_gjk_solver_probe_d3d12$' --output-on-failure
```

The CI workflow runs the two-iteration probe first, then invokes the full solver directly with a 45-second process limit and prints the native Windows exit status. The remaining release, debug-layer, GPU-validation and contact-parity tests remain required.

## Verified observations

| Variant | Observation | CI run |
| --- | --- | --- |
| Binding, support-mapping and single-iteration probes | Finish normally under WARP | 42030200834 |
| Two-step unrolled entry probe | Reaches the second iteration; the ordinary loop stalls | 42030200834 |
| Full 32-iteration, forcibly unrolled DXIL | Compiles; GJK dispatch crashes with Windows exception `0xC0000005` (access violation) | 42036002336 |
| Loop-carried simplex moved to workgroup storage | Does not resolve the second-entry stall | 42028815564 |
| Ordinary DXIL loop, with or without `[loop]` | Hangs on full GJK dispatch | 42036877651 |
| Four-step unrolled blocks inside an outer loop | Still hits the full-solver timeout | 42037900932 |
| Non-inlined simplex/vector helpers | Rejected during DXIL validation (unsupported vector instructions in callable function) | 42038691440 |
| Two complete unrolled simplex iterations | Finish normally; stage-result check passes | 42040300349 |
| Structured single-exit GJK loop | Two-step probe passes; full loop still stalls | 42040998184 |
| Full solver compiled at `-O1` | Two-step probe passes; full loop still stalls | 42041626901 |

The observed `0xC0000005` is **not** the Windows stack-overflow exception `0xC00000FD`. Stack exhaustion or corruption inside WARP is not ruled out, but there is no evidence establishing it as the cause. CTest's generic `SEGFAULT` label alone does not identify an exception code.

The passing two-step test demonstrates execution of two actual simplex reductions, not merely entry into iteration two. It does not establish correct convergence over arbitrary convex pairs or justify reducing the 32-iteration limit.

## Current investigation

The current GJK implementation retains only Minkowski support coordinates and feature IDs in its loop-carried simplex; full shape-space witnesses are reconstructed for contact generation and EPA admission. This reduces the persistent state and avoids recomputing witness coordinates in every iteration. **The full D3D12/WARP runtime and source contact parity must pass before this change is considered verified.**

The remaining distinction to resolve is whether WARP fails because of DXIL loop-carried state, invalid compiler lowering, or the generated shader's execution/resource requirements. Preserve the source geometry, exact witness semantics, bounded iteration count, and direct GPU execution while diagnosing that distinction.

## Acceptance gate

1. The full 32-iteration GJK solver completes within the focused WARP preflight and returns its expected result marker.
2. The unmodified independent D3D12 release, debug-layer, GPU-validation and contact-parity tests pass.
3. Independent Vulkan tests continue to pass; the same convex fixture must not regress in separation, overlap or EPA admission.
4. A performance check confirms that the production shader does not require excessive register, stack, scratch or shader-code resources.

Until these gates pass, PR #328 is **not ready to merge**.
