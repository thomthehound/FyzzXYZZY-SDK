# D3D12 GPU verification

## Test scope

The independent D3D12 build validates DXIL compilation, resource binding,
GPU scheduling, sparse contact generation, convex GJK/EPA, PCM contact
publication, and scene-level coupled solving.

Continuous integration executes bounded native tests on D3D12 WARP.
Full PCM contact parity, EPA, GPU-based validation, and scene-level GPU
solver tests use the `d3d12-extended` CTest label. These tests run
on hardware without changing production iteration limits or assertions.

## Windows hardware run

From the repository root:

```powershell
pwsh -File scripts/verify-d3d12-gpu.ps1
```

The command builds and runs every D3D12 test in the independent SDK.
It requires a physical GPU and rejects software adapters. Build output
stays in `build-gpu` unless `-BuildDir` is specified.

To rerun only the extended suite from a completed build:

```powershell
$env:FYZZXYZZY_D3D12_REQUIRE_HARDWARE = '1'
ctest --test-dir build-gpu -C Release -L '^d3d12-extended$' --output-on-failure
```

## Acceptance

Full Release, debug-layer, and GPU-based-validation contact parity
must pass on hardware, together with the cooked-convex and coupled-solver
scene fixtures. Windows, Linux, and independent Vulkan CI must also pass.
