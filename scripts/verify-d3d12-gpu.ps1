param(
    [string]$BuildDir = 'build-gpu'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($env:FYZZXYZZY_D3D12_FORCE_WARP -eq '1') {
    throw 'Hardware verification cannot run with FYZZXYZZY_D3D12_FORCE_WARP=1'
}
$env:FYZZXYZZY_D3D12_REQUIRE_HARDWARE = '1'

$compiler = Get-Command dxc.exe -ErrorAction SilentlyContinue
$dxc = if ($compiler) { $compiler.Source } else { $null }
if (-not $dxc) {
    $sdkBin = Join-Path ([Environment]::GetFolderPath('ProgramFilesX86')) 'Windows Kits\10\bin'
    $dxc = @(Get-ChildItem -Path $sdkBin -Filter dxc.exe -File -Recurse |
        Where-Object { $_.FullName -match '[\\/]x64[\\/]' } |
        Sort-Object FullName -Descending | Select-Object -First 1 |
        ForEach-Object { $_.FullName }) | Select-Object -First 1
}
if (-not $dxc) { throw 'A Windows x64 DXC compiler is required' }

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $configure = @(
        '-S', '.', '-B', $BuildDir, '-A', 'x64',
        '-DBUILD_TESTING=ON',
        '-DFYZZXYZZY_ENABLE_PHYSX_SOURCE=OFF',
        '-DFYZZXYZZY_BUILD_D3D12=ON',
        '-DFYZZXYZZY_BUILD_SOURCE_GPU_TESTS=ON',
        '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
        "-DFYZZXYZZY_DXC=$dxc"
    )
    & cmake @configure
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    cmake --build $BuildDir --config Release --parallel 4
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    ctest --test-dir $BuildDir -C Release -R '_d3d12' -LE '^d3d12-extended$' --output-on-failure --stop-on-failure --no-tests=error
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    ctest --test-dir $BuildDir -C Release -R '_d3d12' -L '^d3d12-extended$' --output-on-failure --stop-on-failure --no-tests=error
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
