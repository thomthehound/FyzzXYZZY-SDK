# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param(
    [ValidateSet('Both', 'Vulkan', 'D3D12')]
    [string]$Backend = 'Both',
    [string]$VulkanBuild = 'build-profile-vulkan',
    [string]$D3D12Build = 'build-profile-d3d12',
    [string]$OutputDirectory = 'build-hardware-evidence',
    [ValidateRange(1, 50)]
    [int]$Samples = 7,
    [ValidateRange(0, 10)]
    [int]$WarmupRuns = 2,
    [switch]$Scale,
    [switch]$ValidateArtifactsOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

if (-not $IsWindows) {
    throw 'Physical-GPU qualification currently targets Windows PowerShell 7.'
}
if (-not $ValidateArtifactsOnly -and $env:FYZZXYZZY_D3D12_FORCE_WARP -eq '1') {
    throw 'Physical-GPU qualification cannot run with FYZZXYZZY_D3D12_FORCE_WARP=1.'
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
function Resolve-RepositoryPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repo $Path))
}

function Percentile([double[]]$Values, [double]$Fraction) {
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) { throw 'Cannot summarize an empty sample.' }
    $position = ($sorted.Count - 1) * $Fraction
    $low = [int][Math]::Floor($position)
    $high = [int][Math]::Ceiling($position)
    return [double]$sorted[$low] + ($position - $low) *
        ([double]$sorted[$high] - [double]$sorted[$low])
}

function Get-ScenarioMetrics([string[]]$Lines) {
    $metrics = @()
    foreach ($line in $Lines) {
        if (-not $line.StartsWith('FYZZXYZZY_METRIC ')) { continue }
        $fields = [ordered]@{}
        foreach ($token in ($line -split '\s+' | Select-Object -Skip 1)) {
            $pair = $token.Split('=', 2)
            if ($pair.Count -ne 2) { continue }
            if ($pair[1] -match '^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$') {
                $fields[$pair[0]] = [double]::Parse(
                    $pair[1], [Globalization.CultureInfo]::InvariantCulture)
            } else {
                $fields[$pair[0]] = $pair[1]
            }
        }
        $metrics += $fields
    }
    return $metrics
}


function Convert-TraceFields([string]$Line) {
    $fields = [ordered]@{}
    foreach ($token in ($Line -split '\s+' | Select-Object -Skip 1)) {
        $pair = $token.Split('=', 2)
        if ($pair.Count -ne 2) { continue }
        [long]$integer = 0
        [double]$fractional = 0
        if ([long]::TryParse($pair[1], [ref]$integer)) {
            $fields[$pair[0]] = $integer
        } elseif ([double]::TryParse(
            $pair[1], [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$fractional)) {
            $fields[$pair[0]] = $fractional
        } else {
            $fields[$pair[0]] = $pair[1]
        }
    }
    return $fields
}

function Get-Executable([string]$Build, [string]$Name) {
    foreach ($candidate in @(
        (Join-Path $Build "Release/$Name.exe"),
        (Join-Path $Build "$Name.exe")
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "Missing Release executable $Name in $Build. Build the scene test target first."
}

function Get-BackendConfig([string]$Api, [string]$Build) {
    $buildPath = Resolve-RepositoryPath $Build
    $cache = Join-Path $buildPath 'CMakeCache.txt'
    if (-not (Test-Path -LiteralPath $cache -PathType Leaf)) {
        throw "Missing CMakeCache.txt: $cache"
    }
    $cacheText = Get-Content -LiteralPath $cache -Raw
    if ($cacheText -match '(?m)^CMAKE_BUILD_TYPE:STRING=(?!Release\r?$)\S+') {
        throw "The $Api build is not a Release configuration."
    }
    if ($cacheText -notmatch '(?m)^FYZZXYZZY_PROFILE_GPU:BOOL=ON\r?$') {
        throw "The $Api build must enable -DFYZZXYZZY_PROFILE_GPU=ON."
    }
    $suffix = if ($Api -eq 'Vulkan') { 'spv' } else { 'dxil' }
    $manifestType = if ($Api -eq 'Vulkan') { 'spirv' } else { 'dxil' }
    $exeName = if ($Api -eq 'Vulkan') {
        'fyzzxyzzy_source_sc_gpu_pcm_scene'
    } else {
        'fyzzxyzzy_source_sc_gpu_pcm_scene_d3d12'
    }
    $kernels = Join-Path $buildPath 'kernels'
    $shaderArguments = @(
        (Join-Path $kernels "fyzzxyzzy_native_sap_$manifestType.manifest"),
        (Join-Path $kernels "fyzzxyzzy_native_pcm_$suffix.manifest"),
        (Join-Path $kernels "physx_source_dy_distance_pgs_native.$suffix"),
        (Join-Path $kernels "physx_source_dy_convex_native.$suffix")
    )
    $aggregate = Join-Path $kernels "fyzzxyzzy_native_aggregate_$manifestType.manifest"
    foreach ($file in @($shaderArguments) + @($aggregate)) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            throw "Missing compiled shader artifact: $file"
        }
    }
    foreach ($manifest in @($shaderArguments[0], $shaderArguments[1], $aggregate)) {
        foreach ($line in (Get-Content -LiteralPath $manifest | Select-Object -Skip 1)) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            $fields = $line -split '\s+'
            if ($fields.Count -ne 2) {
                throw "Malformed shader manifest entry: $manifest -- $line"
            }
            $shader = Join-Path $kernels $fields[1]
            if (-not (Test-Path -LiteralPath $shader -PathType Leaf)) {
                throw "Missing manifest shader artifact: $shader"
            }
        }
    }
    return [pscustomobject]@{
        Api = $Api
        Build = $buildPath
        Executable = (Get-Executable $buildPath $exeName)
        Arguments = $shaderArguments
        Aggregate = $aggregate
    }
}

$configs = @()
if ($Backend -in @('Both', 'Vulkan')) {
    $configs += Get-BackendConfig 'Vulkan' $VulkanBuild
}
if ($Backend -in @('Both', 'D3D12')) {
    $configs += Get-BackendConfig 'D3D12' $D3D12Build
}
if ($ValidateArtifactsOnly) {
    foreach ($config in $configs) {
        Write-Host "Validated $($config.Api) qualification artifacts: $($config.Build)"
    }
    exit 0
}

$env:FYZZXYZZY_D3D12_REQUIRE_HARDWARE = '1'
$env:FYZZXYZZY_VULKAN_REQUIRE_HARDWARE = '1'
$env:FYZZXYZZY_D3D12_FORCE_WARP = '0'

$root = Resolve-RepositoryPath $OutputDirectory
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$out = Join-Path $root $stamp
New-Item -ItemType Directory -Path $out -Force | Out-Null
$commit = (& git -C $repo rev-parse HEAD 2>$null)
$dirty = @(& git -C $repo status --porcelain 2>$null)
$adapters = @(Get-CimInstance Win32_VideoController |
    Select-Object Name, DriverVersion, PNPDeviceID)
$metadata = [ordered]@{
    created_utc = (Get-Date).ToUniversalTime().ToString('o')
    repository_commit = $commit
    working_tree_dirty = $dirty.Count -ne 0
    host = [Environment]::MachineName
    os = [Environment]::OSVersion.VersionString
    powershell = $PSVersionTable.PSVersion.ToString()
    graphics_controllers = $adapters
    backend = $Backend
    samples = $Samples
    warmup_runs = $WarmupRuns
    scale = [bool]$Scale
    instrumentation = 'FYZZXYZZY_PROFILE_GPU'
    builds = @($configs | ForEach-Object { [ordered]@{
        backend = $_.Api
        directory = $_.Build
        executable = $_.Executable
    } })
}
$metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (
    Join-Path $out 'metadata.json') -Encoding utf8

$records = [System.Collections.Generic.List[object]]::new()
$evidence = Join-Path $out 'runs.jsonl'
$traces = Join-Path $out 'trace.jsonl'
$sequence = 0

function Invoke-Scenario($Config, [string]$Case, [string]$Phase, [int]$Index) {
    $script:sequence++
    $logName = '{0}-{1}-{2}-{3:D3}.log' -f $Config.Api, $Case, $Phase, $script:sequence
    $logPath = Join-Path $out $logName
    $argv = @($Config.Arguments) + @($Case)
    if ($Case -eq 'aggregate_coupled') { $argv += $Config.Aggregate }
    if ($Case.StartsWith('hardware_scene_') -and $Phase -eq 'sample' -and
        ($Index % 2) -eq 1) {
        $argv += 'gpu-first'
    }
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $lines = @(& $Config.Executable @argv 2>&1 | ForEach-Object { "$_" })
    $status = $LASTEXITCODE
    $clock.Stop()
    $lines | Set-Content -LiteralPath $logPath -Encoding utf8
    $adapterLines = @($lines | Where-Object { $_.StartsWith('FYZZXYZZY_ADAPTER ') })
    $pass = @($lines | Where-Object { $_ -eq "Native Sc GPU PCM scene $Case PASS" })
    $selectedAdapter = if ($adapterLines.Count -gt 0) { $adapterLines[0] } else { $null }
    $record = [pscustomobject][ordered]@{
        backend = $Config.Api
        case = $Case
        phase = $Phase
        run = $Index
        exit_code = $status
        process_wall_ms = $clock.Elapsed.TotalMilliseconds
        adapter = $selectedAdapter
        metrics = @(Get-ScenarioMetrics $lines)
        log = $logName
    }
    $records.Add($record)
    $record | ConvertTo-Json -Depth 10 -Compress |
        Add-Content -LiteralPath $evidence -Encoding utf8
    $traceRecords = [System.Collections.Generic.List[string]]::new()
    foreach ($line in $lines) {
        if (-not $line.StartsWith('FYZZXYZZY_TRACE ')) { continue }
        $item = [ordered]@{
            backend = $Config.Api
            case = $Case
            phase = $Phase
            run = $Index
            log = $logName
            event = (Convert-TraceFields $line)
        }
        $traceRecords.Add(($item | ConvertTo-Json -Depth 8 -Compress))
    }
    if ($traceRecords.Count -gt 0) {
        Add-Content -LiteralPath $traces -Value $traceRecords.ToArray() -Encoding utf8
    }
    if (@($lines | Where-Object {
        $_.StartsWith('FYZZXYZZY_TRACE kind=overflow ')
    }).Count -ne 0) {
        throw "Profiling could not retain the complete trace: $logPath"
    }
    if ($Case.StartsWith('hardware_scene_') -and
        -not (@($lines | Where-Object {
            $_.StartsWith('FYZZXYZZY_TRACE kind=gpu ')
        }).Count -gt 0)) {
        throw "Hardware scene produced no GPU timestamps: $logPath"
    }
    if ($status -ne 0 -or $adapterLines.Count -ne 1 -or $pass.Count -ne 1) {
        throw "Qualification failed: $($Config.Api) $Case ($Phase) -- $logPath"
    }
    Write-Host ('{0,-6} {1,-20} {2,-10} process_wall_ms={3,9:N2}' -f
        $Config.Api, $Case, $Phase, $clock.Elapsed.TotalMilliseconds)
}

$correctness = @(
    'scene', 'device_admission', 'indirect_api',
    'coupled_solver', 'mixed_solver', 'legacy_gu_solver', 'dy_switch',
    'plane_only_pcm', 'single_island', 'distance_solver', 'free_integration',
    'sparse_contacts', 'compound_solver', 'sphere_solver',
    'convex_plane', 'convex_finite', 'aggregate_coupled',
    'wide_islands', 'zero_response_wide', 'dual_hub', 'star_island',
    'hardware_scene_1k'
)
$profiles = @('hardware_scene_1k', 'coupled_solver', 'mixed_solver',
    'single_island', 'wide_islands')
if ($Scale) {
    $correctness += @('hardware_scene_10k', 'star_island_10k', 'wide_islands_10k')
    $profiles += @('hardware_scene_10k', 'star_island_10k')
}
foreach ($config in $configs) {
    foreach ($case in $correctness) {
        Invoke-Scenario $config $case 'correctness' 0
    }
    foreach ($case in $profiles) {
        for ($i = 0; $i -lt $WarmupRuns; $i++) {
            Invoke-Scenario $config $case 'warmup' $i
        }
        for ($i = 0; $i -lt $Samples; $i++) {
            Invoke-Scenario $config $case 'sample' $i
        }
    }
}

$summary = @()
foreach ($config in $configs) {
    foreach ($case in $profiles) {
        $times = @($records |
            Where-Object { $_.backend -eq $config.Api -and
                $_.case -eq $case -and $_.phase -eq 'sample' } |
            ForEach-Object { [double]$_.process_wall_ms })
        $entry = [ordered]@{
            backend = $config.Api
            case = $case
            samples = $times.Count
            process_wall_p50_ms = (Percentile $times 0.50)
            process_wall_p95_ms = (Percentile $times 0.95)
            process_wall_p99_ms = (Percentile $times 0.99)
        }
        if ($case.StartsWith('hardware_scene_')) {
            $selected = @($records | Where-Object {
                $_.backend -eq $config.Api -and
                $_.case -eq $case -and $_.phase -eq 'sample'
            })
            foreach ($executor in @('cpu', 'gpu')) {
                $gpuValue = if ($executor -eq 'gpu') { 1.0 } else { 0.0 }
                $frames = @($selected | ForEach-Object {
                    $_.metrics | Where-Object {
                        $_.suite -eq 'hardware_step' -and
                        $_.gpu_dy -eq $gpuValue
                    } | ForEach-Object { [double]$_.step_ms }
                })
                if ($frames.Count -eq 0) {
                    throw "No $executor whole-scene frame samples for $($config.Api) $case."
                }
                $entry[($executor + '_frame_samples')] = $frames.Count
                $entry[($executor + '_frame_p50_ms')] = (Percentile $frames 0.50)
                $entry[($executor + '_frame_p95_ms')] = (Percentile $frames 0.95)
                $entry[($executor + '_frame_p99_ms')] = (Percentile $frames 0.99)
            }
        }
        $summary += [pscustomobject]$entry
    }
}
$summary | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $out 'summary.json') -Encoding utf8

if (-not (Test-Path -LiteralPath $traces -PathType Leaf)) {
    throw "Profiling produced no trace file: $traces"
}
$pipelineLabels = @{}
$frameModes = @{}
foreach ($line in [System.IO.File]::ReadLines($traces)) {
    $record = ConvertFrom-Json -InputObject $line
    if ($record.phase -ne 'sample') { continue }
    $event = $record.event
    if ($event.kind -eq 'pipeline') {
        $key = "$($record.log)|$($event.context)|$($event.pipeline)"
        $pipelineLabels[$key] = $event.label
    } elseif ($event.kind -eq 'frame' -and $event.success -eq 1) {
        $key = "$($record.log)|$($event.trace_id)"
        $frameModes[$key] = [pscustomobject]@{
            mode = if ($event.gpu_dy -eq 1) { 'gpu-dy' } else { 'cpu-dy' }
            frame = [long]$event.frame
        }
    }
}
$stages = [System.Collections.Generic.Dictionary[string,System.Collections.Generic.List[double]]]::new()
foreach ($line in [System.IO.File]::ReadLines($traces)) {
    $record = ConvertFrom-Json -InputObject $line
    if ($record.phase -ne 'sample') { continue }
    $event = $record.event
    if ($event.kind -notin @('cpu','gpu') -or $event.trace_id -eq 0) {
        continue
    }
    $frameKey = "$($record.log)|$($event.trace_id)"
    if (-not $frameModes.ContainsKey($frameKey)) { continue }
    $frame = $frameModes[$frameKey]
    if ($record.case.StartsWith('hardware_scene_') -and $frame.frame -le 4) {
        continue
    }
    if ($event.kind -eq 'cpu') {
        $stage = $event.stage
    } else {
        $pipelineKey = "$($record.log)|$($event.context)|$($event.pipeline)"
        $stage = if ($event.type -eq 'copy') {
            'gpu.copy'
        } elseif ($pipelineLabels.ContainsKey($pipelineKey)) {
            $pipelineLabels[$pipelineKey]
        } else {
            'gpu.unlabeled'
        }
    }
    $key = "$($record.backend)|$($record.case)|$($frame.mode)|$($event.kind)|$stage"
    if (-not $stages.ContainsKey($key)) {
        $stages[$key] = [System.Collections.Generic.List[double]]::new()
    }
    $stages[$key].Add(([double]$event.duration_ns / 1000000.0))
}
$stageSummary = foreach ($key in ($stages.Keys | Sort-Object)) {
    $parts = $key -split '\|', 5
    $samples = $stages[$key]
    $values = [double[]]$samples.ToArray()
    [pscustomobject][ordered]@{
        backend = $parts[0]
        case = $parts[1]
        executor = $parts[2]
        timeline = $parts[3]
        stage = $parts[4]
        segments = $values.Count
        segment_p50_ms = (Percentile $values 0.50)
        segment_p95_ms = (Percentile $values 0.95)
        segment_p99_ms = (Percentile $values 0.99)
        total_segment_ms = [double](($values | Measure-Object -Sum).Sum)
    }
}
$stageSummary | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $out 'stages.json') -Encoding utf8
Write-Host "Hardware qualification passed. Evidence: $out"
Write-Host 'Process timings include device creation and shader setup; in-scene step metrics are in runs.jsonl.'
