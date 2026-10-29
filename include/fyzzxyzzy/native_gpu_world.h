// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/compute.h"
#include "fyzzxyzzy/world.h"

#include <array>
#include <cstddef>

namespace fyzz {

// Compiled source SAP stages in update/fetch dispatch order. Each stage
// contains SPIR-V for Vulkan or DXIL for D3D12, never source HLSL.
enum class NativeGpuSapStage : std::size_t {
    TranslateAabbs,
    CreatedPairs,
    RadixInitialize,
    RadixCount,
    RadixScatter,
    RadixGather,
    SapBoxIndices,
    EndpointHistogramFirst,
    EndpointHistogramSecond,
    FillD32,
    RegionCreate,
    RegionHistogramFirst,
    RegionHistogramWave,
    RegionHistogramSecond,
    ActiveWriteout,
    ActiveWave,
    ActivePortable,
    ActiveOutput,
    OverlapWork,
    OverlapScanFirst,
    OverlapScanSecond,
    NewBoundsPairs,
    ReportFirst,
    ReportSecond,
    ReportCopy,
    ReportWaveFirst,
    ReportWaveSecond,
    RemovedMark,
    RemovedProjections,
    UpdatedBounds,
    IncrementalCountFirst,
    IncrementalCountSecond,
    IncrementalPairs,
    ClearNewFlags,
    Count
};
using NativeGpuSapShaders = std::array<
    compute::ShaderBinary,static_cast<std::size_t>(NativeGpuSapStage::Count)>;

// Construct a World with one native Context, original PhysX GPU buddy heap,
// AABB/SAP owners, stream and source Sc/PCM/Dy scene. No reference PhysX
// scene or CPU broadphase is selected. The World owns all these resources.
World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders shaders,
    WorldOptions options={ExecutionMode::NativePortableGpu});

} // namespace fyzz
