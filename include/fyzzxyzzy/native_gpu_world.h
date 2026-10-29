// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/compute.h"
#include "fyzzxyzzy/world.h"

#include <array>
#include <cstddef>
#include <filesystem>

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

// Original aggregate pre-BP and post-SAP kernels. These are compiled
// alongside source SAP and borrowed by the same native GPU World.
enum class NativeGpuAggregateStage : std::size_t {
    Bounds,MarkBounds,MemberHistory,Projection,SelfCollision,
    PairAdmit,PairPublish,PersistentCollisions,
    PairRetireHistogram,PairRetireMark,PairRetirePublish,CopyReports,
    Count
};
using NativeGpuAggregateShaders = std::array<
    compute::ShaderBinary,static_cast<std::size_t>(NativeGpuAggregateStage::Count)>;

NativeGpuAggregateShaders load_native_gpu_aggregate_shaders(
    compute::Backend backend,const std::filesystem::path& manifest);

// Load the original source SAP shader stages from a build-generated manifest.
// All 34 compiled stages are required in source dispatch order. The manifest
// and binaries can be deployed together, with paths relative to the manifest.
NativeGpuSapShaders load_native_gpu_sap_shaders(
    compute::Backend backend,const std::filesystem::path& manifest);

// Construct a World with one native Context, original PhysX GPU buddy heap,
// AABB/SAP owners, stream and source Sc/PCM/Dy scene. No reference PhysX
// scene or CPU broadphase is selected. The World owns all these resources.
World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders shaders,
    WorldOptions options={ExecutionMode::NativePortableGpu});

World create_native_gpu_world(
    compute::Backend backend,const std::filesystem::path& manifest,
    WorldOptions options={ExecutionMode::NativePortableGpu});

// Explicit source aggregate shader contract: all original aggregate
// pipelines are created under the same native Context and heap owner.
// The existing single-volume World entrypoints remain unchanged.
World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders sap,
    NativeGpuAggregateShaders aggregates,
    WorldOptions options={ExecutionMode::NativePortableGpu});

World create_native_gpu_world(
    compute::Backend backend,const std::filesystem::path& sap_manifest,
    const std::filesystem::path& aggregate_manifest,
    WorldOptions options={ExecutionMode::NativePortableGpu});

} // namespace fyzz
