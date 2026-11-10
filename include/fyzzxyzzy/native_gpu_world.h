// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/compute.h"
#include "fyzzxyzzy/world.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>

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
    BoxTransformBounds,
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

// GuPCMContactBoxBox and sparse GuGJK/GuEPA run in the source Sc
// contact-manager stream. These are separate from the SAP shader bundle.
enum class NativeGpuPcmStage : std::size_t {
    Candidates,EpaCompact,GjkContact,EpaContact,FaceContacts,GjkReduce,GjkPublish,EpaArgsReset,GjkPrimitive,ContactJournal,DeviceIslands,Count
};
using NativeGpuPcmShaders = std::array<
    compute::ShaderBinary,static_cast<std::size_t>(NativeGpuPcmStage::Count)>;
NativeGpuPcmShaders load_native_gpu_pcm_shaders(
    compute::Backend backend,const std::filesystem::path& manifest);

// Load the original source SAP shader stages from a build-generated manifest.
// All compiled stages are required in source dispatch order. The manifest
// and binaries can be deployed together, with paths relative to the manifest.
NativeGpuSapShaders load_native_gpu_sap_shaders(
    compute::Backend backend,const std::filesystem::path& manifest);

// Compiled pipeline stages share one GPU Context, source heap and queue.
struct NativeGpuPipelineShaders {
    NativeGpuSapShaders sap{};
    std::optional<NativeGpuAggregateShaders> aggregates;
    std::optional<NativeGpuPcmShaders> pcm;
    std::optional<compute::ShaderBinary> dy;
    std::optional<compute::ShaderBinary> convex_dy;
};

World create_native_gpu_world(
    compute::Backend backend,NativeGpuPipelineShaders stages,
    WorldOptions options={ExecutionMode::NativePortableGpu});

// Construct a World with one native Context, original PhysX GPU buddy heap,
// AABB/SAP owners, stream and source Sc/PCM/Dy scene. No reference PhysX
// scene or CPU broadphase is selected. The World owns all these resources.
World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders shaders,
    WorldOptions options={ExecutionMode::NativePortableGpu});

World create_native_gpu_world(
    compute::Backend backend,const std::filesystem::path& manifest,
    WorldOptions options={ExecutionMode::NativePortableGpu});

// Explicit source GPU box PCM. Sc owns the real contact managers and
// chooses the existing CPU solver fetch until GPU constraint prep lands.
World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders sap,
    NativeGpuPcmShaders pcm,
    WorldOptions options={ExecutionMode::NativePortableGpu});

// The original source task partitions run Dy 1D, Gu contact and correlated
// friction PGS on the GPU. Native Scene remains the integration owner.
World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders sap,
    compute::ShaderBinary dy_distance,
    WorldOptions options={ExecutionMode::NativePortableGpu});

// Explicit coupled source SAP, GPU PCM and GPU Dy. The GPU narrowphase
// and solver share one Context, stream and original Sc contact managers.
World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders sap,
    NativeGpuPcmShaders pcm,compute::ShaderBinary dy_distance,
    WorldOptions options={ExecutionMode::NativePortableGpu});

// Convex shapes use a dedicated GPU Dy integration kernel; primitive-only
// scenes continue executing the original five-buffer solver.
World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders sap,
    NativeGpuPcmShaders pcm,compute::ShaderBinary dy_distance,
    compute::ShaderBinary convex_dy_distance,
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
