// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/native_gpu_world.h"

#include "native/scene.h"
#include "physx_gpu/source_sc_owned_gpu_broadphase_native.h"

#include <array>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace fyzz {
namespace {

constexpr std::array<std::string_view,
    static_cast<std::size_t>(NativeGpuSapStage::Count)> source_stage_names{{
    "TranslateAabbs",
    "CreatedPairs",
    "RadixInitialize",
    "RadixCount",
    "RadixScatter",
    "RadixGather",
    "SapBoxIndices",
    "EndpointHistogramFirst",
    "EndpointHistogramSecond",
    "FillD32",
    "RegionCreate",
    "RegionHistogramFirst",
    "RegionHistogramWave",
    "RegionHistogramSecond",
    "ActiveWriteout",
    "ActiveWave",
    "ActivePortable",
    "ActiveOutput",
    "OverlapWork",
    "OverlapScanFirst",
    "OverlapScanSecond",
    "NewBoundsPairs",
    "ReportFirst",
    "ReportSecond",
    "ReportCopy",
    "ReportWaveFirst",
    "ReportWaveSecond",
    "RemovedMark",
    "RemovedProjections",
    "UpdatedBounds",
    "IncrementalCountFirst",
    "IncrementalCountSecond",
    "IncrementalPairs",
    "ClearNewFlags",
    "BoxTransformBounds",
}};
constexpr std::array<std::string_view,
    static_cast<std::size_t>(NativeGpuPcmStage::Count)> pcm_stage_names{{
    "Candidates","EpaReset","EpaCompact","EpaContact","ContactStream"
}};
constexpr std::array<std::string_view,
    static_cast<std::size_t>(NativeGpuAggregateStage::Count)> aggregate_stage_names{{
    "Bounds","MarkBounds","MemberHistory","Projection","SelfCollision",
    "PairAdmit","PairPublish","PersistentCollisions","PairRetireHistogram",
    "PairRetireMark","PairRetirePublish","CopyReports"
}};

template<std::size_t Count>
std::array<compute::ShaderBinary,Count> load_source_shader_bundle(
    compute::Backend backend,const std::filesystem::path& manifest_path,
    std::string_view source_name,std::string_view source_header,
    const std::array<std::string_view,Count>& names) {
    if(backend!=compute::Backend::Vulkan &&
       backend!=compute::Backend::D3D12)
        throw std::invalid_argument("Unknown native GPU backend");
    const auto prefix="Native GPU "+std::string(source_name);
    std::ifstream manifest(manifest_path);
    if(!manifest)
        throw std::runtime_error(prefix+" shader manifest is missing");
    std::string line;
    const auto expected_header=std::string(source_header)+
        (backend==compute::Backend::Vulkan?" SPIRV":" DXIL");
    if(!std::getline(manifest,line) || line!=expected_header)
        throw std::invalid_argument(prefix+" shader manifest backend/version mismatch");
    const auto format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv : compute::ShaderFormat::Dxil;
    std::array<compute::ShaderBinary,Count> shaders{};
    for(std::size_t i=0;i<Count;++i) {
        if(!std::getline(manifest,line))
            throw std::invalid_argument(prefix+" shader manifest is incomplete");
        std::istringstream tokens(line);
        std::string stage,filename,extra;
        if(!(tokens>>stage>>filename) || tokens>>extra ||
           stage!=names[i] || filename=="." || filename==".." ||
           std::filesystem::path(filename).filename()!=filename)
            throw std::invalid_argument(prefix+" shader manifest stage or path mismatch");
        const auto asset=manifest_path.parent_path()/filename;
        std::ifstream file(asset,std::ios::binary);
        if(!file)
            throw std::runtime_error(
                "Missing native source "+std::string(source_name)+
                " compiled shader: "+asset.string());
        shaders[i].format=format;
        shaders[i].bytes.assign(std::istreambuf_iterator<char>(file),
                                std::istreambuf_iterator<char>());
        if(shaders[i].bytes.empty() || file.bad())
            throw std::runtime_error("Empty or unreadable native source "+
                std::string(source_name)+" shader: "+asset.string());
    }
    if(std::getline(manifest,line))
        throw std::invalid_argument(prefix+" shader manifest has extra stages");
    return shaders;
}

} // namespace

NativeGpuSapShaders load_native_gpu_sap_shaders(
    compute::Backend backend,const std::filesystem::path& manifest) {
    return load_source_shader_bundle(
        backend,manifest,"SAP","FYZZXYZZY_SOURCE_SAP_1",source_stage_names);
}

NativeGpuAggregateShaders load_native_gpu_aggregate_shaders(
    compute::Backend backend,const std::filesystem::path& manifest) {
    return load_source_shader_bundle(
        backend,manifest,"aggregate","FYZZXYZZY_SOURCE_AGGREGATE",
        aggregate_stage_names);
}

NativeGpuPcmShaders load_native_gpu_pcm_shaders(
    compute::Backend backend,const std::filesystem::path& manifest) {
    return load_source_shader_bundle(
        backend,manifest,"PCM","FYZZXYZZY_SOURCE_PCM_1",
        pcm_stage_names);
}

World create_native_gpu_world(
    compute::Backend backend,const std::filesystem::path& manifest,
    WorldOptions options) {
    return create_native_gpu_world(
        backend,load_native_gpu_sap_shaders(backend,manifest),options);
}

World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders shaders,
    WorldOptions options) {
    if(options.execution!=ExecutionMode::NativePortableGpu)
        throw std::invalid_argument(
            "Native GPU World requires NativePortableGpu execution");
    const auto format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv : compute::ShaderFormat::Dxil;
    for(const auto& shader:shaders)
        if(shader.format!=format || shader.bytes.empty())
            throw std::invalid_argument(
                "Native GPU SAP requires the complete compiled backend shader set");

    return World(create_native_gpu_broadphase_backend(
        std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,std::move(shaders)),
        options.contact_notifications,options.enhanced_determinism,
        options.pcm_plane_box,options.pcm_box_box));
}

World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders sap,
    NativeGpuPcmShaders pcm,WorldOptions options) {
    if(options.execution!=ExecutionMode::NativePortableGpu)
        throw std::invalid_argument(
            "Source GPU PCM World requires NativePortableGpu execution");
    const auto format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv : compute::ShaderFormat::Dxil;
    for(const auto& shader:sap)
        if(shader.format!=format || shader.bytes.empty())
            throw std::invalid_argument("Native GPU SAP shader bundle incomplete");
    for(const auto& shader:pcm)
        if(shader.format!=format || shader.bytes.empty())
            throw std::invalid_argument("Native GPU PCM shader bundle incomplete");
    return World(create_native_gpu_broadphase_backend(
        std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,std::move(sap),32768u,32768u,
            std::nullopt,std::move(pcm)),
        options.contact_notifications,options.enhanced_determinism,
        options.pcm_plane_box,options.pcm_box_box));
}

World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders sap,
    compute::ShaderBinary dy_distance,WorldOptions options) {
    if(options.execution!=ExecutionMode::NativePortableGpu)
        throw std::invalid_argument(
            "Source GPU Dy requires NativePortableGpu execution");
    const auto format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    for(const auto& stage:sap)
        if(stage.format!=format || stage.bytes.empty())
            throw std::invalid_argument("Native GPU SAP bundle incomplete");
    if(dy_distance.format!=format || dy_distance.bytes.empty())
        throw std::invalid_argument(
            "Source GPU Dy shader does not match compute backend");
    return World(create_native_gpu_broadphase_backend(
        std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,std::move(sap),32768u,32768u,
            std::nullopt,std::nullopt,std::move(dy_distance)),
        options.contact_notifications,options.enhanced_determinism,
        options.pcm_plane_box,options.pcm_box_box));
}

World create_native_gpu_world(
    compute::Backend backend,const std::filesystem::path& sap_manifest,
    const std::filesystem::path& aggregate_manifest,WorldOptions options) {
    return create_native_gpu_world(
        backend,load_native_gpu_sap_shaders(backend,sap_manifest),
        load_native_gpu_aggregate_shaders(backend,aggregate_manifest),options);
}

World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders sap,
    NativeGpuAggregateShaders aggregates,WorldOptions options) {
    if(options.execution!=ExecutionMode::NativePortableGpu)
        throw std::invalid_argument(
            "Native GPU World requires NativePortableGpu execution");
    const auto format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    for(const auto& shader:sap)
        if(shader.format!=format || shader.bytes.empty())
            throw std::invalid_argument(
                "Native GPU SAP requires the complete compiled backend shader set");
    for(const auto& shader:aggregates)
        if(shader.format!=format || shader.bytes.empty())
            throw std::invalid_argument(
                "Native GPU aggregate requires the complete compiled backend shader set");
    return World(create_native_gpu_broadphase_backend(
        std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,std::move(sap),32768u,32768u,std::move(aggregates)),
        options.contact_notifications,options.enhanced_determinism,
        options.pcm_plane_box,options.pcm_box_box));
}
} // namespace fyzz
