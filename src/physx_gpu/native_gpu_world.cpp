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
}};

} // namespace

NativeGpuSapShaders load_native_gpu_sap_shaders(
    compute::Backend backend,const std::filesystem::path& manifest_path) {
    if(backend!=compute::Backend::Vulkan &&
       backend!=compute::Backend::D3D12)
        throw std::invalid_argument("Unknown native GPU SAP backend");

    std::ifstream manifest(manifest_path);
    if(!manifest)
        throw std::runtime_error("Native GPU SAP shader manifest is missing");
    std::string line;
    const auto expected_header=backend==compute::Backend::Vulkan
        ? "FYZZXYZZY_SOURCE_SAP_1 SPIRV"
        : "FYZZXYZZY_SOURCE_SAP_1 DXIL";
    if(!std::getline(manifest,line) || line!=expected_header)
        throw std::invalid_argument(
            "Native GPU SAP shader manifest backend/version mismatch");

    NativeGpuSapShaders shaders{};
    const auto format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv : compute::ShaderFormat::Dxil;
    for(std::size_t i=0;i<source_stage_names.size();++i) {
        if(!std::getline(manifest,line))
            throw std::invalid_argument("Native GPU SAP shader manifest is incomplete");
        std::istringstream tokens(line);
        std::string stage,filename,extra;
        if(!(tokens>>stage>>filename) || tokens>>extra ||
           stage!=source_stage_names[i] ||
           filename=="." || filename==".." ||
           std::filesystem::path(filename).filename()!=filename)
            throw std::invalid_argument(
                "Native GPU SAP shader manifest stage or path mismatch");
        const auto asset=manifest_path.parent_path()/filename;
        std::ifstream file(asset,std::ios::binary);
        if(!file)
            throw std::runtime_error(
                "Missing native source SAP compiled shader: "+asset.string());
        shaders[i].format=format;
        shaders[i].bytes.assign(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>());
        if(shaders[i].bytes.empty() || file.bad())
            throw std::runtime_error(
                "Empty or unreadable native source SAP shader: "+asset.string());
    }
    if(std::getline(manifest,line))
        throw std::invalid_argument("Native GPU SAP shader manifest has extra stages");
    return shaders;
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
} // namespace fyzz
