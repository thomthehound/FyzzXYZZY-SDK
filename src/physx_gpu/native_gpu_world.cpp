// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/native_gpu_world.h"

#include "native/scene.h"
#include "physx_gpu/source_sc_owned_gpu_broadphase_native.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace fyzz {

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
