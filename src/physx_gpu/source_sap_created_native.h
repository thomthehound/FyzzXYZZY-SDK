// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_port/gpubroadphase/include/PxgBroadPhaseDesc.h"

#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Source PxgCudaBroadPhaseSap::markCreatedPairsKernel.
// The original buffers retain all ownership and frame histories.
struct SourceSapCreatedNativeLayout {
    std::uint32_t box_capacity=0;
    std::uint32_t endpoint_slots=0;
};
class SourceSapCreatedNativeStage final {
public:
    SourceSapCreatedNativeStage(compute::Context& context,compute::ShaderBinary shader);
    ~SourceSapCreatedNativeStage();
    SourceSapCreatedNativeStage(const SourceSapCreatedNativeStage&)=delete;
    SourceSapCreatedNativeStage& operator=(const SourceSapCreatedNativeStage&)=delete;
    // After removed projections; before the WOR radix sort. The source
    // appends 2 endpoints per new shape to current projections/handles,
    // writes empty old integer bounds, and pads odd handle populations.
    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceSapCreatedNativeLayout layout) const;
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
};
} // namespace fyzz::physx_gpu
