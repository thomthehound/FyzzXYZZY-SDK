// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_port/gpubroadphase/include/PxgBroadPhaseDesc.h"

#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Source PxgCudaBroadPhaseSap::markUpdatedPairsKernel's normal GPU
// AABB-manager bitmap path. Changed, added and removed maps are borrowed
// from the original manager. No CPU-updated handle list is synthesized.
struct SourceSapUpdatedNativeLayout {
    std::uint32_t box_capacity=0;
    std::uint32_t endpoint_slots=0;
};

class SourceSapUpdatedNativeStage final {
public:
    SourceSapUpdatedNativeStage(compute::Context& context,
                                compute::ShaderBinary shader);
    ~SourceSapUpdatedNativeStage();
    SourceSapUpdatedNativeStage(const SourceSapUpdatedNativeStage&)=delete;
    SourceSapUpdatedNativeStage& operator=(const SourceSapUpdatedNativeStage&)=delete;

    // Call only at source gpuStateChanged=true gate, after removals and
    // AABB translation, before the WR ranked sort. Uses source 32x256.
    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceSapUpdatedNativeLayout layout) const;
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
};
} // namespace fyzz::physx_gpu
