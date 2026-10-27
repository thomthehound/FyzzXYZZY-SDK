// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"
#include <cstdint>
#include <vector>
namespace fyzz::physx_gpu {
// The seven real source arrays, not a packed fixture arena.
// Called only at PxgCudaBroadPhaseSap::clearNewFlagKernel's source boundary.
struct SourceClearNewFlagsNativeLayout {
    std::uint32_t box_capacity=0;
    std::uint32_t endpoint_slots=0;
};
class SourceClearNewFlagsNativeStage final {
public:
    SourceClearNewFlagsNativeStage(compute::Context& context,
                                   compute::ShaderBinary shader);
    ~SourceClearNewFlagsNativeStage();
    SourceClearNewFlagsNativeStage(const SourceClearNewFlagsNativeStage&)=delete;
    SourceClearNewFlagsNativeStage& operator=(const SourceClearNewFlagsNativeStage&)=delete;
    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceClearNewFlagsNativeLayout layout) const;
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
};
} // namespace fyzz::physx_gpu
