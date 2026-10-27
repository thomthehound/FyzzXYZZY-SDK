// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"
#include <cstdint>
#include <vector>
namespace fyzz::physx_gpu {

// Two source launches, BP_COMPUTE_ENDPT_HISTOGRAM then
// BP_OUTPUT_ENDPT_HISTOGRAM, each 32x3x256.
// The 48 original source allocations (two frame histories per axis,
// two total-end histories per axis) are never repacked or copied.
struct SourceSapEndpointHistogramsNativeLayout {
    std::uint32_t endpoint_slots=0;
    std::uint32_t ordered_handle_slots=0;
};

class SourceSapEndpointHistogramsNativeStage final {
public:
    SourceSapEndpointHistogramsNativeStage(compute::Context& context,
                                          compute::ShaderBinary first,
                                          compute::ShaderBinary second);
    ~SourceSapEndpointHistogramsNativeStage();
    SourceSapEndpointHistogramsNativeStage(const SourceSapEndpointHistogramsNativeStage&)=delete;
    SourceSapEndpointHistogramsNativeStage& operator=(const SourceSapEndpointHistogramsNativeStage&)=delete;
    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceSapEndpointHistogramsNativeLayout layout,
                bool is_incremental) const;
private:
    compute::Context& context_;
    compute::PipelineId first_=0,second_=0;
    std::size_t alignment_=4u;
    std::uint64_t max_view_=0u;
};
} // namespace fyzz::physx_gpu
