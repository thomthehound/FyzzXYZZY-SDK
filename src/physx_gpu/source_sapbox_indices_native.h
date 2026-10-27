// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {
// The original allocations from PxgCudaBroadPhaseSap::initializeSapBoxKernel.
// This method is called once for the retained previous handles after
// removal and once after the created/removed projection reorder.
struct SourceSapBoxIndicesNativeLayout {
    std::uint32_t box_capacity=0;
    std::uint32_t endpoint_slots=0;
};
class SourceSapBoxIndicesNativeStage final {
public:
    SourceSapBoxIndicesNativeStage(compute::Context& context,
                                   compute::ShaderBinary shader);
    ~SourceSapBoxIndicesNativeStage();
    SourceSapBoxIndicesNativeStage(const SourceSapBoxIndicesNativeStage&)=delete;
    SourceSapBoxIndicesNativeStage& operator=(const SourceSapBoxIndicesNativeStage&)=delete;
    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceSapBoxIndicesNativeLayout layout,
                std::uint32_t num_handles,bool is_new) const;
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t view_alignment_=16u;
    std::uint64_t max_view_bytes_=0u;
    std::uint32_t max_groups_=0u;
};
} // namespace fyzz::physx_gpu
