// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"
#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {
// Source BP_INCREMENTAL_SAP: 256x3x256, after the two comparison
// histogram passes. All 41 resources are ORIGINAL PxgBroadPhaseDesc
// allocations, including the two 32-bit report indices in its device
// descriptor. This stage never owns, copies or publishes report storage.
struct SourceSapIncrementalPairsNativeLayout {
    std::uint32_t box_capacity=0;
    std::uint32_t endpoint_slots=0;
    std::uint32_t ordered_handle_slots=0;
    std::uint32_t report_capacity=0;
};
class SourceSapIncrementalPairsNativeStage final {
public:
    SourceSapIncrementalPairsNativeStage(
        compute::Context& context,compute::ShaderBinary shader);
    ~SourceSapIncrementalPairsNativeStage();
    SourceSapIncrementalPairsNativeStage(const SourceSapIncrementalPairsNativeStage&)=delete;
    SourceSapIncrementalPairsNativeStage& operator=(const SourceSapIncrementalPairsNativeStage&)=delete;
    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                CudaDeviceAddressMap::Address source_device_descriptor,
                const CudaDeviceAddressMap& addresses,
                SourceSapIncrementalPairsNativeLayout layout) const;
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=4u;
    std::uint64_t max_view_=0u;
};
} // namespace fyzz::physx_gpu
