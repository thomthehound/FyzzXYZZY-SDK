// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"

namespace fyzz::physx_gpu {

// Original source capacities, including temporarily retained removed
// projections, and the original mBPDescBuf CUDA virtual address.
struct SourceRegionHistogramsNativeLayout {
    CudaDeviceAddressMap::Address descriptor = 0;
    std::uint32_t projection_slots = 0;
};

// Four GPU launches inside PxgCudaBroadPhaseSap::computeRegionHistogramKernel,
// immediately after the two clears and BP_CREATE_REGIONS:
//
//   BP_COMPUTE_START_REGION_HISTOGRAM
//   BP_OUTPUT_START_REGION_HISTOGRAM
//   BP_COMPUTE_REGION_HISTOGRAM
//   BP_OUTPUT_REGION_HISTOGRAM
//
// Each reads or writes ORIGINAL PxgBroadPhaseDesc allocations (no arena
// packing, synthetic frame runner or added submit/wait). Second passes
// publish the GPU scalar totals to their original PxgBroadPhaseDesc fields.
class SourceRegionHistogramsNativeStage final {
public:
    SourceRegionHistogramsNativeStage(
        compute::Context& context,
        compute::ShaderBinary first_portable,
        compute::ShaderBinary first_wave,
        compute::ShaderBinary second,
        bool force_portable = false);
    ~SourceRegionHistogramsNativeStage();
    SourceRegionHistogramsNativeStage(const SourceRegionHistogramsNativeStage&) = delete;
    SourceRegionHistogramsNativeStage& operator=(const SourceRegionHistogramsNativeStage&) = delete;

    bool using_subgroup_scan() const noexcept { return using_wave_; }
    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceRegionHistogramsNativeLayout layout) const;
private:
    compute::Context& context_;
    compute::PipelineId first_ = 0, second_ = 0;
    bool using_wave_ = false;
};
} // namespace fyzz::physx_gpu
