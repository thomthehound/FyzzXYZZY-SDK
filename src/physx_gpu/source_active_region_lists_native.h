// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"

namespace fyzz::physx_gpu {

// Caller-owned capacities from PxgCudaBroadPhaseSap::gpuDMAUp.
// Projection slots include temporarily retained removed projections.
// No new allocation, descriptor, or frame scheduler is owned here.
struct SourceActiveRegionListsNativeLayout {
    CudaDeviceAddressMap::Address descriptor = 0;
    std::uint32_t box_capacity = 0;
    std::uint32_t projection_slots = 0;
    std::uint32_t region_table_words = 0; // 64 * retained projection slots
};

// Original three launches of
// PxgCudaBroadPhaseSap::computeStartAndActiveHistogramKernel():
// BP_WRITEOUT_ACTIVE_HISTOGRAM (32x256),
// BP_COMPUTE_ACTIVE_HISTOGRAM (64x512),
// BP_OUTPUT_ACTIVE_HISTOGRAM (64x256).
// Receives the same PxgBroadPhaseDesc and CUDA address map used by the
// preceding seven native region operations; append to caller's GPU batch.
class SourceActiveRegionListsNativeStage final {
public:
    SourceActiveRegionListsNativeStage(
        compute::Context& context, compute::ShaderBinary writeout,
        compute::ShaderBinary compute_wave,
        compute::ShaderBinary compute_portable,
        compute::ShaderBinary output, bool force_portable = false);
    ~SourceActiveRegionListsNativeStage();
    SourceActiveRegionListsNativeStage(const SourceActiveRegionListsNativeStage&) = delete;
    SourceActiveRegionListsNativeStage& operator=(const SourceActiveRegionListsNativeStage&) = delete;

    bool using_subgroup_scan() const noexcept { return using_wave_; }
    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceActiveRegionListsNativeLayout layout) const;
private:
    compute::Context& context_;
    compute::PipelineId pipelines_[3] = {};
    bool using_wave_ = false;
};
} // namespace fyzz::physx_gpu
