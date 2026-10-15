// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"

#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Original PxgCudaBroadPhaseSap allocation capacities, distinct from the
// GPU-produced regionAccumTotal. The resident source descriptor remains
// authoritative for that counter and the GPU-written overlap total.
struct SourceOverlapPrefixLayout {
    CudaDeviceAddressMap::Address descriptor = 0;
    std::uint32_t current_boxes = 0;
    std::uint32_t overlap_capacity = 0; // regionOverlapType (uint64) entries
};

// Preserve the original BP_COMPUTE / BP_OUTPUT overlap histogram launches
// (two 32x256 kernels) over source-owned native heap roots, with the existing
// subgroup-based 64-bit scan. Does not own allocations or frame orchestration.
class SourceOverlapPrefixNativeStage final {
public:
    SourceOverlapPrefixNativeStage(compute::Context& context,
                                   compute::ShaderBinary compute_wave,
                                   compute::ShaderBinary output_wave);
    ~SourceOverlapPrefixNativeStage();
    SourceOverlapPrefixNativeStage(const SourceOverlapPrefixNativeStage&) = delete;
    SourceOverlapPrefixNativeStage& operator=(const SourceOverlapPrefixNativeStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceOverlapPrefixLayout layout) const;

private:
    compute::Context& context_;
    compute::PipelineId pipelines_[2] = {};
};
} // namespace fyzz::physx_gpu
