// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"

#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Caller-supplied extents of the ORIGINAL PxgCudaBroadPhaseSap allocations.
// Neither regionAccumTotal nor overlapChecksTotalRegion is host-authoritative.
struct SourceNewBoundsPairsNativeLayout {
    CudaDeviceAddressMap::Address descriptor = 0; // GPU-resident PxgBroadPhaseDesc
    std::uint32_t current_boxes = 0;
    std::uint32_t box_capacity = 0;
    std::uint32_t projection_slots = 0; // includes removed projections
    std::uint32_t overlap_prefix_capacity = 0; // regionOverlapType entries
    std::uint32_t handle_region_capacity = 0; // PxgHandleRegion entries
};

// Source kernel BP_GENERATE_FOUNDPAIR_NEWBOUNDS, 64 x 256. No new frame
// scheduler, ownership model, descriptor copies, or synthetic pair lifecycle.
class SourceNewBoundsPairsNativeStage final {
public:
    SourceNewBoundsPairsNativeStage(compute::Context& context,
                                   compute::ShaderBinary shader);
    ~SourceNewBoundsPairsNativeStage();
    SourceNewBoundsPairsNativeStage(const SourceNewBoundsPairsNativeStage&) = delete;
    SourceNewBoundsPairsNativeStage& operator=(const SourceNewBoundsPairsNativeStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceNewBoundsPairsNativeLayout layout) const;
private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};
} // namespace fyzz::physx_gpu
