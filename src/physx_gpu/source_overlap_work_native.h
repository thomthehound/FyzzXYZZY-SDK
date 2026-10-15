// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"

#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Source allocation capacities and current frame population, supplied by
// PxgCudaBroadPhaseSap. The CUDA descriptor's GPU-written counters are never
// replaced by host snapshots when creating native views.
struct SourceOverlapWorkLayout {
    CudaDeviceAddressMap::Address descriptor = 0; // resident mBPDescBuf
    std::uint32_t current_boxes = 0;
    std::uint32_t box_capacity = 0;
    std::uint32_t projection_slots = 0; // includes removed projections
    std::uint32_t region_work_capacity = 0; // uint64 entries
    std::uint32_t handle_region_capacity = 0; // uint2 entries
};

// Original BP_WRITEOUT_OVERLAPCHECKS_HISTOGRAM_NEWBOUNDS with source-owned
// allocations bound directly as nine native views. Every view may be a slice
// of any PhysX heap root or a standalone allocation. No temporary arena,
// per-dispatch GPU copy, or extra PhysX frame scheduler.
class SourceOverlapWorkNativeStage final {
public:
    SourceOverlapWorkNativeStage(compute::Context& context,
                                 compute::ShaderBinary shader);
    ~SourceOverlapWorkNativeStage();
    SourceOverlapWorkNativeStage(const SourceOverlapWorkNativeStage&) = delete;
    SourceOverlapWorkNativeStage& operator=(const SourceOverlapWorkNativeStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceOverlapWorkLayout layout) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};
} // namespace fyzz::physx_gpu
