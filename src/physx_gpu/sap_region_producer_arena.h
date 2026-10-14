// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fyzzxyzzy/compute.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Source fields read/written by broadphase.cu::createRegionsKernel,
// BP_CREATE_REGIONS. All pointers are byte offsets into caller-owned
// persistent GPU memory; regionRange is two uint4 values per handle.
struct alignas(16) SapRegionProducerArenaDesc {
    std::uint32_t current_x_handles_offset = 0;
    std::uint32_t sap_positions_offset[3] = {};
    std::uint32_t total_end_histogram_offset[3] = {};
    std::uint32_t region_range_offset = 0;
    std::uint32_t start_region_accum_offset = 0;
    std::uint32_t region_accum_offset = 0;
    std::uint32_t current_boxes = 0;
    std::uint32_t created_handles = 0;
    std::uint32_t box_capacity = 0;
    std::uint32_t endpoint_capacity = 0;
    std::uint32_t reserved[2] = {};
};
static_assert(sizeof(SapRegionProducerArenaDesc) == 64,
              "PhysX region producer arena ABI changed");
static_assert(offsetof(SapRegionProducerArenaDesc, region_range_offset) == 28,
              "PhysX region output offset changed");

// Reproduces only BP_CREATE_REGIONS (32x256). Source host invokes it
// conditionally when created handles are present and *after* clearing
// active/start region tables (which this kernel neither touches nor owns).
// The following four histogram scans remain a separate dependency.
class SapRegionProducerArenaStage final {
public:
    SapRegionProducerArenaStage(compute::Context& context,
                                compute::ShaderBinary shader);
    ~SapRegionProducerArenaStage();
    SapRegionProducerArenaStage(const SapRegionProducerArenaStage&) = delete;
    SapRegionProducerArenaStage& operator=(const SapRegionProducerArenaStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset,
                std::uint32_t boxes, std::uint32_t created_handles) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};
} // namespace fyzz::physx_gpu
