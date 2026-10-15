// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "fyzzxyzzy/compute.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {
// Original BP_WRITEOUT_OVERLAPCHECKS_HISTOGRAM_NEWBOUNDS descriptor subset.
// Every address is a byte offset into the caller-owned arena.
struct alignas(16) SapOverlapWorkArenaDesc {
    std::uint32_t sorted_x_handles_offset = 0;
    std::uint32_t x_sap_positions_offset = 0; // min/end projection per handle
    std::uint32_t region_ranges_offset = 0; // two uint4 per handle
    std::uint32_t region_prefix_offset = 0; // exclusive 2*boxes prefix
    std::uint32_t region_total_offset = 0; // one uint32 count
    std::uint32_t active_histograms_offset = 0;
    std::uint32_t start_histograms_offset = 0;
    std::uint32_t overlap_counts_offset = 0; // uint64 per region work item
    std::uint32_t handle_regions_offset = 0; // two uint32 per work item
    std::uint32_t reserved[3] = {};
};
static_assert(sizeof(SapOverlapWorkArenaDesc) == 48, "PhysX overlap producer descriptor ABI");
static_assert(offsetof(SapOverlapWorkArenaDesc, overlap_counts_offset) == 28,
              "PhysX overlap producer count offset changed");

class SapOverlapWorkArenaStage final {
public:
    SapOverlapWorkArenaStage(compute::Context& context, compute::ShaderBinary shader);
    ~SapOverlapWorkArenaStage();
    SapOverlapWorkArenaStage(const SapOverlapWorkArenaStage&) = delete;
    SapOverlapWorkArenaStage& operator=(const SapOverlapWorkArenaStage&) = delete;
    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t boxes,
                std::uint32_t created_handles) const;
private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};
} // namespace fyzz::physx_gpu
