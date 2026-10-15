// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "fyzzxyzzy/compute.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {
// Subset of the source PxgBroadPhaseDesc read by
// broadphase.cu::generateFoundPairsForNewBoundsRegion. All buffer pointers
// are byte offsets into caller-owned GPU arena; scalar fields are inline.
struct alignas(16) SapNewBoundsPairsArenaDesc {
    std::uint32_t new_integer_bounds_offset = 0;      // 6 u32 per box
    std::uint32_t group_ids_offset = 0;                // u32 per box
    std::uint32_t environment_ids_offset = 0;          // u32 per box, optional
    std::uint32_t sorted_x_handles_offset = 0;        // 2*boxes u32
    std::uint32_t x_sap_positions_offset = 0;         // 2 u32 per box
    std::uint32_t region_ranges_offset = 0;           // two uint4 per box
    std::uint32_t overlap_prefix_offset = 0;          // 64-bit per work item
    std::uint32_t region_work_total_offset = 0;       // single u32
    std::uint32_t overlap_total_offset = 0;           // single 64-bit total
    std::uint32_t handle_regions_offset = 0;          // (handle,region base) u32 pairs
    std::uint32_t active_histograms_offset = 0;       // 64*2*boxes u32
    std::uint32_t start_histograms_offset = 0;        // 64*2*boxes u32
    std::uint32_t ordered_active_handles_offset = 0; // 64*2*boxes u32
    std::uint32_t ordered_start_handles_offset = 0;  // 64*2*boxes u32
    std::uint32_t found_reports_offset = 0;           // two u32 per pair
    std::uint32_t found_counter_offset = 0;           // single u32, host-owned lifecycle
    std::uint32_t max_found_lost_pairs = 0;           // cap, counter NOT capped
    std::uint32_t current_boxes = 0;                  // new current handle count
    std::uint32_t box_capacity = 0;                   // bounds/group allocation count
    std::uint32_t has_environment_ids = 0;            // source USE_ENV_IDS runtime selection
    std::uint32_t reserved[4] = {};
};
static_assert(sizeof(SapNewBoundsPairsArenaDesc) == 96,
              "PhysX new-bounds pair ABI changed");
static_assert(offsetof(SapNewBoundsPairsArenaDesc, found_counter_offset) == 60,
              "PhysX found-pair counter offset changed");

// Source BP_GENERATE_FOUNDPAIR_NEWBOUNDS: 32 workgroups x 256 threads,
// gated on created handles. Must follow the three overlap work/prefix
// kernels; this class does not replace their host ordering or buffers.
class SapNewBoundsPairsArenaStage final {
public:
    SapNewBoundsPairsArenaStage(compute::Context& context,
                               compute::ShaderBinary shader);
    ~SapNewBoundsPairsArenaStage();
    SapNewBoundsPairsArenaStage(const SapNewBoundsPairsArenaStage&) = delete;
    SapNewBoundsPairsArenaStage& operator=(const SapNewBoundsPairsArenaStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset,
                std::uint32_t created_handles) const;
private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};
} // namespace fyzz::physx_gpu
