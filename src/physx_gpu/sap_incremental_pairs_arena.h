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

// The original PxgBroadPhaseDesc fields read by performIncrementalSAP.
// Sorted handles/ranks and the two generations of histograms and ordered
// handles remain separate, persistent, caller-owned allocations.
struct SapIncrementalPairsAxisDesc {
    std::uint32_t sorted_handles_offset = 0;
    std::uint32_t projection_ranks_offset = 0;
    std::uint32_t comparison_histogram_offset = 0;
    std::uint32_t current_start_histogram_offset = 0;
    std::uint32_t current_end_histogram_offset = 0;
    std::uint32_t previous_start_histogram_offset = 0;
    std::uint32_t previous_end_histogram_offset = 0;
    std::uint32_t current_start_handles_offset = 0;
    std::uint32_t current_end_handles_offset = 0;
    std::uint32_t previous_start_handles_offset = 0;
    std::uint32_t previous_end_handles_offset = 0;
    std::uint32_t reserved = 0;
};
struct alignas(16) SapIncrementalPairsArenaDesc {
    SapIncrementalPairsAxisDesc axes[3] = {};
    std::uint32_t old_integer_bounds_offset = 0;
    std::uint32_t new_integer_bounds_offset = 0;
    std::uint32_t group_ids_offset = 0;
    std::uint32_t environment_ids_offset = 0;
    std::uint32_t found_report_offset = 0;
    std::uint32_t lost_report_offset = 0;
    std::uint32_t found_lost_counts_offset = 0; // [found, lost]; maintained by host
    std::uint32_t box_capacity = 0;
    std::uint32_t report_capacity = 0; // shared capacity for each report
    std::uint32_t has_environment_ids = 0;
    std::uint32_t reserved[2] = {};
};
static_assert(sizeof(SapIncrementalPairsAxisDesc) == 48, "PhysX pair axis stride changed");
static_assert(sizeof(SapIncrementalPairsArenaDesc) == 192, "PhysX pair descriptor changed");

// Reproduces BP_INCREMENTAL_SAP's 256 x 3 x 256 grid, candidate lookup,
// overlap/filter tests, per-group compaction and shared found/lost counters.
// It does NOT replace the original frame-management or report-publication
// task graph. Report order across workgroups is intentionally unspecified.
class SapIncrementalPairsArenaStage final {
public:
    SapIncrementalPairsArenaStage(compute::Context& context,
                                 compute::ShaderBinary shader,
                                 compute::ShaderBinary wave_shader,
                                 bool force_portable = false);
    ~SapIncrementalPairsArenaStage();
    SapIncrementalPairsArenaStage(const SapIncrementalPairsArenaStage&) = delete;
    SapIncrementalPairsArenaStage& operator=(const SapIncrementalPairsArenaStage&) = delete;

    bool using_wave_reports() const noexcept { return using_wave_; }

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t previous_num_boxes,
                std::uint32_t comparison_totals_offset) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
    bool using_wave_ = false;
};

} // namespace fyzz::physx_gpu
