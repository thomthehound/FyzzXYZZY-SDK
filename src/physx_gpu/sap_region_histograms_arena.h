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

// Only the four arrays and two descriptor totals of the original
// PxgBroadPhaseDesc region scans, as offsets in a caller-owned arena.
struct alignas(16) SapRegionHistogramsArenaDesc {
    std::uint32_t start_region_accum_offset = 0;
    std::uint32_t region_accum_offset = 0;
    std::uint32_t block_start_region_accum_offset = 0; // 32 u32
    std::uint32_t block_region_accum_offset = 0;       // 32 u32
    std::uint32_t start_region_total_offset = 0;       // 1 u32
    std::uint32_t region_total_offset = 0;             // 1 u32
    std::uint32_t reserved[2] = {};
};
static_assert(sizeof(SapRegionHistogramsArenaDesc) == 32,
              "PhysX region histogram descriptor changed");

// The four upstream kernels run in this exact sequence:
// BP_COMPUTE_START_REGION_HISTOGRAM
// BP_OUTPUT_START_REGION_HISTOGRAM
// BP_COMPUTE_REGION_HISTOGRAM
// BP_OUTPUT_REGION_HISTOGRAM
// Each original launch is 32 blocks × 256 threads. The host's
// created-handle gate is respected. Source stage_1 produces local
// exclusive prefixes and 32 block totals. Stage_2 adds previous-block
// totals and writes the single 32-block grand total.
//
// This class does not construct the caller's buffer graph, allocate
// persistent state, or stand in for the subsequent active-region kernels.
class SapRegionHistogramsArenaStage final {
public:
    SapRegionHistogramsArenaStage(compute::Context& context,
                                 compute::ShaderBinary start_first,
                                 compute::ShaderBinary start_second,
                                 compute::ShaderBinary region_first,
                                 compute::ShaderBinary region_second,
                                 compute::ShaderBinary start_first_wave,
                                 compute::ShaderBinary region_first_wave,
                                 bool force_portable = false);
    ~SapRegionHistogramsArenaStage();
    SapRegionHistogramsArenaStage(const SapRegionHistogramsArenaStage&) = delete;
    SapRegionHistogramsArenaStage& operator=(const SapRegionHistogramsArenaStage&) = delete;

    bool using_subgroup_scan() const noexcept { return using_wave_; }

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t boxes,
                std::uint32_t created_handles) const;

private:
    compute::Context& context_;
    compute::PipelineId pipelines_[4] = {};
    bool using_wave_ = false;
};

} // namespace fyzz::physx_gpu
