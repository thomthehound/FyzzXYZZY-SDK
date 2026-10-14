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

// Subset of PxgBroadPhaseDesc consumed by
// BP_WRITEOUT_ACTIVE_HISTOGRAM, BP_COMPUTE_ACTIVE_HISTOGRAM,
// BP_OUTPUT_ACTIVE_HISTOGRAM. Offsets reference caller-owned GPU
// allocations; each dense region array has 64 * (2 * boxes) words.
struct alignas(16) SapActiveRegionListsArenaDesc {
    std::uint32_t sorted_x_handles_offset = 0;
    std::uint32_t x_sap_positions_offset = 0; // 2 u32 per box
    std::uint32_t region_ranges_offset = 0;   // 2 uint4 per box
    std::uint32_t start_region_prefix_offset = 0; // 2*boxes exclusive
    std::uint32_t start_region_total_offset = 0;  // one u32
    std::uint32_t active_histograms_offset = 0;   // 64*2*boxes u32
    std::uint32_t start_histograms_offset = 0;    // 64*2*boxes u32
    std::uint32_t ordered_active_handles_offset = 0; // 64*2*boxes u32
    std::uint32_t ordered_start_handles_offset = 0;  // 64*2*boxes u32
    std::uint32_t reserved[7] = {};
};
static_assert(sizeof(SapActiveRegionListsArenaDesc) == 64,
              "PhysX active-region descriptor ABI changed");

// Original host schedule, created-handles gate, and dispatch dimensions:
//   WRITEOUT: 32 x 256
//   COMPUTE:  64 x 512
//   OUTPUT:   64 x 256
// Original PhysX host initializes both dense histograms to zero before
// BP_CREATE_REGIONS; this stage does not repeat those clears.
// Portable scan keeps independent start/active counts instead of the
// source's packed-16-bit warp arithmetic, preserving output ordering.
class SapActiveRegionListsArenaStage final {
public:
    SapActiveRegionListsArenaStage(compute::Context& context,
                                  compute::ShaderBinary writeout,
                                  compute::ShaderBinary compute,
                                  compute::ShaderBinary output);
    ~SapActiveRegionListsArenaStage();
    SapActiveRegionListsArenaStage(const SapActiveRegionListsArenaStage&) = delete;
    SapActiveRegionListsArenaStage& operator=(const SapActiveRegionListsArenaStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t boxes,
                std::uint32_t created_handles) const;

private:
    compute::Context& context_;
    compute::PipelineId pipelines_[3] = {};
};
} // namespace fyzz::physx_gpu
