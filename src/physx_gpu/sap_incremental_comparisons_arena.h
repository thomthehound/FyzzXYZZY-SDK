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

// Logical pointer lowering of the per-axis PxgBroadPhaseDesc fields
// consumed by BP_COMPUTE_INCREMENTAL_CMP_COUNTS1 and COUNTS2.
// The source has distinct current and previous endpoint histograms.
struct SapIncrementalComparisonAxisDesc {
    std::uint32_t sorted_handles_offset = 0;
    std::uint32_t projection_ranks_offset = 0;
    std::uint32_t current_start_histogram_offset = 0;
    std::uint32_t current_end_histogram_offset = 0;
    std::uint32_t previous_start_histogram_offset = 0;
    std::uint32_t previous_end_histogram_offset = 0;
    std::uint32_t comparison_histogram_offset = 0;
    std::uint32_t block_comparison_histogram_offset = 0;
};
struct alignas(16) SapIncrementalComparisonArenaDesc {
    SapIncrementalComparisonAxisDesc axes[3] = {};
    std::uint32_t total_comparisons_offset = 0; // 3 consecutive PxU32
    std::uint32_t reserved[3] = {};
};
static_assert(sizeof(SapIncrementalComparisonAxisDesc) == 32, "PhysX comparison axis stride changed");
static_assert(sizeof(SapIncrementalComparisonArenaDesc) == 112, "PhysX comparison descriptor layout changed");

// Reproduces PxgCudaBroadPhaseSap::performIncrementalSapKernel's
// first two launches, before BP_INCREMENTAL_SAP itself.
// The caller owns the previous-frame sorted ranks, endpoint histograms,
// and all persistent output storage.
class SapIncrementalComparisonsArenaStage final {
public:
    SapIncrementalComparisonsArenaStage(compute::Context& context,
                                        compute::ShaderBinary stage1_shader,
                                        compute::ShaderBinary stage2_shader,
                                        compute::ShaderBinary stage1_wave_shader,
                                        bool force_portable = false);
    ~SapIncrementalComparisonsArenaStage();
    SapIncrementalComparisonsArenaStage(const SapIncrementalComparisonsArenaStage&) = delete;
    SapIncrementalComparisonsArenaStage& operator=(const SapIncrementalComparisonsArenaStage&) = delete;

    bool using_wave_scan() const noexcept { return using_wave_; }

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset,
                std::uint32_t previous_num_boxes) const;

private:
    compute::Context& context_;
    compute::PipelineId stage1_ = 0;
    compute::PipelineId stage2_ = 0;
    bool using_wave_ = false;
};
} // namespace fyzz::physx_gpu
