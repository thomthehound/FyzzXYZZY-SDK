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

// Arena offsets for one PxgBroadPhaseDesc endpoint-histogram frame/axis.
// Frame zero is current; frame one is the preserved previous-handle view.
struct SapEndpointHistogramFrameDesc {
    std::uint32_t handles_offset = 0;
    std::uint32_t end_histogram_offset = 0;
    std::uint32_t start_histogram_offset = 0;
    std::uint32_t ordered_end_handles_offset = 0;
    std::uint32_t ordered_start_handles_offset = 0;
    std::uint32_t block_end_histogram_offset = 0;
    std::uint32_t block_start_histogram_offset = 0;
    std::uint32_t reserved = 0;
};

struct SapEndpointHistogramAxisDesc {
    SapEndpointHistogramFrameDesc frame[2] = {};
    std::uint32_t total_end_histogram_offset = 0;
    std::uint32_t block_total_end_histogram_offset = 0;
    std::uint32_t reserved[2] = {};
};

struct alignas(16) SapEndpointHistogramArenaDesc {
    SapEndpointHistogramAxisDesc axes[3] = {};
};

static_assert(sizeof(SapEndpointHistogramFrameDesc) == 32, "SAP endpoint frame stride changed");
static_assert(sizeof(SapEndpointHistogramAxisDesc) == 80, "SAP endpoint axis stride changed");
static_assert(sizeof(SapEndpointHistogramArenaDesc) == 240, "SAP endpoint descriptors changed");

// Source host schedule:
// BP_COMPUTE_ENDPT_HISTOGRAM → BP_OUTPUT_ENDPT_HISTOGRAM.
// Both use 32 x 3 blocks, 256 threads per block, and the same frame
// selection (including previous-frame scans when removed handles exist).
class SapEndpointHistogramArenaStage final {
public:
    SapEndpointHistogramArenaStage(compute::Context& context,
                                  compute::ShaderBinary first_pass_shader,
                                  compute::ShaderBinary second_pass_shader,
                                  compute::ShaderBinary first_pass_wave_shader,
                                  bool force_portable = false);
    ~SapEndpointHistogramArenaStage();
    SapEndpointHistogramArenaStage(const SapEndpointHistogramArenaStage&) = delete;
    SapEndpointHistogramArenaStage& operator=(const SapEndpointHistogramArenaStage&) = delete;

    bool using_packed_wave_scan() const noexcept { return using_wave_; }

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t previous_boxes,
                std::uint32_t created_boxes, std::uint32_t removed_boxes,
                bool is_incremental) const;

private:
    compute::Context& context_;
    compute::PipelineId first_pipeline_ = 0;
    compute::PipelineId second_pipeline_ = 0;
    bool using_wave_ = false;
};
} // namespace fyzz::physx_gpu
