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
class RadixCountArenaStage;
class RadixScatterArenaStage;

// PxgBroadPhaseDesc::boxProjectionRanks, boxHandles[1], boxHandles[0].
// The temporary radix rank buffer is separate from boxHandles[1].
struct SapRankedSortAxisDesc {
    std::uint32_t projection_ranks_offset = 0;
    std::uint32_t previous_handles_offset = 0;
    std::uint32_t current_handles_offset = 0;
    std::uint32_t reserved = 0;
};
struct alignas(16) SapRankedSortArenaDesc {
    SapRankedSortAxisDesc axes[3] = {};
};
static_assert(sizeof(SapRankedSortAxisDesc) == 16, "PhysX SAP ranked-sort stride changed");
static_assert(sizeof(SapRankedSortArenaDesc) == 48, "PhysX SAP ranked-sort descriptor changed");

// PxgCudaBroadPhaseSap::sortProjectionAndHandlesWRKernel, including its
// original initialization, eight four-bit radix passes, and handle gather.
class SapRankedSortArenaStage final {
public:
    SapRankedSortArenaStage(compute::Context& context,
                            compute::ShaderBinary initialize_shader,
                            compute::ShaderBinary update_handles_shader);
    ~SapRankedSortArenaStage();
    SapRankedSortArenaStage(const SapRankedSortArenaStage&) = delete;
    SapRankedSortArenaStage& operator=(const SapRankedSortArenaStage&) = delete;

    void append_source_sequence(std::vector<compute::BatchCommand>& commands,
                                const RadixCountArenaStage& count_stage,
                                const RadixScatterArenaStage& scatter_stage,
                                compute::BufferId arena, std::uint32_t arena_bytes,
                                std::uint32_t descriptor_offset,
                                const std::uint32_t (&count_descriptor_offsets)[2],
                                const std::uint32_t (&scatter_descriptor_offsets)[2],
                                std::uint32_t previous_num_boxes) const;

private:
    void append_dispatch(std::vector<compute::BatchCommand>& commands,
                         compute::PipelineId pipeline, compute::BufferId arena,
                         std::uint32_t arena_bytes, std::uint32_t descriptor_offset,
                         std::uint32_t previous_num_boxes) const;

    compute::Context& context_;
    compute::PipelineId initialize_pipeline_ = 0;
    compute::PipelineId update_pipeline_ = 0;
};
} // namespace fyzz::physx_gpu
