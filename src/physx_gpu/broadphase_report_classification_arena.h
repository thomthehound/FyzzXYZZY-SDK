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

// Original broadphase.cu::accumulateReportsStage_{1,2} inputs and outputs.
// One logical, caller-owned arena; these are byte offsets, not CUDA pointers.
// Per-volume tags are Bp::VolumeData::mAggregate values (PX_INVALID_U32
// means isSingleActor()), NOT invented aggregate membership flags.
struct alignas(16) BroadphaseReportClassificationArenaDesc {
    std::uint32_t raw_pair_counts_offset = 0;       // [found,lost] from GPU pair producers
    std::uint32_t aggregate_pair_counts_offset = 0; // [found,lost] written by stage 2
    std::uint32_t found_pairs_offset = 0;
    std::uint32_t lost_pairs_offset = 0;
    std::uint32_t found_aggregate_output_offset = 0;
    std::uint32_t lost_aggregate_output_offset = 0;
    std::uint32_t found_actor_output_offset = 0;
    std::uint32_t lost_actor_output_offset = 0;
    std::uint32_t found_aggregate_block_offset = 0; // 32 u32s
    std::uint32_t found_actor_block_offset = 0;
    std::uint32_t lost_aggregate_block_offset = 0;
    std::uint32_t lost_actor_block_offset = 0;
    std::uint32_t volume_aggregate_tags_offset = 0; // one Bp::VolumeData::mAggregate per volume
    std::uint32_t volume_capacity = 0;
    std::uint32_t raw_report_capacity = 0;          // max_found_lost_pairs
    std::uint32_t aggregate_report_capacity = 0;    // max_found_lost_agg_pairs
    std::uint32_t has_volume_data = 0;               // null pointer => all single actors
    std::uint32_t reserved[3] = {};
};
static_assert(sizeof(BroadphaseReportClassificationArenaDesc) == 80,
              "PhysX report classification descriptor ABI changed");
static_assert(offsetof(BroadphaseReportClassificationArenaDesc, has_volume_data) == 64,
              "PhysX report classification descriptor layout changed");

// Port of BP_ACCUMULATE_REPORT_STAGE_1 followed by STAGE_2.
// Both original launches: 32 × 4 groups with 256 threads per group.
// No allocations or independent broadphase scheduler; the source-owned
// volume state and four separate reports remain caller-owned.
class BroadphaseReportClassificationArenaStage final {
public:
    BroadphaseReportClassificationArenaStage(compute::Context& context,
                                              compute::ShaderBinary first_shader,
                                              compute::ShaderBinary second_shader,
                                              compute::ShaderBinary first_wave_shader,
                                              compute::ShaderBinary second_wave_shader,
                                              bool force_portable = false);
    ~BroadphaseReportClassificationArenaStage();
    BroadphaseReportClassificationArenaStage(const BroadphaseReportClassificationArenaStage&) = delete;
    BroadphaseReportClassificationArenaStage& operator=(const BroadphaseReportClassificationArenaStage&) = delete;

    bool using_wave_reports() const noexcept { return using_wave_; }

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset) const;

private:
    compute::Context& context_;
    compute::PipelineId first_ = 0;
    compute::PipelineId second_ = 0;
    bool using_wave_ = false;
};

} // namespace fyzz::physx_gpu
