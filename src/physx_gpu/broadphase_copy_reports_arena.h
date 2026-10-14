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

// Logical lowering of only the PxgBroadPhaseDesc fields consumed by
// broadphase.cu::copyReports (BP_COPY_REPORTS).
//
// found/lost_actor_pairs are the classified inputs from
// BP_ACCUMULATE_REPORT_STAGE_2, NOT the unclassified pair reports.
// found/lost_map are the logical outputs formerly written through
// CUDA mapped host pointers. Portable backend resource publication and
// PhysX task continuation still belong to the original scene owner.
// All offset values are bytes in one caller-owned device arena.
struct alignas(16) BroadphaseCopyReportsArenaDesc {
    std::uint32_t raw_pair_counts_offset = 0;       // [found, lost]
    std::uint32_t aggregate_pair_counts_offset = 0; // [found, lost]
    std::uint32_t found_actor_pairs_offset = 0;     // PxgBroadPhasePair[capacity]
    std::uint32_t lost_actor_pairs_offset = 0;
    std::uint32_t found_map_offset = 0;             // publication staging buffer
    std::uint32_t lost_map_offset = 0;
    std::uint32_t overflow_flag_offset = 0;         // portable 32-bit bool
    std::uint32_t report_capacity = 0;
    std::uint32_t reserved[8] = {};
};
static_assert(sizeof(BroadphaseCopyReportsArenaDesc) == 64,
              "PhysX copy-reports descriptor ABI changed");
static_assert(offsetof(BroadphaseCopyReportsArenaDesc, report_capacity) == 28,
              "PhysX copy-reports capacity moved");

// Source schedule: a fixed 64 x 256 BP_COPY_REPORTS dispatch following
// both BP_ACCUMULATE_REPORT_STAGE_1 and BP_ACCUMULATE_REPORT_STAGE_2.
// No zero-work elision: the kernel publishes an overflow flag even for
// an empty report.
class BroadphaseCopyReportsArenaStage final {
public:
    BroadphaseCopyReportsArenaStage(compute::Context& context,
                                   compute::ShaderBinary shader);
    ~BroadphaseCopyReportsArenaStage();
    BroadphaseCopyReportsArenaStage(const BroadphaseCopyReportsArenaStage&) = delete;
    BroadphaseCopyReportsArenaStage& operator=(const BroadphaseCopyReportsArenaStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};
} // namespace fyzz::physx_gpu
