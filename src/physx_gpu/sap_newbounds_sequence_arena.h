// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/sap_overlap_work_arena.h"
#include "physx_gpu/sap_overlap_prefix_arena.h"
#include "physx_gpu/sap_newbounds_pairs_arena.h"

#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Derived views of the SAME caller-owned PxgBroadPhaseDesc state.
// Prepare/upload these descriptors when the scene's GPU arena layout is
// established or changed. No duplicate count/prefix/report allocation.
struct SapNewBoundsSequenceArenaDescriptors {
    SapOverlapWorkArenaDesc work{};
    SapOverlapPrefixArenaDesc prefix{};
    SapNewBoundsPairsArenaDesc pairs{};
};

// The pair stage's source field mapping is authoritative for all shared
// offsets. Only the producer's upstream region prefix and the two scan
// kernels' 32 block totals need to be supplied separately.
SapNewBoundsSequenceArenaDescriptors make_sap_newbounds_sequence_descriptors(
    const SapNewBoundsPairsArenaDesc& pairs,
    std::uint32_t region_prefix_offset,
    std::uint32_t block_overlap_counts_offset);

struct SapNewBoundsSequenceDescriptorOffsets {
    std::uint32_t work = 0;
    std::uint32_t prefix = 0;
    std::uint32_t pairs = 0;
};

// Thin source-order adapter for PxgCudaBroadPhaseSap::generatePairsForNewBounds:
// WRITEOUT (32x256), COMPUTE (32x256), OUTPUT (32x256), GENERATE (64x256).
// The caller owns the stages, GPU arena, descriptors, created-handle gate,
// current frame state, submission, task dependencies, and result publication.
// This is NOT a replacement PhysX frame scheduler.
class SapNewBoundsSequenceArenaStage final {
public:
    SapNewBoundsSequenceArenaStage(
        const SapOverlapWorkArenaStage& work,
        const SapOverlapPrefixArenaStage& prefix,
        const SapNewBoundsPairsArenaStage& pairs) noexcept;

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena,
                std::uint32_t arena_bytes,
                SapNewBoundsSequenceDescriptorOffsets descriptor_offsets,
                std::uint32_t current_boxes,
                std::uint32_t created_handles) const;

private:
    const SapOverlapWorkArenaStage& work_;
    const SapOverlapPrefixArenaStage& prefix_;
    const SapNewBoundsPairsArenaStage& pairs_;
};
} // namespace fyzz::physx_gpu
