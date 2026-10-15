// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_newbounds_sequence_arena.h"

#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {

SapNewBoundsSequenceArenaDescriptors make_sap_newbounds_sequence_descriptors(
    const SapNewBoundsPairsArenaDesc& pairs,
    std::uint32_t region_prefix_offset,
    std::uint32_t block_overlap_counts_offset) {
    SapNewBoundsSequenceArenaDescriptors result{};
    result.pairs = pairs;
    auto& w = result.work;
    w.sorted_x_handles_offset = pairs.sorted_x_handles_offset;
    w.x_sap_positions_offset = pairs.x_sap_positions_offset;
    w.region_ranges_offset = pairs.region_ranges_offset;
    w.region_prefix_offset = region_prefix_offset;
    w.region_total_offset = pairs.region_work_total_offset;
    w.active_histograms_offset = pairs.active_histograms_offset;
    w.start_histograms_offset = pairs.start_histograms_offset;
    w.overlap_counts_offset = pairs.overlap_prefix_offset;
    w.handle_regions_offset = pairs.handle_regions_offset;

    auto& p = result.prefix;
    p.overlap_counts_offset = pairs.overlap_prefix_offset;
    p.block_counts_offset = block_overlap_counts_offset;
    p.total_offset = pairs.overlap_total_offset;
    p.region_work_total_offset = pairs.region_work_total_offset;
    return result;
}

SapNewBoundsSequenceArenaStage::SapNewBoundsSequenceArenaStage(
    const SapOverlapWorkArenaStage& work,
    const SapOverlapPrefixArenaStage& prefix,
    const SapNewBoundsPairsArenaStage& pairs) noexcept
    : work_(work), prefix_(prefix), pairs_(pairs) {}

void SapNewBoundsSequenceArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, SapNewBoundsSequenceDescriptorOffsets offsets,
    std::uint32_t current_boxes, std::uint32_t created_handles) const {
    // PhysX gates the complete four-kernel sequence together. In particular
    // never run a stale pair prefix when no handles were created.
    if (!created_handles) return;

    // Stage append methods validate their own descriptor sizes and launch
    // dimensions. Stage into a local vector first so a later rejection does
    // not leave a partially recorded PhysX frame on the caller's command list.
    std::vector<compute::BatchCommand> staged;
    staged.reserve(4u);
    work_.append(staged, arena, arena_bytes, offsets.work,
                 current_boxes, created_handles);
    prefix_.append(staged, arena, arena_bytes, offsets.prefix,
                   created_handles);
    pairs_.append(staged, arena, arena_bytes, offsets.pairs,
                  current_boxes, created_handles);
    if (staged.size() != 4u)
        throw std::logic_error("PhysX new-bounds sequence must issue four dispatches");

    commands.reserve(commands.size() + staged.size());
    for (auto& command : staged)
        commands.emplace_back(std::move(command));
}
} // namespace fyzz::physx_gpu
