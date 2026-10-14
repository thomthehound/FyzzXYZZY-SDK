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

// The port's compact *logical* subset of PxgBroadPhaseDesc, not a memcpy of
// CUDA pointers. All offsets are byte offsets into the caller-owned arena.
// Source: gpubroadphase/include/PxgBroadPhaseDesc.h; used by
// CUDA/broadphase.cu::markRemovedPairsLaunch. The 2x3 handle topology is
// retained, including the previous-frame [1][axis] arrays.
struct alignas(16) MarkRemovedPairsArenaDesc {
    std::uint32_t num_removed_handles = 0;
    std::uint32_t removed_handles_offset = 0;
    std::uint32_t sap_box_offsets[3] = {};
    std::uint32_t box_handle_offsets[2][3] = {};
    std::uint32_t old_integer_bounds_offset = 0;
    std::uint32_t box_capacity = 0;
    std::uint32_t endpoint_capacity = 0;
    std::uint32_t reserved[2] = {};
};

static_assert(sizeof(MarkRemovedPairsArenaDesc) == 64, "Arena descriptor stride changed");
static_assert(alignof(MarkRemovedPairsArenaDesc) == 16, "Arena descriptor alignment changed");
static_assert(offsetof(MarkRemovedPairsArenaDesc, sap_box_offsets) == 8, "SAP field moved");
static_assert(offsetof(MarkRemovedPairsArenaDesc, box_handle_offsets) == 20, "Handle field moved");
static_assert(offsetof(MarkRemovedPairsArenaDesc, old_integer_bounds_offset) == 44,
              "Old integer bounds field moved");

// Each dispatch has one RW arena descriptor, rather than one descriptor for
// each source PhysX pointer. No allocation or world state is owned here.
class MarkRemovedPairsArenaStage final {
public:
    MarkRemovedPairsArenaStage(compute::Context& context, compute::ShaderBinary shader);
    ~MarkRemovedPairsArenaStage();

    MarkRemovedPairsArenaStage(const MarkRemovedPairsArenaStage&) = delete;
    MarkRemovedPairsArenaStage& operator=(const MarkRemovedPairsArenaStage&) = delete;

    // Caller must pack the descriptor at descriptor_offset; count must
    // match desc.num_removed_handles. Zero removes enqueue nothing.
    // Range validation of nested offsets currently belongs to the packer.
    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t count) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};

} // namespace fyzz::physx_gpu
