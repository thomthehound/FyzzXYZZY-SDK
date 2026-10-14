// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/mark_removed_pairs_arena.h"

namespace fyzz::physx_gpu {

// Extends the established compact broadphase descriptor *without changing*
// its original 64-byte prefix. Three boxProjections[axis] pointers are
// lowered into three additional byte offsets. The actual PhysX descriptor
// still owns these arrays; no private simulation representation is created.
struct alignas(16) RemovedProjectionsArenaDesc {
    MarkRemovedPairsArenaDesc common;
    std::uint32_t projection_offsets[3] = {};
    std::uint32_t reserved = 0;
};
static_assert(sizeof(RemovedProjectionsArenaDesc) == 80, "SAP projection descriptor ABI changed");
static_assert(offsetof(RemovedProjectionsArenaDesc, projection_offsets) == 64,
              "SAP projection offsets must follow the existing descriptor prefix");

// PhysX broadphase.cu::markRemovedPairsProjectionsLaunch, called in
// PxgCudaBroadPhaseSap::update after incremental SAP, when any removed
// handles exist. 32 blocks x 256 threads exactly as the pinned host launch.
class RemovedProjectionsArenaStage final {
public:
    RemovedProjectionsArenaStage(compute::Context& context, compute::ShaderBinary shader);
    ~RemovedProjectionsArenaStage();

    RemovedProjectionsArenaStage(const RemovedProjectionsArenaStage&) = delete;
    RemovedProjectionsArenaStage& operator=(const RemovedProjectionsArenaStage&) = delete;

    // The arena contains a complete RemovedProjectionsArenaDesc at the
    // given aligned offset. Caller owns nested offset/range validation.
    // A zero count is the upstream no-dispatch path.
    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t removed_count) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};

} // namespace fyzz::physx_gpu
