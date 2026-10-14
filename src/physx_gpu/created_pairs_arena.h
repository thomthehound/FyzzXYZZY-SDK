// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/removed_projections_arena.h"

namespace fyzz::physx_gpu {

// Extends the existing logical (arena-relative) PxgBroadPhaseDesc subset.
// The 80-byte deletion/projection descriptor prefix remains intact.
// Fields correspond to broadphase.cu::markCreatedPairsLaunch inputs.
struct alignas(16) CreatedPairsArenaDesc {
    RemovedProjectionsArenaDesc prior;
    std::uint32_t num_created_handles = 0;
    std::uint32_t num_previous_handles = 0;
    std::uint32_t created_handles_offset = 0;
    std::uint32_t new_integer_bounds_offset = 0;
    std::uint32_t reserved[4] = {};
};
static_assert(sizeof(CreatedPairsArenaDesc) == 112, "PhysX created-pair descriptor stride changed");
static_assert(offsetof(CreatedPairsArenaDesc, num_created_handles) == 80, "Created count ABI drift");
static_assert(offsetof(CreatedPairsArenaDesc, num_previous_handles) == 84, "Previous count ABI drift");
static_assert(offsetof(CreatedPairsArenaDesc, created_handles_offset) == 88, "Created handle pointer ABI drift");
static_assert(offsetof(CreatedPairsArenaDesc, new_integer_bounds_offset) == 92, "New integer bounds pointer ABI drift");

// PxgCudaBroadPhaseSap::markCreatedPairsKernel -> BP_UPDATE_CREATEDPAIRS.
// The host dispatch is 32 blocks x 256 threads when numCreatedHandles != 0.
// The caller owns the original PhysX persistent buffers and packing.
class CreatedPairsArenaStage final {
public:
    CreatedPairsArenaStage(compute::Context& context, compute::ShaderBinary shader);
    ~CreatedPairsArenaStage();
    CreatedPairsArenaStage(const CreatedPairsArenaStage&) = delete;
    CreatedPairsArenaStage& operator=(const CreatedPairsArenaStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t created_count) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};

} // namespace fyzz::physx_gpu
