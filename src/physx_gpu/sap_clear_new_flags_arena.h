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

// Only the upstream BP_CLEAR_NEWFLAG pointers. These offsets refer to
// PxgBroadPhaseDesc::updateData_createdHandles, boxSapBox1D[axis],
// and boxHandles[0][axis] in caller-owned persistent frame storage.
// Not a replacement PhysX broadphase descriptor or frame owner.
struct alignas(16) SapClearNewFlagsArenaDesc {
    std::uint32_t created_handles_offset = 0;
    std::uint32_t sap_box_offsets[3] = {};
    std::uint32_t current_handle_offsets[3] = {};
    std::uint32_t created_count = 0;
    std::uint32_t box_capacity = 0;
    std::uint32_t endpoint_capacity = 0;
    std::uint32_t reserved = 0;
};
static_assert(sizeof(SapClearNewFlagsArenaDesc) == 48, "PhysX clear-new descriptor stride changed");
static_assert(offsetof(SapClearNewFlagsArenaDesc, sap_box_offsets) == 4,
              "PhysX clear-new SAP offsets moved");
static_assert(offsetof(SapClearNewFlagsArenaDesc, current_handle_offsets) == 16,
              "PhysX clear-new handle offsets moved");
static_assert(offsetof(SapClearNewFlagsArenaDesc, created_count) == 28,
              "PhysX clear-new count moved");

// The source's conditional BP_CLEAR_NEWFLAG / clearNewFlagLaunch:
// 32 workgroups x 256 threads, each created object is rewritten at its
// SAP endpoint indices with createHandle(handle, isStart, false).
// The caller must submit after region pair generation and bounds swap,
// before runCopyResultsKernel. No persistent resource ownership here.
class SapClearNewFlagsArenaStage final {
public:
    SapClearNewFlagsArenaStage(compute::Context& context, compute::ShaderBinary shader);
    ~SapClearNewFlagsArenaStage();
    SapClearNewFlagsArenaStage(const SapClearNewFlagsArenaStage&) = delete;
    SapClearNewFlagsArenaStage& operator=(const SapClearNewFlagsArenaStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t created_count) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};
} // namespace fyzz::physx_gpu
