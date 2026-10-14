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

// Arena-relative pointer lowering of the three current-frame
// boxHandles[0][axis] arrays and both PxgSapBox1D outputs.
// PxgSapBox1D is two consecutive 32-bit endpoint indices per handle.
struct SapBoxIndicesAxisDesc {
    std::uint32_t sorted_handles_offset = 0;
    std::uint32_t current_sap_offset = 0;
    std::uint32_t new_sap_offset = 0;
    std::uint32_t reserved = 0;
};

struct alignas(16) SapBoxIndicesArenaDesc {
    SapBoxIndicesAxisDesc axes[3] = {};
    std::uint32_t box_capacity = 0;
    std::uint32_t reserved[3] = {};
};
static_assert(sizeof(SapBoxIndicesAxisDesc) == 16, "SAP axis descriptor stride changed");
static_assert(sizeof(SapBoxIndicesArenaDesc) == 64, "SAP descriptor layout changed");

// Port of broadphase.cu::initializeSapBox1DLaunch.
// The original host wrapper launches ceil(numHandles*2/256) groups.
// The caller owns the sorted handles and both persistent SAP arrays.
class SapBoxIndicesArenaStage final {
public:
    SapBoxIndicesArenaStage(compute::Context& context, compute::ShaderBinary shader);
    ~SapBoxIndicesArenaStage();
    SapBoxIndicesArenaStage(const SapBoxIndicesArenaStage&) = delete;
    SapBoxIndicesArenaStage& operator=(const SapBoxIndicesArenaStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t num_handles,
                bool is_new) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};

} // namespace fyzz::physx_gpu
