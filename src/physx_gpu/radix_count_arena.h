// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute/compute.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Arena-relative lowering of the inputKeys and radixBlockCounts pointers
// used by the first dispatch of PxgCudaBroadPhaseSap::runRadixSort.
// There is one descriptor per axis; the original six-entry ping-pong table
// and rank/payload pointers remain owned by the caller.
struct RadixCountAxisDesc {
    std::uint32_t input_keys_offset = 0;
    std::uint32_t radix_counts_offset = 0;
    std::uint32_t reserved[2] = {};
};
struct alignas(16) RadixCountArenaDesc {
    RadixCountAxisDesc axes[3] = {};
};
static_assert(sizeof(RadixCountAxisDesc) == 16, "Radix axis layout changed");
static_assert(sizeof(RadixCountArenaDesc) == 48, "Three-axis radix layout changed");

class RadixCountArenaStage final {
public:
    RadixCountArenaStage(compute::Context& context, compute::ShaderBinary shader);
    ~RadixCountArenaStage();
    RadixCountArenaStage(const RadixCountArenaStage&) = delete;
    RadixCountArenaStage& operator=(const RadixCountArenaStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t num_keys,
                std::uint32_t start_bit) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};

} // namespace fyzz::physx_gpu
