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

// Logical pointer lowering for radixSortMultiCalculateRanksLaunchWithCount.
// One axis record carries the same key/payload/count buffer relationships
// as PxgRadixSortDesc; the caller maintains the original ping-pong buffers.
struct RadixScatterAxisDesc {
    std::uint32_t input_keys_offset = 0;
    std::uint32_t input_ranks_offset = 0;
    std::uint32_t output_keys_offset = 0;
    std::uint32_t output_ranks_offset = 0;
    std::uint32_t radix_counts_offset = 0;
    std::uint32_t reserved[3] = {};
};
struct alignas(16) RadixScatterArenaDesc {
    RadixScatterAxisDesc axes[3] = {};
};
static_assert(sizeof(RadixScatterAxisDesc) == 32, "PhysX radix axis stride changed");
static_assert(sizeof(RadixScatterArenaDesc) == 96, "PhysX radix axis array changed");
static_assert(offsetof(RadixScatterAxisDesc, radix_counts_offset) == 16,
              "PhysX radix histogram pointer offset changed");

// Consumes the original per-digit/block histogram produced by
// RS_MULTIBLOCK_COUNT. The caller submits the count before scatter
// and owns submission barriers, source buffers, and later passes.
class RadixScatterArenaStage final {
public:
    RadixScatterArenaStage(compute::Context& context, compute::ShaderBinary shader,
                            compute::ShaderBinary wave_shader = {},
                            bool force_portable = false);
    ~RadixScatterArenaStage();
    RadixScatterArenaStage(const RadixScatterArenaStage&) = delete;
    RadixScatterArenaStage& operator=(const RadixScatterArenaStage&) = delete;

    bool using_wave_scan() const noexcept { return using_wave_; }

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset, std::uint32_t num_keys,
                std::uint32_t start_bit) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
    bool using_wave_ = false;
};

} // namespace fyzz::physx_gpu
