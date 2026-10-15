// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "fyzzxyzzy/compute.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {
// PxgBroadPhaseDesc subset for source BP_COMPUTE/OUTPUT_OVERLAPCHECKS_HISTOGRAM.
// All pointers are byte offsets into caller-owned persistent storage.
// overlap counts and 32 block totals are 64-bit, as in pinned CUDA.
struct alignas(16) SapOverlapPrefixArenaDesc {
    std::uint32_t overlap_counts_offset = 0;  // regionOverlapType (u64)
    std::uint32_t block_counts_offset = 0;    // 32 regionOverlapType (u64)
    std::uint32_t total_offset = 0;           // one regionOverlapType (u64)
    std::uint32_t region_work_total_offset = 0; // GPU-produced u32
    std::uint32_t reserved[4] = {};
};
static_assert(sizeof(SapOverlapPrefixArenaDesc) == 32,
              "PhysX overlap prefix descriptor ABI changed");

// Runs the TWO original CUDA kernels, 32 workgroups x 256 threads each.
// No global uint64 atomics, no per-dispatch feature negotiation, no scene
// ownership: producer input and histogram buffers belong to PhysX.
class SapOverlapPrefixArenaStage final {
public:
    SapOverlapPrefixArenaStage(compute::Context& context,
                              compute::ShaderBinary compute_wave,
                              compute::ShaderBinary output_wave);
    ~SapOverlapPrefixArenaStage();
    SapOverlapPrefixArenaStage(const SapOverlapPrefixArenaStage&) = delete;
    SapOverlapPrefixArenaStage& operator=(const SapOverlapPrefixArenaStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId arena, std::uint32_t arena_bytes,
                std::uint32_t descriptor_offset,
                std::uint32_t created_handles) const;
private:
    compute::Context& context_;
    compute::PipelineId pipelines_[2] = {};
};
} // namespace fyzz::physx_gpu
