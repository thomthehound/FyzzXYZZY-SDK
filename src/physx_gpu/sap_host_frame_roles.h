// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fyzzxyzzy/compute.h"

#include <array>
#include <cstdint>

namespace fyzz::physx_gpu {

// Source: PxgCudaBroadPhaseSap::update, after gpuDMAUp and before any
// conditional kernel dispatch. This is a decision derived from the
// original host code, not a replacement simulation loop or task graph.
struct SapUpdateInputs {
    std::uint32_t previous_boxes = 0;
    std::uint32_t created_handles = 0;
    std::uint32_t removed_handles = 0;
    bool state_changed = false;
    bool gpu_state_changed = false;
    bool force_update = true; // Original constructor initializes mForceUpdate true.
};

struct SapUpdateDecision {
    std::uint32_t previous_boxes = 0;
    std::uint32_t current_boxes = 0;
    bool dispatch_update = false;
    bool forced_idle_update = false;
    bool force_update_next = false;
    bool perform_incremental_sap = false; // Only when gpuStateChanged.
    bool repair_removed_projections = false;
    bool sort_created_removed_projections = false;
    bool clear_created_handle_flags = false;
};

// Does not mutate caller state. The original gpuDMAUp, descriptor uploads,
// and scene task dependencies remain caller-owned even on skipped updates.
// Invalid box-count arithmetic is rejected instead of wrapping PxU32.
SapUpdateDecision decide_sap_update(const SapUpdateInputs& input);

// Each upstream PxgCudaBuffer::swapBuffer exchanges a *resource identity*
// (the backing allocation, alignment and size), not its contents. Here each
// BufferId is an owning Context's opaque resource identity, with ownership
// retained by the original PhysX broadphase adapter, not this role map.
struct SapFrameBufferRoles {
    using History = std::array<compute::BufferId, 6>;

    // The CUDA source indexes current buffers [axis] and previous buffers
    // [axis + 3], for X/Y/Z. These seven histories alone are swapped at
    // PxgCudaBroadPhaseSap::fetchBroadPhaseResults.
    History box_handles{};
    History block_end_histograms{};
    History end_histograms{};
    History ordered_end_handles{};
    History block_start_histograms{};
    History start_histograms{};
    History ordered_start_handles{};

    // The integer bounds are a separate swap inside update, following
    // generateNewPairsKernel and before clearNewFlagKernel. They are not
    // part of the seven families rotated at fetchBroadPhaseResults.
    compute::BufferId old_integer_bounds = 0;
    compute::BufferId new_integer_bounds = 0;

    // Must be called only on the original non-one-shot update path after
    // generateNewPairsKernel. Do not call on the source's early return.
    void swap_bounds_after_generate_new_pairs() noexcept;

    // Must be called after a *successful* original gpuDMABack completion
    // and result retrieval, but before releasing PhysX found/lost tasks.
    // This is independently required even when update's early-return path
    // submitted no GPU kernels: the original fetch rotates histories.
    void swap_histories_after_fetch() noexcept;
};

} // namespace fyzz::physx_gpu
