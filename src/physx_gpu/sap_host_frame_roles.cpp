// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_host_frame_roles.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {

SapUpdateDecision decide_sap_update(const SapUpdateInputs& input) {
    const auto total =
        std::uint64_t{input.previous_boxes} + input.created_handles;
    if (total > std::numeric_limits<std::uint32_t>::max() ||
        input.removed_handles > total)
        throw std::invalid_argument("Invalid PhysX SAP box-count transition");

    SapUpdateDecision decision;
    decision.previous_boxes = input.previous_boxes;
    decision.current_boxes = static_cast<std::uint32_t>(total - input.removed_handles);

    const bool changed =
        input.created_handles != 0 || input.removed_handles != 0 ||
        input.state_changed || input.gpu_state_changed;
    if (!changed && !input.force_update) {
        // The pinned source returns after gpuDMAUp, without updating
        // mForceUpdate or executing the GPU kernel chain.
        decision.force_update_next = false;
        return decision;
    }
    decision.dispatch_update = true;
    decision.forced_idle_update = !changed;
    decision.force_update_next = changed;
    decision.perform_incremental_sap = input.gpu_state_changed;
    decision.repair_removed_projections = input.removed_handles != 0;
    decision.sort_created_removed_projections =
        input.created_handles != 0 || input.removed_handles != 0;
    decision.clear_created_handle_flags = input.created_handles != 0;
    return decision;
}

void SapFrameBufferRoles::swap_bounds_after_generate_new_pairs() noexcept {
    std::swap(old_integer_bounds, new_integer_bounds);
}

void SapFrameBufferRoles::swap_histories_after_fetch() noexcept {
    // Exact seven PxgCudaBuffer::swapBuffer calls, per axis, from
    // PxgCudaBroadPhaseSap::fetchBroadPhaseResults.
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        const auto previous = axis + 3;
        std::swap(box_handles[axis], box_handles[previous]);
        std::swap(block_end_histograms[axis], block_end_histograms[previous]);
        std::swap(end_histograms[axis], end_histograms[previous]);
        std::swap(ordered_end_handles[axis], ordered_end_handles[previous]);
        std::swap(block_start_histograms[axis], block_start_histograms[previous]);
        std::swap(start_histograms[axis], start_histograms[previous]);
        std::swap(ordered_start_handles[axis], ordered_start_handles[previous]);
    }
}

} // namespace fyzz::physx_gpu
