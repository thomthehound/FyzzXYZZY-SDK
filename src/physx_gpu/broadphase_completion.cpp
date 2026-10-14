// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/broadphase_completion.h"

#include <stdexcept>

namespace fyzz::physx_gpu {

BroadPhaseCompletionGate::BroadPhaseCompletionGate(compute::Context& context) noexcept
    : context_(context) {}

void BroadPhaseCompletionGate::arm(compute::SubmissionId submission) {
    if (armed_)
        throw std::logic_error("Broadphase completion already armed");
    submission_ = submission;
    armed_ = true;
}

bool BroadPhaseCompletionGate::gpu_complete() const {
    if (!armed_)
        throw std::logic_error("Broadphase completion not armed");
    return submission_ == 0 || context_.submission_complete(submission_);
}

void BroadPhaseCompletionGate::fetch_and_release(
    const std::function<void()>& fetch_and_resize,
    const std::function<void()>& install_continuations,
    physx::PxBaseTask& found,
    physx::PxBaseTask& lost) {
    if (!armed_)
        throw std::logic_error("Broadphase completion not armed");
    if (!fetch_and_resize || !install_continuations)
        throw std::invalid_argument("Broadphase requires publication and continuation operations");
    if (&found == &lost)
        throw std::invalid_argument("Broadphase pair tasks must be distinct");

    if (submission_ != 0) {
        // Original gpuDMABack waits at the fetch boundary; work queued
        // after this token on the same context is not part of this frame.
        context_.wait_submission(submission_);
        submission_ = 0;
    }

    // Keep armed_ true until publication succeeds. A failure must not
    // release the found/lost tasks against unpublished or stale pair data.
    fetch_and_resize();
    // Exactly the original PxgAABBManager::postBroadPhase order: set both
    // found/lost continuations *after* result retrieval and pair resizing.
    install_continuations();

    // Mark consumed BEFORE calls into PhysX task dispatch. They may execute
    // continuations or even destroy this gate's owning scene immediately.
    armed_ = false;
    found.removeReference();
    lost.removeReference();
}

} // namespace fyzz::physx_gpu
