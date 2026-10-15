// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/broadphase_completion.h"

#include <stdexcept>

namespace fyzz::physx_gpu {

BroadPhaseCompletionGate::BroadPhaseCompletionGate(compute::Context& context) noexcept
    : context_(context) {}

void BroadPhaseCompletionGate::arm(compute::SubmissionId submission) {
    if (continuation_setup_failed_)
        throw std::logic_error("Broadphase continuation setup failed; scene must abort");
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
    // Should installation fail after touching one task, retry would violate
    // PhysX's single-installation reference contract. Poison the frame/scene
    // until the caller takes its original abort/teardown path.
    armed_ = false;
    continuation_setup_failed_ = true;
    install_continuations();

    // Keep the gate terminal until *both* original task releases succeed.
    // removeReference() can dispatch work immediately and partial release
    // cannot safely be replayed if the first task (or its dispatcher) fails.
    found.removeReference();
    lost.removeReference();
    continuation_setup_failed_ = false;
}

void BroadPhaseCompletionGate::fetch_and_run_inline(
    const std::function<void()>& fetch_and_resize,
    const std::function<void()>& run_found_pair_task,
    const std::function<void()>& run_lost_pair_task) {
    if (!armed_)
        throw std::logic_error("Broadphase completion not armed");
    if (!fetch_and_resize || !run_found_pair_task || !run_lost_pair_task)
        throw std::invalid_argument("Broadphase inline publication requires both original pair tasks");

    if (submission_ != 0) {
        context_.wait_submission(submission_);
        submission_ = 0;
    }

    // Source postBroadPhase() always retrieves data and resizes both
    // lists before invoking tasks, including in the null-continuation case.
    // A failed fetch is retryable before either task has executed.
    fetch_and_resize();

    // runInternal() can change scene state immediately. Consume the gate
    // before entering either callback; do not permit replay if only the
    // first original task has run when an exception occurs.
    armed_ = false;
    continuation_setup_failed_ = true;
    run_found_pair_task();
    run_lost_pair_task();
    continuation_setup_failed_ = false;
}

} // namespace fyzz::physx_gpu
