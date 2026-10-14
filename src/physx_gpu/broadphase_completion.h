// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fyzzxyzzy/compute.h"
#include "task/PxTask.h"

#include <functional>

namespace fyzz::physx_gpu {

// Completion boundary for the pinned PhysX 5.11 broadphase host flow:
// PxgCudaBroadPhaseSap::gpuDMABack/fetchBroadPhaseResults waits for the
// results, and PxgAABBManager::postBroadPhase publishes/resizes pair arrays
// before releasing its existing found/lost-pair tasks.
//
// This gate does NOT schedule tasks, copy results or replace PhysX's task
// graph. The caller supplies the original fetch+resize operation and the
// already-initialized PxBaseTask objects from PxgAABBManager.
class BroadPhaseCompletionGate final {
public:
    explicit BroadPhaseCompletionGate(compute::Context& context) noexcept;

    // One frame at a time. Zero is the upstream "no GPU update" case; it
    // still has a publication boundary. Nonzero is a Context submission ID.
    // A subsequent frame can be armed only after successful publication.
    void arm(compute::SubmissionId submission);

    // This is a readiness check only. It never publishes CPU-visible data
    // or removes a PhysX task reference.
    bool gpu_complete() const;

    // Preserve PhysX order: wait for GPU -> fetch/resize results ->
    // install original pair-task continuations -> release each task exactly once.
    // If GPU waiting or publication throws, NO task references are released.
    // After a successful GPU wait the publication step may be retried (e.g.,
    // after the original PhysX abort path has established empty results).
    // install_continuations must implement the original PxgAABBManager
    // continuation setup and is called AFTER fetch/resize (not beforehand).
    void fetch_and_release(const std::function<void()>& fetch_and_resize,
                           const std::function<void()>& install_continuations,
                           physx::PxBaseTask& found,
                           physx::PxBaseTask& lost);

    bool armed() const noexcept { return armed_; }

private:
    compute::Context& context_;
    compute::SubmissionId submission_ = 0;
    bool armed_ = false;
};

} // namespace fyzz::physx_gpu
