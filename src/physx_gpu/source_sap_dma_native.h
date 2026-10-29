// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// Source PxCudaContext::memcpyHtoDAsync arguments from one mStream.
// Host memory is snapshotted by Context::submit_upload_batch before return.
struct SourceSapUpload {
    CudaDeviceAddressMap::Address destination = 0;
    const void* data = nullptr;
    std::size_t bytes = 0;
};

// Single-source-stream lowering of PxgCudaBroadPhaseSap::preBroadPhase /
// gpuDMAUp transfers, preceding the original source GPU launch sequence.
// A Context supplies the ordered native queue; PhysX retains allocation,
// descriptor, stream lifetime and task/scene ownership. This class neither
// creates a scheduler nor introduces cross-stream/event semantics.
//
// Source HtoD calls in one adjacent group submit as one native upload batch;
// GPU copies and shader launches form ordered native command batches. A host
// upload following GPU commands forces their submission first, NOT a wait.
// All submission tokens are reaped only at the source fetch boundary.
class SourceSapDmaNative final {
public:
    struct SubmissionStats {
        std::size_t upload_batches = 0;
        std::size_t command_batches = 0;
        std::size_t upload_spans = 0;
        std::size_t upload_bytes = 0;
    };
    const SubmissionStats& submission_stats() const noexcept { return stats_; }

    SourceSapDmaNative(compute::Context& context,
                       const CudaDeviceAddressMap& addresses) noexcept
        : context_(context), addresses_(addresses) {}

    void queue_uploads(const std::vector<SourceSapUpload>& uploads) {
        resolved_uploads_.clear();
        resolved_uploads_.reserve(uploads.size());
        for (const auto& upload : uploads) {
            // CUDA zero-byte HtoD copies enqueue no transfer.
            if (!upload.bytes) continue;
            if (!upload.data)
                throw std::invalid_argument("PhysX source HtoD upload has no host data");
            const auto target = addresses_.resolve(upload.destination, upload.bytes);
            resolved_uploads_.push_back({target.buffer, target.offset,
                                         upload.data, upload.bytes});
        }
        if (resolved_uploads_.empty()) return;
        tokens_.reserve(tokens_.size() + (pending_.empty() ? 1u : 2u));
        flush_gpu_commands();
        // The Context snapshots host bytes and owns the GPU staging resource
        // until the corresponding completion token is reaped.
        const auto token = context_.submit_upload_batch(resolved_uploads_);
        tokens_.push_back(token);
        ++stats_.upload_batches;
        stats_.upload_spans += resolved_uploads_.size();
        for (const auto& span : resolved_uploads_) stats_.upload_bytes += span.bytes;
    }

    // Original PxgCudaBuffer::allocateCopyOldDataAsync copies RAW mPtr
    // allocation bytes, not its 128-byte-rounded getDevicePtr() view.
    void queue_device_copy(CudaDeviceAddressMap::Address destination_raw,
                           CudaDeviceAddressMap::Address source_raw,
                           std::size_t bytes) {
        if (!bytes) return;
        const auto source = addresses_.resolve(source_raw, bytes);
        const auto destination = addresses_.resolve(destination_raw, bytes);
        if (source.buffer == destination.buffer &&
            source.offset < destination.offset + bytes &&
            destination.offset < source.offset + bytes)
            throw std::invalid_argument("PhysX source GPU copy overlaps itself");
        pending_.emplace_back(compute::CopyCommand{
            source.buffer, source.offset,
            destination.buffer, destination.offset, bytes});
    }

    // Prevent PhysX heap root retirement from accidentally submitting a
    // different scene/context's source commands before releasing memory.
    bool bound_to(const compute::Context& context,
                  const CudaDeviceAddressMap& addresses) const noexcept {
        return &context_ == &context && &addresses_ == &addresses;
    }

    // Source-native kernel stages append into this existing mStream segment.
    // Appenders provide their own all-or-nothing source field validation.
    std::vector<compute::BatchCommand>& commands() noexcept { return pending_; }

    // A complete source update is validated here before being committed to
    // mStream. Retaining the scratch capacity avoids reallocating its
    // command sequence on every frame, including after a failed update.
    std::vector<compute::BatchCommand>& update_scratch() noexcept {
        update_scratch_.clear();
        return update_scratch_;
    }

    // The caller submits each host snapshot before borrowing this storage
    // again. Source heap allocations and GPU submission order are unchanged.
    std::vector<SourceSapUpload>& upload_scratch() noexcept {
        upload_scratch_.clear();
        return upload_scratch_;
    }

    // Submit existing source-ordered device commands without a host wait.
    compute::SubmissionId submit() { return flush_gpu_commands(); }

    // Readiness of the most recent source-ordered GPU submission. The caller
    // must first submit the pending source stage commands (the completion
    // gate's arm_source_stream() performs that submission).
    bool gpu_complete() const {
        if (!pending_.empty())
            throw std::logic_error("PhysX source GPU commands not submitted");
        return tokens_.empty() || context_.submission_complete(tokens_.back());
    }

    // Called at the original GPU fetch/publication boundary, before
    // PxgAABBManager releases found/lost tasks. Reap all staging/command
    // tokens; waiting the last ordered submission implies GPU visibility
    // of the preceding submissions on the same Context queue.
    // At scene teardown there is no caller to propagate a failed source
    // completion to. Keep the Context responsible for already-submitted
    // resources and drop only commands that never entered the native queue.
    // This is not a recovery path: the source scene is being destroyed.
    bool finish_shutdown() noexcept {
        try {
            wait_at_fetch();
            return true;
        } catch (...) {
            pending_.clear();
            return false;
        }
    }

    void wait_at_fetch() {
        flush_gpu_commands();
        if(tokens_.empty())return;
        // Every source command runs on one ordered native queue. Waiting
        // for its newest fence completes every preceding submission,
        // avoiding a separate blocking wait at each intermediate fence.
        // Retire the final token first so a later retirement exception
        // cannot leave an already-consumed final token pending for retry.
        context_.wait_submission(tokens_.back());
        tokens_.pop_back();
        std::size_t completed = 0;
        try {
            for (const auto token : tokens_) {
                context_.wait_submission(token);
                ++completed;
            }
        } catch (...) {
            tokens_.erase(tokens_.begin(), tokens_.begin() + completed);
            throw;
        }
        tokens_.clear();
    }

private:
    compute::SubmissionId flush_gpu_commands() {
        if (pending_.empty()) return tokens_.empty() ? 0 : tokens_.back();
        tokens_.reserve(tokens_.size() + 1u);
        const auto token = context_.submit_batch(pending_);
        pending_.clear();
        tokens_.push_back(token);
        ++stats_.command_batches;
        return token;
    }

    compute::Context& context_;
    const CudaDeviceAddressMap& addresses_;
    std::vector<compute::BatchCommand> pending_;
    std::vector<compute::BatchCommand> update_scratch_;
    std::vector<SourceSapUpload> upload_scratch_;
    std::vector<compute::UploadSpan> resolved_uploads_;
    std::vector<compute::SubmissionId> tokens_;
    SubmissionStats stats_;
};

} // namespace fyzz::physx_gpu
