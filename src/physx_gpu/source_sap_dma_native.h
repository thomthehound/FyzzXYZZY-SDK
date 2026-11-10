// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "hardware_profile.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
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
        std::size_t fetch_batches = 0;
        std::size_t readback_submissions = 0;
        std::size_t fetch_spans = 0;
        std::size_t fetch_bytes = 0;
        // Host wall time includes ordered GPU completion and readback.
        std::uint64_t fetch_wall_ns = 0;
        // Submitted after a witness DMA, but intentionally not joined
        // when its earlier host-facing result becomes visible.
        std::size_t deferred_submissions_after_fetch = 0;
    };
    const SubmissionStats& submission_stats() const noexcept { return stats_; }
    bool has_pending_fetch() const noexcept { return pending_fetch_!=0u; }

    SourceSapDmaNative(compute::Context& context,
                       const CudaDeviceAddressMap& addresses) noexcept
        : context_(context), addresses_(addresses) {}

    void queue_uploads(const std::vector<SourceSapUpload>& uploads) {
        FYZZ_PROFILE_SCOPE("source.upload_enqueue");
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
#if defined(FYZZXYZZY_PROFILE_GPU)
        for (const auto& span : resolved_uploads_)
            FYZZ_PROFILE_COUNTER("source.upload_bytes",span.bytes);
#endif
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

    // Rejected Sc transactions must never submit commands that were
    // staged but never accepted into the native queue. Previously
    // submitted source copy/upload tokens remain owned by Context.
    void discard_unsubmitted_commands() noexcept { pending_.clear(); }

    // Submit existing source-ordered device commands without a host wait.
    compute::SubmissionId submit() { return flush_gpu_commands(); }

    // Arm the host transfer as soon as its final producer dispatch has
    // been submitted. The Context owns readback staging until finish_fetch,
    // so independent source CPU work can overlap both compute and DMA.
    void begin_fetch(std::span<const compute::DownloadSpan> outputs) {
        FYZZ_PROFILE_SCOPE("source.download_enqueue");
        if(pending_fetch_)
            throw std::logic_error("Source GPU readback already pending");
        if(outputs.empty())
            throw std::invalid_argument("Source asynchronous fetch is empty");
        std::size_t bytes=0;
        for(const auto& output:outputs) {
            if(output.bytes>std::numeric_limits<std::size_t>::max()-bytes)
                throw std::overflow_error("Source GPU fetch size overflow");
            bytes+=output.bytes;
        }
        submit();
        const auto prior_token_count=tokens_.size();
        pending_fetch_=context_.submit_download_batch(outputs);
        pending_fetch_token_count_=prior_token_count;
        ++stats_.readback_submissions;
        pending_fetch_bytes_=bytes;
        pending_fetch_spans_=outputs.size();
        FYZZ_PROFILE_COUNTER("source.download_bytes",bytes);
    }

    // The submitted readback fence covers all earlier source commands
    // on this Context queue. Retire the prior submission tokens only
    // after the independent witness DMA has been published to Sc.
    void finish_fetch() {
        FYZZ_PROFILE_SCOPE("source.download_join");
        if(!pending_fetch_)
            throw std::logic_error("Source GPU readback was not armed");
        const auto start=std::chrono::steady_clock::now();
        context_.finish_download_batch(pending_fetch_,true);
        pending_fetch_=0;
        stats_.deferred_submissions_after_fetch+=
            tokens_.size()-pending_fetch_token_count_;
        // Later Sc uploads and device-continuation dispatches may already
        // be queued after this DMA. Their ordered tokens are not host
        // dependencies of the downloaded Gu witnesses.
        if(pending_fetch_token_count_) {
            context_.wait_submissions(std::span<const compute::SubmissionId>(
                tokens_.data(),pending_fetch_token_count_));
            tokens_.erase(tokens_.begin(),
                tokens_.begin()+pending_fetch_token_count_);
        }
        pending_fetch_token_count_=0;
        const auto elapsed=std::chrono::steady_clock::now()-start;
        ++stats_.fetch_batches;
        stats_.fetch_spans+=pending_fetch_spans_;
        stats_.fetch_bytes+=pending_fetch_bytes_;
        stats_.fetch_wall_ns+=static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                elapsed).count());
        pending_fetch_spans_=0;
        pending_fetch_bytes_=0;
    }

    void abandon_fetch() noexcept {
        if(!pending_fetch_)return;
        try {
            context_.finish_download_batch(pending_fetch_,false);
        } catch (...) {
            // A lost GPU cannot publish rejected source data; Context
            // retains any unretired submission until final teardown.
        }
        pending_fetch_=0;
        pending_fetch_token_count_=0;
        pending_fetch_spans_=0;
        pending_fetch_bytes_=0;
        try {wait_at_fetch();} catch (...) {}
    }

    // Existing synchronous fetches use the same staging/transfer path.
    void fetch(std::span<const compute::DownloadSpan> outputs) {
        if(pending_fetch_)
            throw std::logic_error("Source fetch overlaps another readback");
        if(outputs.empty()) {
            const auto start=std::chrono::steady_clock::now();
            wait_at_fetch();
            const auto elapsed=std::chrono::steady_clock::now()-start;
            ++stats_.fetch_batches;
            stats_.fetch_wall_ns+=static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    elapsed).count());
            return;
        }
        begin_fetch(outputs);
        finish_fetch();
    }

    // Readiness of the most recent source-ordered GPU submission. The caller
    // must first submit the pending source stage commands (the completion
    // gate's arm_source_stream() performs that submission).
    bool gpu_complete() const {
        if (!pending_.empty())
            throw std::logic_error("PhysX source GPU commands not submitted");
        // The readback is itself the newest ordered queue submission.
        // Do not announce Sc completion while its witness DMA is pending.
        if(pending_fetch_)
            return context_.submission_complete(pending_fetch_);
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
        abandon_fetch();
        try {
            wait_at_fetch();
            return true;
        } catch (...) {
            pending_.clear();
            return false;
        }
    }

    void wait_at_fetch() {
        FYZZ_PROFILE_SCOPE("source.queue_wait");
        flush_gpu_commands();
        if(tokens_.empty())return;
        // One fence on the last ordered submission completes every earlier
        // GPU upload, copy and kernel. Retire their staging and descriptors
        // together without issuing an OS fence wait for each older token.
        context_.wait_submissions(tokens_);
        tokens_.clear();
    }

private:
    compute::SubmissionId flush_gpu_commands() {
        FYZZ_PROFILE_SCOPE("source.command_submit");
        if (pending_.empty()) return tokens_.empty() ? 0 : tokens_.back();
        tokens_.reserve(tokens_.size() + 1u);
        FYZZ_PROFILE_COUNTER("source.batch_ops",pending_.size());
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
    compute::SubmissionId pending_fetch_=0;
    std::size_t pending_fetch_token_count_=0;
    std::size_t pending_fetch_bytes_=0,pending_fetch_spans_=0;
    SubmissionStats stats_;
};

} // namespace fyzz::physx_gpu
