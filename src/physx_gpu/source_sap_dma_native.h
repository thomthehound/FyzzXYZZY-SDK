// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "hardware_profile.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// Source PxCudaContext::memcpyHtoDAsync arguments from one mStream.
// Host bytes are snapshotted when the upload is enqueued.
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
// Source HtoD bytes are snapshotted into upload pages at enqueue.
// Pages remain immutable through the ordered batch's completion fence.
// All submission tokens are reaped only at the source fetch boundary.
class SourceSapDmaNative final {
public:
    struct SubmissionStats {
        std::size_t upload_batches = 0; // Source enqueue groups, not queue submissions.
        std::size_t command_batches = 0;
        std::size_t upload_spans = 0;
        std::size_t upload_bytes = 0;
        std::size_t fetch_batches = 0;
        std::size_t readback_submissions = 0;
        std::size_t fused_readbacks = 0;
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
    bool fetch_fence_retired() const noexcept {
        return pending_fetch_!=0u && pending_fetch_reaped_;
    }
    std::size_t submitted_token_count() const noexcept { return tokens_.size(); }
    // Advances only when the pending command batch enters the native queue.
    std::size_t command_batch_generation() const noexcept {
        return stats_.command_batches;
    }

    SourceSapDmaNative(compute::Context& context,
                       const CudaDeviceAddressMap& addresses) noexcept
        : context_(context), addresses_(addresses) {}

    SourceSapDmaNative(const SourceSapDmaNative&)=delete;
    SourceSapDmaNative& operator=(const SourceSapDmaNative&)=delete;
    ~SourceSapDmaNative() noexcept {
        discard_unsubmitted_commands();
        abandon_numerical_fetch();
        abandon_fetch();
        for(const auto& page:upload_pages_)
            try { context_.retire_buffer(page.buffer); } catch(...) {}
    }

    // Staged copies borrow an unmapped transfer source through the solve fence.
    void queue_uploads(const std::vector<SourceSapUpload>& uploads,
                       std::span<const compute::CopyCommand> staged={}) {
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
        if (resolved_uploads_.empty() && staged.empty()) return;
        std::size_t upload_bytes=0u;
        for(const auto& span:resolved_uploads_) {
            if(span.destination_offset%4u || span.bytes%4u)
                throw std::invalid_argument(
                    "PhysX source upload requires aligned buffer bytes");
            if(span.bytes>std::numeric_limits<std::size_t>::max()-upload_bytes)
                throw std::overflow_error("Source upload extent overflow");
            upload_bytes+=span.bytes;
        }
        for(const auto& copy:staged)
            if(!copy.source || !copy.destination || copy.source==copy.destination ||
               !copy.bytes || copy.source_offset%4u || copy.destination_offset%4u ||
               copy.bytes%4u)
                throw std::invalid_argument("Invalid staged source upload");
        const auto original_size=pending_.size();
        pending_.reserve(original_size+resolved_uploads_.size()+staged.size());
        UploadPage* page=nullptr;
        std::size_t previous_used=0u;
        try {
            if(upload_bytes) {
                page=&acquire_upload_page(upload_bytes);
                previous_used=page->used;
                auto* mapped=static_cast<std::byte*>(context_.map_buffer(
                    page->buffer,compute::HostMapAccess::Write));
                auto cursor=previous_used;
                for(const auto& span:resolved_uploads_) {
                    std::memcpy(mapped+cursor,span.data,span.bytes);
                    cursor+=span.bytes;
                }
                context_.unmap_buffer(page->buffer);
                cursor=previous_used;
                for(const auto& span:resolved_uploads_) {
                    pending_.emplace_back(compute::CopyCommand{
                        page->buffer,cursor,span.destination,
                        span.destination_offset,span.bytes});
                    cursor+=span.bytes;
                }
                page->used=cursor;
            }
            for(const auto& copy:staged)pending_.emplace_back(copy);
        } catch(...) {
            pending_.erase(pending_.begin()+original_size,pending_.end());
            if(page)page->used=previous_used;
            throw;
        }
        ++stats_.upload_batches;
        stats_.upload_spans += resolved_uploads_.size()+staged.size();
#if defined(FYZZXYZZY_PROFILE_GPU)
        const auto previous_bytes=stats_.upload_bytes;
#endif
        for (const auto& span : resolved_uploads_) stats_.upload_bytes += span.bytes;
        for(const auto& copy:staged)stats_.upload_bytes+=copy.bytes;
#if defined(FYZZXYZZY_PROFILE_GPU)
        FYZZ_PROFILE_COUNTER("source.upload_spans",resolved_uploads_.size()+staged.size());
        FYZZ_PROFILE_COUNTER("source.upload_bytes",
            stats_.upload_bytes-previous_bytes);
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

    // Root retirement belongs to the same context and address map.
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
    void discard_unsubmitted_commands() noexcept {
        pending_.clear();
        release_recorded_roots();
        for(auto& page:upload_pages_)
            if(!page.ticket)page.used=0u;
    }

    // Submit existing source-ordered device commands without a host wait.
    compute::SubmissionId submit() { return flush_gpu_commands(); }

    bool references_recorded(compute::BufferId buffer) const noexcept {
        for(const auto& command:pending_) {
            bool referenced=false;
            if(const auto* copy=std::get_if<compute::CopyCommand>(&command))
                referenced=copy->source==buffer || copy->destination==buffer;
            else if(const auto* upload=std::get_if<compute::UploadCommand>(&command))
                referenced=upload->destination==buffer;
            else if(const auto* dispatch=std::get_if<compute::DispatchCommand>(&command))
                referenced=(dispatch->indirect && dispatch->indirect->buffer==buffer) ||
                    std::any_of(dispatch->buffers.begin(),dispatch->buffers.end(),
                        [buffer](const auto& binding){return binding.buffer==buffer;});
            if(referenced)return true;
        }
        return false;
    }

    bool buffer_in_use(compute::BufferId buffer) const {
        return context_.buffer_in_use(buffer) || references_recorded(buffer);
    }

    // Recorded consumers retain the native buffer until submission or discard.
    void retire_root(compute::BufferId buffer) {
        if(references_recorded(buffer))recorded_roots_.push_back(buffer);
        else context_.retire_buffer(buffer);
    }

    // Append the readback to pending producer commands when possible.
    // Context owns the staging and fence until Sc publishes the results;
    // later GPU continuations can remain queued independently.
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
        tokens_.reserve(tokens_.size()+2u);
        if(pending_.empty()) {
            pending_fetch_=context_.submit_download_batch(outputs);
            ++stats_.readback_submissions;
        }else {
            FYZZ_PROFILE_COUNTER("source.batch_ops",pending_.size());
            pending_fetch_=context_.submit_batch_with_readback(pending_,outputs);
            accept_recorded_commands(pending_fetch_);
            ++stats_.command_batches;
            ++stats_.fused_readbacks;
        }
        pending_fetch_reaped_=false;
        pending_fetch_bytes_=bytes;
        pending_fetch_spans_=outputs.size();
        FYZZ_PROFILE_COUNTER("source.download_bytes",bytes);
    }

    // The readback completion covers every preceding source command on
    // the ordered Context queue. Reap those tokens and the readback with
    // one fence wait; later device continuations remain outstanding.
    void finish_fetch() {
        FYZZ_PROFILE_SCOPE("source.download_join");
        if(!pending_fetch_)
            throw std::logic_error("Source GPU readback was not armed");
        const auto start=std::chrono::steady_clock::now();
        join_readback(pending_fetch_);
        // Retiring the GPU ticket does not publish Sc's borrowed witnesses.
        context_.finish_download_batch(pending_fetch_,true);
        pending_fetch_=0u;
        pending_fetch_reaped_=false;
        stats_.deferred_submissions_after_fetch+=tokens_.size();
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
        cancel_readback(pending_fetch_);
        pending_fetch_=0;
        pending_fetch_reaped_=false;
        pending_fetch_spans_=0;
        pending_fetch_bytes_=0;
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

    // Numerical readback follows its device producers but remains
    // independently owned from Sc's earlier witness publication.
    void begin_numerical_fetch(std::span<const compute::DownloadSpan> outputs) {
        FYZZ_PROFILE_SCOPE("source.numerical_download_enqueue");
        if(pending_numerical_fetch_)
            throw std::logic_error("Source numerical readback already pending");
        if(outputs.empty())
            throw std::invalid_argument("Source numerical readback is empty");
        std::size_t bytes=0u;
        for(const auto& output:outputs) {
            if(output.bytes>std::numeric_limits<std::size_t>::max()-bytes)
                throw std::overflow_error(
                    "Source numerical readback size overflow");
            bytes+=output.bytes;
        }
        tokens_.reserve(tokens_.size()+2u);
        compute::SubmissionId ticket=0u;
        if(pending_.empty()) {
            ticket=context_.submit_download_batch(outputs);
            ++stats_.readback_submissions;
        }else {
            FYZZ_PROFILE_COUNTER("source.batch_ops",pending_.size());
            ticket=context_.submit_batch_with_readback(pending_,outputs);
            accept_recorded_commands(ticket);
            ++stats_.command_batches;
            ++stats_.fused_readbacks;
        }
        pending_numerical_fetch_=ticket;
        pending_numerical_reaped_=false;
        pending_numerical_bytes_=bytes;
        pending_numerical_spans_=outputs.size();
        FYZZ_PROFILE_COUNTER("source.download_bytes",bytes);
    }

    void finish_numerical_fetch() {
        FYZZ_PROFILE_SCOPE("source.numerical_download_join");
        if(!pending_numerical_fetch_)
            throw std::logic_error("Source numerical readback was not armed");
        const auto start=std::chrono::steady_clock::now();
        join_readback(pending_numerical_fetch_);
        context_.finish_download_batch(pending_numerical_fetch_,true);
        pending_numerical_fetch_=0u;
        pending_numerical_reaped_=false;
        const auto elapsed=std::chrono::steady_clock::now()-start;
        ++stats_.fetch_batches;
        stats_.fetch_spans+=pending_numerical_spans_;
        stats_.fetch_bytes+=pending_numerical_bytes_;
        stats_.fetch_wall_ns+=static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                elapsed).count());
        pending_numerical_bytes_=0u;
        pending_numerical_spans_=0u;
    }

    void abandon_numerical_fetch() noexcept {
        if(!pending_numerical_fetch_)return;
        cancel_readback(pending_numerical_fetch_);
        pending_numerical_fetch_=0u;
        pending_numerical_reaped_=false;
        pending_numerical_bytes_=0u;
        pending_numerical_spans_=0u;
    }

    // Readiness of the most recent source-ordered GPU submission. The caller
    // must first submit the pending source stage commands (the completion
    // gate's arm_source_stream() performs that submission).
    bool gpu_complete() const {
        if (!pending_.empty())
            throw std::logic_error("PhysX source GPU commands not submitted");
        // Any outstanding readback or command may own the newest fence.
        auto newest=tokens_.empty()?0u:tokens_.back();
        if(!pending_numerical_reaped_)
            newest=std::max(newest,pending_numerical_fetch_);
        if(!pending_fetch_reaped_)
            newest=std::max(newest,pending_fetch_);
        return !newest || context_.submission_complete(newest);
    }

    // Reject recorded work and borrowed outputs, then retire accepted
    // work once before host-visible storage can be reused or destroyed.
    bool discard_and_wait() noexcept {
        discard_unsubmitted_commands();
        abandon_numerical_fetch();
        abandon_fetch();
        try {
            wait_submitted();
            return true;
        } catch (...) {
            discard_unsubmitted_commands();
            return false;
        }
    }

    void wait_at_fetch() {
        FYZZ_PROFILE_SCOPE("source.queue_wait");
        flush_gpu_commands();
        wait_submitted();
    }

    void wait_submitted() {
        if(tokens_.empty())return;
        const auto newest=tokens_.back();
        context_.wait_submissions(tokens_);
        reclaim_upload_pages(newest);
        tokens_.clear();
    }

private:
    struct UploadPage {
        compute::BufferId buffer=0u;
        std::size_t capacity=0u,used=0u;
        compute::SubmissionId ticket=0u;
        unsigned idle_requests=0u;
    };

    UploadPage& acquire_upload_page(std::size_t bytes) {
        for(auto& page:upload_pages_)
            if(!page.ticket && page.used && bytes<=page.capacity-page.used)
                return page;
        constexpr std::size_t minimum_page=64u*1024u;
        const auto capacity=std::max(bytes,minimum_page);
        // Age only unused pages, and never map a page with a live ticket.
        for(std::size_t i=0u;i<upload_pages_.size();) {
            auto& page=upload_pages_[i];
            if(!page.ticket && !page.used && ++page.idle_requests>=8u) {
                context_.retire_buffer(page.buffer);
                upload_pages_.erase(upload_pages_.begin()+i);
            }else ++i;
        }
        UploadPage* best=nullptr;
        for(auto& page:upload_pages_)
            if(!page.ticket && !page.used &&
               page.capacity>=capacity && page.capacity-capacity<=capacity &&
               (!best || page.capacity<best->capacity))best=&page;
        if(best) {
            best->idle_requests=0u;
            return *best;
        }
        upload_pages_.reserve(upload_pages_.size()+1u);
        const auto buffer=context_.create_buffer(
            capacity,compute::MemoryPlacement::Upload);
        upload_pages_.push_back({buffer,capacity});
        return upload_pages_.back();
    }

    void accept_recorded_commands(compute::SubmissionId ticket) noexcept {
        for(auto& page:upload_pages_)
            if(!page.ticket && page.used)page.ticket=ticket;
        pending_.clear();
        release_recorded_roots();
    }

    void release_recorded_roots() noexcept {
        // Native retirement follows the last accepted GPU reference.
        // Retain ownership if retirement bookkeeping cannot allocate.
        std::erase_if(recorded_roots_,[this](compute::BufferId buffer) noexcept {
            try {
                context_.retire_buffer(buffer);
                return true;
            }catch(const std::bad_alloc&) { return false; }
        });
    }

    void reclaim_upload_pages(compute::SubmissionId completed) noexcept {
        if(pending_.empty() && !recorded_roots_.empty())
            release_recorded_roots();
        for(auto& page:upload_pages_)
            if(page.ticket && page.ticket<=completed) {
                page.ticket=0u;
                page.used=0u;
                page.idle_requests=0u;
            }
    }

    // Cancelled readbacks become ordinary ordered retirement tokens.
    // Submission reserves space for both outstanding readbacks.
    void cancel_readback(compute::SubmissionId ticket) noexcept {
        if(context_.cancel_download_batch(ticket))
            tokens_.insert(std::lower_bound(tokens_.begin(),tokens_.end(),ticket),ticket);
    }

    void join_readback(compute::SubmissionId ticket) {
        const auto end=std::upper_bound(tokens_.begin(),tokens_.end(),ticket);
        fetch_wait_tokens_.assign(tokens_.begin(),end);
        const bool witness=pending_fetch_ && pending_fetch_<=ticket &&
            !pending_fetch_reaped_;
        const bool numerical=pending_numerical_fetch_ &&
            pending_numerical_fetch_<=ticket && !pending_numerical_reaped_;
        if(witness)fetch_wait_tokens_.push_back(pending_fetch_);
        if(numerical)fetch_wait_tokens_.push_back(pending_numerical_fetch_);
        if(fetch_wait_tokens_.empty())return;
        if(witness || numerical)
            std::sort(fetch_wait_tokens_.begin(),fetch_wait_tokens_.end());
        context_.wait_submissions(fetch_wait_tokens_);
        tokens_.erase(tokens_.begin(),end);
        if(witness)pending_fetch_reaped_=true;
        if(numerical)pending_numerical_reaped_=true;
        reclaim_upload_pages(ticket);
    }

    compute::SubmissionId flush_gpu_commands() {
        FYZZ_PROFILE_SCOPE("source.command_submit");
        if (pending_.empty()) return tokens_.empty() ? 0 : tokens_.back();
        tokens_.reserve(tokens_.size()+3u);
        FYZZ_PROFILE_COUNTER("source.batch_ops",pending_.size());
        const auto token = context_.submit_batch(pending_);
        accept_recorded_commands(token);
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
    std::vector<UploadPage> upload_pages_;
    std::vector<compute::BufferId> recorded_roots_;
    std::vector<compute::SubmissionId> tokens_,fetch_wait_tokens_;
    compute::SubmissionId pending_fetch_=0;
    compute::SubmissionId pending_numerical_fetch_=0;
    bool pending_fetch_reaped_=false,pending_numerical_reaped_=false;
    std::size_t pending_numerical_bytes_=0u,pending_numerical_spans_=0u;
    std::size_t pending_fetch_bytes_=0,pending_fetch_spans_=0;
    SubmissionStats stats_;
};

} // namespace fyzz::physx_gpu
