// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"

#include <cstddef>
#include <cstring>
#include <span>
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
// Adjacent source HtoD calls across groups coalesce in one native batch;
// GPU copies and shader launches form ordered native command batches. A host
// upload following GPU commands forces their submission first, NOT a wait.
// All submission tokens are reaped only at the source fetch boundary.
class SourceSapDmaNative final {
public:
    SourceSapDmaNative(compute::Context& context,
                       const CudaDeviceAddressMap& addresses) noexcept
        : context_(context), addresses_(addresses) {}

    // The public source semantics are snapshot-at-enqueue, NOT submit per
    // call. Adjacent HtoD calls share one native submission until a DtoD
    // copy, shader command, explicit submit or fetch establishes an order
    // boundary. This avoids a fresh staging allocation and driver submit
    // for each of the source's handle/descriptor upload groups.
    void queue_uploads(const std::vector<SourceSapUpload>& uploads) {
        std::vector<PendingUpload> staged;
        staged.reserve(uploads.size());
        for (const auto& upload : uploads) {
            if (!upload.bytes) continue;
            if (!upload.data)
                throw std::invalid_argument("PhysX source HtoD upload has no host data");
            const auto target = addresses_.resolve(upload.destination, upload.bytes);
            PendingUpload entry;
            entry.buffer = target.buffer;
            entry.offset = target.offset;
            entry.bytes.resize(upload.bytes);
            std::memcpy(entry.bytes.data(), upload.data, upload.bytes);
            staged.push_back(std::move(entry));
        }
        if (staged.empty()) return;
        // Prior GPU commands must submit before subsequent uploads.
        // Do this only after all new uploads have passed validation.
        pending_uploads_.reserve(pending_uploads_.size()+staged.size());
        flush_gpu_commands();
        for (auto& item : staged) pending_uploads_.push_back(std::move(item));
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
        // DtoD follows all earlier HtoD transfers, not vice versa.
        flush_uploads();
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
    std::vector<compute::BatchCommand>& commands() {
        flush_uploads();
        return pending_;
    }

    // Submit existing source-ordered device commands without a host wait.
    compute::SubmissionId submit() {
        flush_uploads();
        return flush_gpu_commands();
    }

    // Readiness of the most recent source-ordered GPU submission. The caller
    // must first submit the pending source stage commands (the completion
    // gate's arm_source_stream() performs that submission).
    bool gpu_complete() const {
        if (!pending_.empty() || !pending_uploads_.empty())
            throw std::logic_error("PhysX source GPU commands not submitted");
        return tokens_.empty() || context_.submission_complete(tokens_.back());
    }

    // Called at the original GPU fetch/publication boundary, before
    // PxgAABBManager releases found/lost tasks. Reap all staging/command
    // tokens; waiting the last ordered submission implies GPU visibility
    // of the preceding submissions on the same Context queue.
    void wait_at_fetch() {
        flush_uploads();
        flush_gpu_commands();
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

    // Source frame instrumentation, not a second scheduling state machine.
    // Lifetime totals allow CI and later hardware tests to catch submission
    // inflation when composing actual PhysX operations.
    struct SubmissionStats {
        std::size_t upload_batches = 0;
        std::size_t command_batches = 0;
        std::size_t upload_spans = 0;
        std::size_t upload_bytes = 0;
    };
    const SubmissionStats& submission_stats() const noexcept { return stats_; }

private:
    struct PendingUpload {
        compute::BufferId buffer = 0;
        std::size_t offset = 0;
        std::vector<std::byte> bytes;
    };

    void flush_uploads() {
        if (pending_uploads_.empty()) return;
        std::vector<compute::UploadSpan> spans;
        spans.reserve(pending_uploads_.size());
        std::size_t total_bytes = 0;
        for (const auto& upload : pending_uploads_) {
            spans.push_back({upload.buffer, upload.offset,
                             upload.bytes.data(), upload.bytes.size()});
            total_bytes += upload.bytes.size();
        }
        tokens_.reserve(tokens_.size()+1u);
        // Context snapshots into GPU-safe staging synchronously before
        // returning. Only NOW can our CPU snapshots be discarded.
        const auto token = context_.submit_upload_batch(spans);
        tokens_.push_back(token);
        ++stats_.upload_batches;
        stats_.upload_spans += spans.size();
        stats_.upload_bytes += total_bytes;
        pending_uploads_.clear();
    }

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
    std::vector<PendingUpload> pending_uploads_;
    SubmissionStats stats_;
    std::vector<compute::SubmissionId> tokens_;
};

} // namespace fyzz::physx_gpu
