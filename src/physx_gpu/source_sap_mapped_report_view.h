// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace fyzz::physx_gpu {

// Narrow fetch-boundary view of original PhysX's hostMappedAlloc output.
// This is NOT another GPU allocator or scheduler. The owning source heap
// retains the actual allocations, GPU address mappings and lifetime.
// Broadphase report/classification code must not need CUDA-aware allocator
// headers simply to include the source gpuDMABack/fetch interface.
struct SourceSapMappedReportSpan {
    void* host=nullptr;
    std::size_t bytes=0;
};
class SourceSapMappedReportView {
public:
    virtual ~SourceSapMappedReportView() = default;
    virtual std::uint64_t device_address(const void* host,
                                          std::size_t bytes) const = 0;
    virtual void publish_at_fetch(void* host, std::size_t bytes) = 0;
    virtual void publish_batch_at_fetch(
        std::span<const SourceSapMappedReportSpan> spans) {
        for(const auto& span:spans)
            if(span.bytes)publish_at_fetch(span.host,span.bytes);
    }
};

} // namespace fyzz::physx_gpu
