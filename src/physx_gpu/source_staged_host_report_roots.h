// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "CmVirtualAllocatorCallback.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>

namespace fyzz::physx_gpu {

// A backend-neutral LOWERING of the *source broadphase report arrays'*
// GPU-mapped host allocation contract. This is a STAGED implementation,
// not CUDA zero-copy host mapping or a generic GPU-pinned host allocator.
//
// PhysX's original Cm::PinnableArray and PxgHeapMemoryAllocator continue
// owning the CPU pointer and their original buddy-block lifetimes. For each
// lower-level host root, this callback owns a native GPU-side write target,
// registered as a CUDA-style virtual root. Source host pointer + offset and
// source GPU virtual root + offset refer to matching byte ranges.
//
// At the ORIGINAL fetch boundary, after mStream completes, publish_at_fetch()
// transfers the GPU-written bytes into the original CPU pointer. No CPU
// data are read before the fetch, and no GPU work is detached from its stream.
//
// This only satisfies the mapped actor-report publication use case. It does
// NOT replace direct GPU/CPU coherent flag polling (mEventMapped), nor does
// it promise pinned/persistent mapping to arbitrary application code.
class SourceStagedHostReportRoots final : public physx::Cm::VirtualAllocatorCallback {
public:
    using Address = CudaDeviceAddressMap::Address;

    SourceStagedHostReportRoots(compute::Context& context,
                                CudaDeviceAddressMap& addresses,
                                SourceSapDmaNative& source) noexcept
        : context_(context), addresses_(addresses), source_(source) {}

    SourceStagedHostReportRoots(const SourceStagedHostReportRoots&) = delete;
    SourceStagedHostReportRoots& operator=(const SourceStagedHostReportRoots&) = delete;

    ~SourceStagedHostReportRoots() override { cleanup_noexcept(); }

    void* allocate(std::size_t bytes, int, const char*, int) override {
        if (!bytes) return nullptr;
        if (!source_.bound_to(context_, addresses_))
            throw std::invalid_argument("Mapped PhysX report stream belongs to another GPU context");

        // Host/virtual ranges have distinct ownership and are only paired by
        // this original lower-level allocation callback.
        void* host = ::operator new(bytes, std::align_val_t{256});
        compute::BufferId device = 0;
        Address virtual_root = 0;
        try {
            device = context_.create_buffer(bytes, compute::MemoryPlacement::DeviceLocal);
            virtual_root = addresses_.register_allocation(device, bytes);
            const auto key = reinterpret_cast<std::uintptr_t>(host);
            const auto inserted = roots_.emplace(key, Entry{host, bytes, device, virtual_root});
            if (!inserted.second)
                throw std::logic_error("PhysX mapped report root address reused while live");
        } catch (...) {
            if (virtual_root) addresses_.unregister_allocation(virtual_root);
            if (device) context_.destroy_buffer(device);
            ::operator delete(host, std::align_val_t{256});
            throw;
        }
        return host;
    }

    void deallocate(void* host) override {
        if (!host) return;
        const auto key = reinterpret_cast<std::uintptr_t>(host);
        auto it = roots_.find(key);
        if (it == roots_.end())
            throw std::out_of_range("PhysX mapped-report free requires an exact host root");
        // Respect original scene queue ownership; no wait on deallocation.
        source_.submit();
        context_.retire_buffer(it->second.device);
        addresses_.unregister_allocation(it->second.virtual_root);
        ::operator delete(host, std::align_val_t{256});
        roots_.erase(it);
    }

    Address device_address(const void* host, std::size_t bytes) const {
        if (!host || !bytes)
            throw std::invalid_argument("PhysX mapped report view must be nonempty");
        const auto location = find(host, bytes);
        return location.root->virtual_root + location.offset;
    }

    // Source mFoundActorPairsMapped/mLostActorPairsMapped host arrays become
    // readable only at the existing source GPU fetch/publication boundary.
    // This method reaps mStream first, so no speculative CPU read or copy
    // before original GPU task completion is possible.
    void publish_at_fetch(void* host, std::size_t bytes) {
        if (!host || !bytes) return;
        if (!source_.bound_to(context_, addresses_))
            throw std::invalid_argument("Foreign PhysX source stream for mapped report publication");
        const auto location = find(host, bytes);
        source_.wait_at_fetch();
        context_.download(location.root->device, location.offset, host, bytes);
    }

    std::size_t live_roots() const noexcept { return roots_.size(); }

private:
    struct Entry {
        void* host;
        std::size_t bytes;
        compute::BufferId device;
        Address virtual_root;
    };
    struct Location {
        const Entry* root;
        std::size_t offset;
    };

    Location find(const void* host, std::size_t bytes) const {
        const auto ptr = reinterpret_cast<std::uintptr_t>(host);
        auto it = roots_.upper_bound(ptr);
        if (it == roots_.begin()) throw std::out_of_range("PhysX mapped report pointer is unknown");
        --it;
        const auto offset = ptr - it->first;
        if (offset >= it->second.bytes || bytes > it->second.bytes - offset)
            throw std::out_of_range("PhysX mapped report view crosses source root");
        return {&it->second, static_cast<std::size_t>(offset)};
    }

    void cleanup_noexcept() noexcept {
        try { source_.submit(); } catch (...) {}
        while (!roots_.empty()) {
            const auto it = roots_.begin();
            const auto device = it->second.device;
            // Usually nonblocking: the Context owns the backing resource
            // until its last submitted fence completes. If retirement
            // bookkeeping fails during exception unwinding, destroy the
            // native buffer explicitly rather than dropping its last ID.
            try {
                context_.retire_buffer(device);
            } catch (...) {
                try { context_.destroy_buffer(device); }
                catch (...) {
                    // Context failure is unrecoverable during a noexcept
                    // destructor. Normal deallocate() remains retryable:
                    // it does not erase ownership on retirement failure.
                }
            }
            try { addresses_.unregister_allocation(it->second.virtual_root); }
            catch (...) { /* May already have been invalidated. */ }
            ::operator delete(it->second.host, std::align_val_t{256});
            roots_.erase(it);
        }
    }

    compute::Context& context_;
    CudaDeviceAddressMap& addresses_;
    SourceSapDmaNative& source_;
    std::map<std::uintptr_t, Entry> roots_;
};

} // namespace fyzz::physx_gpu
