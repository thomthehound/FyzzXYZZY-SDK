// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_sap_mapped_report_view.h"
#include "physx_port/common/src/CmVirtualAllocatorCallback.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>

namespace fyzz::physx_gpu {

// One original PhysX Cm::VirtualAllocatorCallback for the source
// pinned/mapped host heap: GPU-WRITTEN/CPU-READ broadphase reports AND
// CPU-WRITTEN/GPU-READ pinned input descriptors. The GPU heap manager and
// PxgHeapMemoryAllocator own the host buddy blocks. This callback owns ONLY
// lower-level host roots, native GPU resources, and GPU address aliases.
//
// PreferDirect: GPU writes directly to a host-visible native resource,
// which is mapped once and supplies PhysX's real CPU allocation pointer.
// No separate shadow pair list, staged download, memcpy, or CPU polling.
// The CPU must not dereference reports until source mStream's GPU fence
// is complete at the original fetch/publication boundary.
//
// ForceStaged: separate CPU root and native GPU resource, with publication
// ONLY after mStream's original fence. Used when the native device does
// not expose GPU-writable, CPU-readable mapped storage, and for conformance.
//
// This is NOT a generic coherent allocation callback for arbitrary CUDA
// pointers/flags. Original mEventMapped busy polling is not supported;
// BroadPhaseCompletionGate owns source GPU completion and task publication.
class SourcePhysxMappedReportRoots final : public physx::Cm::VirtualAllocatorCallback,
                                         public SourceSapMappedReportView {
public:
    using Address = CudaDeviceAddressMap::Address;
    enum class Mode { PreferDirect, ForceStaged };

    SourcePhysxMappedReportRoots(compute::Context& context,
                                 CudaDeviceAddressMap& addresses,
                                 SourceSapDmaNative& source,
                                 Mode mode = Mode::PreferDirect) noexcept
        : context_(context), addresses_(addresses), source_(source), mode_(mode) {}

    SourcePhysxMappedReportRoots(const SourcePhysxMappedReportRoots&) = delete;
    SourcePhysxMappedReportRoots& operator=(const SourcePhysxMappedReportRoots&) = delete;
    ~SourcePhysxMappedReportRoots() override { cleanup_noexcept(); }

    void* allocate(std::size_t bytes, int, const char*, int) override {
        if (!bytes) return nullptr;
        if (!source_.bound_to(context_, addresses_))
            throw std::invalid_argument("PhysX mapped report stream uses another compute context");

        Entry entry{};
        entry.bytes = bytes;
        std::size_t backing_bytes = bytes;
        try {
            bool direct = mode_ == Mode::PreferDirect &&
                context_.device_info().supports_gpu_writable_host_visible;
            if (direct) {
                // The adapter's general memory-type probe can be more
                // permissive than a particular Vulkan STORAGE_BUFFER's
                // memoryTypeBits. Only an explicit unsupported-placement
                // error may select staged fallback; OOM must propagate.
                if (bytes > std::numeric_limits<std::size_t>::max() - 255u)
                    throw std::overflow_error("Mapped PhysX host root size overflows alignment");
                backing_bytes = bytes + 255u;
                try {
                    entry.device = context_.create_buffer(
                        backing_bytes, compute::MemoryPlacement::HostVisible);
                } catch (const compute::UnsupportedMemoryPlacement&) {
                    direct = false;
                    backing_bytes = bytes;
                }
            }
            if (direct) {
                // CUDA host-root alignment and original PhysX buddy-block
                // arithmetic must remain 256-byte-safe even if a native
                // Map() pointer has weaker platform alignment guarantees.
                // In normal PhysX mode the manager aliases hostAlloc and
                // hostMappedAlloc: original pinned descriptors can be
                // CPU-written while actor reports are GPU-written. Both
                // directions are part of this host root's contract.
                auto* base = context_.map_buffer(
                    entry.device, compute::HostMapAccess::ReadWrite);
                entry.cpu_mapped = true;
                const auto raw = reinterpret_cast<std::uintptr_t>(base);
                if (raw > std::numeric_limits<std::uintptr_t>::max() - 255u)
                    throw std::overflow_error("Mapped PhysX host root pointer overflows alignment");
                const auto aligned = (raw + 255u) & ~std::uintptr_t{255u};
                entry.host = reinterpret_cast<void*>(aligned);
                entry.gpu_offset = static_cast<std::size_t>(aligned - raw);
                entry.direct = true;
            } else {
                // This is the ONLY path allocating a separate CPU root.
                entry.host = ::operator new(bytes, std::align_val_t{256});
                entry.device = context_.create_buffer(
                    bytes, compute::MemoryPlacement::DeviceLocal);
            }

            entry.virtual_root = addresses_.register_allocation(
                entry.device, backing_bytes);
            const auto inserted = roots_.emplace(
                reinterpret_cast<std::uintptr_t>(entry.host), entry);
            if (!inserted.second)
                throw std::logic_error("PhysX mapped report root pointer already live");
        } catch (...) {
            if (entry.virtual_root) {
                try { (void)addresses_.unregister_allocation(entry.virtual_root); }
                catch (...) {}
            }
            if (entry.device) {
                if (entry.cpu_mapped) {
                    try { context_.unmap_buffer(entry.device); } catch (...) {}
                }
                // The resource was never submitted to mStream before
                // allocation returned; synchronous destroy is safe here.
                try { context_.destroy_buffer(entry.device); } catch (...) {}
            }
            if (entry.host && !entry.direct)
                ::operator delete(entry.host, std::align_val_t{256});
            throw;
        }
        return entry.host;
    }

    void deallocate(void* host) override {
        if (!host) return;
        const auto it = roots_.find(reinterpret_cast<std::uintptr_t>(host));
        if (it == roots_.end())
            throw std::out_of_range("PhysX mapped report free requires the exact host root");

        // Submit all source commands referencing this root before retirement.
        // This does not block on a GPU fence.
        source_.submit();
        if (it->second.cpu_mapped) {
            context_.unmap_buffer(it->second.device);
            it->second.cpu_mapped = false;
        }
        // May throw without losing ownership: the entry and CUDA address
        // remain available for teardown retry, even if Map() was released.
        context_.retire_buffer(it->second.device);
        addresses_.unregister_allocation(it->second.virtual_root);
        if (!it->second.direct)
            ::operator delete(it->second.host, std::align_val_t{256});
        roots_.erase(it);
    }

    Address device_address(const void* host, std::size_t bytes) const override {
        if (!host || !bytes)
            throw std::invalid_argument("PhysX mapped report GPU view must be nonempty");
        const auto location = find(host, bytes);
        return location.root->virtual_root + location.root->gpu_offset +
               location.offset;
    }

    bool is_direct(const void* host, std::size_t bytes = 1) const {
        return find(host, bytes).root->direct;
    }

    // Source mFoundActorPairsMapped/mLostActorPairsMapped CPU arrays become
    // valid only after the source GPU queue fence at the original fetch.
    // Direct: no copies, the *same memory* was written by GPU.
    // The same source heap also owns CPU-written pinned host inputs; their
    // visibility to GPU is ordered by the caller's source submission.
    // Staged: fetch then download into the *original PhysX CPU array*.
    void publish_at_fetch(void* host, std::size_t bytes) override {
        if (!host || !bytes) return;
        if (!source_.bound_to(context_, addresses_))
            throw std::invalid_argument("Foreign source stream for mapped report fetch");
        const auto location = find(host, bytes);
        source_.wait_at_fetch(); // backend fence wait, never a polling loop
        if (!location.root->direct)
            context_.download(location.root->device,
                              location.root->gpu_offset + location.offset,
                              host, bytes);
    }

    std::size_t live_roots() const noexcept { return roots_.size(); }

private:
    struct Entry {
        void* host = nullptr;                  // original PhysX heap CPU root
        std::size_t bytes = 0;                 // source heap root capacity
        compute::BufferId device = 0;         // native backing resource
        Address virtual_root = 0;             // exact CUDA-style allocation root
        std::size_t gpu_offset = 0;           // aligned CPU base -> GPU offset
        bool direct = false;
        bool cpu_mapped = false;
    };
    struct Location {
        const Entry* root;
        std::size_t offset;
    };

    Location find(const void* host, std::size_t bytes) const {
        if (!host || !bytes)
            throw std::invalid_argument("Empty PhysX mapped report host address");
        const auto ptr = reinterpret_cast<std::uintptr_t>(host);
        auto it = roots_.upper_bound(ptr);
        if (it == roots_.begin())
            throw std::out_of_range("Unknown PhysX mapped report host pointer");
        --it;
        const auto offset = ptr - it->first;
        if (offset >= it->second.bytes || bytes > it->second.bytes - offset)
            throw std::out_of_range("PhysX mapped report view exceeds source heap root");
        return {&it->second, static_cast<std::size_t>(offset)};
    }

    void cleanup_noexcept() noexcept {
        try { source_.submit(); } catch (...) {}
        while (!roots_.empty()) {
            const auto it = roots_.begin();
            if (it->second.cpu_mapped) {
                try { context_.unmap_buffer(it->second.device); } catch (...) {}
            }
            try {
                context_.retire_buffer(it->second.device);
            } catch (...) {
                // Rare retirement bookkeeping failure during scene unwind:
                // use the existing blocking destroy fallback rather than
                // dropping the only reference to a GPU allocation.
                try { context_.destroy_buffer(it->second.device); } catch (...) {}
            }
            try { (void)addresses_.unregister_allocation(it->second.virtual_root); }
            catch (...) {}
            if (!it->second.direct)
                ::operator delete(it->second.host, std::align_val_t{256});
            roots_.erase(it);
        }
    }

    compute::Context& context_;
    CudaDeviceAddressMap& addresses_;
    SourceSapDmaNative& source_;
    Mode mode_;
    std::map<std::uintptr_t, Entry> roots_;
};

} // namespace fyzz::physx_gpu
