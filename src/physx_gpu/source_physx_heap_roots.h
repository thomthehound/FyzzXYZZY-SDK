// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace fyzz::physx_gpu {

// Native backing for ROOT allocations made by the original
// PxgCudaAllocatorCallbackBase::allocate / deallocate path.
//
// This is not a replacement for PxgHeapMemoryAllocator. The original PhysX
// heap remains responsible for its 128-byte buddy splits, exceptional roots,
// raw mPtr keys, in-heap reuse, deferred block frees and simulation-step flush.
// It can do its normal root+offset arithmetic on these opaque CUDA-style
// virtual addresses; CudaDeviceAddressMap::resolve resolves interior addresses
// to the actual native resource. Do not dereference these as CPU pointers.
//
// Roots are owned here (unlike borrow-only source allocation slices).
// The original heap callback forwards root allocate/deallocate events through
// this boundary. The caller still owns stream/fetch/task lifecycle.
class SourcePhysxHeapRoots final {
public:
    using Address = CudaDeviceAddressMap::Address;

    SourcePhysxHeapRoots(compute::Context& context,
                         CudaDeviceAddressMap& addresses) noexcept
        : context_(context), addresses_(addresses) {}

    SourcePhysxHeapRoots(const SourcePhysxHeapRoots&) = delete;
    SourcePhysxHeapRoots& operator=(const SourcePhysxHeapRoots&) = delete;

    // One PhysX heap page or exceptional root, not one PxgCudaBuffer block.
    // Same GPU-local allocation lifecycle as the original device allocator,
    // but returns a stable CUDA-style virtual root, never a native BufferId.
    Address allocate_root(std::size_t bytes) {
        if (!bytes)
            throw std::invalid_argument("PhysX heap root cannot be zero sized");
        const auto buffer =
            context_.create_buffer(bytes, compute::MemoryPlacement::DeviceLocal);
        Address root = 0;
        try {
            root = addresses_.register_allocation(buffer, bytes);
            roots_.emplace(root, buffer);
        } catch (...) {
            if (root) addresses_.unregister_allocation(root);
            context_.destroy_buffer(buffer);
            throw;
        }
        return root;
    }

    // Called for EXACT ROOT identities returned to the original allocator,
    // not the buddy suballocation pointers passed to PxgCudaBuffer::deallocate.
    // If the original source has queued native commands, submit (not wait)
    // before retirement so they retain their original GPU buffer identity.
    // The Context retains in-flight memory until its queue fence completes.
    void release_root(Address root, SourceSapDmaNative* stream = nullptr) {
        const auto it = roots_.find(root);
        if (it == roots_.end())
            throw std::out_of_range("Free requires original PhysX heap root");
        if (stream) stream->submit();
        // Do the potentially failing retirement BEFORE invalidating the
        // original CUDA address; no stale mapping after successful retire.
        context_.retire_buffer(it->second);
        addresses_.unregister_allocation(root);
        roots_.erase(it);
    }

    std::size_t live_roots() const noexcept { return roots_.size(); }

private:
    compute::Context& context_;
    CudaDeviceAddressMap& addresses_;
    std::map<Address, compute::BufferId> roots_;
};

} // namespace fyzz::physx_gpu
