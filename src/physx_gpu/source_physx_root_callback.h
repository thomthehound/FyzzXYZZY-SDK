// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/source_physx_heap_roots.h"
#include "CmVirtualAllocatorCallback.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace fyzz::physx_gpu {

// Exact allocator virtual interface used by PhysX's GPU heap:
// Cm::VirtualAllocatorCallback::{allocate(size_t,int,const char*,int),
// deallocate(void*)}. The source's PxgCudaDeviceMemoryAllocatorCallback is
// a CUDA-context-aware subclass; that subclass must be lowered in the
// independent PhysX-derived runtime before this is installed in PxgHeap.
//
// The source buddy allocator, NOT this callback, decides block splits,
// exceptional roots, deferred block frees, reuse and accounting. This
// callback is called only when its original source allocator requests
// or releases a physical heap page / exceptional root.
//
// Never CPU-dereference the returned opaque device pointer.
class SourcePhysxDeviceRootCallback final
    : public physx::Cm::VirtualAllocatorCallback {
public:
    explicit SourcePhysxDeviceRootCallback(
        SourcePhysxHeapRoots& roots,
        SourceSapDmaNative* stream = nullptr) noexcept
        : roots_(roots), stream_(stream) {}

    SourcePhysxDeviceRootCallback(const SourcePhysxDeviceRootCallback&) = delete;
    SourcePhysxDeviceRootCallback& operator=(const SourcePhysxDeviceRootCallback&) = delete;

    void* allocate(std::size_t size, int, const char*, int) override {
        if (!size) return nullptr;
        const auto address = roots_.allocate_root(size);
        // PhysX's original allocator has a void* API even for GPU roots;
        // 64-bit opaque CUdeviceptr arithmetic must survive this round trip.
        if (address > std::numeric_limits<std::uintptr_t>::max()) {
            roots_.release_root(address, stream_);
            throw std::overflow_error("Original PhysX GPU address cannot fit host pointer ABI");
        }
        return reinterpret_cast<void*>(static_cast<std::uintptr_t>(address));
    }

    void deallocate(void* ptr) override {
        if (!ptr) return;
        roots_.release_root(
            static_cast<SourcePhysxHeapRoots::Address>(
                reinterpret_cast<std::uintptr_t>(ptr)),
            stream_);
    }

private:
    SourcePhysxHeapRoots& roots_;
    SourceSapDmaNative* stream_;
};

} // namespace fyzz::physx_gpu
