// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpucommon/PxgCudaBuffer.h"
#include "physx_port/gpucommon/PxgHeapMemAllocator.h"
#include "physx_gpu/source_sap_dma_native.h"

#include "foundation/PxMath.h"

#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>

namespace physx {

PxgCudaBuffer::PxgCudaBuffer(PxgHeapMemoryAllocator& deviceAlloc, PxI32 statGroup)
    : mPtr(0), mDeviceAlloc(deviceAlloc), mSize(0), mStatGroup(statGroup) {}

PxgCudaBuffer::~PxgCudaBuffer() { deallocate(); }

void PxgCudaBuffer::allocate(PxU64 size, const char* filename, PxI32 line) {
    if (mSize >= size) return;
    if (mSize > std::numeric_limits<PxU64>::max() / 2u)
        throw std::overflow_error("PhysX GPU buffer growth overflows");
    const PxU64 capacity = PxMax(size, mSize * 2u);
    if (capacity > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("PhysX GPU buffer exceeds native address range");

    // Do not discard a live raw allocation on OOM. Normal source policy
    // remains doubling and freeing the previous raw key, without a copy.
    void* next = mDeviceAlloc.allocate(
        static_cast<std::size_t>(capacity), mStatGroup, filename, line);
    if (!next) throw std::bad_alloc();
    if (mPtr)
        mDeviceAlloc.deallocate(reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(mPtr)));
    mPtr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(next));
    mSize = capacity;
}

void PxgCudaBuffer::allocateCopyOldDataAsync(
    PxU64 size, fyzz::physx_gpu::SourceSapDmaNative& stream,
    const char* filename, PxI32 line) {
    if (size <= mSize) return;
    if (mSize > std::numeric_limits<PxU64>::max() / 2u)
        throw std::overflow_error("PhysX GPU buffer copy growth overflows");
    const PxU64 capacity = PxMax(size, mSize * 2u);
    if (capacity > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("PhysX GPU buffer exceeds native address range");

    void* next = mDeviceAlloc.allocate(
        static_cast<std::size_t>(capacity), mStatGroup, filename, line);
    if (!next) throw std::bad_alloc();
    const auto next_raw =
        static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(next));
    try {
        if (mPtr && mSize)
            stream.queue_device_copy(next_raw, mPtr,
                static_cast<std::size_t>(mSize));
    } catch (...) {
        mDeviceAlloc.deallocate(next);
        throw;
    }

    // The old allocation is still GPU-visible to the ordered copy. The
    // original buddy heap, not the native address map, decides when its
    // deferred blocks can be reused.
    if (mPtr && mSize)
        mDeviceAlloc.deallocateDeferred(reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(mPtr)));
    mPtr = next_raw;
    mSize = capacity;
}

void PxgCudaBuffer::deallocate() {
    if (mPtr && mSize) {
        mDeviceAlloc.deallocate(reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(mPtr)));
        mPtr = 0;
        mSize = 0;
    }
}

void PxgCudaBuffer::deallocateDeferred() {
    if (mPtr && mSize) {
        mDeviceAlloc.deallocateDeferred(reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(mPtr)));
        mPtr = 0;
        mSize = 0;
    }
}

} // namespace physx
