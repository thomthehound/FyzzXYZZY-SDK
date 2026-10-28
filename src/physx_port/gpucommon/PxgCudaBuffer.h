// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef PXG_CUDA_BUFFER_H
#define PXG_CUDA_BUFFER_H

#include "foundation/PxPreprocessor.h"
#include "foundation/PxAssert.h"
#include "foundation/PxAllocator.h"

#include <cstdint>

// The original PhysX CUdeviceptr stores an opaque 64-bit device address.
// This source-port definition is a virtual address resolved by the native
// compute adapter, not a CUDA driver allocation.
using CUdeviceptr = unsigned long long;
static_assert(sizeof(CUdeviceptr) == sizeof(std::uintptr_t),
              "Native PhysX source buffer requires a 64-bit host address ABI");
#include "physx_port/gpucommon/include/PxgDevicePointer.h"

namespace fyzz { namespace physx_gpu { class SourceSapDmaNative; } }

namespace physx {

class PxgHeapMemoryAllocator;

// Original PhysX buffer ownership, allocation growth, raw-pointer identity,
// deferred free, swap, and transfer semantics. Only its CUDA stream/copy
// dependency is lowered onto the native source stream.
class PxgCudaBuffer {
    PX_NOCOPY(PxgCudaBuffer)
public:
    PxgCudaBuffer(PxgHeapMemoryAllocator& deviceAlloc, PxI32 statGroup);
    ~PxgCudaBuffer();

    void allocate(PxU64 size, const char* filename, PxI32 line);
    void allocateCopyOldDataAsync(PxU64 size,
                                  fyzz::physx_gpu::SourceSapDmaNative& stream,
                                  const char* filename, PxI32 line);

    void deallocate();
    void deallocateDeferred();

    PX_FORCE_INLINE CUdeviceptr getDevicePtr() const { return (mPtr + 127) & (~CUdeviceptr(127)); }
    PX_FORCE_INLINE PxU64 getSize() const { return mSize; }

    static void swapBuffer(PxgCudaBuffer& buf0, PxgCudaBuffer& buf1) {
        // Original heap keys are the raw mPtr values, never aligned views.
        const CUdeviceptr ptr = buf0.mPtr;
        const PxU64 size = buf0.mSize;
        buf0.mPtr = buf1.mPtr;
        buf0.mSize = buf1.mSize;
        buf1.mPtr = ptr;
        buf1.mSize = size;
    }

    void assign(PxgCudaBuffer& other) {
        PX_ASSERT(&mDeviceAlloc == &other.mDeviceAlloc);
        PX_ASSERT(mStatGroup == other.mStatGroup);
        if (this == &other) return;
        deallocate();
        mPtr = other.mPtr;
        mSize = other.mSize;
        other.mPtr = 0;
        other.mSize = 0;
    }

protected:
    CUdeviceptr mPtr;
    PxgHeapMemoryAllocator& mDeviceAlloc;
    PxU64 mSize;
    const PxU32 mStatGroup;
};

template <typename T>
class PxgTypedCudaBuffer : public PxgCudaBuffer {
public:
    PxgTypedCudaBuffer(PxgHeapMemoryAllocator& alloc, PxI32 group)
        : PxgCudaBuffer(alloc, group) {}
    PX_FORCE_INLINE void allocateElements(PxU64 count, const char* file, PxI32 line) {
        allocate(count * sizeof(T), file, line);
    }
    PX_FORCE_INLINE PxU64 getNbElements() const { return mSize / sizeof(T); }
    PX_FORCE_INLINE PxgDevicePointer<T> getTypedDevicePtr() const {
        return PxgDevicePointer<T>(getDevicePtr());
    }
    PX_FORCE_INLINE T* getTypedPtr() const {
        return reinterpret_cast<T*>(getDevicePtr());
    }
};

template <unsigned int NbBuffers>
class PxgCudaBufferN {
    alignas(PxgCudaBuffer) PxU8 mCudaArrays[sizeof(PxgCudaBuffer) * NbBuffers];
public:
    PxgCudaBufferN(PxgHeapMemoryAllocator& alloc, PxI32 group) {
        auto* buffers = reinterpret_cast<PxgCudaBuffer*>(mCudaArrays);
        for (PxU32 i = 0; i < NbBuffers; ++i)
            PX_PLACEMENT_NEW(&buffers[i], PxgCudaBuffer)(alloc, group);
    }
    ~PxgCudaBufferN() {
        auto* buffers = reinterpret_cast<PxgCudaBuffer*>(mCudaArrays);
        for (PxU32 i = 0; i < NbBuffers; ++i)
            buffers[i].~PxgCudaBuffer();
    }
    PxgCudaBuffer& operator[](PxU32 i) {
        PX_ASSERT(i < NbBuffers);
        return reinterpret_cast<PxgCudaBuffer*>(mCudaArrays)[i];
    }
    const PxgCudaBuffer& operator[](PxU32 i) const {
        PX_ASSERT(i < NbBuffers);
        return reinterpret_cast<const PxgCudaBuffer*>(mCudaArrays)[i];
    }
    PxgCudaBuffer* begin() { return reinterpret_cast<PxgCudaBuffer*>(mCudaArrays); }
    PxU32 size() { return NbBuffers; }
};

} // namespace physx
#endif
