// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "PxgHeapMemAllocator.h"

#include <memory>

using namespace physx;

// Original PhysX PxgHeapMemoryAllocatorManager policy, preserved: one device
// heap, separate pinned and mapped host heaps in zero-capacity debug mode,
// and one shared *mapped* host heap in ordinary mode. The original
// PxgMemoryManager CUDA allocator selection is supplied here by caller-owned
// Cm::VirtualAllocatorCallback implementations instead.
PxgHeapMemoryAllocatorManager::PxgHeapMemoryAllocatorManager(
    PxU32 heapCapacity,
    Cm::VirtualAllocatorCallback& deviceRoots,
    Cm::VirtualAllocatorCallback& hostPinnedRoots,
    Cm::VirtualAllocatorCallback& hostMappedRoots,
    PxgHeapMemoryAllocator::AbortCallback abort,
    void* abortData)
{
    // Construction may fail halfway through scene setup. Keep partial heaps
    // owned locally until all required source allocations succeed; they must
    // not leak a CUDA-style root if construction unwinds.
    std::unique_ptr<PxgHeapMemoryAllocator> device(
        PX_NEW(PxgHeapMemoryAllocator)(heapCapacity, deviceRoots, abort, abortData));
    std::unique_ptr<PxgHeapMemoryAllocator> host;
    std::unique_ptr<PxgHeapMemoryAllocator> mapped;

    if (heapCapacity == 0u)
    {
        host.reset(PX_NEW(PxgHeapMemoryAllocator)(
            heapCapacity, hostPinnedRoots, abort, abortData));
        mapped.reset(PX_NEW(PxgHeapMemoryAllocator)(
            heapCapacity, hostMappedRoots, abort, abortData));
    }
    else
    {
        // Source uses the mapped allocator for BOTH exposed host views in
        // normal mode. Creating two heaps or selecting the plain pinned
        // allocator here would change identity, ownership and mapping.
        mapped.reset(PX_NEW(PxgHeapMemoryAllocator)(
            heapCapacity, hostMappedRoots, abort, abortData));
    }

    mDeviceMemoryAllocator = device.release();
    if (heapCapacity == 0u)
    {
        mPinnedHostMemoryAllocator = host.release();
        mPinnedHostMappedMemoryAllocator = mapped.release();
    }
    else
    {
        mPinnedHostMappedMemoryAllocator = mapped.release();
        mPinnedHostMemoryAllocator = mPinnedHostMappedMemoryAllocator;
    }
}

PxgHeapMemoryAllocatorManager::~PxgHeapMemoryAllocatorManager()
{
    delete mDeviceMemoryAllocator;
    if (mPinnedHostMappedMemoryAllocator != mPinnedHostMemoryAllocator)
        delete mPinnedHostMappedMemoryAllocator;
    delete mPinnedHostMemoryAllocator;
}

PxU64 PxgHeapMemoryAllocatorManager::getDeviceMemorySize() const
{
    return mDeviceMemoryAllocator ? mDeviceMemoryAllocator->getTotalSize() : 0u;
}

PxsHeapStats PxgHeapMemoryAllocatorManager::getDeviceHeapStats() const
{
    return mDeviceMemoryAllocator ? mDeviceMemoryAllocator->getHeapStats()
                                  : PxsHeapStats();
}

void PxgHeapMemoryAllocatorManager::flushDeferredDeallocs()
{
    if (mDeviceMemoryAllocator)
        mDeviceMemoryAllocator->flushDeferredDeallocs();
}
