// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef	PXG_HEAP_MEM_ALLOCATOR_H
#define	PXG_HEAP_MEM_ALLOCATOR_H

#include "foundation/PxAssert.h"
#include "foundation/PxBitUtils.h"
#include "foundation/PxHashMap.h"
#include "foundation/PxArray.h"
#include "foundation/PxPool.h"
#include "foundation/PxMutex.h"
#include "CmVirtualAllocatorCallback.h"
#include "PxsHeapStats.h"
#include "PxsHeapMemoryAllocator.h"
#include "common/PxPhysXCommonConfig.h"

#if PX_DEBUG
#include "PxgMemoryTracker.h"
#endif

namespace physx
{
	class PxgMemoryManager;
	// Native callbacks use the existing Cm::VirtualAllocatorCallback ABI.

	class BlockHeader
	{
	public:
		PX_FORCE_INLINE BlockHeader(const PxU32 offset, PxU32 rootIndex) : mOffset(offset), mRootIndex(rootIndex), mPrev(NULL), mNext(NULL)
		{
		}

		PX_FORCE_INLINE void initialize(const PxU32 rootIndex, const PxU32 offset)
		{
			mOffset = offset;
			mRootIndex = rootIndex;
			mPrev = NULL;
			mNext = NULL;
		}

		PxU32 mOffset;
		PxU32 mRootIndex;
		BlockHeader* mPrev;
		BlockHeader* mNext;
	};

#define PXG_INVALID_BLOCK 0xFFFFFFFF

	class Block
	{
	public:
		PX_FORCE_INLINE Block() : mStartHeader(NULL), mEndHeader(NULL), mHeaderSizes(0) {}
		
		PX_FORCE_INLINE bool isEmpty() { return mHeaderSizes == 0; }
		void insertBlockHeader(const PxU32 rootIndex, const PxU32 offset, PxPool<BlockHeader>& pool);
		void removeBlockHeader(BlockHeader* header, PxPool<BlockHeader>& pool);
		BlockHeader* findBuddy(const PxU32 offsetToFind, const PxU32 rootIndex);
		BlockHeader* getFreeBlocks(){ return mEndHeader; }

		bool isValid();

		BlockHeader* mStartHeader;
		BlockHeader* mEndHeader;
		PxU32 mHeaderSizes;
		PxU32 mBlockSize;
		PxU32 mBlockIndex;
	};

	class AllocationValue
	{
	public:
		PX_FORCE_INLINE AllocationValue(const PxU32 blockIndex, const PxU32 rootIndex, const size_t byteSize, const int group)
			: mBlockIndex(blockIndex), mRootIndex(rootIndex), mByteSize(byteSize), mGroup(group)
		{}
		PxU32 mBlockIndex;
		PxU32 mRootIndex;
		size_t mByteSize;
		int mGroup;
	};

	struct ExceptionalAlloc
	{
		void* address;
		size_t size;
	};

	class PxgHeapMemoryAllocator : public Cm::VirtualAllocatorCallback, public PxUserAllocated
	{
	public:

		using AbortCallback = void (*)(void*) noexcept;
		// This is the ORIGINAL PhysX buddy heap with its CUDA-context dependency
		// injected, not a new allocator. The callback must mark its owning scene
		// aborted if an internal duplicate raw allocation key is discovered.
		PxgHeapMemoryAllocator(const PxU32 byteSize,
			Cm::VirtualAllocatorCallback& allocator,
			AbortCallback abortCallback, void* abortUserData);
		~PxgHeapMemoryAllocator();
	
		void initializeBlocks(const PxU32 rootIndex);
		//return a free block index
		PxU32 getNextFreeBlock(const PxU32 blockIndex, const PxU32 allocationSize, const char* file, const int line);
	
		// Cm::VirtualAllocatorCallback
		virtual	void* allocate(const size_t byteSize, const int group, const char* file, const int line)	PX_OVERRIDE;
		virtual	void deallocate(void* ptr)																	PX_OVERRIDE;
		//~Cm::VirtualAllocatorCallback

		void deallocateDeferred(void* ptr);

		void flushDeferredDeallocs();

		PxU64 getTotalSize();
		PxsHeapStats& getHeapStats() { return mHeapStats; }

		//Used for memcheck support
#if PX_DEBUG || PX_STOMP_ALLOCATED_MEMORY
		Cm::VirtualAllocatorCallback* getAllocator() { return mAllocator; }
#endif

	private:

		PxHashMap<void*, AllocationValue> mHashMap;//this is used to look up where the block is
		PxArray<Block> mBlocks; //this is used to store fix size slots 
		Cm::VirtualAllocatorCallback* mAllocator;
		AbortCallback mAbortCallback;
		void* mAbortUserData;
		PxArray<void*> mRoots;
		PxArray<ExceptionalAlloc> mExceptionalAllocs;
		PxArray<void*> deferredDeallocs;
		PxU32 mAllocationSize;
		PxU32 mBitfield;
		PxU64 mTotalMem;
		PxsHeapStats mHeapStats;

		PxPool<BlockHeader> mBlockHeaderPool;

		PxMutex mMutex;

#if PX_DEBUG
		MemTracker mMemTracker;
#endif
		PX_NOCOPY(PxgHeapMemoryAllocator)
	};

    // The original PxgHeapMemoryAllocatorManager lifetime and host-view
    // aliasing rules, with its CUDA PxgMemoryManager factory dependency
    // replaced by three borrowed, backend-neutral source allocator callbacks.
    // The two host callbacks MUST implement actual host-pinned / GPU-mapped
    // guarantees before a live GPU scene uses these addresses.
    class PxgHeapMemoryAllocatorManager final
#if PX_SUPPORT_GPU_PHYSX
        : public PxsHeapMemoryAllocatorManager
#else
        : public PxUserAllocated
#endif
    {
    public:
        PxgHeapMemoryAllocatorManager(
            PxU32 heapCapacity,
            Cm::VirtualAllocatorCallback& deviceRoots,
            Cm::VirtualAllocatorCallback& hostPinnedRoots,
            Cm::VirtualAllocatorCallback& hostMappedRoots,
            PxgHeapMemoryAllocator::AbortCallback abort,
            void* abortData);
        ~PxgHeapMemoryAllocatorManager()
#if PX_SUPPORT_GPU_PHYSX
            override
#endif
        ;

        PxU64 getDeviceMemorySize() const
#if PX_SUPPORT_GPU_PHYSX
            override
#endif
        ;
        PxsHeapStats getDeviceHeapStats() const
#if PX_SUPPORT_GPU_PHYSX
            override
#endif
        ;
        void flushDeferredDeallocs()
#if PX_SUPPORT_GPU_PHYSX
            override
#endif
        ;

        PxgHeapMemoryAllocator* mDeviceMemoryAllocator = nullptr;
#if !PX_SUPPORT_GPU_PHYSX
        // These slots live in PxsHeapMemoryAllocatorManager when GPU PhysX
        // is enabled, but are absent from the CPU-only source build.
        Cm::VirtualAllocatorCallback* mPinnedHostMemoryAllocator = nullptr;
        Cm::VirtualAllocatorCallback* mPinnedHostMappedMemoryAllocator = nullptr;
#endif

    private:
        PX_NOCOPY(PxgHeapMemoryAllocatorManager)
    };

}

#endif
