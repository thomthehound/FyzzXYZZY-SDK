// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_port/gpubroadphase/PxgSapDmaUpFrameNative.h"
#include "physx_port/gpubroadphase/PxgSapPreBroadPhaseNative.h"
#include "physx_port/gpubroadphase/PxgSapUpdateFetchNative.h"
#include "physx_port/gpucommon/PxgHeapMemAllocator.h"
#include "PxgBroadPhasePairReport.h"
#include "CmPinnableObject.h"
#include "CmPinnableArray.h"
#include <array>
#include <cstdint>

namespace fyzz::physx_gpu {

// Original PxgCudaBroadPhaseSap member block, with only CUDA context,
// stream and pinnable-host dependencies replaced by the existing native
// Context/SourceSapDmaNative and borrowed original mapped-host views.
// All GPU buffers use the ORIGINAL PxgCudaBuffer and PhysX buddy heap.
// No parallel endpoint/rank/report storage and no independent scheduler.
class PxgSapSourceMembersNative final {
public:
    PxgSapSourceMembersNative(physx::PxgAllocatorDesc& allocators,
                              physx::PxU32 max_pairs,
                              physx::PxU32 max_aggregate_pairs);
    PxgSapSourceMembersNative(const PxgSapSourceMembersNative&)=delete;
    PxgSapSourceMembersNative& operator=(const PxgSapSourceMembersNative&)=delete;

    SourceSapPreBroadPhaseBuffers pre_broadphase_buffers() noexcept;
    SourceSapDmaUpFrameMembers dma_up_members(
        const SourceSapAabbManagerDescriptorView* aabb_manager,
        CUdeviceptr original_found_map,
        CUdeviceptr original_lost_map);
    SourceSapFetchHistoryBuffers fetch_histories() noexcept;

    physx::PxgBroadPhaseDesc& descriptor() noexcept {return pinnedDescriptor_.get();}
    const physx::PxgBroadPhaseDesc& descriptor() const noexcept {return pinnedDescriptor_.get();}
    physx::PxgCudaBuffer& new_integer_bounds() noexcept {return newIntegerBounds_;}
    physx::PxgCudaBuffer& old_integer_bounds() noexcept {return oldIntegerBounds_;}
    bool& force_update() noexcept {return forceUpdate_;}
    physx::PxU32 num_boxes() const noexcept {return numBoxes_;}
    physx::PxgBroadPhasePair* mapped_found_actor_pairs() noexcept {
        return mappedFound_.begin();
    }
    physx::PxgBroadPhasePair* mapped_lost_actor_pairs() noexcept {
        return mappedLost_.begin();
    }
    physx::PxU32 max_found_lost_pairs() const noexcept {return maxPairs_;}


private:
    physx::PxU32 numBoxes_=0, boxesCapacity_=0;
    physx::PxU32 createdCount_=0, removedCount_=0, updatedCount_=0;
    physx::PxU32 maxPairs_=0,maxAggregatePairs_=0;
    physx::PxU32 regionAccumTotal_=0,startRegionAccumTotal_=0;
    physx::regionOverlapType overlapChecksTotal_=0;
    bool forceUpdate_=true;
    physx::Cm::PinnableObject<physx::PxgBroadPhaseDesc> pinnedDescriptor_;
    physx::Cm::PinnableArray<physx::PxgRadixSortDesc> radixDescs_,radixWorDescs_;
    physx::Cm::PinnableArray<physx::PxgBroadPhasePair> mappedFound_,mappedLost_;
    physx::PxgTypedCudaBuffer<physx::PxU32> createdHandles_;
    physx::PxgTypedCudaBuffer<physx::PxU32> removedHandles_;
    physx::PxgTypedCudaBuffer<physx::PxU32> updatedHandles_;
    physx::PxgTypedCudaBuffer<physx::PxBounds3> fpBounds_;
    physx::PxgTypedCudaBuffer<physx::PxReal> contactDistances_;
    physx::PxgTypedCudaBuffer<physx::PxU32> groups_;
    physx::PxgTypedCudaBuffer<physx::PxU32> environmentIds_;
    physx::PxgCudaBuffer newIntegerBounds_;
    physx::PxgCudaBuffer oldIntegerBounds_;
    physx::PxgCudaBufferN<3> projections_;
    physx::PxgCudaBufferN<3> projectionRanks_;
    physx::PxgCudaBufferN<3> tempProjections_;
    physx::PxgCudaBufferN<3> tempHandles_;
    physx::PxgCudaBufferN<3> radixCounts_;
    physx::PxgCudaBufferN<3> sapBoxes_;
    physx::PxgCudaBufferN<3> newSapBoxes_;
    physx::PxgCudaBufferN<3> incrementalComparisons_;
    physx::PxgCudaBufferN<3> incrementalBlockComparisons_;
    physx::PxgCudaBufferN<6> projectionHandles_;
    physx::PxgCudaBufferN<6> endHistograms_;
    physx::PxgCudaBufferN<6> blockEndHistograms_;
    physx::PxgCudaBufferN<6> endHandles_;
    physx::PxgCudaBufferN<6> startHistograms_;
    physx::PxgCudaBufferN<6> blockStartHistograms_;
    physx::PxgCudaBufferN<6> startHandles_;
    physx::PxgCudaBufferN<6> totalEndHistograms_;
    physx::PxgCudaBufferN<6> blockTotalEndHistograms_;
    physx::PxgCudaBufferN<2> aggregateReportBlocks_;
    physx::PxgCudaBufferN<2> actorReportBlocks_;
    physx::PxgCudaBuffer activeRegions_;
    physx::PxgCudaBuffer startRegions_;
    physx::PxgCudaBuffer orderedActiveHandles_;
    physx::PxgCudaBuffer orderedStartHandles_;
    physx::PxgCudaBuffer overlapChecks_;
    physx::PxgCudaBuffer blockOverlapChecks_;
    physx::PxgCudaBuffer overlapHandleRegions_;
    physx::PxgCudaBuffer regionRange_;
    physx::PxgCudaBuffer startRegionAccum_;
    physx::PxgCudaBuffer blockStartRegionAccum_;
    physx::PxgCudaBuffer regionAccum_;
    physx::PxgCudaBuffer blockRegionAccum_;
    physx::PxgTypedCudaBuffer<physx::PxgBroadPhasePair> foundPairs_;
    physx::PxgTypedCudaBuffer<physx::PxgBroadPhasePair> lostPairs_;
    physx::PxgTypedCudaBuffer<physx::PxgBroadPhasePair> foundAggregate_;
    physx::PxgTypedCudaBuffer<physx::PxgBroadPhasePair> lostAggregate_;
    physx::PxgTypedCudaBuffer<physx::PxgBroadPhasePair> foundActor_;
    physx::PxgTypedCudaBuffer<physx::PxgBroadPhasePair> lostActor_;
    physx::PxgTypedCudaBuffer<physx::PxgBroadPhaseDesc> broadphaseDescriptorBuffer_;
    physx::PxgTypedCudaBuffer<physx::PxgRadixSortDesc> radixDescriptorBuffer_;
    physx::PxgTypedCudaBuffer<physx::PxgRadixSortDesc> radixWithoutRanksBuffer_;
};
} // namespace fyzz::physx_gpu
