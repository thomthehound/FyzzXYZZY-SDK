// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"
#include <stdexcept>

namespace fyzz::physx_gpu {
PxgSapSourceMembersNative::PxgSapSourceMembersNative(
    physx::PxgHeapMemoryAllocator& heap,physx::PxU32 max_pairs,
    physx::PxU32 max_aggregate_pairs)
    : maxPairs_(max_pairs),maxAggregatePairs_(max_aggregate_pairs),
    createdHandles_(heap,physx::PxsHeapStats::eBROADPHASE),
    removedHandles_(heap,physx::PxsHeapStats::eBROADPHASE),
    updatedHandles_(heap,physx::PxsHeapStats::eBROADPHASE),
    fpBounds_(heap,physx::PxsHeapStats::eBROADPHASE),
    contactDistances_(heap,physx::PxsHeapStats::eBROADPHASE),
    groups_(heap,physx::PxsHeapStats::eBROADPHASE),
    environmentIds_(heap,physx::PxsHeapStats::eBROADPHASE),
    newIntegerBounds_(heap,physx::PxsHeapStats::eBROADPHASE),
    oldIntegerBounds_(heap,physx::PxsHeapStats::eBROADPHASE),
    projections_(heap,physx::PxsHeapStats::eBROADPHASE),
    projectionRanks_(heap,physx::PxsHeapStats::eBROADPHASE),
    tempProjections_(heap,physx::PxsHeapStats::eBROADPHASE),
    tempHandles_(heap,physx::PxsHeapStats::eBROADPHASE),
    radixCounts_(heap,physx::PxsHeapStats::eBROADPHASE),
    sapBoxes_(heap,physx::PxsHeapStats::eBROADPHASE),
    newSapBoxes_(heap,physx::PxsHeapStats::eBROADPHASE),
    incrementalComparisons_(heap,physx::PxsHeapStats::eBROADPHASE),
    incrementalBlockComparisons_(heap,physx::PxsHeapStats::eBROADPHASE),
    projectionHandles_(heap,physx::PxsHeapStats::eBROADPHASE),
    endHistograms_(heap,physx::PxsHeapStats::eBROADPHASE),
    blockEndHistograms_(heap,physx::PxsHeapStats::eBROADPHASE),
    endHandles_(heap,physx::PxsHeapStats::eBROADPHASE),
    startHistograms_(heap,physx::PxsHeapStats::eBROADPHASE),
    blockStartHistograms_(heap,physx::PxsHeapStats::eBROADPHASE),
    startHandles_(heap,physx::PxsHeapStats::eBROADPHASE),
    totalEndHistograms_(heap,physx::PxsHeapStats::eBROADPHASE),
    blockTotalEndHistograms_(heap,physx::PxsHeapStats::eBROADPHASE),
    aggregateReportBlocks_(heap,physx::PxsHeapStats::eBROADPHASE),
    actorReportBlocks_(heap,physx::PxsHeapStats::eBROADPHASE),
    activeRegions_(heap,physx::PxsHeapStats::eBROADPHASE),
    startRegions_(heap,physx::PxsHeapStats::eBROADPHASE),
    orderedActiveHandles_(heap,physx::PxsHeapStats::eBROADPHASE),
    orderedStartHandles_(heap,physx::PxsHeapStats::eBROADPHASE),
    overlapChecks_(heap,physx::PxsHeapStats::eBROADPHASE),
    blockOverlapChecks_(heap,physx::PxsHeapStats::eBROADPHASE),
    overlapHandleRegions_(heap,physx::PxsHeapStats::eBROADPHASE),
    regionRange_(heap,physx::PxsHeapStats::eBROADPHASE),
    startRegionAccum_(heap,physx::PxsHeapStats::eBROADPHASE),
    blockStartRegionAccum_(heap,physx::PxsHeapStats::eBROADPHASE),
    regionAccum_(heap,physx::PxsHeapStats::eBROADPHASE),
    blockRegionAccum_(heap,physx::PxsHeapStats::eBROADPHASE),
    foundPairs_(heap,physx::PxsHeapStats::eBROADPHASE),
    lostPairs_(heap,physx::PxsHeapStats::eBROADPHASE),
    foundAggregate_(heap,physx::PxsHeapStats::eBROADPHASE),
    lostAggregate_(heap,physx::PxsHeapStats::eBROADPHASE),
    foundActor_(heap,physx::PxsHeapStats::eBROADPHASE),
    lostActor_(heap,physx::PxsHeapStats::eBROADPHASE),
    broadphaseDescriptorBuffer_(heap,physx::PxsHeapStats::eBROADPHASE),
    radixDescriptorBuffer_(heap,physx::PxsHeapStats::eBROADPHASE),
    radixWithoutRanksBuffer_(heap,physx::PxsHeapStats::eBROADPHASE) {
    if(!maxPairs_ || !maxAggregatePairs_)
        throw std::invalid_argument("Original SAP report capacities must be nonzero");
    // PxgCudaBroadPhaseSap constructor allocates only these permanent
    // resources. All projection, endpoint and region allocations grow
    // later, in the original gpuDMAUp ordering.
    for(physx::PxU32 i=0;i<3u;++i)
        radixCounts_[i].allocate(sizeof(physx::PxU32)*32u*16u,PX_FL);
    broadphaseDescriptorBuffer_.allocate(sizeof(physx::PxgBroadPhaseDesc),PX_FL);
    radixDescriptorBuffer_.allocate(sizeof(physx::PxgRadixSortDesc)*6u,PX_FL);
    radixWithoutRanksBuffer_.allocate(sizeof(physx::PxgRadixSortDesc)*6u,PX_FL);
    foundPairs_.allocate(sizeof(physx::PxgBroadPhasePair)*maxPairs_,PX_FL);
    lostPairs_.allocate(sizeof(physx::PxgBroadPhasePair)*maxPairs_,PX_FL);
    foundAggregate_.allocate(sizeof(physx::PxgBroadPhasePair)*maxAggregatePairs_,PX_FL);
    lostAggregate_.allocate(sizeof(physx::PxgBroadPhasePair)*maxAggregatePairs_,PX_FL);
    foundActor_.allocate(sizeof(physx::PxgBroadPhasePair)*maxPairs_,PX_FL);
    lostActor_.allocate(sizeof(physx::PxgBroadPhasePair)*maxPairs_,PX_FL);
}
SourceSapPreBroadPhaseBuffers
PxgSapSourceMembersNative::pre_broadphase_buffers() noexcept {
    return {fpBounds_,contactDistances_,groups_,environmentIds_,boxesCapacity_};
}
SourceSapFetchHistoryBuffers
PxgSapSourceMembersNative::fetch_histories() noexcept {
    return {projectionHandles_,blockEndHistograms_,endHistograms_,
        endHandles_,blockStartHistograms_,startHistograms_,startHandles_};
}
SourceSapDmaUpFrameMembers PxgSapSourceMembersNative::dma_up_members(
    const SourceSapAabbManagerDescriptorView* aabb_manager,
    CUdeviceptr original_found_map,CUdeviceptr original_lost_map) {
    if(!original_found_map || !original_lost_map)
        throw std::invalid_argument(
            "Source GPU SAP requires original pinned mapped actor report addresses");
    return {
        {numBoxes_,boxesCapacity_,createdCount_,removedCount_,
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
         updatedCount_
#endif
        },
        {createdHandles_,removedHandles_,
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
         updatedHandles_
#endif
        },
        oldIntegerBounds_,
        {{
            {sapBoxes_[0],newSapBoxes_[0],projections_[0],
         projectionRanks_[0],tempProjections_[0],tempHandles_[0]},
            {sapBoxes_[1],newSapBoxes_[1],projections_[1],
         projectionRanks_[1],tempProjections_[1],tempHandles_[1]},
            {sapBoxes_[2],newSapBoxes_[2],projections_[2],
         projectionRanks_[2],tempProjections_[2],tempHandles_[2]}
        }},
        {projectionHandles_,endHistograms_,blockEndHistograms_,
         endHandles_,startHistograms_,blockStartHistograms_,startHandles_,
         incrementalComparisons_,incrementalBlockComparisons_,
         totalEndHistograms_,blockTotalEndHistograms_},
        {aggregateReportBlocks_,actorReportBlocks_,
         activeRegions_,startRegions_,orderedActiveHandles_,orderedStartHandles_,
         overlapChecks_,blockOverlapChecks_,overlapHandleRegions_,regionRange_,
         startRegionAccum_,blockStartRegionAccum_,regionAccum_,blockRegionAccum_},
        newIntegerBounds_,
        {
         createdHandles_,removedHandles_,updatedHandles_,oldIntegerBounds_,newIntegerBounds_,fpBounds_,
         contactDistances_,groups_,environmentIds_,foundPairs_,lostPairs_,foundAggregate_,
         lostAggregate_,foundActor_,lostActor_,sapBoxes_,newSapBoxes_,projectionRanks_,
         projections_,projectionHandles_,totalEndHistograms_,blockTotalEndHistograms_,endHistograms_,blockEndHistograms_,
         startHistograms_,blockStartHistograms_,endHandles_,startHandles_,incrementalComparisons_,incrementalBlockComparisons_,
         aggregateReportBlocks_,actorReportBlocks_,activeRegions_,startRegions_,orderedActiveHandles_,orderedStartHandles_,
         blockOverlapChecks_,overlapChecks_,overlapHandleRegions_,regionRange_,startRegionAccum_,blockStartRegionAccum_,
         regionAccum_,blockRegionAccum_},
        {numBoxes_,createdCount_,removedCount_,updatedCount_,
         original_found_map,original_lost_map,regionAccumTotal_,
         overlapChecksTotal_,maxPairs_,maxAggregatePairs_,aabb_manager},
        {projections_,projectionRanks_,tempProjections_,tempHandles_,
         radixCounts_,projectionHandles_},
        {broadphaseDescriptorBuffer_,radixDescriptorBuffer_,radixWithoutRanksBuffer_},
        descriptor_,radixDescs_.data(),radixWorDescs_.data()
    };
}
} // namespace fyzz::physx_gpu
