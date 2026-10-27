// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"
#include <stdexcept>

namespace fyzz::physx_gpu {
PxgSapSourceMembersNative::PxgSapSourceMembersNative(
    physx::PxgAllocatorDesc& allocators,physx::PxU32 max_pairs,
    physx::PxU32 max_aggregate_pairs)
    : maxPairs_(max_pairs),maxAggregatePairs_(max_aggregate_pairs),
    pinnedDescriptor_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
    radixDescs_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
    radixWorDescs_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
    mappedFound_(allocators.hostMappedAlloc,physx::PxsHeapStats::eBROADPHASE,
                 physx::Cm::PinnableAllocatorFallback::eDISABLED),
    mappedLost_(allocators.hostMappedAlloc,physx::PxsHeapStats::eBROADPHASE,
                physx::Cm::PinnableAllocatorFallback::eDISABLED),
    createdHandles_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    removedHandles_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    updatedHandles_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    fpBounds_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    contactDistances_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    groups_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    environmentIds_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    newIntegerBounds_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    oldIntegerBounds_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    projections_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    projectionRanks_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    tempProjections_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    tempHandles_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    radixCounts_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    sapBoxes_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    newSapBoxes_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    incrementalComparisons_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    incrementalBlockComparisons_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    projectionHandles_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    endHistograms_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    blockEndHistograms_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    endHandles_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    startHistograms_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    blockStartHistograms_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    startHandles_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    totalEndHistograms_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    blockTotalEndHistograms_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    aggregateReportBlocks_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    actorReportBlocks_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    activeRegions_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    startRegions_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    orderedActiveHandles_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    orderedStartHandles_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    overlapChecks_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    blockOverlapChecks_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    overlapHandleRegions_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    regionRange_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    startRegionAccum_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    blockStartRegionAccum_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    regionAccum_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    blockRegionAccum_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    foundPairs_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    lostPairs_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    foundAggregate_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    lostAggregate_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    foundActor_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    lostActor_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    broadphaseDescriptorBuffer_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    radixDescriptorBuffer_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
    radixWithoutRanksBuffer_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE) {
    if(!maxPairs_ || !maxAggregatePairs_)
        throw std::invalid_argument("Original SAP report capacities must be nonzero");
    if(!pinnedDescriptor_.isValid() ||
       !mappedFound_.reserve(maxPairs_) || !mappedLost_.reserve(maxPairs_))
        throw std::bad_alloc();
    radixDescs_.resize(6u);
    radixWorDescs_.resize(6u);
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
    const SourceSapMappedReportView& mapped_host) {
    const auto pair_bytes=sizeof(physx::PxgBroadPhasePair)*maxPairs_;
    const auto original_found_map=mapped_host.device_address(
        mappedFound_.begin(),pair_bytes);
    const auto original_lost_map=mapped_host.device_address(
        mappedLost_.begin(),pair_bytes);
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
        pinnedDescriptor_.get(),radixDescs_.begin(),radixWorDescs_.begin()
    };
}
} // namespace fyzz::physx_gpu
