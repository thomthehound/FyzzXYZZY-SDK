// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// aggregate.cu::addAggPairsStage1/addAggPairsStage2: original source
// aggregate-root pair admission and free-index pool ownership.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer sourceDescriptor : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer foundAggregatePairs : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer aggregatePairs : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer freeIndices : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer freeList : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer aggregatePairCount : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer overflowReports : register(u6,space0);

struct Parameters {
    uint maxRootReports, pairCapacity, foundCounterOffset, gridCount;
    uint descriptorBias, foundBias, pairsBias, freeIndicesBias;
    uint freeListBias, pairCountBias, overflowBias, reserved;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

// One launch on the source stream, before the serial free-buffer-list
// update, exactly as the original addAggPairsStage1.
[numthreads(256,1,1)]
void admit(uint3 block:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint index=block.x*256u+lane.x;
    const uint stride=p.gridCount*256u;
    const uint reports=min(
        sourceDescriptor.Load(p.descriptorBias+p.foundCounterOffset),
        p.maxRootReports);
    const uint numFree=freeList.Load(p.freeListBias);
    const uint maxIndex=freeList.Load(p.freeListBias+4u);
    for(uint i=index;i<reports;i+=stride) {
        uint destination=0u;
        if(i<numFree)
            destination=freeIndices.Load(
                p.freeIndicesBias+(numFree-i-1u)*4u);
        else
            destination=maxIndex+(i-numFree);
        if(destination>=p.pairCapacity)continue;
        const uint2 rootPair=foundAggregatePairs.Load2(
            p.foundBias+i*8u);
        // Native storage is the original 12-byte PxgAggregatePair:
        // two source handles, isNew=true, isDead=false, zero pad.
        aggregatePairs.Store3(
            p.pairsBias+destination*12u,
            uint3(rootPair.x,rootPair.y,1u));
    }
}

// Original addAggPairsStage2 has one writer for free-list accounting,
// aggregate count, overflow flags and simulation-stat candidate count.
[numthreads(1,1,1)]
void publish(uint3 block:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint found=sourceDescriptor.Load(
        p.descriptorBias+p.foundCounterOffset);
    const uint safe=min(found,p.maxRootReports);
    const uint numFree=freeList.Load(p.freeListBias);
    const uint maxIndex=freeList.Load(p.freeListBias+4u);
    const uint remaining=numFree>safe?numFree-safe:0u;
    const uint added=safe>numFree?safe-numFree:0u;
    // Source bounds use PxU32 counts; keep untruncated aggregate overflow
    // information visible rather than silently wrapping maxIndex.
    const bool wouldOverflow=maxIndex>0xffffffffu-added;
    const uint requested=wouldOverflow?0xffffffffu:maxIndex+added;
    const uint newMax=min(requested,p.pairCapacity);
    freeList.Store(p.freeListBias,remaining);
    freeList.Store(p.freeListBias+4u,newMax);
    aggregatePairCount.Store(p.pairCountBias,newMax);
    overflowReports.Store(p.overflowBias,requested);
    overflowReports.Store(p.overflowBias+4u,
        (wouldOverflow || requested>p.pairCapacity)?1u:0u);
    overflowReports.Store(p.overflowBias+8u,
        found>p.maxRootReports?1u:0u);
}
