// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// aggregate.cu::doSelfCollision/generatePairs/generateAllPairs.
// Source per-aggregate X endpoint windows, group filtering, full encoded
// AABB intersections and current/previous found/lost comparison.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer currentBounds : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer previousBounds : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer aggregates : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer currentHandles : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer currentSap : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer currentMembers : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer currentMasks : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer previousHandles : register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer previousSap : register(u8,space0);
#ifdef __spirv__
[[vk::binding(9,0)]]
#endif
RWByteAddressBuffer previousMembers : register(u9,space0);
#ifdef __spirv__
[[vk::binding(10,0)]]
#endif
RWByteAddressBuffer previousMasks : register(u10,space0);
#ifdef __spirv__
[[vk::binding(11,0)]]
#endif
RWByteAddressBuffer filterGroups : register(u11,space0);
#ifdef __spirv__
[[vk::binding(12,0)]]
#endif
RWByteAddressBuffer previousSizes : register(u12,space0);
#ifdef __spirv__
[[vk::binding(13,0)]]
#endif
RWByteAddressBuffer foundPairs : register(u13,space0);
#ifdef __spirv__
[[vk::binding(14,0)]]
#endif
RWByteAddressBuffer lostPairs : register(u14,space0);
#ifdef __spirv__
[[vk::binding(15,0)]]
#endif
RWByteAddressBuffer pairCounts : register(u15,space0);

struct Parameters {
    uint aggregateCount, memberCapacity, boundsCapacity, reportCapacity;
    uint newBias, oldBias, aggregateBias, currentHandlesBias;
    uint currentSapBias, currentMemberBias, currentMaskBias, oldHandlesBias;
    uint oldSapBias, oldMemberBias, oldMaskBias, groupsBias;
    uint previousSizesBias, foundBias, lostBias, countBias;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

bool intersects(uint handleA,uint handleB,bool previous) {
    const uint a=(previous?p.oldBias:p.newBias)+handleA*24u;
    const uint b=(previous?p.oldBias:p.newBias)+handleB*24u;
    uint3 minA,maxA,minB,maxB;
    if(previous) {
        minA=previousBounds.Load3(a);
        maxA=previousBounds.Load3(a+12u);
        minB=previousBounds.Load3(b);
        maxB=previousBounds.Load3(b+12u);
    }else {
        minA=currentBounds.Load3(a);
        maxA=currentBounds.Load3(a+12u);
        minB=currentBounds.Load3(b);
        maxB=currentBounds.Load3(b+12u);
    }
    return all(minA<=maxB)&&all(minB<=maxA);
}
[numthreads(32,4,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint aggregate=group.x*4u+lane.y;
    if(aggregate>=p.aggregateCount)return;
    const uint4 record=aggregates.Load4(p.aggregateBias+aggregate*16u);
    const bool valid=record.x!=0xffffffffu && record.z!=0u;
    const uint n=valid?record.z:0u;
    const uint first=record.y;
    // PxGetAggregateSelfCollisionBit(filterHint) is the low source bit.
    const bool collide=valid && (record.w&1u)!=0u;
    const uint prior=previousSizes.Load(p.previousSizesBias+aggregate*4u);
    // The same source GPU aggregate slot owns prevSize. Do not consume old
    // sorted arrays until a previous frame has successfully published them.
    if(collide) {
        for(uint pass=0u;pass<2u;++pass) {
            const bool lost=pass!=0u;
            const uint count=lost?prior:n;
            const uint hb=lost?p.oldHandlesBias:p.currentHandlesBias;
            const uint sb=lost?p.oldSapBias:p.currentSapBias;
            const uint mb=lost?p.oldMemberBias:p.currentMemberBias;
            const uint kb=lost?p.oldMaskBias:p.currentMaskBias;
            for(uint local=lane.x;local<count;local+=32u) {
                const uint pairBase=first*2u+local*2u;
                const uint start=lost
                    ? previousSap.Load(sb+pairBase*4u)
                    : currentSap.Load(sb+pairBase*4u);
                const uint end=lost
                    ? previousSap.Load(sb+(pairBase+1u)*4u)
                    : currentSap.Load(sb+(pairBase+1u)*4u);
                const uint a=lost
                    ? previousMembers.Load(mb+(first+local)*4u)
                    : currentMembers.Load(mb+(first+local)*4u);
                const uint ga=filterGroups.Load(p.groupsBias+a*4u);
                // Source comparison interval: start+1 through end-1,
                // selecting only other X start projections from the mask.
                for(uint index=start+1u;index<end && index<count*2u;++index) {
                    const uint mask=lost
                        ? previousMasks.Load(kb+(first+(index>>5u))*4u)
                        : currentMasks.Load(kb+(first+(index>>5u))*4u);
                    if((mask&(1u<<(index&31u)))==0u)continue;
                    const uint h=lost
                        ? previousHandles.Load(hb+(first*2u+index)*4u)
                        : currentHandles.Load(hb+(first*2u+index)*4u);
                    const uint other=h>>3u;
                    const uint b=lost
                        ? previousMembers.Load(mb+(first+other)*4u)
                        : currentMembers.Load(mb+(first+other)*4u);
                    if(ga==filterGroups.Load(p.groupsBias+b*4u))continue;
                    // Same source test in reverse history for lost pairs.
                    if(!intersects(a,b,lost) || intersects(a,b,!lost))
                        continue;
                    const uint volA=min(a,b),volB=max(a,b);
                    uint position=0u;
                    if(lost) {
                        pairCounts.InterlockedAdd(p.countBias+4u,1u,position);
                        if(position<p.reportCapacity)
                            lostPairs.Store2(p.lostBias+position*8u,
                                             uint2(volA,volB));
                    }else {
                        pairCounts.InterlockedAdd(p.countBias,1u,position);
                        if(position<p.reportCapacity)
                            foundPairs.Store2(p.foundBias+position*8u,
                                              uint2(volA,volB));
                    }
                }
            }
        }
    }
    if(lane.x==0u)
        previousSizes.Store(p.previousSizesBias+aggregate*4u,n);
}
