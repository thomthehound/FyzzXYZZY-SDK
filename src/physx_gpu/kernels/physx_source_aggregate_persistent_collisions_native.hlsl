// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// aggregate.cu::doAggPairCollisions/boxPruningProjection.
// Existing GPU root pairs use the original VolumeData aggregate tags and
// current/previous sorted aggregate projections to emit member reports.
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
RWByteAddressBuffer volumeData : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer aggregates : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer currentKeys : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer currentHandles : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer currentMembers : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer currentMasks : register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer previousKeys : register(u8,space0);
#ifdef __spirv__
[[vk::binding(9,0)]]
#endif
RWByteAddressBuffer previousHandles : register(u9,space0);
#ifdef __spirv__
[[vk::binding(10,0)]]
#endif
RWByteAddressBuffer previousMembers : register(u10,space0);
#ifdef __spirv__
[[vk::binding(11,0)]]
#endif
RWByteAddressBuffer previousMasks : register(u11,space0);
#ifdef __spirv__
[[vk::binding(12,0)]]
#endif
RWByteAddressBuffer previousSizes : register(u12,space0);
#ifdef __spirv__
[[vk::binding(13,0)]]
#endif
RWByteAddressBuffer groups : register(u13,space0);
#ifdef __spirv__
[[vk::binding(14,0)]]
#endif
RWByteAddressBuffer pairs : register(u14,space0);
#ifdef __spirv__
[[vk::binding(15,0)]]
#endif
RWByteAddressBuffer pairCount : register(u15,space0);
#ifdef __spirv__
[[vk::binding(16,0)]]
#endif
RWByteAddressBuffer removedRoots : register(u16,space0);
#ifdef __spirv__
[[vk::binding(17,0)]]
#endif
RWByteAddressBuffer removePairs : register(u17,space0);
#ifdef __spirv__
[[vk::binding(18,0)]]
#endif
RWByteAddressBuffer found : register(u18,space0);
#ifdef __spirv__
[[vk::binding(19,0)]]
#endif
RWByteAddressBuffer lost : register(u19,space0);
#ifdef __spirv__
[[vk::binding(20,0)]]
#endif
RWByteAddressBuffer reports : register(u20,space0);

struct Parameters {
    uint pairCapacity, aggregateCapacity, memberCapacity, boundsCapacity;
    uint reportCapacity, currentBoundsBias, previousBoundsBias, volumeBias;
    uint aggregateBias, currentKeysBias, currentHandlesBias, currentMembersBias;
    uint currentMasksBias, previousKeysBias, previousHandlesBias, previousMembersBias;
    uint previousMasksBias, previousSizesBias, groupsBias, pairsBias;
    uint countBias, removedRootsBias, removePairsBias, foundBias;
    uint lostBias, reportsBias, gridCount, reserved1;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

bool intersects(uint a,uint b,bool old) {
    const uint startA=(old?p.previousBoundsBias:p.currentBoundsBias)+a*24u;
    const uint startB=(old?p.previousBoundsBias:p.currentBoundsBias)+b*24u;
    uint3 amin,amax,bmin,bmax;
    if(old) {
        amin=previousBounds.Load3(startA);
        amax=previousBounds.Load3(startA+12u);
        bmin=previousBounds.Load3(startB);
        bmax=previousBounds.Load3(startB+12u);
    } else {
        amin=currentBounds.Load3(startA);
        amax=currentBounds.Load3(startA+12u);
        bmin=currentBounds.Load3(startB);
        bmax=currentBounds.Load3(startB+12u);
    }
    return all(amin<=bmax) && all(bmin<=amax);
}
uint projectionMin(uint h,bool old) {
    const uint index=(old?p.previousBoundsBias:p.currentBoundsBias)+h*24u;
    return old?previousBounds.Load(index):currentBounds.Load(index);
}
uint projectionMax(uint h,bool old) {
    const uint index=(old?p.previousBoundsBias:p.currentBoundsBias)+h*24u+12u;
    return old?previousBounds.Load(index):currentBounds.Load(index);
}
void emitPair(uint a,uint b,bool old) {
    // The original shared aggregate found/lost counters are never
    // saturated; overflow is observable by the source fetch boundary.
    uint index=0u;
    if(old) {
        reports.InterlockedAdd(p.reportsBias+4u,1u,index);
        if(index<p.reportCapacity)
            lost.Store2(p.lostBias+index*8u,uint2(min(a,b),max(a,b)));
    } else {
        reports.InterlockedAdd(p.reportsBias,1u,index);
        if(index<p.reportCapacity)
            found.Store2(p.foundBias+index*8u,uint2(min(a,b),max(a,b)));
    }
}
uint lowerProjection(uint first,uint count,uint x,bool old) {
    uint low=0u,high=count*2u;
    const uint bias=old?p.previousKeysBias:p.currentKeysBias;
    while(low<high) {
        const uint mid=(low+high)>>1u;
        const uint val=old
            ? previousKeys.Load(bias+(first*2u+mid)*4u)
            : currentKeys.Load(bias+(first*2u+mid)*4u);
        if(val<x)low=mid+1u;
        else high=mid;
    }
    return low;
}
void scanDirection(
    uint outerRoot,uint otherRoot,
    bool outerAggregate,bool otherAggregate,
    uint4 outerDesc,uint4 otherDesc,
    uint oldCountOuter,uint oldCountOther,
    bool old,bool firstDirection,uint lane) {
    const uint outerSize=outerAggregate?(old?oldCountOuter:outerDesc.z):1u;
    const uint targetSize=otherAggregate?(old?oldCountOther:otherDesc.z):1u;
    if(!outerSize || !targetSize)return;
    for(uint i=lane;i<outerSize;i+=32u) {
        const uint a=outerAggregate
            ? (old
                ? previousMembers.Load(p.previousMembersBias+(outerDesc.y+i)*4u)
                : currentMembers.Load(p.currentMembersBias+(outerDesc.y+i)*4u))
            : outerRoot;
        if(a>=p.boundsCapacity)continue;
        const uint minA=projectionMin(a,old),maxA=projectionMax(a,old);
        if(minA>maxA)continue;
        const uint groupA=groups.Load(p.groupsBias+a*4u);
        // A standalone root is the source's one-member synthetic
        // aggregate. Otherwise enumerate the original sorted X starts.
        if(!otherAggregate) {
            const uint b=otherRoot;
            if(a==b || b>=p.boundsCapacity)continue;
            const uint minB=projectionMin(b,old),maxB=projectionMax(b,old);
            if(minB<minA || minB>maxA ||
               (!firstDirection && minB==minA))continue;
            if(groupA==groups.Load(p.groupsBias+b*4u))continue;
            if(intersects(a,b,old) && !intersects(a,b,!old))
                emitPair(a,b,old);
            continue;
        }
        const uint start=lowerProjection(otherDesc.y,targetSize,minA,old);
        const uint bias=old?p.previousKeysBias:p.currentKeysBias;
        const uint handleBias=old?p.previousHandlesBias:p.currentHandlesBias;
        const uint maskBias=old?p.previousMasksBias:p.currentMasksBias;
        const uint memberBias=old?p.previousMembersBias:p.currentMembersBias;
        for(uint j=start;j<targetSize*2u;++j) {
            const uint projection=old
                ? previousKeys.Load(bias+(otherDesc.y*2u+j)*4u)
                : currentKeys.Load(bias+(otherDesc.y*2u+j)*4u);
            if(projection>maxA)break;
            const uint mask=old
                ? previousMasks.Load(maskBias+(otherDesc.y+(j>>5u))*4u)
                : currentMasks.Load(maskBias+(otherDesc.y+(j>>5u))*4u);
            if((mask&(1u<<(j&31u)))==0u)continue;
            if(!firstDirection && projection==minA)continue;
            const uint encoded=old
                ? previousHandles.Load(handleBias+(otherDesc.y*2u+j)*4u)
                : currentHandles.Load(handleBias+(otherDesc.y*2u+j)*4u);
            const uint local=encoded>>3u;
            if(local>=targetSize)continue;
            const uint b=old
                ? previousMembers.Load(memberBias+(otherDesc.y+local)*4u)
                : currentMembers.Load(memberBias+(otherDesc.y+local)*4u);
            if(a==b || b>=p.boundsCapacity ||
               groupA==groups.Load(p.groupsBias+b*4u))continue;
            if(intersects(a,b,old) && !intersects(a,b,!old))
                emitPair(a,b,old);
        }
    }
}
// The source pair's isNew byte is read by every logical lane. Snapshot it
// before any subgroup may clear it, even when native waves are <32 lanes.
groupshared uint sourcePairFlags[4];
[numthreads(32,4,1)]
void main(uint3 block:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint count=min(pairCount.Load(p.countBias),p.pairCapacity);
    for(uint start=block.x*4u;start<count;start+=p.gridCount*4u) {
        const uint id=start+lane.y;
        const bool active=id<count;
        const uint base=p.pairsBias+id*12u;
        if(lane.x==0u) {
            uint flags=0u;
            if(active)flags=pairs.Load(base+8u);
            sourcePairFlags[lane.y]=flags;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint flags=sourcePairFlags[lane.y];
        bool remove=false,clearNew=false;
        if(active && (flags&0x0000ff00u)==0u) {
            const uint a=pairs.Load(base),b=pairs.Load(base+4u);
            if(a>=p.boundsCapacity || b>=p.boundsCapacity) {
                remove=true;
            } else {
                // Bp::VolumeData's original mAggregate tag follows the
                // pointer-sized mUserData at byte 8, stride 16 on x64.
                const uint tagA=volumeData.Load(p.volumeBias+a*16u+8u);
                const uint tagB=volumeData.Load(p.volumeBias+b*16u+8u);
                const bool aggA=tagA!=0xffffffffu && (tagA&1u)!=0u;
                const bool aggB=tagB!=0xffffffffu && (tagB&1u)!=0u;
                const uint indexA=tagA>>1u,indexB=tagB>>1u;
                const uint4 infoA=aggA && indexA<p.aggregateCapacity
                    ? aggregates.Load4(p.aggregateBias+indexA*16u)
                    : uint4(0xffffffffu,0u,0u,0u);
                const uint4 infoB=aggB && indexB<p.aggregateCapacity
                    ? aggregates.Load4(p.aggregateBias+indexB*16u)
                    : uint4(0xffffffffu,0u,0u,0u);
                const bool valid=
                    (!aggA || (indexA<p.aggregateCapacity && infoA.x==a)) &&
                    (!aggB || (indexB<p.aggregateCapacity && infoB.x==b));
                const bool retired=valid &&
                    (((removedRoots.Load(p.removedRootsBias+(a>>5u)*4u)>>(a&31u))&1u)!=0u ||
                     ((removedRoots.Load(p.removedRootsBias+(b>>5u)*4u)>>(b&31u))&1u)!=0u);
                if(!valid || retired || (!aggA && !aggB)) {
                    remove=true;
                } else {
                    remove=!intersects(a,b,false);
                    const bool isNew=(flags&0xffu)!=0u;
                    const uint prevA=aggA
                        ? previousSizes.Load(p.previousSizesBias+indexA*4u):1u;
                    const uint prevB=aggB
                        ? previousSizes.Load(p.previousSizesBias+indexB*4u):1u;
                    for(uint pass=0u;pass<2u;++pass) {
                        const bool old=pass!=0u;
                        if(old && isNew)continue;
                        // Original two-sided bipartite X sweep; ties only
                        // report in the first direction.
                        scanDirection(a,b,aggA,aggB,infoA,infoB,
                            prevA,prevB,old,true,lane.x);
                        scanDirection(b,a,aggB,aggA,infoB,infoA,
                            prevB,prevA,old,false,lane.x);
                    }
                    clearNew=true;
                }
            }
        }
        // All physical subgroups must finish pair comparisons before lane
        // zero publishes the remove bitmap or clears the source isNew byte.
        // Inactive logical warps participate in every group barrier.
        GroupMemoryBarrierWithGroupSync();
        if(active && lane.x==0u) {
            removePairs.Store(p.removePairsBias+id*4u,remove?1u:0u);
            if(clearNew)pairs.Store(base+8u,flags&0xffffff00u);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
