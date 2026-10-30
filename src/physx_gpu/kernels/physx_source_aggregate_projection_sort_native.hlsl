// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// aggregate.cu::sortAndUpdateAggregateProjections.
// Four logical source warps own four independent aggregates. The original
// eight stable four-bit radix passes preserve equal-key endpoint ordering;
// explicit group memory replaces CUDA warp-synchronous shared storage.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer integerBounds : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer aggregateSlots : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer sourceMembers : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer projections : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer handles : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer scratchProjections : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer scratchHandles : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer sapBoxes : register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer startMasks : register(u8,space0);
#ifdef __spirv__
[[vk::binding(9,0)]]
#endif
RWByteAddressBuffer boundIndicesHistory : register(u9,space0);

// All storage uses original per-aggregate reserved member slices. Each
// current/previous slot has 2 endpoints per reserved member, one SAP box
// and one member index per member, and at least one mask word per member.
// The current slot is selected by its source-owned device pointers.
// The shader never touches the other historical slot.
struct Parameters {
    uint aggregateCount, memberCapacity, boundsCapacity, currentSlot;
    uint boundsBias, aggregateBias, memberBias, projectionBias;
    uint handleBias, scratchProjectionBias, scratchHandleBias, sapBias;
    uint maskBias, historyBias, reserved0, reserved1;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

groupshared uint projectionCounts[4];
groupshared uint digitHistogram[4][16];
groupshared uint digitBases[4][16];
groupshared uint digitCompleted[4][16];
groupshared uint laneDigits[4][32];

[numthreads(32,4,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint aggregateIndex=group.x*4u+lane.y;
    const uint4 descriptor=(aggregateIndex<p.aggregateCount)
        ? aggregateSlots.Load4(p.aggregateBias+16u*aggregateIndex)
        : uint4(0xffffffffu,0u,0u,0u);
    const bool active=aggregateIndex<p.aggregateCount &&
        descriptor.x!=0xffffffffu && descriptor.z!=0u;
    const uint numMembers=active?descriptor.z:0u;
    const uint count=numMembers*2u;
    const uint firstMember=descriptor.y;
    const uint currentProjectionStart=firstMember*2u;
    const uint currentMemberStart=firstMember;
    const uint scratchStart=firstMember*2u;

    if(lane.x==0u)projectionCounts[lane.y]=count;
    GroupMemoryBarrierWithGroupSync();
    const uint maxCount=max(max(projectionCounts[0],projectionCounts[1]),
                            max(projectionCounts[2],projectionCounts[3]));
    const uint chunks=(maxCount+31u)/32u;

    // Copy original updateBoundIndices into the current history slot.
    // Only the current slot is written; previous members survive swaps.
    for(uint index=lane.x;index<maxCount;index+=32u) {
        if(index>=count)continue;
        const uint local=index>>1u;
        const uint member=sourceMembers.Load(
            p.memberBias+(firstMember+local)*4u);
        const uint endpoint=integerBounds.Load(
            p.boundsBias+member*24u+(index&1u)*12u);
        projections.Store(
            p.projectionBias+(currentProjectionStart+index)*4u,endpoint);
        handles.Store(
            p.handleBias+(currentProjectionStart+index)*4u,
            (local<<3u)|((index&1u)==0u?1u:0u));
        if((index&1u)==0u)
            boundIndicesHistory.Store(
                p.historyBias+(currentMemberStart+local)*4u,member);
    }
    DeviceMemoryBarrierWithGroupSync();

    // Eight source LSD nibble-radix passes. Histogram and stable per-lane
    // ranks are scoped to each logical aggregate; no native subgroup-width
    // assumption or serialization through a CPU aggregate sort.
    [unroll] for(uint pass=0u;pass<8u;++pass) {
        if(lane.x<16u) {
            digitHistogram[lane.y][lane.x]=0u;
            digitCompleted[lane.y][lane.x]=0u;
        }
        GroupMemoryBarrierWithGroupSync();
        for(uint index=lane.x;index<maxCount;index+=32u) {
            if(index>=count)continue;
            const uint inIndex=(pass&1u)!=0u
                ? scratchStart+index:currentProjectionStart+index;
            const uint key=(pass&1u)!=0u
                ? scratchProjections.Load(p.scratchProjectionBias+inIndex*4u)
                : projections.Load(p.projectionBias+inIndex*4u);
            const uint digit=(key>>(pass*4u))&15u;
            InterlockedAdd(digitHistogram[lane.y][digit],1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if(lane.x<16u) {
            uint exclusive=0u;
            for(uint digit=0u;digit<lane.x;++digit)
                exclusive+=digitHistogram[lane.y][digit];
            digitBases[lane.y][lane.x]=exclusive;
        }
        GroupMemoryBarrierWithGroupSync();

        for(uint chunk=0u;chunk<chunks;++chunk) {
            const uint index=chunk*32u+lane.x;
            const bool valid=index<count;
            const uint inIndex=(pass&1u)!=0u
                ? scratchStart+index:currentProjectionStart+index;
            uint key=0u,sourceHandle=0u,digit=0xffffffffu;
            if(valid) {
                if((pass&1u)!=0u) {
                    key=scratchProjections.Load(p.scratchProjectionBias+inIndex*4u);
                    sourceHandle=scratchHandles.Load(p.scratchHandleBias+inIndex*4u);
                }else {
                    key=projections.Load(p.projectionBias+inIndex*4u);
                    sourceHandle=handles.Load(p.handleBias+inIndex*4u);
                }
                digit=(key>>(pass*4u))&15u;
            }
            laneDigits[lane.y][lane.x]=digit;
            GroupMemoryBarrierWithGroupSync();
            if(valid) {
                uint localRank=0u;
                [unroll] for(uint j=0u;j<32u;++j)
                    if(j<lane.x && laneDigits[lane.y][j]==digit)
                        ++localRank;
                const uint destination=digitBases[lane.y][digit]+
                    digitCompleted[lane.y][digit]+localRank;
                const uint outIndex=(pass&1u)!=0u
                    ? currentProjectionStart+destination:scratchStart+destination;
                if((pass&1u)!=0u) {
                    projections.Store(p.projectionBias+outIndex*4u,key);
                    handles.Store(p.handleBias+outIndex*4u,sourceHandle);
                }else {
                    scratchProjections.Store(
                        p.scratchProjectionBias+outIndex*4u,key);
                    scratchHandles.Store(
                        p.scratchHandleBias+outIndex*4u,sourceHandle);
                }
            }
            // Chunk destinations are disjoint within this radix pass.
            // Shared digit ranks need group synchronization every chunk;
            // GPU-visible scatter writes require a device barrier only
            // before the next pass reads that completed destination.
            if(chunk+1u<chunks)
                GroupMemoryBarrierWithGroupSync();
            else
                DeviceMemoryBarrierWithGroupSync();
            if(lane.x<16u) {
                uint completed=0u;
                [unroll] for(uint j=0u;j<32u;++j)
                    if(laneDigits[lane.y][j]==lane.x)
                        ++completed;
                digitCompleted[lane.y][lane.x]+=completed;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // The final even number of passes leaves sorted endpoint keys and
    // original createHandle local IDs in the current projection slot.
    for(uint index=lane.x;index<maxCount;index+=32u) {
        if(index>=count)continue;
        const uint localHandle=handles.Load(
            p.handleBias+(currentProjectionStart+index)*4u);
        const uint local=localHandle>>3u;
        const bool isStart=(localHandle&1u)!=0u;
        sapBoxes.Store(
            p.sapBias+(currentProjectionStart+local*2u+
                       (isStart?0u:1u))*4u,index);
    }
    DeviceMemoryBarrierWithGroupSync();

    // Source ballot masks, one bit per sorted projection. Each logical
    // warp writes its aggregate's own words; unused reservations stay
    // untouched for later source-history comparisons.
    const uint maxMaskWords=(maxCount+31u)/32u;
    for(uint word=lane.x;word<maxMaskWords;word+=32u) {
        if(word>=(count+31u)/32u)continue;
        uint mask=0u;
        [unroll] for(uint bit=0u;bit<32u;++bit) {
            const uint index=word*32u+bit;
            if(index<count) {
                const uint h=handles.Load(
                    p.handleBias+(currentProjectionStart+index)*4u);
                if((h&1u)!=0u)mask|=1u<<bit;
            }
        }
        startMasks.Store(
            p.maskBias+(currentMemberStart+word)*4u,mask);
    }
}
