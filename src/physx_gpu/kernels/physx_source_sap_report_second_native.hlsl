// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_source_sap_report_multi_root_common.hlsli"
RWByteAddressBuffer rawCounts : register(u0,space0);
RWByteAddressBuffer aggregateCounts : register(u1,space0);
RWByteAddressBuffer foundRaw : register(u2,space0);
RWByteAddressBuffer lostRaw : register(u3,space0);
RWByteAddressBuffer volume : register(u4,space0);
RWByteAddressBuffer block0 : register(u5,space0);
RWByteAddressBuffer block1 : register(u6,space0);
RWByteAddressBuffer block2 : register(u7,space0);
RWByteAddressBuffer block3 : register(u8,space0);
RWByteAddressBuffer foundAggregate : register(u9,space0);
RWByteAddressBuffer foundActor : register(u10,space0);
RWByteAddressBuffer lostAggregate : register(u11,space0);
RWByteAddressBuffer lostActor : register(u12,space0);
uint2 readPair(uint stream,uint index) {
    return stream==0u ? foundRaw.Load2(params.bias2+index*8u)
                      : lostRaw.Load2(params.bias3+index*8u);
}
bool selectPair(uint2 pair,bool aggregate) {
    bool singleA=true,singleB=true;
    if(params.hasVolumes!=0u) {
        singleA=volume.Load(params.bias4+pair.x*params.volumeStride+
                            params.aggregateOffset)==0xffffffffu;
        singleB=volume.Load(params.bias4+pair.y*params.volumeStride+
                            params.aggregateOffset)==0xffffffffu;
    }
    return (!(singleA&&singleB))==aggregate;
}
uint readBlock(uint kind,uint index) {
    if(kind==0u) return block0.Load(params.bias5+index*4u);
    if(kind==1u) return block1.Load(params.bias6+index*4u);
    if(kind==2u) return block2.Load(params.bias7+index*4u);
    return block3.Load(params.bias8+index*4u);
}
void storeReport(uint kind,uint index,uint2 pair) {
    if(kind==0u) foundAggregate.Store2(params.bias9+index*8u,pair);
    else if(kind==1u) foundActor.Store2(params.bias10+index*8u,pair);
    else if(kind==2u) lostAggregate.Store2(params.bias11+index*8u,pair);
    else lostActor.Store2(params.bias12+index*8u,pair);
}

// Source BP_ACCUMULATE_REPORT_STAGE_2; original 32x4x256 stable compaction.
groupshared uint scanA[256],scanB[256],blockBase,carry;
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    const uint stream=group.y/2u;
    const bool aggregate=(group.y&1u)==0u;
    const uint raw=rawCounts.Load(params.bias0+stream*4u);
    const uint safeCount=min(raw,params.rawCapacity);
    const uint capacity=aggregate?params.aggCapacity:params.rawCapacity;
    if(thread.x==0u) {
        uint exclusive=0u,all=0u;
        [unroll]
        for(uint block=0u;block<32u;++block) {
            const uint n=readBlock(group.y,block);
            if(block<group.x) exclusive+=n;
            all+=n;
        }
        blockBase=exclusive;
        carry=0u;
        if(aggregate&&group.x==0u)
            aggregateCounts.Store(params.bias1+stream*4u,all);
    }
    GroupMemoryBarrierWithGroupSync();
    const uint blocksRequired=(raw+255u)/256u;
    const uint iterations=(blocksRequired+31u)/32u;
    for(uint step=0u;step<iterations;++step) {
        const uint index=(group.x*iterations+step)*256u+thread.x;
        bool eligible=false;
        uint2 pair=uint2(0u,0u);
        if(index<safeCount) {
            pair=readPair(stream,index);
            eligible=selectPair(pair,aggregate);
        }
        scanA[thread.x]=eligible?1u:0u;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for(uint shift=1u;shift<256u;shift<<=1u) {
            if(shift==1u||shift==4u||shift==16u||shift==64u) {
                uint value=scanA[thread.x];
                if(thread.x>=shift)value+=scanA[thread.x-shift];
                scanB[thread.x]=value;
            } else {
                uint value=scanB[thread.x];
                if(thread.x>=shift)value+=scanB[thread.x-shift];
                scanA[thread.x]=value;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if(eligible) {
            const uint destination=blockBase+carry+scanA[thread.x]-1u;
            if(destination<capacity)storeReport(group.y,destination,pair);
        }
        GroupMemoryBarrierWithGroupSync();
        if(thread.x==0u)carry+=scanA[255u];
        GroupMemoryBarrierWithGroupSync();
    }
}
