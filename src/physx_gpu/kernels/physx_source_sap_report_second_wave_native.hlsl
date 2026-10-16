// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_source_sap_report_multi_root_common.hlsli"
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer rawCounts : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1, 0)]]
#endif
RWByteAddressBuffer aggregateCounts : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2, 0)]]
#endif
RWByteAddressBuffer foundRaw : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3, 0)]]
#endif
RWByteAddressBuffer lostRaw : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4, 0)]]
#endif
RWByteAddressBuffer volume : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5, 0)]]
#endif
RWByteAddressBuffer block0 : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6, 0)]]
#endif
RWByteAddressBuffer block1 : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7, 0)]]
#endif
RWByteAddressBuffer block2 : register(u7,space0);
#ifdef __spirv__
[[vk::binding(8, 0)]]
#endif
RWByteAddressBuffer block3 : register(u8,space0);
#ifdef __spirv__
[[vk::binding(9, 0)]]
#endif
RWByteAddressBuffer foundAggregate : register(u9,space0);
#ifdef __spirv__
[[vk::binding(10, 0)]]
#endif
RWByteAddressBuffer foundActor : register(u10,space0);
#ifdef __spirv__
[[vk::binding(11, 0)]]
#endif
RWByteAddressBuffer lostAggregate : register(u11,space0);
#ifdef __spirv__
[[vk::binding(12, 0)]]
#endif
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

// Source BP_ACCUMULATE_REPORT_STAGE_2; original 32x4x256 native-wave variant.
groupshared uint nextWave;
groupshared uint waveTotals[64],waveBases[64];
groupshared uint blockBase,carry,iterationCarry,localBlockPrefix;
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    const uint stream=group.y/2u;
    const bool aggregate=(group.y&1u)==0u;
    const uint raw=rawCounts.Load(params.bias0+stream*4u);
    const uint safeCount=min(raw,params.rawCapacity);
    const uint capacity=aggregate?params.aggCapacity:params.rawCapacity;
    if(thread.x==0u)nextWave=0u;
    GroupMemoryBarrierWithGroupSync();
    const uint laneCount=WaveGetLaneCount();
    const uint laneIndex=WaveGetLaneIndex();
    const uint waveCount=256u/laneCount;
    uint waveId=0u;
    if(WaveIsFirstLane())InterlockedAdd(nextWave,1u,waveId);
    waveId=WaveReadLaneFirst(waveId);
    const uint logicalThread=waveId*laneCount+laneIndex;
    const uint blockValue=logicalThread<32u?
        readBlock(group.y,logicalThread):0u;
    const uint blockPrefix=WavePrefixSum(blockValue);
    const uint blockWaveTotal=WaveActiveSum(blockValue);
    if(WaveIsFirstLane())waveTotals[waveId]=blockWaveTotal;
    if(logicalThread==group.x)localBlockPrefix=blockPrefix;
    GroupMemoryBarrierWithGroupSync();
    if(thread.x==0u) {
        uint all=0u,earlier=0u;
        const uint selectedWave=group.x/laneCount;
        for(uint w=0u;w<waveCount;++w) {
            const uint n=waveTotals[w];
            if(w<selectedWave)earlier+=n;
            all+=n;
        }
        blockBase=earlier+localBlockPrefix;
        carry=0u;
        if(aggregate&&group.x==0u)
            aggregateCounts.Store(params.bias1+stream*4u,all);
    }
    GroupMemoryBarrierWithGroupSync();
    const uint blocksRequired=(raw+255u)/256u;
    const uint iterations=(blocksRequired+31u)/32u;
    for(uint step=0u;step<iterations;++step) {
        const uint index=(group.x*iterations+step)*256u+
                         waveId*laneCount+laneIndex;
        bool eligible=false;
        uint2 pair=uint2(0u,0u);
        if(index<safeCount) {
            pair=readPair(stream,index);
            eligible=selectPair(pair,aggregate);
        }
        const uint flag=eligible?1u:0u;
        const uint wavePrefix=WavePrefixSum(flag);
        const uint waveTotal=WaveActiveSum(flag);
        if(WaveIsFirstLane())waveTotals[waveId]=waveTotal;
        GroupMemoryBarrierWithGroupSync();
        if(thread.x==0u) {
            uint subtotal=0u;
            for(uint w=0u;w<waveCount;++w) {
                waveBases[w]=subtotal;
                subtotal+=waveTotals[w];
            }
            iterationCarry=carry;
            carry+=subtotal;
        }
        GroupMemoryBarrierWithGroupSync();
        if(eligible) {
            const uint destination=blockBase+iterationCarry+
                                   waveBases[waveId]+wavePrefix;
            if(destination<capacity)storeReport(group.y,destination,pair);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
