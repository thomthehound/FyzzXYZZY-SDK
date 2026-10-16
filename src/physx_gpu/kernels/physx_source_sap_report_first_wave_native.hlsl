// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_source_sap_report_multi_root_common.hlsli"
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer counts : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1, 0)]]
#endif
RWByteAddressBuffer rawFound : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2, 0)]]
#endif
RWByteAddressBuffer rawLost : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3, 0)]]
#endif
RWByteAddressBuffer volume : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4, 0)]]
#endif
RWByteAddressBuffer block0 : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5, 0)]]
#endif
RWByteAddressBuffer block1 : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6, 0)]]
#endif
RWByteAddressBuffer block2 : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7, 0)]]
#endif
RWByteAddressBuffer block3 : register(u7,space0);
uint2 readPair(uint stream, uint index) {
    return stream == 0u ? rawFound.Load2(params.bias1 + index * 8u)
                        : rawLost.Load2(params.bias2 + index * 8u);
}
bool selectPair(uint2 pair, bool aggregate) {
    bool singleA = true, singleB = true;
    if (params.hasVolumes != 0u) {
        singleA = volume.Load(params.bias3 + pair.x * params.volumeStride +
                              params.aggregateOffset) == 0xffffffffu;
        singleB = volume.Load(params.bias3 + pair.y * params.volumeStride +
                              params.aggregateOffset) == 0xffffffffu;
    }
    return (!(singleA && singleB)) == aggregate;
}
void storeBlock(uint kind, uint index, uint value) {
    if (kind == 0u) block0.Store(params.bias4 + index * 4u,value);
    else if (kind == 1u) block1.Store(params.bias5 + index * 4u,value);
    else if (kind == 2u) block2.Store(params.bias6 + index * 4u,value);
    else block3.Store(params.bias7 + index * 4u,value);
}

// Source BP_ACCUMULATE_REPORT_STAGE_1, native-wave reduction variant.
groupshared uint groupTotal;
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    const uint stream=group.y / 2u;
    const bool aggregate=(group.y & 1u)==0u;
    const uint raw=counts.Load(params.bias0 + stream * 4u);
    const uint safeCount=min(raw,params.rawCapacity);
    const uint blocksRequired=(raw + 255u) / 256u;
    const uint iterations=(blocksRequired + 31u) / 32u;
    if(thread.x==0u) groupTotal=0u;
    GroupMemoryBarrierWithGroupSync();
    uint subtotal=0u;
    for(uint step=0u;step<iterations;++step) {
        const uint index=(group.x * iterations + step)*256u+thread.x;
        if(index<safeCount && selectPair(readPair(stream,index),aggregate))
            ++subtotal;
    }
    const uint waveTotal=WaveActiveSum(subtotal);
    if(WaveIsFirstLane()) {
        uint ignored;
        InterlockedAdd(groupTotal,waveTotal,ignored);
    }
    GroupMemoryBarrierWithGroupSync();
    if(thread.x==0u) storeBlock(group.y,group.x,groupTotal);
}
