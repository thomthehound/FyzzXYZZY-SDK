// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_source_sap_report_multi_root_common.hlsli"

// Source BP_COPY_REPORTS: original 64x256 launch. The physical resources
// are original PhysX allocations; mapped destinations are native views.
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
RWByteAddressBuffer foundActor : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3, 0)]]
#endif
RWByteAddressBuffer lostActor : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4, 0)]]
#endif
RWByteAddressBuffer foundMap : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5, 0)]]
#endif
RWByteAddressBuffer lostMap : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6, 0)]]
#endif
RWByteAddressBuffer overflow : register(u6,space0);
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    const uint rawFound=rawCounts.Load(params.bias0);
    const uint rawLost=rawCounts.Load(params.bias0+4u);
    const uint aggFound=aggregateCounts.Load(params.bias1);
    const uint aggLost=aggregateCounts.Load(params.bias1+4u);
    const uint found=min(rawFound,params.rawCapacity)-aggFound;
    const uint lost=min(rawLost,params.rawCapacity)-aggLost;
    const uint index=group.x*256u+thread.x;
    if(index==0u)
        overflow.Store(params.bias6,
                       (rawFound>params.rawCapacity ||
                        rawLost>params.rawCapacity)?1u:0u);
    for(uint i=index;i<found*2u;i+=64u*256u)
        foundMap.Store(params.bias4+i*4u,
                       foundActor.Load(params.bias2+i*4u));
    for(uint i=index;i<lost*2u;i+=64u*256u)
        lostMap.Store(params.bias5+i*4u,
                      lostActor.Load(params.bias3+i*4u));
}
