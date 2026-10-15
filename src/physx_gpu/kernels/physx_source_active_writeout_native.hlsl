// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Source: broadphase.cu::writeOutStartAndActiveRegionHistogram
// Original BP_WRITEOUT_ACTIVE_HISTOGRAM, 32 groups x 256 threads.
// Original GPU descriptor supplies the GPU-written startRegionAccumTotal.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer sortedX : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1, 0)]]
#endif
RWByteAddressBuffer ranges : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2, 0)]]
#endif
RWByteAddressBuffer prefix : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3, 0)]]
#endif
RWByteAddressBuffer sourceDesc : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4, 0)]]
#endif
RWByteAddressBuffer active : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5, 0)]]
#endif
RWByteAddressBuffer starts : register(u5,space0);
struct Parameters {
    uint boxes; uint sortedBias; uint sapBias; uint rangesBias;
    uint prefixBias; uint descriptorBias; uint activeBias; uint startsBias;
    uint orderedActiveBias; uint orderedStartsBias; uint totalFieldOffset; uint reserved;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0,space0);
#endif

uint prefix_to_x_position(uint workIndex,uint projections){
    uint low=0u, high=projections;
    while(low<high){
        const uint mid=low+(high-low)/2u;
        if(prefix.Load(params.prefixBias+mid*4u)<=workIndex)low=mid+1u;
        else high=mid;
    }
    return low-1u;
}
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID){
    const uint n=params.boxes*2u;
    const uint total=sourceDesc.Load(params.descriptorBias+params.totalFieldOffset);
    const uint loops=((total+255u)/256u+31u)/32u;
    for(uint k=0u;k<loops;++k){
        const uint work=(group.x*loops+k)*256u+thread.x;
        if(work>=total)continue;
        const uint xPos=prefix_to_x_position(work,n);
        const uint tagged=sortedX.Load(params.sortedBias+xPos*4u);
        const uint handle=tagged>>3u;
        const uint range=params.rangesBias+handle*32u;
        const uint sy=ranges.Load(range+4u);
        const uint sz=ranges.Load(range+8u);
        const uint ez=ranges.Load(range+24u);
        const uint zWidth=ez-sz+1u;
        const uint ordinal=work-prefix.Load(params.prefixBias+xPos*4u);
        const uint yz=(ordinal/zWidth+sy)*4u+(ordinal%zWidth+sz)*16u;
        const uint rx=ranges.Load(range);
        const uint regionId=rx+yz;
        const uint index=(regionId*n+xPos)*4u;
        starts.Store(params.startsBias+index,1u);
        active.Store(params.activeBias+index,(tagged>>1u)&1u);
    }
}
