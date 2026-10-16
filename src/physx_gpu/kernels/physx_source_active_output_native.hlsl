// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Source: broadphase.cu::outputOrderedActiveRegionHistogram.
// Original BP_OUTPUT_ACTIVE_HISTOGRAM, 64 groups x 256 threads.
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
RWByteAddressBuffer sapX : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5, 0)]]
#endif
RWByteAddressBuffer starts : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6, 0)]]
#endif
RWByteAddressBuffer active : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7, 0)]]
#endif
RWByteAddressBuffer orderedStarts : register(u7,space0);
#ifdef __spirv__
[[vk::binding(8, 0)]]
#endif
RWByteAddressBuffer orderedActive : register(u8,space0);
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
    uint low=0u,high=projections;
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
    const uint loops=((total+255u)/256u+63u)/64u;
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
        const uint regionId=ranges.Load(range)+
            (ordinal/zWidth+sy)*4u+(ordinal%zWidth+sz)*16u;
        const uint begin=sapX.Load(params.sapBias+handle*8u);
        const uint offset=regionId*n*4u;
        const uint startRank=starts.Load(params.startsBias+offset+begin*4u);
        orderedStarts.Store(params.orderedStartsBias+offset+startRank*4u,handle);
        if(((tagged>>1u)&1u)!=0u){
            const uint activeRank=active.Load(params.activeBias+offset+begin*4u);
            orderedActive.Store(params.orderedActiveBias+offset+activeRank*4u,handle);
        }
    }
}
