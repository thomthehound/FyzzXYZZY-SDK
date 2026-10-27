// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// broadphase.cu::performIncrementalSAP, BP_INCREMENTAL_SAP, 256x3x256.
// All inputs, outputs and counters belong to the original GPU descriptor.
// Original 32-bit group report reservation and cross-axis overlap rules.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer handlesX:register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer ranksX:register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer comparisonX:register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer currentStartHistX:register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer currentEndHistX:register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer previousStartHistX:register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer previousEndHistX:register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer currentStartHandlesX:register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer currentEndHandlesX:register(u8,space0);
#ifdef __spirv__
[[vk::binding(9,0)]]
#endif
RWByteAddressBuffer previousStartHandlesX:register(u9,space0);
#ifdef __spirv__
[[vk::binding(10,0)]]
#endif
RWByteAddressBuffer previousEndHandlesX:register(u10,space0);
#ifdef __spirv__
[[vk::binding(11,0)]]
#endif
RWByteAddressBuffer handlesY:register(u11,space0);
#ifdef __spirv__
[[vk::binding(12,0)]]
#endif
RWByteAddressBuffer ranksY:register(u12,space0);
#ifdef __spirv__
[[vk::binding(13,0)]]
#endif
RWByteAddressBuffer comparisonY:register(u13,space0);
#ifdef __spirv__
[[vk::binding(14,0)]]
#endif
RWByteAddressBuffer currentStartHistY:register(u14,space0);
#ifdef __spirv__
[[vk::binding(15,0)]]
#endif
RWByteAddressBuffer currentEndHistY:register(u15,space0);
#ifdef __spirv__
[[vk::binding(16,0)]]
#endif
RWByteAddressBuffer previousStartHistY:register(u16,space0);
#ifdef __spirv__
[[vk::binding(17,0)]]
#endif
RWByteAddressBuffer previousEndHistY:register(u17,space0);
#ifdef __spirv__
[[vk::binding(18,0)]]
#endif
RWByteAddressBuffer currentStartHandlesY:register(u18,space0);
#ifdef __spirv__
[[vk::binding(19,0)]]
#endif
RWByteAddressBuffer currentEndHandlesY:register(u19,space0);
#ifdef __spirv__
[[vk::binding(20,0)]]
#endif
RWByteAddressBuffer previousStartHandlesY:register(u20,space0);
#ifdef __spirv__
[[vk::binding(21,0)]]
#endif
RWByteAddressBuffer previousEndHandlesY:register(u21,space0);
#ifdef __spirv__
[[vk::binding(22,0)]]
#endif
RWByteAddressBuffer handlesZ:register(u22,space0);
#ifdef __spirv__
[[vk::binding(23,0)]]
#endif
RWByteAddressBuffer ranksZ:register(u23,space0);
#ifdef __spirv__
[[vk::binding(24,0)]]
#endif
RWByteAddressBuffer comparisonZ:register(u24,space0);
#ifdef __spirv__
[[vk::binding(25,0)]]
#endif
RWByteAddressBuffer currentStartHistZ:register(u25,space0);
#ifdef __spirv__
[[vk::binding(26,0)]]
#endif
RWByteAddressBuffer currentEndHistZ:register(u26,space0);
#ifdef __spirv__
[[vk::binding(27,0)]]
#endif
RWByteAddressBuffer previousStartHistZ:register(u27,space0);
#ifdef __spirv__
[[vk::binding(28,0)]]
#endif
RWByteAddressBuffer previousEndHistZ:register(u28,space0);
#ifdef __spirv__
[[vk::binding(29,0)]]
#endif
RWByteAddressBuffer currentStartHandlesZ:register(u29,space0);
#ifdef __spirv__
[[vk::binding(30,0)]]
#endif
RWByteAddressBuffer currentEndHandlesZ:register(u30,space0);
#ifdef __spirv__
[[vk::binding(31,0)]]
#endif
RWByteAddressBuffer previousStartHandlesZ:register(u31,space0);
#ifdef __spirv__
[[vk::binding(32,0)]]
#endif
RWByteAddressBuffer previousEndHandlesZ:register(u32,space0);
#ifdef __spirv__
[[vk::binding(33,0)]]
#endif
RWByteAddressBuffer oldBounds:register(u33,space0);
#ifdef __spirv__
[[vk::binding(34,0)]]
#endif
RWByteAddressBuffer newBounds:register(u34,space0);
#ifdef __spirv__
[[vk::binding(35,0)]]
#endif
RWByteAddressBuffer groupIds:register(u35,space0);
#ifdef __spirv__
[[vk::binding(36,0)]]
#endif
RWByteAddressBuffer envIds:register(u36,space0);
#ifdef __spirv__
[[vk::binding(37,0)]]
#endif
RWByteAddressBuffer found:register(u37,space0);
#ifdef __spirv__
[[vk::binding(38,0)]]
#endif
RWByteAddressBuffer lost:register(u38,space0);
#ifdef __spirv__
[[vk::binding(39,0)]]
#endif
RWByteAddressBuffer reportCounts:register(u39,space0);
#ifdef __spirv__
[[vk::binding(40,0)]]
#endif
RWByteAddressBuffer totals:register(u40,space0);
struct Parameters {
    uint4 biasPairs[6]; // two 16-bit native view biases per DWORD
    uint4 dimensions; // previous boxes, box capacity, report capacity, env IDs
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p:register(b0,space0);
#endif
uint bias(uint slot) {
    const uint packed=p.biasPairs[slot>>3u][(slot>>1u)&3u];
    return (slot&1u)!=0u?packed>>16u:packed&0xffffu;
}
uint load_word(uint slot,uint index) {
    const uint byteOffset=bias(slot)+index*4u;
    switch(slot) {
    case 0u:return handlesX.Load(byteOffset);
    case 1u:return ranksX.Load(byteOffset);
    case 2u:return comparisonX.Load(byteOffset);
    case 3u:return currentStartHistX.Load(byteOffset);
    case 4u:return currentEndHistX.Load(byteOffset);
    case 5u:return previousStartHistX.Load(byteOffset);
    case 6u:return previousEndHistX.Load(byteOffset);
    case 7u:return currentStartHandlesX.Load(byteOffset);
    case 8u:return currentEndHandlesX.Load(byteOffset);
    case 9u:return previousStartHandlesX.Load(byteOffset);
    case 10u:return previousEndHandlesX.Load(byteOffset);
    case 11u:return handlesY.Load(byteOffset);
    case 12u:return ranksY.Load(byteOffset);
    case 13u:return comparisonY.Load(byteOffset);
    case 14u:return currentStartHistY.Load(byteOffset);
    case 15u:return currentEndHistY.Load(byteOffset);
    case 16u:return previousStartHistY.Load(byteOffset);
    case 17u:return previousEndHistY.Load(byteOffset);
    case 18u:return currentStartHandlesY.Load(byteOffset);
    case 19u:return currentEndHandlesY.Load(byteOffset);
    case 20u:return previousStartHandlesY.Load(byteOffset);
    case 21u:return previousEndHandlesY.Load(byteOffset);
    case 22u:return handlesZ.Load(byteOffset);
    case 23u:return ranksZ.Load(byteOffset);
    case 24u:return comparisonZ.Load(byteOffset);
    case 25u:return currentStartHistZ.Load(byteOffset);
    case 26u:return currentEndHistZ.Load(byteOffset);
    case 27u:return previousStartHistZ.Load(byteOffset);
    case 28u:return previousEndHistZ.Load(byteOffset);
    case 29u:return currentStartHandlesZ.Load(byteOffset);
    case 30u:return currentEndHandlesZ.Load(byteOffset);
    case 31u:return previousStartHandlesZ.Load(byteOffset);
    case 32u:return previousEndHandlesZ.Load(byteOffset);
    case 33u:return oldBounds.Load(byteOffset);
    case 34u:return newBounds.Load(byteOffset);
    case 35u:return groupIds.Load(byteOffset);
    case 36u:return envIds.Load(byteOffset);
    case 37u:return found.Load(byteOffset);
    case 38u:return lost.Load(byteOffset);
    case 39u:return reportCounts.Load(byteOffset);
    case 40u:return totals.Load(byteOffset);
    default:return 0u;
    }
}
uint endpoint_for_work(uint axis,uint work) {
    const uint prefix=axis*11u+2u;
    uint lo=0u,hi=p.dimensions.x*2u;
    while(lo<hi) {
        const uint mid=lo+(hi-lo)/2u;
        if(load_word(prefix,mid)<=work)lo=mid+1u;
        else hi=mid;
    }
    return lo-1u;
}
uint bound(uint source,uint object,uint axis,bool upper) {
    return load_word(source,object*6u+axis+(upper?3u:0u));
}
bool overlap_axis(uint boundsIndex,uint a,uint b,uint axis) {
    return bound(boundsIndex,b,axis,false)<=bound(boundsIndex,a,axis,true) &&
           bound(boundsIndex,a,axis,false)<=bound(boundsIndex,b,axis,true);
}
bool overlap_3d(uint boundsIndex,uint a,uint b) {
    return overlap_axis(boundsIndex,a,b,0u) &&
           overlap_axis(boundsIndex,a,b,1u) &&
           overlap_axis(boundsIndex,a,b,2u);
}
bool separated_before(uint boundsIndex,uint a,uint b,uint axis) {
    for(uint k=0u;k<axis;++k)
        if(!overlap_axis(boundsIndex,a,b,k))return true;
    return false;
}
groupshared uint2 scanA[256];
groupshared uint2 scanB[256];
groupshared uint2 groupBase;
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    const uint axis=group.y,base=axis*11u;
    const uint total=load_word(40u,axis);
    const uint blocks=(total+255u)/256u;
    if(group.x>=blocks)return; // idle fixed-grid partition: no barriers
    for(uint block=group.x;block<blocks;block+=256u) {
        const uint work=block*256u+thread.x;
        uint handle=0u,other=0u,reportType=0u;
        bool emit=false;
        if(work<total) {
            const uint endpoint=endpoint_for_work(axis,work);
            const uint tagged=load_word(base,endpoint);
            handle=tagged>>3u;
            const bool start=(tagged&1u)!=0u;
            const uint previousIndex=load_word(base+1u,endpoint);
            const bool down=previousIndex>endpoint;
            const uint first=down?endpoint:previousIndex+1u;
            const uint local=work-load_word(base+2u,endpoint);
            const uint histogram=base+(down?(start?6u:5u):
                                               (start?4u:3u));
            const uint ordered=base+(down?(start?10u:9u):
                                             (start?8u:7u));
            const uint candidate=load_word(histogram,first)+local;
            other=load_word(ordered,candidate);
            const bool willFind=(!start)!=down;
            reportType=willFind?0u:1u;
            if(handle<p.dimensions.y&&other<p.dimensions.y) {
                bool permitted=load_word(35u,handle)!=load_word(35u,other);
                if(p.dimensions.w!=0u) {
                    const uint ea=load_word(36u,handle);
                    const uint eb=load_word(36u,other);
                    permitted=permitted &&
                        (ea==eb||ea==0xffffffffu||eb==0xffffffffu);
                }
                if(permitted) {
                    const uint present=willFind?34u:33u;
                    const uint absent=willFind?33u:34u;
                    emit=overlap_3d(present,handle,other) &&
                         !overlap_axis(absent,handle,other,axis) &&
                         !separated_before(absent,handle,other,axis);
                }
            }
        }
        const uint2 flags=emit?(reportType==0u?uint2(1u,0u):
                                                    uint2(0u,1u)):
                               uint2(0u,0u);
        scanA[thread.x]=flags;
        GroupMemoryBarrierWithGroupSync();
        [unroll]for(uint shift=1u;shift<256u;shift<<=1u) {
            if(shift==1u||shift==4u||shift==16u||shift==64u) {
                uint2 v=scanA[thread.x];
                if(thread.x>=shift)v+=scanA[thread.x-shift];
                scanB[thread.x]=v;
            }else{
                uint2 v=scanB[thread.x];
                if(thread.x>=shift)v+=scanB[thread.x-shift];
                scanA[thread.x]=v;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if(thread.x==0u) {
            const uint2 sum=scanA[255u];
            uint2 previous=uint2(0u,0u);
            if(sum.x!=0u)reportCounts.InterlockedAdd(
                bias(39u),sum.x,previous.x);
            if(sum.y!=0u)reportCounts.InterlockedAdd(
                bias(39u)+4u,sum.y,previous.y);
            groupBase=previous;
        }
        GroupMemoryBarrierWithGroupSync();
        if(emit) {
            const uint output=groupBase[reportType]+
                scanA[thread.x][reportType]-1u;
            if(output<p.dimensions.z) {
                const uint2 pair=uint2(min(handle,other),max(handle,other));
                if(reportType==0u)found.Store2(bias(37u)+output*8u,pair);
                else lost.Store2(bias(38u)+output*8u,pair);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
