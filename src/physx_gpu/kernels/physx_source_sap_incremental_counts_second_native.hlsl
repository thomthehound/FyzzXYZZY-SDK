// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// BP_COMPUTE_INCREMENTAL_CMP_COUNTS[1,2]: 32x256, original SAP
// current/previous histograms and projection rank histories.
// View slots per axis: handles, ranks, current starts/ends,
// previous starts/ends, incremental prefixes, block prefixes.
// Slot 24 is the GPU PxgBroadPhaseDesc::totalIncrementalComparisons.
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
RWByteAddressBuffer currentStartX:register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer currentEndX:register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer previousStartX:register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer previousEndX:register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer prefixX:register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer blockX:register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer handlesY:register(u8,space0);
#ifdef __spirv__
[[vk::binding(9,0)]]
#endif
RWByteAddressBuffer ranksY:register(u9,space0);
#ifdef __spirv__
[[vk::binding(10,0)]]
#endif
RWByteAddressBuffer currentStartY:register(u10,space0);
#ifdef __spirv__
[[vk::binding(11,0)]]
#endif
RWByteAddressBuffer currentEndY:register(u11,space0);
#ifdef __spirv__
[[vk::binding(12,0)]]
#endif
RWByteAddressBuffer previousStartY:register(u12,space0);
#ifdef __spirv__
[[vk::binding(13,0)]]
#endif
RWByteAddressBuffer previousEndY:register(u13,space0);
#ifdef __spirv__
[[vk::binding(14,0)]]
#endif
RWByteAddressBuffer prefixY:register(u14,space0);
#ifdef __spirv__
[[vk::binding(15,0)]]
#endif
RWByteAddressBuffer blockY:register(u15,space0);
#ifdef __spirv__
[[vk::binding(16,0)]]
#endif
RWByteAddressBuffer handlesZ:register(u16,space0);
#ifdef __spirv__
[[vk::binding(17,0)]]
#endif
RWByteAddressBuffer ranksZ:register(u17,space0);
#ifdef __spirv__
[[vk::binding(18,0)]]
#endif
RWByteAddressBuffer currentStartZ:register(u18,space0);
#ifdef __spirv__
[[vk::binding(19,0)]]
#endif
RWByteAddressBuffer currentEndZ:register(u19,space0);
#ifdef __spirv__
[[vk::binding(20,0)]]
#endif
RWByteAddressBuffer previousStartZ:register(u20,space0);
#ifdef __spirv__
[[vk::binding(21,0)]]
#endif
RWByteAddressBuffer previousEndZ:register(u21,space0);
#ifdef __spirv__
[[vk::binding(22,0)]]
#endif
RWByteAddressBuffer prefixZ:register(u22,space0);
#ifdef __spirv__
[[vk::binding(23,0)]]
#endif
RWByteAddressBuffer blockZ:register(u23,space0);
#ifdef __spirv__
[[vk::binding(24,0)]]
#endif
RWByteAddressBuffer totals:register(u24,space0);
struct Parameters { uint4 offsets[7]; };
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p:register(b0,space0);
#endif
uint bias(uint slot) {return p.offsets[slot>>2u][slot&3u];}
uint previous_boxes(){return p.offsets[6].y;}
uint field_load(uint axis,uint field,uint index) {
    const uint offset=bias(axis*8u+field)+index*4u;
    if(axis==0u && field==0u) return handlesX.Load(offset);
    if(axis==0u && field==1u) return ranksX.Load(offset);
    if(axis==0u && field==2u) return currentStartX.Load(offset);
    if(axis==0u && field==3u) return currentEndX.Load(offset);
    if(axis==0u && field==4u) return previousStartX.Load(offset);
    if(axis==0u && field==5u) return previousEndX.Load(offset);
    if(axis==0u && field==6u) return prefixX.Load(offset);
    if(axis==0u && field==7u) return blockX.Load(offset);
    if(axis==1u && field==0u) return handlesY.Load(offset);
    if(axis==1u && field==1u) return ranksY.Load(offset);
    if(axis==1u && field==2u) return currentStartY.Load(offset);
    if(axis==1u && field==3u) return currentEndY.Load(offset);
    if(axis==1u && field==4u) return previousStartY.Load(offset);
    if(axis==1u && field==5u) return previousEndY.Load(offset);
    if(axis==1u && field==6u) return prefixY.Load(offset);
    if(axis==1u && field==7u) return blockY.Load(offset);
    if(axis==2u && field==0u) return handlesZ.Load(offset);
    if(axis==2u && field==1u) return ranksZ.Load(offset);
    if(axis==2u && field==2u) return currentStartZ.Load(offset);
    if(axis==2u && field==3u) return currentEndZ.Load(offset);
    if(axis==2u && field==4u) return previousStartZ.Load(offset);
    if(axis==2u && field==5u) return previousEndZ.Load(offset);
    if(axis==2u && field==6u) return prefixZ.Load(offset);
    if(axis==2u && field==7u) return blockZ.Load(offset);
    return 0u;
}
void field_store(uint axis,uint field,uint index,uint value) {
    const uint offset=bias(axis*8u+field)+index*4u;
    if(axis==0u && field==6u) prefixX.Store(offset,value);
    if(axis==0u && field==7u) blockX.Store(offset,value);
    if(axis==1u && field==6u) prefixY.Store(offset,value);
    if(axis==1u && field==7u) blockY.Store(offset,value);
    if(axis==2u && field==6u) prefixZ.Store(offset,value);
    if(axis==2u && field==7u) blockZ.Store(offset,value);
}

groupshared uint3 blockBase;
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    if(thread.x==0u) {
        uint3 prior=uint3(0u,0u,0u);
        uint3 sums=uint3(0u,0u,0u);
        [unroll]for(uint axis=0u;axis<3u;++axis) {
            uint prefix=0u;
            [loop]for(uint b=0u;b<group.x;++b)prefix+=field_load(axis,7u,b);
            uint total=prefix;
            [loop]for(uint b=group.x;b<32u;++b)total+=field_load(axis,7u,b);
            prior[axis]=prefix;
            sums[axis]=total;
        }
        blockBase=prior;
        if(group.x==0u)totals.Store3(bias(24u),sums);
    }
    GroupMemoryBarrierWithGroupSync();
    const uint projections=previous_boxes()*2u;
    const uint required=(projections+255u)/256u;
    const uint iterations=(required+31u)/32u;
    for(uint iteration=0u;iteration<iterations;++iteration) {
        const uint index=(group.x*iterations+iteration)*256u+thread.x;
        if(index<projections) {
            [unroll]for(uint axis=0u;axis<3u;++axis) {
                const uint previous=field_load(axis,6u,index);
                field_store(axis,6u,index,previous+blockBase[axis]);
            }
        }
    }
}
