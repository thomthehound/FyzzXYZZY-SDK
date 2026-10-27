// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// PxgCudaBroadPhaseSap::calculateEndPtHistogramKernel
// 32 x 3 x 256, original 2-frame endpoint histories and current totals.
// Each source view retains its original GPU allocation; only 16-bit RAW
// binding offsets are packed, avoiding any GPU-side descriptor upload.
#ifdef __spirv__
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer currenthandlesX:register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer currentendsX:register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer currentstartsX:register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer currentorderedEndsX:register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer currentorderedStartsX:register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer currentblockEndsX:register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer currentblockStartsX:register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer previoushandlesX:register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer previousendsX:register(u8,space0);
#ifdef __spirv__
[[vk::binding(9,0)]]
#endif
RWByteAddressBuffer previousstartsX:register(u9,space0);
#ifdef __spirv__
[[vk::binding(10,0)]]
#endif
RWByteAddressBuffer previousorderedEndsX:register(u10,space0);
#ifdef __spirv__
[[vk::binding(11,0)]]
#endif
RWByteAddressBuffer previousorderedStartsX:register(u11,space0);
#ifdef __spirv__
[[vk::binding(12,0)]]
#endif
RWByteAddressBuffer previousblockEndsX:register(u12,space0);
#ifdef __spirv__
[[vk::binding(13,0)]]
#endif
RWByteAddressBuffer previousblockStartsX:register(u13,space0);
#ifdef __spirv__
[[vk::binding(14,0)]]
#endif
RWByteAddressBuffer totalEndX:register(u14,space0);
#ifdef __spirv__
[[vk::binding(15,0)]]
#endif
RWByteAddressBuffer blockTotalX:register(u15,space0);
#ifdef __spirv__
[[vk::binding(16,0)]]
#endif
RWByteAddressBuffer currenthandlesY:register(u16,space0);
#ifdef __spirv__
[[vk::binding(17,0)]]
#endif
RWByteAddressBuffer currentendsY:register(u17,space0);
#ifdef __spirv__
[[vk::binding(18,0)]]
#endif
RWByteAddressBuffer currentstartsY:register(u18,space0);
#ifdef __spirv__
[[vk::binding(19,0)]]
#endif
RWByteAddressBuffer currentorderedEndsY:register(u19,space0);
#ifdef __spirv__
[[vk::binding(20,0)]]
#endif
RWByteAddressBuffer currentorderedStartsY:register(u20,space0);
#ifdef __spirv__
[[vk::binding(21,0)]]
#endif
RWByteAddressBuffer currentblockEndsY:register(u21,space0);
#ifdef __spirv__
[[vk::binding(22,0)]]
#endif
RWByteAddressBuffer currentblockStartsY:register(u22,space0);
#ifdef __spirv__
[[vk::binding(23,0)]]
#endif
RWByteAddressBuffer previoushandlesY:register(u23,space0);
#ifdef __spirv__
[[vk::binding(24,0)]]
#endif
RWByteAddressBuffer previousendsY:register(u24,space0);
#ifdef __spirv__
[[vk::binding(25,0)]]
#endif
RWByteAddressBuffer previousstartsY:register(u25,space0);
#ifdef __spirv__
[[vk::binding(26,0)]]
#endif
RWByteAddressBuffer previousorderedEndsY:register(u26,space0);
#ifdef __spirv__
[[vk::binding(27,0)]]
#endif
RWByteAddressBuffer previousorderedStartsY:register(u27,space0);
#ifdef __spirv__
[[vk::binding(28,0)]]
#endif
RWByteAddressBuffer previousblockEndsY:register(u28,space0);
#ifdef __spirv__
[[vk::binding(29,0)]]
#endif
RWByteAddressBuffer previousblockStartsY:register(u29,space0);
#ifdef __spirv__
[[vk::binding(30,0)]]
#endif
RWByteAddressBuffer totalEndY:register(u30,space0);
#ifdef __spirv__
[[vk::binding(31,0)]]
#endif
RWByteAddressBuffer blockTotalY:register(u31,space0);
#ifdef __spirv__
[[vk::binding(32,0)]]
#endif
RWByteAddressBuffer currenthandlesZ:register(u32,space0);
#ifdef __spirv__
[[vk::binding(33,0)]]
#endif
RWByteAddressBuffer currentendsZ:register(u33,space0);
#ifdef __spirv__
[[vk::binding(34,0)]]
#endif
RWByteAddressBuffer currentstartsZ:register(u34,space0);
#ifdef __spirv__
[[vk::binding(35,0)]]
#endif
RWByteAddressBuffer currentorderedEndsZ:register(u35,space0);
#ifdef __spirv__
[[vk::binding(36,0)]]
#endif
RWByteAddressBuffer currentorderedStartsZ:register(u36,space0);
#ifdef __spirv__
[[vk::binding(37,0)]]
#endif
RWByteAddressBuffer currentblockEndsZ:register(u37,space0);
#ifdef __spirv__
[[vk::binding(38,0)]]
#endif
RWByteAddressBuffer currentblockStartsZ:register(u38,space0);
#ifdef __spirv__
[[vk::binding(39,0)]]
#endif
RWByteAddressBuffer previoushandlesZ:register(u39,space0);
#ifdef __spirv__
[[vk::binding(40,0)]]
#endif
RWByteAddressBuffer previousendsZ:register(u40,space0);
#ifdef __spirv__
[[vk::binding(41,0)]]
#endif
RWByteAddressBuffer previousstartsZ:register(u41,space0);
#ifdef __spirv__
[[vk::binding(42,0)]]
#endif
RWByteAddressBuffer previousorderedEndsZ:register(u42,space0);
#ifdef __spirv__
[[vk::binding(43,0)]]
#endif
RWByteAddressBuffer previousorderedStartsZ:register(u43,space0);
#ifdef __spirv__
[[vk::binding(44,0)]]
#endif
RWByteAddressBuffer previousblockEndsZ:register(u44,space0);
#ifdef __spirv__
[[vk::binding(45,0)]]
#endif
RWByteAddressBuffer previousblockStartsZ:register(u45,space0);
#ifdef __spirv__
[[vk::binding(46,0)]]
#endif
RWByteAddressBuffer totalEndZ:register(u46,space0);
#ifdef __spirv__
[[vk::binding(47,0)]]
#endif
RWByteAddressBuffer blockTotalZ:register(u47,space0);
#else
// Same source binding slots; native D3D12 indexes the descriptor table
// instead of lowering 48 resource-selection branches into every load.
RWByteAddressBuffer sourceViews[48] : register(u0,space0);
#endif
struct Parameters {
    uint4 biasPairs[6];
    uint4 counts; // previous, created, removed, incremental
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p:register(b0,space0);
#endif
uint bias(uint slot) {
    const uint packed=p.biasPairs[slot>>3u][(slot>>1u)&3u];
    return (slot&1u)!=0u?(packed>>16u):(packed&0xffffu);
}
uint load_field(uint slot,uint index) {
    const uint address=bias(slot)+4u*index;
#ifdef __spirv__
    switch(slot) {
    case 0u:return currenthandlesX.Load(address);
    case 1u:return currentendsX.Load(address);
    case 2u:return currentstartsX.Load(address);
    case 3u:return currentorderedEndsX.Load(address);
    case 4u:return currentorderedStartsX.Load(address);
    case 5u:return currentblockEndsX.Load(address);
    case 6u:return currentblockStartsX.Load(address);
    case 7u:return previoushandlesX.Load(address);
    case 8u:return previousendsX.Load(address);
    case 9u:return previousstartsX.Load(address);
    case 10u:return previousorderedEndsX.Load(address);
    case 11u:return previousorderedStartsX.Load(address);
    case 12u:return previousblockEndsX.Load(address);
    case 13u:return previousblockStartsX.Load(address);
    case 14u:return totalEndX.Load(address);
    case 15u:return blockTotalX.Load(address);
    case 16u:return currenthandlesY.Load(address);
    case 17u:return currentendsY.Load(address);
    case 18u:return currentstartsY.Load(address);
    case 19u:return currentorderedEndsY.Load(address);
    case 20u:return currentorderedStartsY.Load(address);
    case 21u:return currentblockEndsY.Load(address);
    case 22u:return currentblockStartsY.Load(address);
    case 23u:return previoushandlesY.Load(address);
    case 24u:return previousendsY.Load(address);
    case 25u:return previousstartsY.Load(address);
    case 26u:return previousorderedEndsY.Load(address);
    case 27u:return previousorderedStartsY.Load(address);
    case 28u:return previousblockEndsY.Load(address);
    case 29u:return previousblockStartsY.Load(address);
    case 30u:return totalEndY.Load(address);
    case 31u:return blockTotalY.Load(address);
    case 32u:return currenthandlesZ.Load(address);
    case 33u:return currentendsZ.Load(address);
    case 34u:return currentstartsZ.Load(address);
    case 35u:return currentorderedEndsZ.Load(address);
    case 36u:return currentorderedStartsZ.Load(address);
    case 37u:return currentblockEndsZ.Load(address);
    case 38u:return currentblockStartsZ.Load(address);
    case 39u:return previoushandlesZ.Load(address);
    case 40u:return previousendsZ.Load(address);
    case 41u:return previousstartsZ.Load(address);
    case 42u:return previousorderedEndsZ.Load(address);
    case 43u:return previousorderedStartsZ.Load(address);
    case 44u:return previousblockEndsZ.Load(address);
    case 45u:return previousblockStartsZ.Load(address);
    case 46u:return totalEndZ.Load(address);
    case 47u:return blockTotalZ.Load(address);
    default:return 0u;
    }
#else
    return sourceViews[slot].Load(address);
#endif
}
void store_field(uint slot,uint index,uint value) {
    const uint address=bias(slot)+4u*index;
#ifdef __spirv__
    switch(slot) {
    case 0u:currenthandlesX.Store(address,value);return;
    case 1u:currentendsX.Store(address,value);return;
    case 2u:currentstartsX.Store(address,value);return;
    case 3u:currentorderedEndsX.Store(address,value);return;
    case 4u:currentorderedStartsX.Store(address,value);return;
    case 5u:currentblockEndsX.Store(address,value);return;
    case 6u:currentblockStartsX.Store(address,value);return;
    case 7u:previoushandlesX.Store(address,value);return;
    case 8u:previousendsX.Store(address,value);return;
    case 9u:previousstartsX.Store(address,value);return;
    case 10u:previousorderedEndsX.Store(address,value);return;
    case 11u:previousorderedStartsX.Store(address,value);return;
    case 12u:previousblockEndsX.Store(address,value);return;
    case 13u:previousblockStartsX.Store(address,value);return;
    case 14u:totalEndX.Store(address,value);return;
    case 15u:blockTotalX.Store(address,value);return;
    case 16u:currenthandlesY.Store(address,value);return;
    case 17u:currentendsY.Store(address,value);return;
    case 18u:currentstartsY.Store(address,value);return;
    case 19u:currentorderedEndsY.Store(address,value);return;
    case 20u:currentorderedStartsY.Store(address,value);return;
    case 21u:currentblockEndsY.Store(address,value);return;
    case 22u:currentblockStartsY.Store(address,value);return;
    case 23u:previoushandlesY.Store(address,value);return;
    case 24u:previousendsY.Store(address,value);return;
    case 25u:previousstartsY.Store(address,value);return;
    case 26u:previousorderedEndsY.Store(address,value);return;
    case 27u:previousorderedStartsY.Store(address,value);return;
    case 28u:previousblockEndsY.Store(address,value);return;
    case 29u:previousblockStartsY.Store(address,value);return;
    case 30u:totalEndY.Store(address,value);return;
    case 31u:blockTotalY.Store(address,value);return;
    case 32u:currenthandlesZ.Store(address,value);return;
    case 33u:currentendsZ.Store(address,value);return;
    case 34u:currentstartsZ.Store(address,value);return;
    case 35u:currentorderedEndsZ.Store(address,value);return;
    case 36u:currentorderedStartsZ.Store(address,value);return;
    case 37u:currentblockEndsZ.Store(address,value);return;
    case 38u:currentblockStartsZ.Store(address,value);return;
    case 39u:previoushandlesZ.Store(address,value);return;
    case 40u:previousendsZ.Store(address,value);return;
    case 41u:previousstartsZ.Store(address,value);return;
    case 42u:previousorderedEndsZ.Store(address,value);return;
    case 43u:previousorderedStartsZ.Store(address,value);return;
    case 44u:previousblockEndsZ.Store(address,value);return;
    case 45u:previousblockStartsZ.Store(address,value);return;
    case 46u:totalEndZ.Store(address,value);return;
    case 47u:blockTotalZ.Store(address,value);return;
    default:return;
    }
#else
    sourceViews[slot].Store(address,value);
#endif
}
groupshared uint3 scanA[256];
groupshared uint3 scanB[256];
groupshared uint3 carry;
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    const uint axisBase=group.y*16u;
    const bool incremental=p.counts.w!=0u;
    const uint currentBoxes=incremental?p.counts.x:
        p.counts.x+p.counts.y-p.counts.z;
    [loop]for(uint frame=0u;frame<2u;++frame) {
        if(frame==1u&&(!incremental||p.counts.z==0u))break;
        const uint boxes=frame==0u?currentBoxes:p.counts.x;
        const uint n=boxes*2u;
        const uint base=axisBase+frame*7u;
        const uint required=(n+255u)/256u;
        const uint iterations=(required+31u)/32u;
        // The source fixed grid includes many empty blocks for small
        // populations. Empty blocks must publish zero counts, but need
        // no group scan or synchronization rounds.
        if(group.x*iterations*256u>=n) {
            if(thread.x==0u) {
                store_field(base+5u,group.x,0u);
                store_field(base+6u,group.x,0u);
                if(frame==0u)store_field(axisBase+15u,group.x,0u);
            }
            continue;
        }
        if(thread.x==0u)carry=uint3(0u,0u,0u);
        GroupMemoryBarrierWithGroupSync();
        for(uint iteration=0u;iteration<iterations;++iteration) {
            const uint index=(group.x*iterations+iteration)*256u+thread.x;
            const bool valid=index<n;
            const uint tagged=valid?load_field(base,index):0u;
            const bool start=(tagged&1u)!=0u;
            const bool deleted=(tagged&4u)!=0u;
            const uint3 flags=uint3(
                (valid&&start&&!deleted)?1u:0u,
                (valid&&!start&&!deleted)?1u:0u,
                (valid&&!start)?1u:0u);
            scanA[thread.x]=flags;
            GroupMemoryBarrierWithGroupSync();
            [unroll]for(uint shift=1u;shift<256u;shift<<=1u) {
                if(shift==1u||shift==4u||shift==16u||shift==64u) {
                    uint3 v=scanA[thread.x];
                    if(thread.x>=shift)v+=scanA[thread.x-shift];
                    scanB[thread.x]=v;
                } else {
                    uint3 v=scanB[thread.x];
                    if(thread.x>=shift)v+=scanB[thread.x-shift];
                    scanA[thread.x]=v;
                }
                GroupMemoryBarrierWithGroupSync();
            }
            if(valid) {
                const uint3 result=carry+scanA[thread.x]-flags;
                store_field(base+1u,index,result.y);
                store_field(base+2u,index,result.x);
                if(frame==0u)
                    store_field(axisBase+14u,index,result.z);
            }
            GroupMemoryBarrierWithGroupSync();
            if(thread.x==0u)carry+=scanA[255u];
            GroupMemoryBarrierWithGroupSync();
        }
        if(thread.x==0u) {
            store_field(base+5u,group.x,carry.y);
            store_field(base+6u,group.x,carry.x);
            if(frame==0u)store_field(axisBase+15u,group.x,carry.z);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
