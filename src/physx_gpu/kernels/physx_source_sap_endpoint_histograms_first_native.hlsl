// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// PxgCudaBroadPhaseSap::calculateEndPtHistogramKernel
// 32 x 3 x 256, original 2-frame endpoint histories and current totals.
// Each source view retains its original GPU allocation; only 16-bit RAW
// binding offsets are packed, avoiding any GPU-side descriptor upload.
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
    if(slot==0u)return currenthandlesX.Load(address);
    if(slot==1u)return currentendsX.Load(address);
    if(slot==2u)return currentstartsX.Load(address);
    if(slot==3u)return currentorderedEndsX.Load(address);
    if(slot==4u)return currentorderedStartsX.Load(address);
    if(slot==5u)return currentblockEndsX.Load(address);
    if(slot==6u)return currentblockStartsX.Load(address);
    if(slot==7u)return previoushandlesX.Load(address);
    if(slot==8u)return previousendsX.Load(address);
    if(slot==9u)return previousstartsX.Load(address);
    if(slot==10u)return previousorderedEndsX.Load(address);
    if(slot==11u)return previousorderedStartsX.Load(address);
    if(slot==12u)return previousblockEndsX.Load(address);
    if(slot==13u)return previousblockStartsX.Load(address);
    if(slot==14u)return totalEndX.Load(address);
    if(slot==15u)return blockTotalX.Load(address);
    if(slot==16u)return currenthandlesY.Load(address);
    if(slot==17u)return currentendsY.Load(address);
    if(slot==18u)return currentstartsY.Load(address);
    if(slot==19u)return currentorderedEndsY.Load(address);
    if(slot==20u)return currentorderedStartsY.Load(address);
    if(slot==21u)return currentblockEndsY.Load(address);
    if(slot==22u)return currentblockStartsY.Load(address);
    if(slot==23u)return previoushandlesY.Load(address);
    if(slot==24u)return previousendsY.Load(address);
    if(slot==25u)return previousstartsY.Load(address);
    if(slot==26u)return previousorderedEndsY.Load(address);
    if(slot==27u)return previousorderedStartsY.Load(address);
    if(slot==28u)return previousblockEndsY.Load(address);
    if(slot==29u)return previousblockStartsY.Load(address);
    if(slot==30u)return totalEndY.Load(address);
    if(slot==31u)return blockTotalY.Load(address);
    if(slot==32u)return currenthandlesZ.Load(address);
    if(slot==33u)return currentendsZ.Load(address);
    if(slot==34u)return currentstartsZ.Load(address);
    if(slot==35u)return currentorderedEndsZ.Load(address);
    if(slot==36u)return currentorderedStartsZ.Load(address);
    if(slot==37u)return currentblockEndsZ.Load(address);
    if(slot==38u)return currentblockStartsZ.Load(address);
    if(slot==39u)return previoushandlesZ.Load(address);
    if(slot==40u)return previousendsZ.Load(address);
    if(slot==41u)return previousstartsZ.Load(address);
    if(slot==42u)return previousorderedEndsZ.Load(address);
    if(slot==43u)return previousorderedStartsZ.Load(address);
    if(slot==44u)return previousblockEndsZ.Load(address);
    if(slot==45u)return previousblockStartsZ.Load(address);
    if(slot==46u)return totalEndZ.Load(address);
    if(slot==47u)return blockTotalZ.Load(address);
    return 0u;
}
void store_field(uint slot,uint index,uint value) {
    const uint address=bias(slot)+4u*index;
    if(slot==1u)currentendsX.Store(address,value);
    if(slot==2u)currentstartsX.Store(address,value);
    if(slot==3u)currentorderedEndsX.Store(address,value);
    if(slot==4u)currentorderedStartsX.Store(address,value);
    if(slot==5u)currentblockEndsX.Store(address,value);
    if(slot==6u)currentblockStartsX.Store(address,value);
    if(slot==8u)previousendsX.Store(address,value);
    if(slot==9u)previousstartsX.Store(address,value);
    if(slot==10u)previousorderedEndsX.Store(address,value);
    if(slot==11u)previousorderedStartsX.Store(address,value);
    if(slot==12u)previousblockEndsX.Store(address,value);
    if(slot==13u)previousblockStartsX.Store(address,value);
    if(slot==14u)totalEndX.Store(address,value);
    if(slot==15u)blockTotalX.Store(address,value);
    if(slot==17u)currentendsY.Store(address,value);
    if(slot==18u)currentstartsY.Store(address,value);
    if(slot==19u)currentorderedEndsY.Store(address,value);
    if(slot==20u)currentorderedStartsY.Store(address,value);
    if(slot==21u)currentblockEndsY.Store(address,value);
    if(slot==22u)currentblockStartsY.Store(address,value);
    if(slot==24u)previousendsY.Store(address,value);
    if(slot==25u)previousstartsY.Store(address,value);
    if(slot==26u)previousorderedEndsY.Store(address,value);
    if(slot==27u)previousorderedStartsY.Store(address,value);
    if(slot==28u)previousblockEndsY.Store(address,value);
    if(slot==29u)previousblockStartsY.Store(address,value);
    if(slot==30u)totalEndY.Store(address,value);
    if(slot==31u)blockTotalY.Store(address,value);
    if(slot==33u)currentendsZ.Store(address,value);
    if(slot==34u)currentstartsZ.Store(address,value);
    if(slot==35u)currentorderedEndsZ.Store(address,value);
    if(slot==36u)currentorderedStartsZ.Store(address,value);
    if(slot==37u)currentblockEndsZ.Store(address,value);
    if(slot==38u)currentblockStartsZ.Store(address,value);
    if(slot==40u)previousendsZ.Store(address,value);
    if(slot==41u)previousstartsZ.Store(address,value);
    if(slot==42u)previousorderedEndsZ.Store(address,value);
    if(slot==43u)previousorderedStartsZ.Store(address,value);
    if(slot==44u)previousblockEndsZ.Store(address,value);
    if(slot==45u)previousblockStartsZ.Store(address,value);
    if(slot==46u)totalEndZ.Store(address,value);
    if(slot==47u)blockTotalZ.Store(address,value);
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
