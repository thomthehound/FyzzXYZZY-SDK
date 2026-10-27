// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// PhysX RS_MULTIBLOCK_COUNT. Exact 32x3x1024 uint4 partitions,
// 16x32 radix histogram, no CUDA pointers or intermediate packing.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer keysX:register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer keysY:register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer keysZ:register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer countsX:register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer countsY:register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer countsZ:register(u5,space0);
struct Params {
    uint kx;
    uint ky;
    uint kz;
    uint hx;
    uint hy;
    uint hz;
    uint numKeys;
    uint startBit;
};
#ifdef __spirv__
[[vk::push_constant]] Params p;
#else
ConstantBuffer<Params> p:register(b0,space0);
#endif

groupshared uint totals[16];
uint key(uint axis,uint byteOffset) {
    if(axis==0u)return keysX.Load(byteOffset);
    if(axis==1u)return keysY.Load(byteOffset);
    return keysZ.Load(byteOffset);
}
void count_store(uint axis,uint byteOffset,uint value) {
    if(axis==0u)countsX.Store(byteOffset,value);
    else if(axis==1u)countsY.Store(byteOffset,value);
    else countsZ.Store(byteOffset,value);
}
[numthreads(1024,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    const uint block=group.x,axis=group.y;
    const uint input=axis==0u?p.kx:(axis==1u?p.ky:p.kz);
    const uint output=axis==0u?p.hx:(axis==1u?p.hy:p.hz);
    if(thread.x<16u)totals[thread.x]=0u;
    GroupMemoryBarrierWithGroupSync();
    const uint vectors=(p.numKeys+3u)/4u;
    const uint required=(vectors+1023u)/1024u;
    const uint iterations=(required+31u)/32u;
    const uint first=min(iterations*block*1024u,vectors);
    const uint last=min(first+iterations*1024u,vectors);
    for(uint v=first+thread.x;v<last;v+=1024u) {
        [unroll]for(uint lane=0u;lane<4u;++lane) {
            const uint index=v*4u+lane;
            const uint value=index<p.numKeys?key(axis,input+index*4u):0xffffffffu;
            uint unused;
            InterlockedAdd(totals[(value>>p.startBit)&15u],1u,unused);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if(thread.x<16u)
        count_store(axis,output+(thread.x*32u+block)*4u,totals[thread.x]);
}
