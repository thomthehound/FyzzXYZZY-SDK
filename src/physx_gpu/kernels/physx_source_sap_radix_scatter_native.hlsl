// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// PxgCudaBroadPhaseSap::runRadixSort / RS_CALCULATERANKS_MULTIBLOCK_COUNT.
// Stable 4-bit LSD radix, source 32x3x1024 work distribution. Original
// key/payload ping-pong pointers; histograms stay in the source allocation.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer inKeyX:register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer inRankX:register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer outKeyX:register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer outRankX:register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer countX:register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer inKeyY:register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer inRankY:register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer outKeyY:register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer outRankY:register(u8,space0);
#ifdef __spirv__
[[vk::binding(9,0)]]
#endif
RWByteAddressBuffer countY:register(u9,space0);
#ifdef __spirv__
[[vk::binding(10,0)]]
#endif
RWByteAddressBuffer inKeyZ:register(u10,space0);
#ifdef __spirv__
[[vk::binding(11,0)]]
#endif
RWByteAddressBuffer inRankZ:register(u11,space0);
#ifdef __spirv__
[[vk::binding(12,0)]]
#endif
RWByteAddressBuffer outKeyZ:register(u12,space0);
#ifdef __spirv__
[[vk::binding(13,0)]]
#endif
RWByteAddressBuffer outRankZ:register(u13,space0);
#ifdef __spirv__
[[vk::binding(14,0)]]
#endif
RWByteAddressBuffer countZ:register(u14,space0);
struct Params {
    uint ikx;
    uint irx;
    uint okx;
    uint orx;
    uint cx;
    uint iky;
    uint iry;
    uint oky;
    uint ory;
    uint cy;
    uint ikz;
    uint irz;
    uint okz;
    uint orz;
    uint cz;
    uint numKeys;
    uint startBit;
    uint pad0;
    uint pad1;
    uint pad2;
};
#ifdef __spirv__
[[vk::push_constant]] Params p;
#else
ConstantBuffer<Params> p:register(b0,space0);
#endif

uint load_key(uint axis,uint byteOffset) {if(axis==0u)return inKeyX.Load(byteOffset);if(axis==1u)return inKeyY.Load(byteOffset);return inKeyZ.Load(byteOffset);}
uint load_rank(uint axis,uint byteOffset) {if(axis==0u)return inRankX.Load(byteOffset);if(axis==1u)return inRankY.Load(byteOffset);return inRankZ.Load(byteOffset);}
uint load_count(uint axis,uint byteOffset) {if(axis==0u)return countX.Load(byteOffset);if(axis==1u)return countY.Load(byteOffset);return countZ.Load(byteOffset);}
void store_key(uint axis,uint byteOffset,uint value) {if(axis==0u)outKeyX.Store(byteOffset,value);else if(axis==1u)outKeyY.Store(byteOffset,value);else outKeyZ.Store(byteOffset,value);}
void store_rank(uint axis,uint byteOffset,uint value) {if(axis==0u)outRankX.Store(byteOffset,value);else if(axis==1u)outRankY.Store(byteOffset,value);else outRankZ.Store(byteOffset,value);}
groupshared uint bucketBase[16];
groupshared uint carriedCount[16];
groupshared uint scanValues[2048];
[numthreads(1024,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    const uint block=group.x,axis=group.y;
    const uint kIn=axis==0u?p.ikx:(axis==1u?p.iky:p.ikz);
    const uint rIn=axis==0u?p.irx:(axis==1u?p.iry:p.irz);
    const uint kOut=axis==0u?p.okx:(axis==1u?p.oky:p.okz);
    const uint rOut=axis==0u?p.orx:(axis==1u?p.ory:p.orz);
    const uint hist=axis==0u?p.cx:(axis==1u?p.cy:p.cz);
    const uint vectors=(p.numKeys+3u)/4u;
    const uint required=(vectors+1023u)/1024u;
    const uint iterations=(required+31u)/32u;
    if(iterations*block*1024u>=vectors)return;
    if(thread.x<16u) {
        uint prefix=0u;
        for(uint digit=0u;digit<thread.x;++digit)
            for(uint b=0u;b<32u;++b)
                prefix+=load_count(axis,hist+(digit*32u+b)*4u);
        for(uint b=0u;b<block;++b)
            prefix+=load_count(axis,hist+(thread.x*32u+b)*4u);
        bucketBase[thread.x]=prefix;
        carriedCount[thread.x]=0u;
    }
    GroupMemoryBarrierWithGroupSync();
    for(uint iteration=0u;iteration<iterations;++iteration) {
        const uint vectorIndex=iterations*block*1024u+iteration*1024u+thread.x;
        const bool valid=vectorIndex<vectors;
        uint4 keys=uint4(0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu);
        uint4 ranks=uint4(0u,0u,0u,0u);
        if(valid) {
            [unroll]for(uint lane=0u;lane<4u;++lane) {
                const uint index=vectorIndex*4u+lane;
                if(index<p.numKeys) {
                    keys[lane]=load_key(axis,kIn+index*4u);
                    ranks[lane]=load_rank(axis,rIn+index*4u);
                }
            }
        }
        const uint4 digits=(keys>>p.startBit)&15u;
        [loop]for(uint digit=0u;digit<16u;++digit) {
            uint mine=0u;
            if(valid) {
                [unroll]for(uint lane=0u;lane<4u;++lane)
                    mine+=(digits[lane]==digit)?1u:0u;
            }
            scanValues[thread.x]=mine;
            GroupMemoryBarrierWithGroupSync();
            uint sourceBank=0u,targetBank=1024u;
            [unroll]for(uint distance=1u;distance<1024u;distance<<=1u) {
                uint sum=scanValues[sourceBank+thread.x];
                if(thread.x>=distance)sum+=scanValues[sourceBank+thread.x-distance];
                scanValues[targetBank+thread.x]=sum;
                GroupMemoryBarrierWithGroupSync();
                const uint temp=sourceBank;sourceBank=targetBank;targetBank=temp;
            }
            const uint exclusive=scanValues[sourceBank+thread.x]-mine;
            const uint base=bucketBase[digit]+carriedCount[digit]+exclusive;
            uint preceding=0u;
            if(valid) {
                [unroll]for(uint lane=0u;lane<4u;++lane) {
                    if(digits[lane]==digit) {
                        const uint index=vectorIndex*4u+lane;
                        if(index<p.numKeys) {
                            const uint dst=base+preceding;
                            store_key(axis,kOut+dst*4u,keys[lane]);
                            store_rank(axis,rOut+dst*4u,ranks[lane]);
                        }
                        ++preceding;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            if(thread.x==0u)
                carriedCount[digit]+=scanValues[sourceBank+1023u];
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
