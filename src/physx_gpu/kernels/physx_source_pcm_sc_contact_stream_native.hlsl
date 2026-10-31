// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Sc's original contact-manager edges own persistent Gu PCM witnesses.
// Publish only live Sc pairs into a compact, pair-ordered GPU contact stream.
// The source edge cache remains device-resident and is not retired here.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer pairs : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer results : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer edgeCaches : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer streamContacts : register(u3,space0);

struct Parameters {
    uint pairCount;
    uint pairBias;
    uint resultBias;
    uint edgeBias;
    uint outputBias;
    uint edgeCapacity;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

[numthreads(128,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint i=thread.x;
    if(i>=p.pairCount)return;
    uint edge=pairs.Load(p.pairBias+i*20u+8u);
    uint dst=p.outputBias+i*176u;
    uint result=p.resultBias+i*32u;
    uint resultEdge=results.Load(result+12u);
    uint status=results.Load(result+8u);
    uint selected=results.Load(result+20u);
    uint regenerated=results.Load(result+28u);
    // Dy/Sc publication uses the same complete result validation without
    // a second readback of the GPU's per-pair result buffer.
    uint flags=min(selected,0x7fffffffu) |
        (regenerated!=0u ? 0x80000000u : 0u);
    if(edge>=p.edgeCapacity) {
        streamContacts.Store4(dst,uint4(resultEdge,0xffffffffu,status,flags));
        return;
    }
    uint src=p.edgeBias+edge*240u;
    uint count=edgeCaches.Load(src+12u);
    streamContacts.Store4(dst,uint4(resultEdge,count,status,flags));
    if(count>4u)return;
    [unroll]
    for(uint j=0u;j<4u;++j) {
        if(j>=count)break;
        uint from=src+80u+j*40u;
        uint to=dst+16u+j*40u;
        streamContacts.Store4(to,edgeCaches.Load4(from));
        streamContacts.Store4(to+16u,edgeCaches.Load4(from+16u));
        streamContacts.Store2(to+32u,edgeCaches.Load2(from+32u));
    }
}
