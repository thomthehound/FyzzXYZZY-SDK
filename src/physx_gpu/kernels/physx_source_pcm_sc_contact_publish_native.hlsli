// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Publish Gu world contacts from the lane that finalizes the Sc EdgeIndex.
void publishScContacts(uint i,uint first,uint second,uint edge,uint result,
                       uint outputBias,uint transformBias,uint geometryBias,
                       uint edgeBias,uint shapeCapacity,uint edgeCapacity) {
    uint resultEdge=results.Load(result+12u);
    uint status=results.Load(result+8u);
    uint selected=results.Load(result+20u);
    uint regenerated=results.Load(result+28u);
    uint flags=min(selected,0x7fffffffu) |
        (regenerated!=0u ? 0x80000000u : 0u);
    uint dst=outputBias+i*96u;
    if(edge>=edgeCapacity || second>=shapeCapacity) {
        streamContacts.Store4(dst,uint4(resultEdge,0xffffffffu,1u,flags));
        return;
    }
    uint src=edgeBias+edge*240u;
    uint count=edgeCaches.Load(src+12u);
    uint emitted=0u;
    float3 normal=float3(0.f,0.f,0.f);
    if(status==0u && count>0u && count<=4u) {
        uint tb=transformBias+second*32u;
        float4 qb=asfloat(transforms.Load4(tb));
        float3 pb=asfloat(transforms.Load3(tb+16u));
        float distance=0.f;
        if(first<shapeCapacity) {
            uint ga=geometryBias+first*16u;
            uint gb=geometryBias+second*16u;
            distance=asfloat(geometry.Load(ga+12u))+
                     asfloat(geometry.Load(gb+12u));
        } else {
            status=1u;
        }
        if(status==0u) {
            float3 localNormal=float3(0.f,0.f,0.f);
            [unroll]
            for(uint j=0u;j<4u;++j) {
                if(j>=count)break;
                float3 current=asfloat(edgeCaches.Load3(
                    src+80u+j*40u+24u));
                if(regenerated!=0u) {
                    if(j==0u)localNormal=current;
                }else
                    localNormal+=current;
            }
            float sq=dot(localNormal,localNormal);
            if(!(sq>0.f) || !isfinite(sq)) {
                status=4u;
            }else {
                normal=rotateSource(qb,localNormal*(1.f/sqrt(sq)));
                [unroll]
                for(uint j=0u;j<4u;++j) {
                    if(j>=count)break;
                    uint contact=src+80u+j*40u;
                    float depth=asfloat(edgeCaches.Load(contact+36u));
                    if(regenerated==0u && depth>distance)continue;
                    float3 b=asfloat(edgeCaches.Load3(contact+12u));
                    float3 world=rotateSource(qb,b)+pb;
                    streamContacts.Store4(dst+32u+emitted*16u,
                        asuint(float4(world,depth)));
                    ++emitted;
                }
            }
        }
    }
    streamContacts.Store4(dst,uint4(resultEdge,count,status,flags));
    streamContacts.Store3(dst+16u,asuint(normal));
    streamContacts.Store(dst+28u,emitted);
}
