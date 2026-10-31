// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// GuGJKPenetration / GuEPA sparse source continuation. One indirect
// workgroup owns one Sc pair; no EPA facet storage enters the SAT shader.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer transforms : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer geometry : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer pairs : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer results : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer candidates : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer edgeCaches : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer work : register(u6,space0);

struct Parameters {
    uint pairCount;
    uint transformBias;
    uint geometryBias;
    uint pairBias;
    uint resultBias;
    uint candidateBias;
    uint edgeCacheBias;
    uint workBias;
    uint shapeCapacity;
    uint candidateCapacity;
    uint edgeCapacity;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

#include "physx_source_pcm_box_math_native.hlsli"
#include "physx_source_pcm_box_gjk_native.hlsli"
#include "physx_source_pcm_box_epa_native.hlsli"

[numthreads(1,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint i=work.Load(p.workBias+thread.x*4u);
    if(i>=p.pairCount)return;
    uint result=p.resultBias+i*32u;
    if(results.Load(result+8u)!=3u)return;
    uint source=p.pairBias+i*20u;
    uint a=pairs.Load(source),b=pairs.Load(source+4u);
    uint edge=pairs.Load(source+8u);
    uint first=pairs.Load(source+12u),limit=pairs.Load(source+16u);
    if(a>=p.shapeCapacity || b>=p.shapeCapacity || edge>=p.edgeCapacity ||
       first>p.candidateCapacity || limit>p.candidateCapacity-first)return;
    uint ta=p.transformBias+a*32u,tb=p.transformBias+b*32u;
    float3 pa=asfloat(transforms.Load3(ta+16u));
    float3 pb=asfloat(transforms.Load3(tb+16u));
    float4 qa=asfloat(transforms.Load4(ta));
    float4 qb=asfloat(transforms.Load4(tb));
    uint ga=p.geometryBias+a*16u,gb=p.geometryBias+b*16u;
    GjkBoxPair boxPair;
    boxPair.halfA=asfloat(geometry.Load3(ga));
    boxPair.halfB=asfloat(geometry.Load3(gb));
    boxPair.aInB=rotateSource(conjugate(qb),pa-pb);
    boxPair.relative=quatProduct(conjugate(qb),qa);
    float distance=asfloat(geometry.Load(ga+12u))+
                   asfloat(geometry.Load(gb+12u));
    GjkWitness contact;
    GjkSupportPoint simplex[4];
    uint simplexCount;
    uint outcome=gjkBoxContinuation(
        boxPair,distance,contact,simplex,simplexCount);
    bool admitted=outcome==1u;
    if(outcome==3u)
        admitted=epaBoxContinuation(
            boxPair,simplex,simplexCount,contact);
    uint cache=p.edgeCacheBias+edge*240u;
    if(admitted && limit==0u) {
        // A valid contact with no destination must regenerate later.
        results.Store(result+8u,2u);
        edgeCaches.Store(cache+12u,0u);
        edgeCaches.Store(cache+64u,0u);
        edgeCaches.Store(cache+68u,0u);
        return;
    }
    if(admitted) {
        uint destination=p.candidateBias+first*40u;
        candidates.Store3(destination,asuint(contact.localA));
        candidates.Store3(destination+12u,asuint(contact.localB));
        candidates.Store3(destination+24u,asuint(contact.normalB));
        candidates.Store(destination+36u,asuint(contact.separation));
        uint persistent=cache+80u;
        edgeCaches.Store3(persistent,asuint(contact.localA));
        edgeCaches.Store3(persistent+12u,asuint(contact.localB));
        edgeCaches.Store3(persistent+24u,asuint(contact.normalB));
        edgeCaches.Store(persistent+36u,asuint(contact.separation));
        edgeCaches.Store(cache+12u,1u);
        results.Store(result,1u);
        results.Store(result+20u,1u);
        results.Store(result+24u,0u);
    }else {
        edgeCaches.Store(cache+12u,0u);
        results.Store(result,0u);
        results.Store(result+4u,0u);
        results.Store(result+20u,0u);
        results.Store(result+24u,0u);
    }
    edgeCaches.Store(cache+68u,0u);
    results.Store(result+8u,0u);
}
