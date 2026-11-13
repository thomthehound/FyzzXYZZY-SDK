// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Sparse GuEPA continuation for a compacted penetration worklist.
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
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer streamContacts : register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer hullVertices : register(u8,space0);
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
    uint outputBias;
    uint hullBias;
    uint hullCount;    uint headerBias;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

#include "physx_source_pcm_box_math_native.hlsli"
#include "physx_source_pcm_box_gjk_native.hlsli"
#ifndef FYZZXYZZY_EPA_EMPTY_DISPATCH_PROBE
#include "physx_source_pcm_box_epa_native.hlsli"
#include "physx_source_pcm_sc_contact_publish_native.hlsli"
#endif

// Sparse admission is recorded in the dense Sc pair result buffer.
// One lane owns each pending pair. No simplex data is shared between lanes,
// and GJK requires no workgroup barriers or private global scratch.
[numthreads(64,1,1)]
void main(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID) {
#ifdef FYZZXYZZY_EPA_EMPTY_DISPATCH_PROBE
    const uint i=work.Load(p.workBias+group.x*4u);
    if(i<p.pairCount && lane.x==0u)
        results.Store(p.resultBias+i*32u+24u,0xE001u);
#else
    uint i=work.Load(p.workBias+group.x*4u);
    if(i>=p.pairCount)return;
    uint result=p.resultBias+i*32u;
    if(results.Load(result+8u)!=4u)return;
    uint source=p.pairBias+i*20u;
    uint a=pairs.Load(source),b=pairs.Load(source+4u);
    uint edge=pairs.Load(source+8u);
    uint first=pairs.Load(source+12u);
    uint limit=pairs.Load(source+16u)&0x1fffffffu;
    if(a>=p.shapeCapacity || b>=p.shapeCapacity || edge>=p.edgeCapacity ||
       first>p.candidateCapacity || limit>p.candidateCapacity-first)return;
    uint ta=p.transformBias+a*32u,tb=p.transformBias+b*32u;
    float3 pa=asfloat(transforms.Load3(ta+16u));
    float3 pb=asfloat(transforms.Load3(tb+16u));
    float4 qa=asfloat(transforms.Load4(ta));
    float4 qb=asfloat(transforms.Load4(tb));
    uint ga=p.geometryBias+a*64u,gb=p.geometryBias+b*64u;
    GjkBoxPair boxPair=(GjkBoxPair)0;
    boxPair.halfA=asfloat(geometry.Load3(ga));
    boxPair.halfB=asfloat(geometry.Load3(gb));
    boxPair.aInB=rotateSource(conjugate(qb),pa-pb);
    boxPair.relative=quatProduct(conjugate(qb),qa);
    boxPair.kindA=geometry.Load(ga+28u);
    boxPair.kindB=geometry.Load(gb+28u);
    if(boxPair.kindA>2u || boxPair.kindB>2u)return;
    boxPair.radiusA=boxPair.kindA==1u?boxPair.halfA.x:0.f;
    boxPair.radiusB=boxPair.kindB==1u?boxPair.halfB.x:0.f;
    boxPair.firstA=geometry.Load(ga+52u);
    boxPair.countA=geometry.Load(ga+56u);
    boxPair.firstB=geometry.Load(gb+52u);
    boxPair.countB=geometry.Load(gb+56u);
    if(boxPair.kindA==2u &&
       (boxPair.countA<4u || boxPair.firstA>p.hullCount ||
        boxPair.countA>p.hullCount-boxPair.firstA))return;
    if(boxPair.kindB==2u &&
       (boxPair.countB<4u || boxPair.firstB>p.hullCount ||
        boxPair.countB>p.hullCount-boxPair.firstB))return;
    if(boxPair.kindA==1u)boxPair.halfA=boxPair.radiusA;
    if(boxPair.kindB==1u)boxPair.halfB=boxPair.radiusB;
    // Support vertices retain their original local coordinates; EPA
    // convergence uses hull dimensions, not their origin-relative reach.
    if(boxPair.kindA==2u) {
        const float extent=asfloat(geometry.Load(ga+60u));
        if(!(extent>0.f) || !isfinite(extent))return;
        boxPair.halfA=float3(extent,extent,extent);
    }
    if(boxPair.kindB==2u) {
        const float extent=asfloat(geometry.Load(gb+60u));
        if(!(extent>0.f) || !isfinite(extent))return;
        boxPair.halfB=float3(extent,extent,extent);
    }
    float distance=asfloat(geometry.Load(ga+12u))+
                   asfloat(geometry.Load(gb+12u));
    const uint pending=p.edgeCacheBias+edge*240u;
    const uint simplexCount=results.Load(result+24u);
    GjkSupportPoint simplex[4];
    if(simplexCount==0u || simplexCount>4u) {
        if(lane.x==0u) {
            results.Store(result+8u,5u);
            publishScContacts(i,a,b,edge,result,p.outputBias,p.headerBias,
                p.transformBias,p.geometryBias,p.edgeCacheBias,
                p.shapeCapacity,p.edgeCapacity);
        }
        return;
    }
    for(uint j=0u;j<simplexCount;++j) {
        const uint slot=pending+80u+j*40u;
        simplex[j].a=asfloat(edgeCaches.Load3(slot));
        simplex[j].b=asfloat(edgeCaches.Load3(slot+12u));
        simplex[j].q=simplex[j].a-simplex[j].b;
        simplex[j].ai=edgeCaches.Load(slot+24u);
        simplex[j].bi=edgeCaches.Load(slot+28u);
    }
    GjkWitness contact;
    const bool admitted=epaBoxContinuation(
        boxPair,simplex,simplexCount,lane.x,contact);
    if(lane.x!=0u)return;
    if(!admitted) {
        // Failure to resolve a penetrating simplex is not separation.
        // Report an unresolved manifold rather than silently removing it.
        edgeCaches.Store(pending+12u,0u);
        edgeCaches.Store(pending+68u,1u);
        results.Store(result,0u);
        results.Store(result+20u,0u);
        results.Store(result+24u,0u);
        results.Store(result+8u,5u);
        publishScContacts(i,a,b,edge,result,p.outputBias,p.headerBias,
            p.transformBias,p.geometryBias,p.edgeCacheBias,
            p.shapeCapacity,p.edgeCapacity);
        return;
    }
    uint cache=p.edgeCacheBias+edge*240u;
    if(admitted && limit==0u) {
        // A valid contact with no destination must regenerate later.
        results.Store(result+8u,2u);
        edgeCaches.Store(cache+12u,0u);
        edgeCaches.Store(cache+64u,0u);
        edgeCaches.Store(cache+68u,0u);
        publishScContacts(i,a,b,edge,result,p.outputBias,p.headerBias,
        p.transformBias,p.geometryBias,p.edgeCacheBias,
        p.shapeCapacity,p.edgeCapacity);
        return;
    }
    if(admitted) {
        const uint destination=p.candidateBias+first*40u;
        candidates.Store3(destination,asuint(contact.localA));
        candidates.Store3(destination+12u,asuint(contact.localB));
        candidates.Store3(destination+24u,asuint(contact.normalB));
        candidates.Store(destination+36u,asuint(contact.separation));
        const uint persistent=cache+80u;
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
    publishScContacts(i,a,b,edge,result,p.outputBias,p.headerBias,
        p.transformBias,p.geometryBias,p.edgeCacheBias,
        p.shapeCapacity,p.edgeCapacity);
#endif
}
