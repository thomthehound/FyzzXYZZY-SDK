// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Sparse GuGJK continuation across independent Sc pair lanes.
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
    uint hullCount;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

#include "physx_source_pcm_box_math_native.hlsli"
#include "physx_source_pcm_box_gjk_native.hlsli"
#include "physx_source_pcm_sc_contact_publish_native.hlsli"

// Sparse admission is recorded in the dense Sc pair result buffer.
// One lane owns each pending pair. No simplex data is shared between lanes,
// and GJK requires no workgroup barriers or private global scratch.
[numthreads(32,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    const uint i=thread.x;
    if(i>=p.pairCount)return;
    uint result=p.resultBias+i*32u;
    if(results.Load(result+8u)!=3u)return;
    uint source=p.pairBias+i*20u;
    uint a=pairs.Load(source),b=pairs.Load(source+4u);
    uint edge=pairs.Load(source+8u);
    uint first=pairs.Load(source+12u);
    uint limit=pairs.Load(source+16u)&0x7fffffffu;
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
#ifdef FYZZXYZZY_GJK_PROBE_BINDINGS
    // Dispatch probe: complete the same Sc pending-pair read and
    // geometry admission path without invoking the simplex solver.
    results.Store(result+24u,0xB001u);
    results.Store(result+8u,0u);
    return;
#endif
#ifdef FYZZXYZZY_GJK_PROBE_SUPPORT
    const float3 direction=gjkLength2(boxPair.aInB)>0.f
        ?-boxPair.aInB:float3(-1.f,0.f,0.f);
    const GjkSupportPoint support=gjkSupport(boxPair,direction);
    results.Store(result+24u,0xB100u|(support.ai&0xffu));
    results.Store(result+8u,0u);
    return;
#endif
    GjkWitness contact;
    GjkSupportPoint simplex[4];
    uint simplexCount;
    const uint outcome=gjkBoxContinuation(
        boxPair,distance,contact,simplex,simplexCount);
#if defined(FYZZXYZZY_GJK_PROBE_ITERATION)
    results.Store(result+24u,0xB300u|outcome);
    results.Store(result+8u,0u);
    return;
#elif defined(FYZZXYZZY_GJK_PROBE_SOLVER)
    // Distinguish simplex execution from subsequent Sc publication.
    results.Store(result+24u,0xB200u|outcome);
    results.Store(result+8u,0u);
    return;
#endif
    if(outcome==3u) {
        const uint pending=p.edgeCacheBias+edge*240u;
        if(simplexCount==0u || simplexCount>4u) {
            results.Store(result+8u,5u);
            publishScContacts(i,a,b,edge,result,p.outputBias,
                p.transformBias,p.geometryBias,p.edgeCacheBias,
                p.shapeCapacity,p.edgeCapacity);
            return;
        }
        // The Sc manager has no live contacts while EPA is pending.
        // Reuse its four 40-byte witness slots for the four simplex
        // supports, retaining both local points and exact feature IDs.
        for(uint j=0u;j<simplexCount;++j) {
            const uint slot=pending+80u+j*40u;
            edgeCaches.Store3(slot,asuint(simplex[j].a));
            edgeCaches.Store3(slot+12u,asuint(simplex[j].b));
            edgeCaches.Store(slot+24u,simplex[j].ai);
            edgeCaches.Store(slot+28u,simplex[j].bi);
        }
        results.Store(result+24u,simplexCount);
        results.Store(result+8u,4u);
        return;
    }
    const bool admitted=outcome==1u;
    uint cache=p.edgeCacheBias+edge*240u;
    if(admitted && limit==0u) {
        // A valid contact with no destination must regenerate later.
        results.Store(result+8u,2u);
        edgeCaches.Store(cache+12u,0u);
        edgeCaches.Store(cache+64u,0u);
        edgeCaches.Store(cache+68u,0u);
        publishScContacts(i,a,b,edge,result,p.outputBias,
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
    publishScContacts(i,a,b,edge,result,p.outputBias,
        p.transformBias,p.geometryBias,p.edgeCacheBias,
        p.shapeCapacity,p.edgeCapacity);
}

