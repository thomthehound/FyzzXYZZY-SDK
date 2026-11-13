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
#ifdef FYZZXYZZY_GJK_STAGED_ADVANCE
    uint gjkPass;
#endif    uint headerBias;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

#include "physx_source_pcm_box_math_native.hlsli"
#include "physx_source_pcm_box_gjk_native.hlsli"
#ifdef FYZZXYZZY_GJK_STAGED_ADVANCE
#include "physx_source_pcm_gjk_state_native.hlsli"
#endif
#include "physx_source_pcm_sc_contact_publish_native.hlsli"

// Sparse admission is recorded in the dense Sc pair result buffer.
// One lane owns each pending pair. No simplex data is shared between lanes,
// and GJK requires no workgroup barriers or private global scratch.
[numthreads(32,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
#ifdef FYZZXYZZY_GJK_STAGED_ADVANCE
    if(thread.x>=work.Load(p.workBias))return;
    const uint i=work.Load(p.workBias+(thread.x+1u)*4u);
#else
    const uint i=thread.x;
#endif
    if(i>=p.pairCount)return;
    uint result=p.resultBias+i*32u;
    if(results.Load(result+8u)!=3u)return;
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
    GjkWitness contact=(GjkWitness)0;
    GjkSupportPoint simplex[4];
    uint simplexCount=0u;
#ifdef FYZZXYZZY_GJK_STAGED_ADVANCE
    const uint cacheBase=p.edgeCacheBias+edge*240u;
    GjkGpuState state;
#ifdef FYZZXYZZY_GJK_STAGED_PUBLISH
    state=gjkLoadGpuState(cacheBase);
    if(state.phase!=2u)return;
    uint outcome=state.outcome;
#else
    if(p.gjkPass==0u)state=gjkInitialState(boxPair,distance);
    else state=gjkLoadGpuState(cacheBase);
    uint outcome=0u;
    if(state.phase==2u) {
        outcome=state.outcome;
    }else if(p.gjkPass==32u) {
        outcome=state.pointCore!=0u && state.simplex.count>0u?3u:0u;
    }else {
        const float dist=sqrt(gjkLength2(state.closest));
        const float epsilon=0.005f*min(
            min(boxPair.halfA.x,min(boxPair.halfA.y,boxPair.halfA.z)),
            min(boxPair.halfB.x,min(boxPair.halfB.y,boxPair.halfB.z)));
        if(state.simplex.count>0u && dist<=epsilon && state.pointCore==0u) {
            outcome=3u;
        }else if(!(dist>0.f)) {
            outcome=state.pointCore!=0u && state.simplex.count>0u?3u:0u;
        }else {
            const GjkSupportPoint support=gjkSupport(boxPair,-state.closest);
            const float projection=gjkDot(state.closest*(1.f/dist),support.q);
            if(projection>state.contactLimit) {
                outcome=0u;
            }else if(state.simplex.count>0u &&
                     projection>dist*(1.f-0.000225f)) {
                outcome=1u;
            }else {
                const bool duplicate=
                    (state.simplex.count>0u && support.ai==state.simplex.point0.ai &&
                     support.bi==state.simplex.point0.bi) ||
                    (state.simplex.count>1u && support.ai==state.simplex.point1.ai &&
                     support.bi==state.simplex.point1.bi) ||
                    (state.simplex.count>2u && support.ai==state.simplex.point2.ai &&
                     support.bi==state.simplex.point2.bi) ||
                    (state.simplex.count>3u && support.ai==state.simplex.point3.ai &&
                     support.bi==state.simplex.point3.bi);
                if(duplicate)outcome=projection>0.f?1u:(state.pointCore!=0u?3u:0u);
                else if(state.simplex.count==4u)outcome=3u;
                else {
                    gjkAppend(state.simplex,support);
                    state.projection=projection;
                    state.phase=1u;
                    gjkStoreGpuState(cacheBase,state);
                    return;
                }
            }
        }
    }
    state.phase=2u;
    state.outcome=outcome;
    gjkStoreGpuState(cacheBase,state);
    return;
#endif
    if(outcome==1u) {
        const GjkClosest nearest=gjkCachedClosest(state);
        outcome=nearest.mask!=0u &&
                gjkContact(boxPair,nearest,state.simplex,contact) &&
                contact.separation<=distance?1u:0u;
    }
    simplexCount=state.simplex.count;
    [unroll]
    for(uint j=0u;j<4u;++j)simplex[j]=(GjkSupportPoint)0;
    if(outcome==3u) {
        if(simplexCount>0u)simplex[0]=gjkRestoreSupport(boxPair,state.simplex.point0);
        if(simplexCount>1u)simplex[1]=gjkRestoreSupport(boxPair,state.simplex.point1);
        if(simplexCount>2u)simplex[2]=gjkRestoreSupport(boxPair,state.simplex.point2);
        if(simplexCount>3u)simplex[3]=gjkRestoreSupport(boxPair,state.simplex.point3);
    }
#else
    const uint outcome=gjkBoxContinuation(
        boxPair,distance,contact,simplex,simplexCount);
#endif
    if(outcome==3u) {
        const uint pending=p.edgeCacheBias+edge*240u;
        if(simplexCount==0u || simplexCount>4u) {
            results.Store(result+8u,5u);
            publishScContacts(i,a,b,edge,result,p.outputBias,p.headerBias,
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
}

