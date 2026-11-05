// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Sparse GuGJK and GuEPA continuations. The GJK simplex and EPA facet
// search compile as separate entrypoints; both consume one Sc worklist.
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
#include "physx_source_pcm_box_epa_native.hlsli"
#include "physx_source_pcm_sc_contact_publish_native.hlsli"

// One Sc pair owns one 32-thread group. Independent simplex subsets
// retain their source evaluation order through a stable final reduction.
groupshared GjkSupportPoint gjkGroupSimplex[4];
groupshared GjkClosest gjkGroupChoices[15];
groupshared GjkClosest gjkGroupBest;
groupshared GjkWitness gjkGroupWitness;
groupshared uint gjkGroupCount,gjkGroupState;
groupshared float3 gjkGroupClosest;
groupshared float gjkGroupPrevious,gjkGroupProjection;

GjkClosest gjkGroupSubset(uint mask,uint count) {
    GjkClosest candidate=(GjkClosest)0;
    candidate.dist2=3.402823466e+38f;
    if(mask>=(1u<<count))return candidate;
    uint ids[4]={0u,0u,0u,0u};
    uint n=0u;
    for(uint i=0u;i<count;++i)
        if((mask&(1u<<i))!=0u)ids[n++]=i;
    float weights[4]={0.f,0.f,0.f,0.f};
    weights[ids[0]]=1.f;
    if(n>1u) {
        const float3 origin=gjkGroupSimplex[ids[0]].q;
        const float3 d1=gjkGroupSimplex[ids[1]].q-origin;
        const float a=gjkDot(d1,d1);
        const float x=-gjkDot(d1,origin);
        const float epsilon=1.1920928955078125e-7f;
        float u=0.f,v=0.f,w=0.f;
        if(n==2u) {
            if(!(a>0.f) || !isfinite(a))return candidate;
            u=x/a;
        }else {
            const float3 d2=gjkGroupSimplex[ids[2]].q-origin;
            const float b=gjkDot(d1,d2);
            const float c=gjkDot(d2,d2);
            const float y=-gjkDot(d2,origin);
            if(n==3u) {
                const float scale=max(a,c);
                const float determinant=a*c-b*b;
                if(!(scale>0.f) ||
                   !(determinant>epsilon*scale*scale) ||
                   !isfinite(determinant))
                    return candidate;
                u=(x*c-y*b)/determinant;
                v=(y*a-x*b)/determinant;
            }else {
                const float3 d3=gjkGroupSimplex[ids[3]].q-origin;
                const float third=gjkDot(d3,d3);
                const float scale=max(a,max(c,third));
                const float3 cross23=cross(d2,d3);
                const float determinant=gjkDot(d1,cross23);
                // The Gram determinant is the squared tetrahedral
                // volume. Match the relative pivot rejection scale,
                // avoiding dynamic Gaussian-elimination scratch.
                if(!(scale>0.f) ||
                   !(determinant*determinant>
                     epsilon*scale*scale*scale) ||
                   !isfinite(determinant))
                    return candidate;
                const float3 minusOrigin=-origin;
                u=gjkDot(minusOrigin,cross23)/determinant;
                v=gjkDot(minusOrigin,cross(d3,d1))/determinant;
                w=gjkDot(minusOrigin,cross(d1,d2))/determinant;
            }
        }
        weights[ids[0]]=1.f-u-v-w;
        weights[ids[1]]=u;
        if(n>=3u)weights[ids[2]]=v;
        if(n==4u)weights[ids[3]]=w;
    }
    for(uint j=0u;j<n;++j)
        if(weights[ids[j]]< -1.e-5f)return candidate;
    precise float3 q=0.f,pa=0.f,pb=0.f;
    for(uint j=0u;j<count;++j) {
        precise float3 dq=gjkGroupSimplex[j].q*weights[j];
        precise float3 da=gjkGroupSimplex[j].a*weights[j];
        precise float3 db=gjkGroupSimplex[j].b*weights[j];
        q=q+dq;
        pa=pa+da;
        pb=pb+db;
    }
    candidate.closestPoint=q;
    candidate.a=pa;
    candidate.b=pb;
    candidate.mask=mask;
    candidate.dist2=gjkLength2(q);
    for(uint j=0u;j<4u;++j)candidate.weights[j]=weights[j];
    return candidate;
}

uint gjkGroupContinuation(GjkBoxPair pair,float contactDistance,
                          uint lane) {
    const float epsilon=0.005f*min(
        min(pair.halfA.x,min(pair.halfA.y,pair.halfA.z)),
        min(pair.halfB.x,min(pair.halfB.y,pair.halfB.z)));
    const float progressFactor=1.f-0.000225f;
    const bool pointCore=pair.radiusA>0.f || pair.radiusB>0.f;
    if(lane==0u) {
        gjkGroupCount=0u;
        gjkGroupState=0u;
        gjkGroupWitness=(GjkWitness)0;
        gjkGroupPrevious=3.402823466e+38f;
        gjkGroupClosest=gjkLength2(pair.aInB)>0.f
            ?pair.aInB:float3(1.f,0.f,0.f);
    }
    GroupMemoryBarrierWithGroupSync();
    // Terminal decisions are broadcast before a uniform wave exit.
    for(uint iteration=0u;iteration<32u;++iteration) {
        if(lane==0u && gjkGroupState==0u) {
            const float distance=sqrt(gjkLength2(gjkGroupClosest));
            if(gjkGroupCount>0u && distance<=epsilon && !pointCore)
                gjkGroupState=3u;
            else if(distance<=0.f)
                gjkGroupState=pointCore && gjkGroupCount>0u?3u:2u;
            else {
                const float3 normal=gjkGroupClosest*(1.f/distance);
                const GjkSupportPoint support=gjkSupport(pair,-gjkGroupClosest);
                const float projection=gjkDot(normal,support.q);
                gjkGroupProjection=projection;
                if(projection>contactDistance+pair.radiusA+pair.radiusB)
                    gjkGroupState=2u;
                else if(gjkGroupCount>0u &&
                        projection>distance*progressFactor) {
                    GjkWitness contact;
                    const bool accepted=gjkGroupBest.mask!=0u &&
                        gjkContact(pair,gjkGroupBest,contact) &&
                        contact.separation<=contactDistance;
                    if(accepted)gjkGroupWitness=contact;
                    gjkGroupState=accepted?1u:2u;
                }else {
                    bool duplicate=false;
                    for(uint j=0u;j<gjkGroupCount;++j)
                        if(support.ai==gjkGroupSimplex[j].ai &&
                           support.bi==gjkGroupSimplex[j].bi) {
                            duplicate=true;
                            break;
                        }
                    if(duplicate) {
                        if(projection>0.f) {
                            // Projection and simplex distance bound the
                            // true gap when support can no longer advance.
                            GjkWitness contact;
                            const bool accepted=gjkGroupBest.mask!=0u &&
                                gjkContact(pair,gjkGroupBest,contact) &&
                                contact.separation<=contactDistance;
                            if(accepted)gjkGroupWitness=contact;
                            gjkGroupState=accepted?1u:2u;
                        }else
                            gjkGroupState=pointCore?3u:2u;
                    }else if(gjkGroupCount==4u)
                        gjkGroupState=3u;
                    else
                        gjkGroupSimplex[gjkGroupCount++]=support;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if(WaveActiveAllTrue(gjkGroupState!=0u))break;
        // Each lane evaluates one simplex feature. Lane zero selects
        // the strict minimum in mask order after all candidates publish.
        if(lane<15u)
            gjkGroupChoices[lane]=gjkGroupSubset(lane+1u,gjkGroupCount);
        GroupMemoryBarrierWithGroupSync();
        if(lane==0u && gjkGroupState==0u) {
            GjkClosest best=(GjkClosest)0;
            best.dist2=3.402823466e+38f;
            for(uint mask=0u;mask<15u;++mask)
                if(gjkGroupChoices[mask].dist2<best.dist2)
                    best=gjkGroupChoices[mask];
            if(best.mask==0u)
                gjkGroupState=2u;
            else {
                gjkGroupBest=best;
                gjkGroupClosest=best.closestPoint;
                const float nextDistance=sqrt(best.dist2);
                uint write=0u;
                const uint count=gjkGroupCount;
                for(uint read=0u;read<count;++read)
                    if((best.mask&(1u<<read))!=0u &&
                       best.weights[read]>1.e-6f)
                        gjkGroupSimplex[write++]=gjkGroupSimplex[read];
                if(write==0u && count>0u)write=1u;
                gjkGroupCount=write;
                if(nextDistance>=gjkGroupPrevious) {
                    if(pointCore && gjkGroupProjection>0.f) {
                        GjkWitness contact;
                        const bool accepted=gjkContact(pair,best,contact) &&
                            contact.separation<=contactDistance;
                        if(accepted)gjkGroupWitness=contact;
                        gjkGroupState=accepted?1u:2u;
                    }else
                        gjkGroupState=pointCore?3u:2u;
                }else
                    gjkGroupPrevious=nextDistance;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if(WaveActiveAllTrue(gjkGroupState!=0u))break;
    }
    if(lane==0u && gjkGroupState==0u)
        gjkGroupState=pointCore && gjkGroupCount>0u?3u:2u;
    GroupMemoryBarrierWithGroupSync();
    return gjkGroupState;
}

[numthreads(32,1,1)]
void gjk(uint3 thread : SV_DispatchThreadID) {
    const uint lane=thread.x&31u;
    const uint workIndex=thread.x>>5u;
    uint i=work.Load(p.workBias+workIndex*4u);
    if(i>=p.pairCount)return;
    uint result=p.resultBias+i*32u;
    if(results.Load(result+8u)!=3u)return;
    if(lane==0u)results.Store(result+24u,0xA001u);
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
    if(lane==0u)results.Store(result+24u,0xA002u);
    const uint outcome=gjkGroupContinuation(boxPair,distance,lane);
    if(lane==0u)results.Store(result+24u,0xA003u);
    if(lane!=0u)return;
    const GjkWitness contact=gjkGroupWitness;
    const uint simplexCount=gjkGroupCount;
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
            edgeCaches.Store3(slot,asuint(gjkGroupSimplex[j].a));
            edgeCaches.Store3(slot+12u,asuint(gjkGroupSimplex[j].b));
            edgeCaches.Store(slot+24u,gjkGroupSimplex[j].ai);
            edgeCaches.Store(slot+28u,gjkGroupSimplex[j].bi);
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

[numthreads(1,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint i=work.Load(p.workBias+thread.x*4u);
    if(i>=p.pairCount)return;
    uint result=p.resultBias+i*32u;
    if(results.Load(result+8u)!=4u)return;
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
    const uint pending=p.edgeCacheBias+edge*240u;
    const uint simplexCount=results.Load(result+24u);
    GjkSupportPoint simplex[4];
    if(simplexCount==0u || simplexCount>4u) {
        results.Store(result+8u,5u);
        publishScContacts(i,a,b,edge,result,p.outputBias,
            p.transformBias,p.geometryBias,p.edgeCacheBias,
            p.shapeCapacity,p.edgeCapacity);
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
        boxPair,simplex,simplexCount,contact);
    if(!admitted) {
        // Failure to resolve a penetrating simplex is not separation.
        // Report an unresolved manifold rather than silently removing it.
        edgeCaches.Store(pending+12u,0u);
        edgeCaches.Store(pending+68u,1u);
        results.Store(result,0u);
        results.Store(result+20u,0u);
        results.Store(result+24u,0u);
        results.Store(result+8u,5u);
        publishScContacts(i,a,b,edge,result,p.outputBias,
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
