// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Device-resident convex face expansion follows GuGJK/GuEPA rather than
// sharing its large per-thread simplex/facet working set.
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
#ifdef __spirv__
[[vk::binding(9,0)]]
#endif
RWByteAddressBuffer hullSurfaces : register(u9,space0);

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
    uint surfaceBias;
    uint surfaceFacesBase;
    uint surfaceIndicesBase;
    uint surfaceBytes;    uint headerBias;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

#include "physx_source_pcm_box_math_native.hlsli"
#include "physx_source_pcm_box_gjk_native.hlsli"
#include "physx_source_pcm_convex_face_native.hlsli"
#include "physx_source_pcm_sc_contact_publish_native.hlsli"

[numthreads(64,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    if(p.surfaceBytes==0u)return;
    const uint i=thread.x;
    if(i>=p.pairCount)return;
    const uint source=p.pairBias+i*20u;
    if((pairs.Load(source+16u)&0x10000000u)!=0u)return;
    const uint result=p.resultBias+i*32u;
    // Direct GJK and sparse EPA both publish into Sc's result stream.
    // Quiet cached managers retain their existing four-point manifold.
    if(results.Load(result+8u)!=0u ||
       results.Load(result)!=1u ||
       results.Load(result+20u)!=1u ||
       results.Load(result+28u)==0u)return;
    const uint a=pairs.Load(source),b=pairs.Load(source+4u);
    const uint edge=pairs.Load(source+8u);
    const uint first=pairs.Load(source+12u);
    const uint limit=pairs.Load(source+16u)&0x0fffffffu;
    if(a>=p.shapeCapacity || b>=p.shapeCapacity ||
       edge>=p.edgeCapacity || first>p.candidateCapacity ||
       limit>p.candidateCapacity-first)return;
    const uint ga=p.geometryBias+a*64u,gb=p.geometryBias+b*64u;
    GjkBoxPair pair=(GjkBoxPair)0;
    pair.kindA=geometry.Load(ga+28u);
    pair.kindB=geometry.Load(gb+28u);
    if(pair.kindA!=2u && pair.kindB!=2u)return;
    if(pair.kindA>2u || pair.kindB>2u ||
       pair.kindA==1u || pair.kindB==1u)return;
    pair.halfA=asfloat(geometry.Load3(ga));
    pair.halfB=asfloat(geometry.Load3(gb));
    pair.firstA=geometry.Load(ga+52u);
    pair.countA=geometry.Load(ga+56u);
    pair.firstB=geometry.Load(gb+52u);
    pair.countB=geometry.Load(gb+56u);
    if(pair.kindA==2u &&
       (pair.countA<4u || pair.firstA>p.hullCount ||
        pair.countA>p.hullCount-pair.firstA))return;
    if(pair.kindB==2u &&
       (pair.countB<4u || pair.firstB>p.hullCount ||
        pair.countB>p.hullCount-pair.firstB))return;
    if(pair.kindA==2u) {
        const float extent=asfloat(geometry.Load(ga+60u));
        if(!(extent>0.f) || !isfinite(extent))return;
        pair.halfA=float3(extent,extent,extent);
    }
    if(pair.kindB==2u) {
        const float extent=asfloat(geometry.Load(gb+60u));
        if(!(extent>0.f) || !isfinite(extent))return;
        pair.halfB=float3(extent,extent,extent);
    }
    const uint ta=p.transformBias+a*32u,tb=p.transformBias+b*32u;
    const float3 pa=asfloat(transforms.Load3(ta+16u));
    const float3 pb=asfloat(transforms.Load3(tb+16u));
    const float4 qa=asfloat(transforms.Load4(ta));
    const float4 qb=asfloat(transforms.Load4(tb));
    pair.aInB=rotateSource(conjugate(qb),pa-pb);
    pair.relative=quatProduct(conjugate(qb),qa);
    const float contactDistance=
        asfloat(geometry.Load(ga+12u))+asfloat(geometry.Load(gb+12u));
    const uint cache=p.edgeCacheBias+edge*240u;
    if(edgeCaches.Load(cache+12u)!=1u)return;
    GjkWitness witness;
    witness.localA=asfloat(edgeCaches.Load3(cache+80u));
    witness.localB=asfloat(edgeCaches.Load3(cache+92u));
    witness.normalB=asfloat(edgeCaches.Load3(cache+104u));
    witness.separation=asfloat(edgeCaches.Load(cache+116u));
    GjkWitness contacts[4];
    const uint count=convexFaceContacts(
        pair,contactDistance,witness,contacts);
    if(count==0u)return;
    if(count>limit) {
        edgeCaches.Store(cache+12u,0u);
        edgeCaches.Store(cache+68u,1u);
        results.Store(result+8u,2u);
        publishScContacts(i,a,b,edge,result,p.outputBias,p.headerBias,
            p.transformBias,p.geometryBias,p.edgeCacheBias,
            p.shapeCapacity,p.edgeCapacity);
        return;
    }
    uint selectedIndices=0u;
    for(uint j=0u;j<count;++j) {
        const GjkWitness c=contacts[j];
        const uint candidate=p.candidateBias+(first+j)*40u;
        candidates.Store3(candidate,asuint(c.localA));
        candidates.Store3(candidate+12u,asuint(c.localB));
        candidates.Store3(candidate+24u,asuint(c.normalB));
        candidates.Store(candidate+36u,asuint(c.separation));
        const uint persistent=cache+80u+j*40u;
        edgeCaches.Store3(persistent,asuint(c.localA));
        edgeCaches.Store3(persistent+12u,asuint(c.localB));
        edgeCaches.Store3(persistent+24u,asuint(c.normalB));
        edgeCaches.Store(persistent+36u,asuint(c.separation));
        selectedIndices|=j<<(j*8u);
    }
    edgeCaches.Store(cache+12u,count);
    results.Store(result,count);
    results.Store(result+20u,count);
    results.Store(result+24u,selectedIndices);
    publishScContacts(i,a,b,edge,result,p.outputBias,p.headerBias,
        p.transformBias,p.geometryBias,p.edgeCacheBias,
        p.shapeCapacity,p.edgeCapacity);
}
