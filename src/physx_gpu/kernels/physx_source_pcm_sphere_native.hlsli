// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Gu PCM sphere/sphere and sphere/box produce one world-space contact.
// Store its B-local witness in the existing Sc EdgeIndex cache so the
// source GPU contact publisher and Dy consume the same device stream.
bool sourcePcmSpherePair(uint i,uint a,uint b,uint edge,uint first,
                         uint limit,uint result,
                         uint ga,uint gb,float3 pa,float3 pb,
                         float4 qa,float4 qb,float contactDistance) {
    const uint kindA=geometry.Load(ga+28u);
    const uint kindB=geometry.Load(gb+28u);
    if(kindA==0u && kindB==0u)return false;

    const uint edgeBase=p.edgeCacheBias+edge*240u;
    uint status=0u;
    bool touching=false;
    float3 normal=float3(0.f,0.f,0.f);
    float3 witness=float3(0.f,0.f,0.f);
    float separation=0.f;
    if(kindA>1u || kindB>1u) {
        status=1u;
    }else if(kindA==1u && kindB==1u) {
        const float radiusA=asfloat(geometry.Load(ga));
        const float radiusB=asfloat(geometry.Load(gb));
        const float3 delta=pa-pb;
        const float squared=dot(delta,delta);
        const float sum=radiusA+radiusB;
        const float inflated=sum+contactDistance;
        if(inflated*inflated>squared) {
            const float length=sqrt(squared);
            normal=length<=0.00001f
                ? float3(1.f,0.f,0.f):delta/length;
            witness=pb+normal*radiusB;
            separation=length-sum;
            touching=true;
        }
    }else {
        const bool firstSphere=kindA==1u;
        const float3 sphere=firstSphere?pa:pb;
        const float radius=asfloat(geometry.Load(firstSphere?ga:gb));
        const float3 box=firstSphere?pb:pa;
        const float4 boxQ=firstSphere?qb:qa;
        const float3 extents=asfloat(geometry.Load3(firstSphere?gb:ga));
        const float3 localCenter=rotateSource(conjugate(boxQ),sphere-box);
        const float3 closest=clamp(localCenter,-extents,extents);
        const float3 gap=localCenter-closest;
        const float squared=dot(gap,gap);
        const float inflated=radius+contactDistance;
        if(inflated*inflated>squared) {
            float3 outward;
            if(all(abs(localCenter)<=extents)) {
                const float3 distances=extents-abs(closest);
                float3 localNormal=float3(0.f,0.f,0.f);
                float depth;
                if(distances.z<=distances.x && distances.z<=distances.y) {
                    localNormal.z=closest.z>=0.f?1.f:-1.f;
                    depth=distances.z;
                }else if(distances.x<=distances.y && distances.x<=distances.z) {
                    localNormal.x=closest.x>=0.f?1.f:-1.f;
                    depth=distances.x;
                }else {
                    localNormal.y=closest.y>=0.f?1.f:-1.f;
                    depth=distances.y;
                }
                outward=rotateSource(boxQ,localNormal);
                witness=sphere+outward*depth;
                separation=-depth-radius;
            }else {
                const float length=sqrt(squared);
                outward=rotateSource(boxQ,gap/length);
                witness=box+rotateSource(boxQ,closest);
                separation=length-radius;
            }
            normal=firstSphere?outward:-outward;
            touching=true;
        }
    }

    if(touching && (!all(isfinite(normal)) ||
                    !all(isfinite(witness)) || !isfinite(separation)))
        status=1u;
    if(touching && limit==0u)status=2u;
    if(status!=0u)touching=false;
    edgeCaches.Store(edgeBase+12u,touching?1u:0u);
    edgeCaches.Store(edgeBase+64u,touching?1u:0u);
    edgeCaches.Store(edgeBase+68u,0u);
    if(touching) {
        const float3 localA=rotateSource(conjugate(qa),witness-pa);
        const float3 localB=rotateSource(conjugate(qb),witness-pb);
        const float3 normalB=rotateSource(conjugate(qb),normal);
        const uint dst=edgeBase+80u;
        edgeCaches.Store3(dst,asuint(localA));
        edgeCaches.Store3(dst+12u,asuint(localB));
        edgeCaches.Store3(dst+24u,asuint(normalB));
        edgeCaches.Store(dst+36u,asuint(separation));
    }
    publishPair(result,touching?1u:0u,touching?1u:0u,status,
                edge,first,touching?1u:0u,0u,1u);
    publishScContacts(i,a,b,edge,result,p.outputBias,p.headerBias,
                      p.transformBias,p.geometryBias,p.edgeCacheBias,
                      p.shapeCapacity,p.edgeCapacity);
    return true;
}
