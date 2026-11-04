// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// The plane has no finite broadphase volume. Its height and the combined
// contact offset arrive in the existing Sc PCM pair descriptor. The box
// transform and geometry remain the source ElementSim device records.
uint planeCachePoint(uint edgeBase,uint slot) {
    return edgeBase+80u+slot*40u;
}

void planeCacheStore(uint edgeBase,uint slot,float3 localA,
                     float3 projected,float separation) {
    const uint dst=planeCachePoint(edgeBase,slot);
    edgeCaches.Store3(dst,asuint(localA));
    edgeCaches.Store3(dst+12u,asuint(projected));
    edgeCaches.Store3(dst+24u,asuint(float3(0.f,1.f,0.f)));
    edgeCaches.Store(dst+36u,asuint(separation));
}

// The Sc plane-box contact manager retains the same projected manifold
// witnesses as the CPU geometry path. This is not an AABB approximation.
void sourcePcmPlaneBox(uint index,uint boxElement,uint edge,uint result,
                       float planeHeight,float contactDistance,uint flags) {
    const uint dst=p.outputBias+index*84u;
    if(boxElement>=p.shapeCapacity || edge>=p.edgeCapacity ||
       !isfinite(planeHeight) || !isfinite(contactDistance) ||
       contactDistance<0.f) {
        publishPair(result,0u,0u,1u,edge,0u,0u,0u,0u);
        streamContacts.Store2(dst,uint2(edge,8u));
        return;
    }
    const uint boxGeo=p.geometryBias+boxElement*64u;
    const uint boxPose=p.transformBias+boxElement*32u;
    if(geometry.Load(boxGeo+28u)!=0u) {
        publishPair(result,0u,0u,1u,edge,0u,0u,0u,0u);
        streamContacts.Store2(dst,uint2(edge,8u));
        return;
    }
    const float3 extent=asfloat(geometry.Load3(boxGeo));
    const float3 center=asfloat(transforms.Load3(boxPose+16u));
    const float4 rotation=asfloat(transforms.Load4(boxPose));
    if(!all(isfinite(extent)) || !all(extent>0.f) ||
       !all(isfinite(center)) || !all(isfinite(rotation))) {
        publishPair(result,0u,0u,1u,edge,0u,0u,0u,0u);
        streamContacts.Store2(dst,uint2(edge,8u));
        return;
    }

    const uint cache=p.edgeCacheBias+edge*240u;
    if((flags&0x80000000u)!=0u) {
        edgeCaches.Store(cache+12u,0u);
        edgeCaches.Store(cache+64u,0u);
        edgeCaches.Store(cache+68u,0u);
    }
    uint count=edgeCaches.Load(cache+12u);
    bool initialized=edgeCaches.Load(cache+64u)!=0u;
    if(count>4u) {
        publishPair(result,0u,0u,1u,edge,0u,0u,0u,0u);
        streamContacts.Store2(dst,uint2(edge,8u));
        return;
    }

    const float3 relative=center-float3(0.f,planeHeight,0.f);
    const float margin=min(0.15f*min(extent.x,min(extent.y,extent.z)),0.15f);
    const float breaking=margin*0.2f;
    const uint originalCount=count;
    if(initialized) {
        const float breakingSquared=breaking*breaking;
        for(uint j=count;j>0u;--j) {
            const uint source=planeCachePoint(cache,j-1u);
            const float3 localA=asfloat(edgeCaches.Load3(source));
            const float3 localB=asfloat(edgeCaches.Load3(source+12u));
            const float3 delta=relative+rotateSource(rotation,localA)-localB;
            const float3 tangential=float3(delta.x,0.f,delta.z);
            if(dot(tangential,tangential)>breakingSquared) {
                --count;
                if(j-1u!=count) {
                    const uint from=planeCachePoint(cache,count);
                    [unroll]
                    for(uint k=0u;k<10u;++k)
                        edgeCaches.Store(source+k*4u,
                                        edgeCaches.Load(from+k*4u));
                }
            }else {
                edgeCaches.Store(source+36u,asuint(delta.y));
            }
        }
    }
    const float3 oldRelative=initialized
        ? asfloat(edgeCaches.Load3(cache)):float3(0.f,0.f,0.f);
    const float4 oldRotation=initialized
        ? asfloat(edgeCaches.Load4(cache+16u)):float4(0.f,0.f,0.f,1.f);
    const float3 motion=abs(relative-oldRelative);
    const float maxDelta=max(motion.x,max(motion.y,motion.z));
    const bool regenerate=!initialized || count!=originalCount ||
        maxDelta>breaking || dot(rotation,oldRotation)<0.99996f;
    if(regenerate) {
        float3 localPoints[8];
        float3 planePoints[8];
        float depth[8];
        uint total=0u;
        [unroll]
        for(uint corner=0u;corner<8u;++corner) {
            const float3 localA=float3(
                (corner&4u)!=0u?-extent.x:extent.x,
                (corner&2u)!=0u?-extent.y:extent.y,
                (corner&1u)!=0u?-extent.z:extent.z);
            const float3 inPlane=relative+rotateSource(rotation,localA);
            if(inPlane.y>=contactDistance)continue;
            localPoints[total]=localA;
            planePoints[total]=float3(inPlane.x,0.f,inPlane.z);
            depth[total]=inPlane.y;
            ++total;
        }
        uint chosenIndex[4]={0u,0u,0u,0u};
        if(total<=4u) {
            for(uint j=0u;j<total;++j)chosenIndex[j]=j;
        }else {
            // Four-contact batch reduction: deepest, farthest, segment
            // extrema, then cluster residuals around the original four.
            bool chosen[8]={false,false,false,false,
                            false,false,false,false};
            uint first=0u;
            for(uint j=1u;j<total;++j)
                if(depth[j]<depth[first])first=j;
            chosenIndex[0]=first;
            chosen[first]=true;
            uint farthest=0u;
            float maximum=dot(planePoints[0]-planePoints[first],
                              planePoints[0]-planePoints[first]);
            for(uint j=1u;j<total;++j) {
                const float3 d=planePoints[j]-planePoints[first];
                const float length=dot(d,d);
                if(length>maximum) {maximum=length;farthest=j;}
            }
            chosenIndex[1]=farthest;
            chosen[farthest]=true;
            const float3 segment=planePoints[farthest]-planePoints[first];
            float3 sideways=float3(-segment.z,0.f,segment.x);
            const float normSquared=dot(sideways,sideways);
            sideways=normSquared>0.f
                ?sideways*rsqrt(normSquared):float3(0.f,1.f,0.f);
            float largest=-3.402823466e38f;
            float smallest=3.402823466e38f;
            uint third=0u,fourth=0u;
            for(uint j=0u;j<total;++j) {
                if(chosen[j])continue;
                const float d=dot(planePoints[j]-planePoints[first],sideways);
                if(d>largest){largest=d;third=j;}
                if(d<smallest){smallest=d;fourth=j;}
            }
            chosenIndex[2]=third;
            chosen[third]=true;
            if(smallest*largest>0.f) {
                largest=-3.402823466e38f;
                for(uint j=0u;j<total;++j) {
                    if(chosen[j])continue;
                    const float d=dot(planePoints[j]-planePoints[first],sideways);
                    if(d>largest){largest=d;fourth=j;}
                }
            }
            chosenIndex[3]=fourth;
            chosen[fourth]=true;
            const float3 centers[4]={
                planePoints[chosenIndex[0]],planePoints[chosenIndex[1]],
                planePoints[chosenIndex[2]],planePoints[chosenIndex[3]]};
            for(uint j=0u;j<total;++j) {
                if(chosen[j])continue;
                float nearest=3.402823466e38f;
                uint slot=0u;
                for(uint k=0u;k<4u;++k) {
                    const float3 d=planePoints[j]-centers[k];
                    const float length=dot(d,d);
                    if(length<nearest){nearest=length;slot=k;}
                }
                if(depth[j]<depth[chosenIndex[slot]])
                    chosenIndex[slot]=j;
            }
        }
        count=min(total,4u);
        for(uint j=0u;j<count;++j) {
            const uint chosen=chosenIndex[j];
            planeCacheStore(cache,j,localPoints[chosen],
                            planePoints[chosen],depth[chosen]);
        }
        edgeCaches.Store3(cache,asuint(relative));
        edgeCaches.Store4(cache+16u,asuint(rotation));
        edgeCaches.Store(cache+64u,1u);
        edgeCaches.Store(cache+68u,0u);
    }
    edgeCaches.Store(cache+12u,count);
    uint emitted=0u;
    for(uint j=0u;j<count;++j) {
        const uint entry=planeCachePoint(cache,j);
        const float separation=asfloat(edgeCaches.Load(entry+36u));
        if(separation>contactDistance)continue;
        const float3 projected=asfloat(edgeCaches.Load3(entry+12u));
        const float3 world=float3(projected.x,planeHeight,projected.z);
        streamContacts.Store4(dst+20u+emitted*16u,
            asuint(float4(world,separation)));
        ++emitted;
    }
    publishPair(result,count,count!=0u?1u:0u,0u,edge,0u,
                count,0u,regenerate?1u:0u);
    streamContacts.Store2(dst,uint2(edge,
        count | (count<<6u) | ((regenerate?1u:0u)<<9u) |
        (emitted<<10u)));
    streamContacts.Store3(dst+8u,asuint(float3(0.f,-1.f,0.f)));
}
