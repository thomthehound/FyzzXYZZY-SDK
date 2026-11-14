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

// The plane generator publishes final Sc contacts directly, including
// its compact classification header for the host admission boundary.
void planePublishHeader(uint index,uint edge,uint counts,float3 normal) {
    const uint dst=p.outputBias+index*84u;
    streamContacts.Store2(dst,uint2(edge,counts));
    streamContacts.Store3(dst+8u,asuint(normal));
    if(p.headerBias!=0xffffffffu) {
        const uint header=p.headerBias+index*20u;
        streamContacts.Store2(header,uint2(edge,counts));
        streamContacts.Store3(header+8u,asuint(normal));
    }
}

// Source PCM plane/box and plane/sphere share the Sc contact manager and
// resident Gu stream. The infinite plane has no AABB ElementSim volume.
void sourcePcmPlanePair(uint index,uint boxElement,uint edge,uint result,
                       float planeHeight,float contactDistance,uint flags) {
    const uint dst=p.outputBias+index*84u;
    if(boxElement>=p.shapeCapacity || edge>=p.edgeCapacity ||
       !isfinite(planeHeight) || !isfinite(contactDistance) ||
       contactDistance<0.f) {
        publishPair(result,0u,0u,1u,edge,0u,0u,0u,0u);
        planePublishHeader(index,edge,8u,float3(0.f,0.f,0.f));
        return;
    }
    const uint boxGeo=p.geometryBias+boxElement*64u;
    const uint boxPose=p.transformBias+boxElement*32u;
    const uint kind=geometry.Load(boxGeo+28u);
    if(kind==1u) {
        // Gu::pcmContactSpherePlane: inclusive contact-distance test,
        // sphere-surface witness, and source separation. Its Sc contact
        // normal is already in the final solver orientation.
        const float radius=asfloat(geometry.Load(boxGeo));
        const float3 center=asfloat(transforms.Load3(boxPose+16u));
        if(!isfinite(radius) || radius<=0.f || !all(isfinite(center))) {
            publishPair(result,0u,0u,1u,edge,0u,0u,0u,0u);
            planePublishHeader(index,edge,8u,float3(0.f,0.f,0.f));
            return;
        }
        const float separation=center.y-planeHeight-radius;
        if(!isfinite(separation)) {
            publishPair(result,0u,0u,1u,edge,0u,0u,0u,0u);
            planePublishHeader(index,edge,8u,float3(0.f,0.f,0.f));
            return;
        }
        const bool touching=contactDistance>=separation;
        const uint cache=p.edgeCacheBias+edge*240u;
        edgeCaches.Store(cache+12u,touching?1u:0u);
        edgeCaches.Store(cache+64u,1u);
        edgeCaches.Store(cache+68u,0u);
        if(touching) {
            const float3 witness=center-float3(0.f,radius,0.f);
            planeCacheStore(cache,0,float3(0.f,-radius,0.f),
                float3(witness.x,0.f,witness.z),separation);
            streamContacts.Store4(dst+20u,asuint(float4(witness,separation)));
        }
        const uint count=touching?1u:0u;
        publishPair(result,count,count,0u,edge,0u,count,0u,1u);
        const uint transition=sourceScTouchTransition(
            edge,count,0u,p.edgeCacheBias,p.edgeCapacity);
        planePublishHeader(index,edge,count|(count<<6u)|
            (1u<<9u)|(count<<10u)|(transition<<13u),
            float3(0.f,1.f,0.f));
        return;
    }
    if(kind!=0u && kind!=2u) {
        publishPair(result,0u,0u,1u,edge,0u,0u,0u,0u);
        planePublishHeader(index,edge,8u,float3(0.f,0.f,0.f));
        return;
    }
    const float3 extent=asfloat(geometry.Load3(boxGeo));
    const float3 center=asfloat(transforms.Load3(boxPose+16u));
    const float4 rotation=asfloat(transforms.Load4(boxPose));
    if(!all(isfinite(extent)) || !all(extent>0.f) ||
       !all(isfinite(center)) || !all(isfinite(rotation))) {
        publishPair(result,0u,0u,1u,edge,0u,0u,0u,0u);
        planePublishHeader(index,edge,8u,float3(0.f,0.f,0.f));
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
        planePublishHeader(index,edge,8u,float3(0.f,0.f,0.f));
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
        if(kind==2u) {
            // Cooked hull vertices stay resident in the source GPU
            // geometry arena. Never substitute corners of its AABB.
            const uint start=geometry.Load(boxGeo+52u);
            const uint vertexCount=geometry.Load(boxGeo+56u);
            if(vertexCount<4u || start>p.hullCount ||
               vertexCount>p.hullCount-start) {
                publishPair(result,0u,0u,1u,edge,0u,0u,0u,0u);
                planePublishHeader(index,edge,8u,float3(0.f,0.f,0.f));
                return;
            }
            float deepest=3.402823466e38f;
            uint eligible=0u,first=0u;
            for(uint j=0u;j<vertexCount;++j) {
                const float3 v=asfloat(hullVertices.Load3(
                    p.hullBias+(start+j)*12u));
                const float3 world=relative+rotateSource(rotation,v);
                if(world.y>=contactDistance)continue;
                ++eligible;
                if(world.y<deepest) {
                    deepest=world.y;
                    first=j;
                }
            }
            count=min(eligible,4u);
            if(eligible<=4u) {
                uint slot=0u;
                for(uint j=0u;j<vertexCount;++j) {
                    const float3 v=asfloat(hullVertices.Load3(
                        p.hullBias+(start+j)*12u));
                    const float3 world=relative+rotateSource(rotation,v);
                    if(world.y>=contactDistance)continue;
                    planeCacheStore(cache,slot++,v,
                        float3(world.x,0.f,world.z),world.y);
                }
            }else {
                const float3 firstLocal=asfloat(hullVertices.Load3(
                    p.hullBias+(start+first)*12u));
                const float3 firstWorld=relative+
                    rotateSource(rotation,firstLocal);
                uint index[4]={first,first,first,first};
                float bestDistance=-1.f;
                for(uint j=0u;j<vertexCount;++j) {
                    if(j==first)continue;
                    const float3 v=asfloat(hullVertices.Load3(
                        p.hullBias+(start+j)*12u));
                    const float3 world=relative+rotateSource(rotation,v);
                    if(world.y>=contactDistance)continue;
                    const float3 d=float3(world.x-firstWorld.x,0.f,
                                          world.z-firstWorld.z);
                    const float distance=dot(d,d);
                    if(distance>bestDistance) {
                        bestDistance=distance;
                        index[1]=j;
                    }
                }
                const float3 farLocal=asfloat(hullVertices.Load3(
                    p.hullBias+(start+index[1])*12u));
                const float3 farWorld=relative+
                    rotateSource(rotation,farLocal);
                const float3 segment=float3(
                    farWorld.x-firstWorld.x,0.f,
                    farWorld.z-firstWorld.z);
                const float3 tangent=float3(-segment.z,0.f,segment.x);
                const float len2=dot(tangent,tangent);
                const float3 sideways=len2>0.f
                    ?tangent*rsqrt(len2):float3(0.f,1.f,0.f);
                float high=-3.402823466e38f;
                float low=3.402823466e38f;
                for(uint j=0u;j<vertexCount;++j) {
                    if(j==index[0] || j==index[1])continue;
                    const float3 v=asfloat(hullVertices.Load3(
                        p.hullBias+(start+j)*12u));
                    const float3 world=relative+rotateSource(rotation,v);
                    if(world.y>=contactDistance)continue;
                    const float d=dot(float3(world.x-firstWorld.x,0.f,
                                             world.z-firstWorld.z),sideways);
                    if(d>high){high=d;index[2]=j;}
                    if(d<low){low=d;index[3]=j;}
                }
                if(low*high>0.f) {
                    // The source batch reducer replaces the fourth
                    // witness when both extrema lie on one side.
                    float widest=-3.402823466e38f;
                    for(uint j=0u;j<vertexCount;++j) {
                        if(j==index[0] || j==index[1] || j==index[2])
                            continue;
                        const float3 v=asfloat(hullVertices.Load3(
                            p.hullBias+(start+j)*12u));
                        const float3 world=relative+rotateSource(rotation,v);
                        if(world.y>=contactDistance)continue;
                        const float d=dot(float3(world.x-firstWorld.x,0.f,
                                                 world.z-firstWorld.z),
                                          sideways);
                        if(d>widest){widest=d;index[3]=j;}
                    }
                }
                float3 centers[4];
                for(uint k=0u;k<4u;++k) {
                    const float3 v=asfloat(hullVertices.Load3(
                        p.hullBias+(start+index[k])*12u));
                    const float3 world=relative+rotateSource(rotation,v);
                    centers[k]=float3(world.x,0.f,world.z);
                }
                // Cluster once around the original four witnesses:
                // replace by the deeper member without drifting centers.
                for(uint j=0u;j<vertexCount;++j) {
                    if(j==index[0] || j==index[1] ||
                       j==index[2] || j==index[3])continue;
                    const float3 v=asfloat(hullVertices.Load3(
                        p.hullBias+(start+j)*12u));
                    const float3 world=relative+rotateSource(rotation,v);
                    if(world.y>=contactDistance)continue;
                    const float3 projection=float3(world.x,0.f,world.z);
                    uint nearest=0u;
                    float best=3.402823466e38f;
                    for(uint k=0u;k<4u;++k) {
                        const float3 d=projection-centers[k];
                        const float sq=dot(d,d);
                        if(sq<best){best=sq;nearest=k;}
                    }
                    const float3 previous=asfloat(hullVertices.Load3(
                        p.hullBias+(start+index[nearest])*12u));
                    const float3 oldWorld=relative+
                        rotateSource(rotation,previous);
                    if(world.y<oldWorld.y)index[nearest]=j;
                }
                for(uint k=0u;k<4u;++k) {
                    const float3 v=asfloat(hullVertices.Load3(
                        p.hullBias+(start+index[k])*12u));
                    const float3 world=relative+rotateSource(rotation,v);
                    planeCacheStore(cache,k,v,
                        float3(world.x,0.f,world.z),world.y);
                }
            }
        }else {
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
    const uint transition=sourceScTouchTransition(
        edge,emitted,0u,p.edgeCacheBias,p.edgeCapacity);
    // Device Dy consumes the already-oriented Sc normal.
    planePublishHeader(index,edge,
        count | (count<<6u) | ((regenerate?1u:0u)<<9u) |
        (emitted<<10u) | (transition<<13u),float3(0.f,1.f,0.f));
}
