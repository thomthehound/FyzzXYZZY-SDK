// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Source DySolverConstraintsShared.h contact and friction PGS, executing
// correlated Gu contact runs in source DyConstraintPartition order.
double contactDouble(uint addr) {
    uint2 v=arena.Load2(addr);
    return asdouble(v.x,v.y);
}
void storeContactDouble(uint addr,double value,uint pointIndex,bool finalPass) {
    uint lo,hi;asuint(value,lo,hi);
    uint2 packed=uint2(lo,hi);
    arena.Store2(addr,packed);
    if(finalPass)
        arena.Store2(p.writebackBias+p.resultContactOffset+
            pointIndex*8u,packed);
}
double frictionDouble(uint addr) {
    uint2 v=arena.Load2(addr);
    return asdouble(v.x,v.y);
}
void storeFrictionDouble(uint addr,double value,uint rowIndex,bool finalPass) {
    uint lo,hi;asuint(value,lo,hi);
    uint2 packed=uint2(lo,hi);
    arena.Store2(addr,packed);
    if(finalPass) {
        uint target=p.writebackBias+p.resultFrictionOffset+rowIndex*20u;
        // The source Sc friction anchor writeback combines this tangent
        // with the accumulated impulse. Both are GPU-prepared now.
        arena.Store3(target,
            arena.Load3(p.frictionBias+rowIndex*72u));
        arena.Store2(target+12u,packed);
    }
}
bool contactImpulse(uint a,uint b,uint ao,uint bo,
                    float3 n,float3 ja,float3 jb,
                    inout float3 va,inout float3 vb,
                    inout float3 aa,inout float3 ab,
                    double delta,float ma,float mb,uint errorAddress) {
    double da=delta*(double)ma,db=delta*(double)mb;
    float3 v0=float3((float)((double)va.x+da*(double)n.x),
                     (float)((double)va.y+da*(double)n.y),
                     (float)((double)va.z+da*(double)n.z));
    float3 v1=float3((float)((double)vb.x-db*(double)n.x),
                     (float)((double)vb.y-db*(double)n.y),
                     (float)((double)vb.z-db*(double)n.z));
    float3 w0=float3((float)((double)aa.x+delta*(double)ja.x),
                     (float)((double)aa.y+delta*(double)ja.y),
                     (float)((double)aa.z+delta*(double)ja.z));
    float3 w1=float3((float)((double)ab.x-delta*(double)jb.x),
                     (float)((double)ab.y-delta*(double)jb.y),
                     (float)((double)ab.z-delta*(double)jb.z));
    if(!all(isfinite(v0)) || !all(isfinite(v1)) ||
       !all(isfinite(w0)) || !all(isfinite(w1))) {
        arena.InterlockedOr(errorAddress,16u);
        return false;
    }
    if(a!=invalidBody && ma>0.f) {
        arena.Store3(ao,asuint(v0));
        arena.Store3(ao+12u,asuint(w0));
        va=v0;
        aa=w0;
    }
    if(b!=invalidBody && mb>0.f) {
        arena.Store3(bo,asuint(v1));
        arena.Store3(bo+12u,asuint(w1));
        vb=v1;
        ab=w1;
    }
    return true;
}
void solveContact(uint4 descriptor,uint errorAddress,bool biased,
                  bool frictionEnabled,bool finalPass) {
    if(descriptor.x>=p.manifoldCount ||
       (descriptor.y!=invalidBody && descriptor.y>=p.bodyCount) ||
       (descriptor.z!=invalidBody && descriptor.z>=p.bodyCount)) {
        arena.InterlockedOr(errorAddress,32u);
        return;
    }
    uint4 manager=arena.Load4(
        p.manifoldBias+descriptor.x*20u);
    if(manager.x>p.patchCount || manager.y>p.patchCount-manager.x ||
       manager.z>p.orderCount || manager.w>p.orderCount-manager.z) {
        arena.InterlockedOr(errorAddress,64u);
        return;
    }
    if(manager.y==0u)return;
    uint a=descriptor.y,b=descriptor.z;
    uint ao=p.bodyBias+a*52u,bo=p.bodyBias+b*52u;
    float ma=0.f,mb=0.f;
    float3 va=float3(0,0,0),vb=float3(0,0,0);
    float3 aa=float3(0,0,0),ab=float3(0,0,0);
    // A source conflict partition gives this descriptor exclusive writes
    // to its dynamic bodies. Reuse committed velocities through all
    // correlated normal and friction rows; static participants stay fixed.
    if(a!=invalidBody) {
        ma=asfloat(arena.Load(ao+48u));
        va=asfloat(arena.Load3(ao));
        aa=asfloat(arena.Load3(ao+12u));
    }
    if(b!=invalidBody) {
        mb=asfloat(arena.Load(bo+48u));
        vb=asfloat(arena.Load3(bo));
        ab=asfloat(arena.Load3(bo+12u));
    }
    uint cursor=manager.z;
    for(uint patchIndex=manager.x;patchIndex<manager.x+manager.y;++patchIndex) {
        uint po=p.patchBias+patchIndex*32u;
        uint4 patch=arena.Load4(po);
        uint4 settings=arena.Load4(po+16u);
        if(patch.x>p.frictionCount || patch.z>2u ||
           patch.z*2u>p.frictionCount-patch.x ||
           cursor>p.orderCount || patch.y>p.orderCount-cursor ||
           settings.z!=cursor) {
            arena.InterlockedOr(errorAddress,128u);
            return;
        }
        double normalImpulse=0.0;
        for(uint i=0;i<patch.y;++i) {
            uint pointIndex=arena.Load(
                p.orderBias+(cursor++)*4u);
            if(pointIndex>=p.contactCount) {
                arena.InterlockedOr(errorAddress,256u);
                return;
            }
            uint co=p.contactBias+pointIndex*88u;
            float3 n=asfloat(arena.Load3(co));
            float3 ja=asfloat(arena.Load3(co+12u));
            float3 jb=asfloat(arena.Load3(co+24u));
            float normalVelocity=dot(n,va)-dot(n,vb)+
                dot(aa,ja)-dot(ab,jb);
            double oldForce=contactDouble(co+72u);
            double error=biased?contactDouble(co+56u):
                                contactDouble(co+64u);
            double candidate=max(
                -(double)normalVelocity*contactDouble(co+40u)+error,
                -oldForce);
            double nextForce=min(contactDouble(co+80u),
                contactDouble(co+48u)*oldForce+candidate);
            if(!isfinite(nextForce)) {
                arena.InterlockedOr(errorAddress,1024u);
                return;
            }
            if(!contactImpulse(a,b,ao,bo,n,ja,jb,
                               va,vb,aa,ab,nextForce-oldForce,
                               ma,mb,errorAddress))return;
            storeContactDouble(co+72u,nextForce,pointIndex,finalPass);
            normalImpulse+=nextForce;
        }
        if(!frictionEnabled || patch.z==0u)continue;
        double coeff=patch.z==2u?0.5:1.0;
        double maxStatic=(double)asfloat(settings.x)*coeff*normalImpulse;
        double maxDynamic=(double)asfloat(settings.y)*coeff*normalImpulse;
        bool broken=false;
        for(uint k=patch.x;k<patch.x+patch.z*2u;++k) {
            uint fo=p.frictionBias+k*72u;
            float3 n=asfloat(arena.Load3(fo));
            float3 ja=asfloat(arena.Load3(fo+12u));
            float3 jb=asfloat(arena.Load3(fo+24u));
            float tangentDelta=dot(n,va)-dot(n,vb)+
                dot(aa,ja)-dot(ab,jb);
            double oldForce=frictionDouble(fo+64u);
            double bias=biased?frictionDouble(fo+48u):0.0;
            double candidate=oldForce+
                (frictionDouble(fo+56u)-bias-(double)tangentDelta)*
                frictionDouble(fo+40u);
            bool saturated=abs(candidate)>maxStatic;
            // Source signed FMin/FMax also handles negative compliant bounds.
            double nextForce=saturated
                ? min(maxDynamic,max(-maxDynamic,candidate)):candidate;
            if(!isfinite(nextForce)) {
                arena.InterlockedOr(errorAddress,2048u);
                return;
            }
            if(!contactImpulse(a,b,ao,bo,n,ja,jb,
                               va,vb,aa,ab,nextForce-oldForce,
                               ma,mb,errorAddress))return;
            storeFrictionDouble(fo+64u,nextForce,k,finalPass);
            broken=broken||saturated;
        }
        if(finalPass) {
            bool gpuGrowth=(patch.w&0x80000000u)!=0u;
            arena.Store(po+12u,broken?1u:0u);
            uint reportAddress=p.writebackBias+p.resultPatchOffset+
                patchIndex*4u;
            uint report=gpuGrowth?arena.Load(reportAddress):0u;
            arena.Store(reportAddress,(report&~1u)|(broken?1u:0u));
        }
    }
    if(cursor!=manager.z+manager.w)
        arena.InterlockedOr(errorAddress,512u);
}

float3 prepSlop(float3 value,float slop) {
    return float3(abs(value.x)<slop?0.0f:value.x,
                  abs(value.y)<slop?0.0f:value.y,
                  abs(value.z)<slop?0.0f:value.z);
}
// Low-bit-tagged references reuse one active integrateCore body for every
// contact manager. Even references address the inline immovable-body payload
// relative to the packed source contact-prep block.
uint prepBodyOffset(uint source,uint offset) {
    if((source&1u)==0u)return p.prepBias+source+offset;
    uint ordinal=source>>1u;
    uint base=p.integrationBias+16u+ordinal*104u;
    if(offset==12u)return base+28u;
    if(offset==24u)return base+40u;
    if(offset>=36u && offset<=60u)return base+offset+16u;
    if(offset==72u)return p.bodyBias+ordinal*52u+48u;
    if(offset==80u)return base+12u;
    return base+offset;
}
float3 prepBody3(uint source,uint offset) {
    return asfloat(arena.Load3(prepBodyOffset(source,offset)));
}
float4 prepBody4(uint source,uint offset) {
    return asfloat(arena.Load4(prepBodyOffset(source,offset)));
}
float prepBodyFloat(uint source,uint offset) {
    return asfloat(arena.Load(prepBodyOffset(source,offset)));
}
float3 prepInertia(uint source,float3 v) {
    float3 c0=prepBody3(source,36u);
    float3 c1=prepBody3(source,48u);
    float3 c2=prepBody3(source,60u);
    return c0*v.x+c1*v.y+c2*v.z;
}
void prepStoreDouble(uint address,double value) {
    uint lo,hi;
    asuint(value,lo,hi);
    arena.Store2(address,uint2(lo,hi));
}
void prepareContactRow(uint index,uint pairIndex,uint contactBegin,
                       uint contactCount,uint errorAddress) {
    if(index>=p.contactCount || pairIndex>=p.prepCount) {
        arena.InterlockedOr(errorAddress,4096u);
        return;
    }
    uint co=p.contactBias+index*88u;
    uint pair=p.prepBias+pairIndex*72u;
    uint b0=arena.Load(pair),b1=arena.Load(pair+4u);
    // The source correlation fixes patch order and body inputs. The GPU
    // reconstructs PCM normal rows without uploading a placeholder row.
    float3 n;
    float3 pos;
    float separation;
    uint pcmIndex=arena.Load(pair+48u);
    if(pcmIndex!=0xffffffffu) {
        uint local=index-contactBegin;
        uint source=p.pcmBias+pcmIndex*84u;
        uint packed=pcmContacts.Load(source+4u);
        if(index<contactBegin || local>=contactCount ||
           local>=4u ||
           pcmContacts.Load(source)!=arena.Load(pair+52u) ||
           ((packed>>3u)&7u)!=0u ||
           local>=((packed>>10u)&7u)) {
            arena.InterlockedOr(errorAddress,65536u);
            return;
        }
        n=asfloat(pcmContacts.Load3(source+8u));
        uint contact=source+20u+local*16u;
        pos=asfloat(pcmContacts.Load3(contact));
        separation=asfloat(pcmContacts.Load(contact+12u));
        arena.Store3(co,asuint(n));
    } else {
        uint cpuOffset=arena.Load(pair+64u);
        if(cpuOffset==0xffffffffu || index<contactBegin ||
           index-contactBegin>=contactCount) {
            arena.InterlockedOr(errorAddress,131072u);
            return;
        }
        uint raw=p.prepBias+p.prepCount*72u+
            (cpuOffset+index-contactBegin)*28u;
        n=asfloat(arena.Load3(raw));
        pos=asfloat(arena.Load3(raw+12u));
        separation=asfloat(arena.Load(raw+24u));
        arena.Store3(co,asuint(n));
    }
    float3 p0=prepBody3(b0,0u);
    float3 p1=prepBody3(b1,0u);
    float3 v0=prepBody3(b0,12u);
    float3 v1=prepBody3(b1,12u);
    float3 w0=prepBody3(b0,24u);
    float3 w1=prepBody3(b1,24u);
    float m0=prepBodyFloat(b0,72u);
    float m1=prepBodyFloat(b1,72u);
    float3 r0=cross(pos-p0,n);
    float3 r1=cross(pos-p1,n);
    float linearRel=dot(n,v0-v1);
    float pairSlop=asfloat(arena.Load(pair+12u));
    if(pairSlop>0.0f) {
        float angularRel=dot(r0,w0)-dot(r1,w1);
        float ratio=linearRel==0.0f
            ? 3.402823466e+38f : angularRel/linearRel;
        float slop=pairSlop*max(ratio,1.0f);
        r0=prepSlop(r0,slop);
        r1=prepSlop(r1,slop);
    }
    float3 ja=prepInertia(b0,r0);
    float3 jb=prepInertia(b1,r1);
    float angularRel=dot(r0,w0)-dot(r1,w1);
    float vrel=linearRel+angularRel;
    double response=(double)m0+(double)m1+
        (double)dot(ja,ja)+(double)dot(jb,jb);
    double multiplier=response>0.0?1.0/response:0.0;
    float restDistance=asfloat(arena.Load(pair+8u));
    float restitution=asfloat(arena.Load(pair+16u));
    float damping=asfloat(arena.Load(pair+20u));
    float maxPenBias=asfloat(arena.Load(pair+24u));
    float invDt=asfloat(arena.Load(pair+28u));
    float seconds=asfloat(arena.Load(pair+32u));
    float rigidBias=asfloat(arena.Load(pair+36u));
    float bounceThreshold=asfloat(arena.Load(pair+40u));
    bool acceleration=arena.Load(pair+44u)!=0u;
    float penetration=separation-restDistance;
    float penetrationInvDt=penetration*invDt;
    bool separated=penetration>=0.0f;
    bool closing=-vrel>penetrationInvDt;
    bool bouncing=restitution>0.0f &&
        bounceThreshold>vrel && closing;
    float targetVelocity=bouncing?-vrel*restitution:0.0f;
    double target=(double)(targetVelocity-vrel);
    double impulseMultiplier=1.0;
    double biasedError,unbiasedError;
    if(restitution<0.0f) {
        float enabledDamping=separated&&!closing?0.0f:damping;
        double nrdt=(double)seconds*(double)restitution;
        double aCoeff=(double)seconds*
            ((double)enabledDamping-nrdt);
        double massFactor=acceleration?multiplier:1.0;
        double x=1.0/(1.0+aCoeff*(acceleration?1.0:response));
        double bias=x*(-nrdt*(double)penetration*massFactor);
        multiplier=x*aCoeff*massFactor;
        impulseMultiplier=1.0-x;
        biasedError=target*multiplier-bias;
        unbiasedError=biasedError;
    } else {
        float penetrationScaled=separated?penetrationInvDt:
            penetration*(invDt*rigidBias);
        double scaledBias=bouncing?0.0:
            multiplier*(double)max(maxPenBias,penetrationScaled);
        biasedError=target*multiplier-scaledBias;
        unbiasedError=target*multiplier-
            (bouncing?0.0:max(scaledBias,0.0));
    }
    if(!isfinite(multiplier) || !isfinite(impulseMultiplier) ||
       !isfinite(biasedError) || !isfinite(unbiasedError) ||
       !all(isfinite(ja)) || !all(isfinite(jb))) {
        arena.InterlockedOr(errorAddress,8192u);
        return;
    }
    // A freshly prepared source Gu normal row begins with zero impulse.
    // Its limit is common to the entire contact-manager pair.
    arena.Store(co+36u,0u);
    arena.Store2(co+72u,uint2(0u,0u));
    arena.Store2(co+80u,arena.Load2(pair+56u));
    arena.Store3(co+12u,asuint(ja));
    arena.Store3(co+24u,asuint(jb));
    prepStoreDouble(co+40u,multiplier);
    prepStoreDouble(co+48u,impulseMultiplier);
    prepStoreDouble(co+56u,biasedError);
    prepStoreDouble(co+64u,unbiasedError);
}

float3 prepRotate(uint body,float3 local) {
    float4 q=prepBody4(body,80u);
    float3 u=q.xyz;
    float3 t=cross(u,local)*2.0f;
    return local+q.w*t+cross(u,t);
}
void prepareFrictionRow(uint index,uint pairIndex,uint errorAddress) {
    if(index>=p.frictionCount || pairIndex>=p.prepCount) {
        arena.InterlockedOr(errorAddress,16384u);
        return;
    }
    uint fo=p.frictionBias+index*72u;
    uint pair=p.prepBias+pairIndex*72u;
    uint b0=arena.Load(pair),b1=arena.Load(pair+4u);
    // Source DyFrictionCorrelation selected these body-local anchors.
    // Source DyContactPrep selected the two tangents and their order.
    float3 tangent=asfloat(arena.Load3(fo));
    // The original Gu correlation owns two local anchors per patch
    // anchor, shared by its two tangent constraints.
    uint anchor=p.contactBias-(p.frictionCount/2u)*24u+
        (index/2u)*24u;
    float3 r0=prepRotate(b0,asfloat(arena.Load3(anchor)));
    float3 r1=prepRotate(b1,asfloat(arena.Load3(anchor+12u)));
    float3 p0=prepBody3(b0,0u);
    float3 p1=prepBody3(b1,0u);
    float3 v0=prepBody3(b0,12u);
    float3 v1=prepBody3(b1,12u);
    float3 w0=prepBody3(b0,24u);
    float3 w1=prepBody3(b1,24u);
    float m0=prepBodyFloat(b0,72u);
    float m1=prepBodyFloat(b1,72u);
    float slop=asfloat(arena.Load(pair+12u));
    float invDt=asfloat(arena.Load(pair+28u));
    float rigidBias=asfloat(arena.Load(pair+36u));
    float3 error=prepSlop((r0+p0)-(r1+p1),slop);
    float3 ax0=prepSlop(cross(r0,tangent),slop);
    float3 ax1=prepSlop(cross(r1,tangent),slop);
    float3 ja=prepInertia(b0,ax0);
    float3 jb=prepInertia(b1,ax1);
    double response=(double)m0+(double)m1+
        (double)dot(ja,ja)+(double)dot(jb,jb);
    double velocityMultiplier=response>0.0
        ? (double)rigidBias/response:0.0;
    double bias=(double)(dot(tangent,error)*invDt);
    float3 relative=v0-v1;
    double targetVelocity=(double)(-(dot(tangent,relative)+
        dot(ax0,w0)-dot(ax1,w1)));
    if(!isfinite(velocityMultiplier) || !isfinite(bias) ||
       !isfinite(targetVelocity) || !all(isfinite(ja)) ||
       !all(isfinite(jb))) {
        arena.InterlockedOr(errorAddress,32768u);
        return;
    }
    arena.Store3(fo+12u,asuint(ja));
    arena.Store3(fo+24u,asuint(jb));
    arena.Store(fo+36u,0u);
    arena.Store2(fo+64u,uint2(0u,0u));
    prepStoreDouble(fo+40u,velocityMultiplier);
    prepStoreDouble(fo+48u,bias);
    prepStoreDouble(fo+56u,targetVelocity);
}

// GPU PCM supplies exactly one Gu normal/contact patch for each active
// box pair. DyFrictionCorrelation's retained anchors remain Sc-owned;
// this device pass chooses replacements directly from the original PCM
// contact stream, before constructing the same source tangent rows.
// Source output: broken bit 0, resulting count bits 1-2, and changed
// Gu contact indices (+1) in bits 3-5 and 6-8. Zero means retained.
void growPcmFrictionAnchors(uint patchIndex,uint pairIndex,
                            uint errorAddress) {
    uint po=p.patchBias+patchIndex*32u;
    uint4 patch=arena.Load4(po);
    uint pair=p.prepBias+pairIndex*72u;
    uint pcmIndex=arena.Load(pair+48u);
    if(pcmIndex==0xffffffffu || patch.y>4u || patch.z>2u ||
       (patch.x&1u)!=0u || patch.x+4u>p.frictionCount) {
        arena.InterlockedOr(errorAddress,262144u);
        return;
    }
    uint stream=p.pcmBias+pcmIndex*84u;
    uint packed=pcmContacts.Load(stream+4u);
    if(((packed>>3u)&7u)!=0u || ((packed>>10u)&7u)!=patch.y ||
       pcmContacts.Load(stream)!=arena.Load(pair+52u)) {
        arena.InterlockedOr(errorAddress,262144u);
        return;
    }
    uint body0=arena.Load(pair);
    uint body1=arena.Load(pair+4u);
    uint anchorBase=p.contactBias-(p.frictionCount/2u)*24u+
        (patch.x/2u)*24u;
    // This source-heap prefix is a physical mirror of Sc's original
    // EdgeIndex friction history. A rejected frame explicitly reloads
    // Sc's last successful anchors; ordinary frames reuse device memory.
    uint history=p.writebackBias+arena.Load(pair+52u)*64u;
    uint count=patch.z;
    if(count>0u && (patch.w&0x40000000u)==0u) {
        if(arena.Load(history)!=count) {
            arena.InterlockedOr(errorAddress,262144u);
            return;
        }
        [unroll]
        for(uint k=0u;k<2u;++k) {
            if(k>=count)break;
            arena.Store3(anchorBase+k*24u,
                         arena.Load3(history+8u+k*24u));
            arena.Store3(anchorBase+k*24u+12u,
                         arena.Load3(history+20u+k*24u));
        }
    }
    float3 local0=count>0u?asfloat(arena.Load3(anchorBase)):
                            float3(0.f,0.f,0.f);
    float3 local1=count>1u?asfloat(arena.Load3(anchorBase+24u)):
                            float3(0.f,0.f,0.f);
    bool keepPair=false;
    if(count==2u) {
        float3 delta=local0-local1;
        float distance=delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
        keepPair=distance*4.0f>=asfloat(arena.Load(po+28u));
        if(!keepPair)count=0u;
    }
    float3 anchors[2];
    uint selected[2]={0u,0u};
    bool changed[2]={false,false};
    float pointDistance=0.0f;
    if(!keepPair) {
        if(count==1u)anchors[0]=prepRotate(body0,local0)+
                                 prepBody3(body0,0u);
        float limit=asfloat(arena.Load(pair+8u))+0.04f;
        [loop]
        for(uint j=0u;j<patch.y;++j) {
            uint src=stream+20u+j*16u;
            float separation=asfloat(pcmContacts.Load(src+12u));
            if(separation>=limit)continue;
            float3 world=asfloat(pcmContacts.Load3(src));
            if(count==0u) {
                anchors[0]=world;
                selected[0]=j;
                changed[0]=true;
                ++count;
            }else if(count==1u) {
                float3 v=world-anchors[0];
                float distance=v.x*v.x+v.y*v.y+v.z*v.z;
                if(distance>1.0e-8f) {
                    anchors[1]=world;
                    selected[1]=j;
                    changed[1]=true;
                    pointDistance=distance;
                    ++count;
                }
            }else {
                float3 v0=world-anchors[0];
                float3 v1=world-anchors[1];
                float d0=v0.x*v0.x+v0.y*v0.y+v0.z*v0.z;
                float d1=v1.x*v1.x+v1.y*v1.y+v1.z*v1.z;
                if(d0>d1 && d0>pointDistance) {
                    anchors[1]=world;
                    selected[1]=j;
                    changed[1]=true;
                    pointDistance=d0;
                } else if(d1>=d0 && d1>pointDistance) {
                    anchors[0]=world;
                    selected[0]=j;
                    changed[0]=true;
                    pointDistance=d1;
                }
            }
        }
        [unroll]
        for(uint k=0u;k<2u;++k) {
            if(k>=count || !changed[k])continue;
            float3 w=anchors[k];
            float3 a=prepBody3(body0,0u);
            float3 b=prepBody3(body1,0u);
            float4 qa=prepBody4(body0,80u);
            float4 qb=prepBody4(body1,80u);
            float3 da=w-a,db=w-b;
            float3 ta=cross(-qa.xyz,da)*2.0f;
            float3 tb=cross(-qb.xyz,db)*2.0f;
            float3 la=da+qa.w*ta+cross(-qa.xyz,ta);
            float3 lb=db+qb.w*tb+cross(-qb.xyz,tb);
            arena.Store3(anchorBase+k*24u,asuint(la));
            arena.Store3(anchorBase+k*24u+12u,asuint(lb));
        }
    }
    [unroll]
    for(uint k=0u;k<2u;++k) {
        if(k>=count)break;
        arena.Store3(history+8u+k*24u,
                     arena.Load3(anchorBase+k*24u));
        arena.Store3(history+20u+k*24u,
                     arena.Load3(anchorBase+k*24u+12u));
    }
    arena.Store(history,count);
    arena.Store(po+8u,count);
    uint report=(count<<1u) |
        ((changed[0]?(selected[0]+1u):0u)<<3u) |
        ((changed[1]?(selected[1]+1u):0u)<<6u);
    arena.Store(p.writebackBias+p.resultPatchOffset+
                patchIndex*4u,report);
}

void prepareFrictionPatch(uint patchIndex,uint pairIndex,uint errorAddress) {
    if(patchIndex>=p.patchCount || pairIndex>=p.prepCount) {
        arena.InterlockedOr(errorAddress,16384u);
        return;
    }
    uint po=p.patchBias+patchIndex*32u;
    uint4 patch=arena.Load4(po);
    if((patch.w&0x80000000u)!=0u) {
        growPcmFrictionAnchors(patchIndex,pairIndex,errorAddress);
        patch=arena.Load4(po);
    }
    if(patch.z==0u)return;
    uint orderOffset=arena.Load(po+24u);
    if(patch.y==0u || orderOffset>=p.orderCount ||
       patch.x>p.frictionCount || patch.z>2u ||
       patch.z*2u>p.frictionCount-patch.x) {
        arena.InterlockedOr(errorAddress,16384u);
        return;
    }
    // setupFinalizeSolverConstraints chooses the first normal from the
    // source's reverse correlated contact-run list for this patch.
    uint first=arena.Load(p.orderBias+orderOffset*4u);
    if(first>=p.contactCount) {
        arena.InterlockedOr(errorAddress,16384u);
        return;
    }
    float3 n=asfloat(arena.Load3(p.contactBias+first*88u));
    uint pair=p.prepBias+pairIndex*72u;
    float3 a=prepBody3(arena.Load(pair),12u);
    float3 b=prepBody3(arena.Load(pair+4u),12u);
    float3 relative=a-b;
    float projection=dot(relative,n);
    float3 tangent0=relative-float3(
        n.x*projection,n.y*projection,n.z*projection);
    float squared=tangent0.x*tangent0.x+
        tangent0.y*tangent0.y+tangent0.z*tangent0.z;
    if(squared<=0.0001f)
        tangent0=abs(n.x)<0.70710678f
            ? float3(0.0f,-n.z,n.y)
            : float3(-n.y,n.x,0.0f);
    float lengthSq=tangent0.x*tangent0.x+
        tangent0.y*tangent0.y+tangent0.z*tangent0.z;
    float inverseLength=1.0f/sqrt(lengthSq);
    tangent0=tangent0*inverseLength;
    float3 tangent1=cross(n,tangent0);
    if(!all(isfinite(tangent0)) || !all(isfinite(tangent1))) {
        arena.InterlockedOr(errorAddress,32768u);
        return;
    }
    for(uint anchor=0u;anchor<patch.z;++anchor) {
        uint row=patch.x+anchor*2u;
        arena.Store3(p.frictionBias+row*72u,asuint(tangent0));
        arena.Store3(p.frictionBias+(row+1u)*72u,asuint(tangent1));
        prepareFrictionRow(row,pairIndex,errorAddress);
        prepareFrictionRow(row+1u,pairIndex,errorAddress);
    }
}
