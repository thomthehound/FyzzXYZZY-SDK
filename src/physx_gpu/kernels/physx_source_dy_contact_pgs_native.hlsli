// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Source DySolverConstraintsShared.h contact and friction PGS, executing
// correlated Gu contact runs in source DyConstraintPartition order.
float contactScalar(uint addr) {
    return asfloat(arena.Load(addr));
}
void storeContactScalar(uint addr,float value,uint pointIndex,bool finalPass) {
    arena.Store(addr,asuint(value));
    if(finalPass)
        arena.Store(p.writebackBias+p.resultContactOffset+pointIndex*4u,
                    asuint(value));
}
float frictionScalar(uint addr) {
    return asfloat(arena.Load(addr));
}
void storeFrictionScalar(uint addr,float value,uint rowIndex,bool finalPass) {
    arena.Store(addr,asuint(value));
    if(finalPass && (p.launchControl&0x20000000u)==0u) {
        uint target=p.writebackBias+p.resultFrictionOffset+rowIndex*16u;
        arena.Store3(target,arena.Load3(p.frictionBias+rowIndex*56u));
        arena.Store(target+12u,asuint(value));
    }
}
// The source PCM producer already writes contact points in Gu order.
// The high bit selects that resident range; CPU Gu patches retain
// their explicit correlated-order upload.
static const uint sourceDyDirectContactOrder=0x80000000u;
bool sourceDyContactOrderValid(uint order,uint count) {
    const uint begin=order&~sourceDyDirectContactOrder;
    const uint limit=(order&sourceDyDirectContactOrder)!=0u
        ?p.contactCount:p.orderCount;
    return begin<=limit && count<=limit-begin;
}
uint sourceDyContactOrderPoint(uint order,uint relative) {
    const uint index=(order&~sourceDyDirectContactOrder)+relative;
    return (order&sourceDyDirectContactOrder)!=0u
        ?index:arena.Load(p.orderBias+index*4u);
}
bool contactImpulse(uint a,uint b,uint ao,uint bo,
                    float3 n,float3 ja,float3 jb,
                    inout float3 va,inout float3 vb,
                    inout float3 aa,inout float3 ab,
                    float delta,float ma,float mb,uint errorAddress) {
    // Source contact PGS multiplies impulse by inverse mass first.
    precise float da=delta*ma,db=delta*mb;
    precise float3 v0=va+n*da;
    precise float3 v1=vb-n*db;
    precise float3 w0=aa+ja*delta;
    precise float3 w1=ab-jb*delta;
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
    // Zero-response descriptors carry source endpoint ordinals in the
    // otherwise unused patch offsets. They have no contact rows to read.
    if(manager.y==0u)return;
    if(manager.x>p.patchCount || manager.y>p.patchCount-manager.x ||
       manager.z>p.contactCount ||
       manager.w>p.contactCount-manager.z) {
        arena.InterlockedOr(errorAddress,64u);
        return;
    }
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
    uint contactCursor=0u;
    for(uint patchIndex=manager.x;patchIndex<manager.x+manager.y;++patchIndex) {
        uint po=p.patchBias+patchIndex*32u;
        uint4 patch=arena.Load4(po);
        uint4 settings=arena.Load4(po+16u);
        if(patch.x>p.frictionCount || patch.z>2u ||
           patch.z*2u>p.frictionCount-patch.x ||
           contactCursor>manager.w ||
           patch.y>manager.w-contactCursor ||
           !sourceDyContactOrderValid(settings.z,patch.y)) {
            arena.InterlockedOr(errorAddress,128u);
            return;
        }
        precise float normalImpulse=0.0;
        for(uint i=0;i<patch.y;++i) {
            uint pointIndex=sourceDyContactOrderPoint(settings.z,i);
            if(pointIndex>=p.contactCount) {
                arena.InterlockedOr(errorAddress,256u);
                return;
            }
            uint co=p.contactBias+pointIndex*64u;
            float3 n=asfloat(arena.Load3(co));
            float3 ja=asfloat(arena.Load3(co+12u));
            float3 jb=asfloat(arena.Load3(co+24u));
            precise float normalVelocity=
                sourceDyDot3(n,va)-sourceDyDot3(n,vb)+
                sourceDyDot3(aa,ja)-sourceDyDot3(ab,jb);
            float oldForce=contactScalar(co+56u);
            float error=biased?contactScalar(co+48u):
                                contactScalar(co+52u);
            precise float proposed=
                -normalVelocity*contactScalar(co+40u)+error;
            if(!isfinite(proposed) || !isfinite(oldForce)) {
                arena.InterlockedOr(errorAddress,1024u);
                return;
            }
            precise float candidate=max(proposed,-oldForce);
            precise float nextForce=min(contactScalar(co+60u),
                contactScalar(co+44u)*oldForce+candidate);
            if(!isfinite(nextForce)) {
                arena.InterlockedOr(errorAddress,1024u);
                return;
            }
            if(!contactImpulse(a,b,ao,bo,n,ja,jb,
                               va,vb,aa,ab,nextForce-oldForce,
                               ma,mb,errorAddress))return;
            storeContactScalar(co+56u,nextForce,pointIndex,finalPass);
            normalImpulse+=nextForce;
        }
        contactCursor+=patch.y;
        if(!frictionEnabled || patch.z==0u)continue;
        float coeff=patch.z==2u?0.5:1.0;
        precise float maxStatic=asfloat(settings.x)*coeff*normalImpulse;
        precise float maxDynamic=asfloat(settings.y)*coeff*normalImpulse;
        bool broken=false;
        for(uint k=patch.x;k<patch.x+patch.z*2u;++k) {
            uint fo=p.frictionBias+k*56u;
            float3 n=asfloat(arena.Load3(fo));
            float3 ja=asfloat(arena.Load3(fo+12u));
            float3 jb=asfloat(arena.Load3(fo+24u));
            precise float tangentDelta=
                sourceDyDot3(n,va)-sourceDyDot3(n,vb)+
                sourceDyDot3(aa,ja)-sourceDyDot3(ab,jb);
            float oldForce=frictionScalar(fo+52u);
            float bias=biased?frictionScalar(fo+44u):0.0;
            precise float candidate=oldForce+
                (frictionScalar(fo+48u)-bias-tangentDelta)*
                frictionScalar(fo+40u);
            if(!isfinite(candidate) || !isfinite(maxStatic) ||
               !isfinite(maxDynamic)) {
                arena.InterlockedOr(errorAddress,2048u);
                return;
            }
            bool saturated=abs(candidate)>maxStatic;
            // Source signed FMin/FMax also handles negative compliant bounds.
            precise float nextForce=saturated
                ? min(maxDynamic,max(-maxDynamic,candidate)):candidate;
            if(!isfinite(nextForce)) {
                arena.InterlockedOr(errorAddress,2048u);
                return;
            }
            if(!contactImpulse(a,b,ao,bo,n,ja,jb,
                               va,vb,aa,ab,nextForce-oldForce,
                               ma,mb,errorAddress))return;
            storeFrictionScalar(fo+52u,nextForce,k,finalPass);
            broken=broken||saturated;
        }
        if(finalPass) {
            if((p.launchControl&0x20000000u)!=0u) {
                // Report one world impulse per source friction anchor.
                // The full solved tangent rows remain device-resident.
                for(uint anchor=0u;anchor<patch.z;++anchor) {
                    const uint row=patch.x+anchor*2u;
                    const uint x=p.frictionBias+row*56u;
                    const uint y=x+56u;
                    precise float3 fx=asfloat(arena.Load3(x))*
                        frictionScalar(x+52u);
                    precise float3 fy=asfloat(arena.Load3(y))*
                        frictionScalar(y+52u);
                    precise float3 impulse=fx+fy;
                    if(!all(isfinite(impulse))) {
                        arena.InterlockedOr(errorAddress,2048u);
                        return;
                    }
                    arena.Store3(p.writebackBias+p.resultFrictionOffset+
                        (row/2u)*12u,asuint(impulse));
                }
            }
            bool gpuGrowth=(patch.w&0x80000000u)!=0u;
            arena.Store(po+12u,broken?1u:0u);
            uint reportAddress=p.writebackBias+p.resultPatchOffset+
                patchIndex*4u;
            uint report=gpuGrowth?arena.Load(reportAddress):0u;
            arena.Store(reportAddress,(report&~1u)|(broken?1u:0u));
        }
    }
    if(contactCursor!=manager.w)
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
    const uint ordinal=source>>1u;
    const uint base=p.integrationBias+40u+ordinal*76u;
    if(offset>=36u && offset<=60u)return base+offset-36u;
    if(offset==72u)return p.bodyBias+ordinal*52u+48u;
    return base+offset;
}
// Resolve the identical Sc mass frame for fixed and integrated participants.
// A tagged body with no host pose reads the authorized device transform.
bool prepBodyPose(uint source,out float3 position,out float4 orientation) {
    if((source&1u)==0u) {
        const uint base=p.prepBias+source;
        position=asfloat(arena.Load3(base));
        orientation=asfloat(arena.Load4(base+80u));
        return all(isfinite(position)) && all(isfinite(orientation));
    }
    const uint input=p.integrationBias+40u+(source>>1u)*76u;
    return sourceDyMassPose(input,position,orientation);
}
float3 prepBody3(uint source,uint offset) {
    if((source&1u)!=0u && (offset==12u || offset==24u)) {
        const uint input=p.integrationBias+40u+(source>>1u)*76u;
        return sourceDyPreparedVelocity(input,offset==24u);
    }
    return asfloat(arena.Load3(prepBodyOffset(source,offset)));
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
void prepStoreScalar(uint address,float value) {
    arena.Store(address,asuint(value));
}
void prepareContactRow(uint index,uint pairIndex,uint contactBegin,
                       uint contactCount,uint errorAddress) {
    if(index>=p.contactCount || pairIndex>=p.prepCount) {
        arena.InterlockedOr(errorAddress,4096u);
        return;
    }
    uint co=p.contactBias+index*64u;
    uint pair=p.prepBias+pairIndex*64u;
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
           contactCount!=((packed>>10u)&7u)) {
            arena.InterlockedOr(errorAddress,65536u);
            return;
        }
        n=asfloat(pcmContacts.Load3(source+8u));
        uint contact=source+20u+local*16u;
        pos=asfloat(pcmContacts.Load3(contact));
        separation=asfloat(pcmContacts.Load(contact+12u));
        arena.Store3(co,asuint(n));
    } else {
        uint cpuOffset=arena.Load(pair+60u);
        if(cpuOffset==0xffffffffu || index<contactBegin ||
           index-contactBegin>=contactCount) {
            arena.InterlockedOr(errorAddress,131072u);
            return;
        }
        uint raw=p.prepBias+p.prepCount*64u+
            (cpuOffset+index-contactBegin)*28u;
        n=asfloat(arena.Load3(raw));
        pos=asfloat(arena.Load3(raw+12u));
        separation=asfloat(arena.Load(raw+24u));
        arena.Store3(co,asuint(n));
    }
    float3 p0,p1;
    float4 q0,q1;
    if(!prepBodyPose(b0,p0,q0) || !prepBodyPose(b1,p1,q1)) {
        arena.InterlockedOr(errorAddress,524288u);
        return;
    }
    float3 v0=prepBody3(b0,12u);
    float3 v1=prepBody3(b1,12u);
    float3 w0=prepBody3(b0,24u);
    float3 w1=prepBody3(b1,24u);
    float m0=prepBodyFloat(b0,72u);
    float m1=prepBodyFloat(b1,72u);
    float3 r0=cross(pos-p0,n);
    float3 r1=cross(pos-p1,n);
    float linearRel=sourceDyDot3(n,v0-v1);
    float pairSlop=asfloat(arena.Load(pair+12u));
    if(pairSlop>0.0f) {
        float angularRel=sourceDyDot3(r0,w0)-sourceDyDot3(r1,w1);
        float ratio=linearRel==0.0f
            ? 3.402823466e+38f : angularRel/linearRel;
        float slop=pairSlop*max(ratio,1.0f);
        r0=prepSlop(r0,slop);
        r1=prepSlop(r1,slop);
    }
    float3 ja=prepInertia(b0,r0);
    float3 jb=prepInertia(b1,r1);
    float angularRel=sourceDyDot3(r0,w0)-sourceDyDot3(r1,w1);
    float vrel=linearRel+angularRel;
    precise float response=(m0+sourceDyDot3(ja,ja))+(m1+sourceDyDot3(jb,jb));
    precise float multiplier=response>0.0?1.0/response:0.0;
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
    float target=(targetVelocity-vrel);
    float impulseMultiplier=1.0;
    precise float biasedError,unbiasedError;
    if(restitution<0.0f) {
        float enabledDamping=separated&&!closing?0.0f:damping;
        float nrdt=seconds*restitution;
        float aCoeff=seconds*
            (enabledDamping-nrdt);
        float massFactor=acceleration?multiplier:1.0;
        float x=1.0/(1.0+aCoeff*(acceleration?1.0:response));
        float bias=x*(-nrdt*penetration*massFactor);
        multiplier=x*aCoeff*massFactor;
        impulseMultiplier=1.0-x;
        biasedError=target*multiplier-bias;
        unbiasedError=biasedError;
    } else {
        float penetrationScaled=separated?penetrationInvDt:
            penetration*(invDt*rigidBias);
        float scaledBias=bouncing?0.0:
            multiplier*max(maxPenBias,penetrationScaled);
        biasedError=target*multiplier-scaledBias;
        unbiasedError=target*multiplier-
            (bouncing?0.0:max(scaledBias,0.0));
    }
    if(!isfinite(response) || !isfinite(multiplier) ||
       !isfinite(impulseMultiplier) ||
       !isfinite(biasedError) || !isfinite(unbiasedError) ||
       !all(isfinite(ja)) || !all(isfinite(jb))) {
        arena.InterlockedOr(errorAddress,8192u);
        return;
    }
    // A freshly prepared source Gu normal row begins with zero impulse.
    // Its limit is common to the entire contact-manager pair.
    arena.Store(co+36u,0u);
    arena.Store(co+56u,0u);
    arena.Store(co+60u,arena.Load(pair+56u));
    arena.Store3(co+12u,asuint(ja));
    arena.Store3(co+24u,asuint(jb));
    prepStoreScalar(co+40u,multiplier);
    prepStoreScalar(co+44u,impulseMultiplier);
    prepStoreScalar(co+48u,biasedError);
    prepStoreScalar(co+52u,unbiasedError);
}

void prepareFrictionRow(uint index,uint patchIndex,uint pairIndex,
                        uint errorAddress) {
    if(index>=p.frictionCount || pairIndex>=p.prepCount) {
        arena.InterlockedOr(errorAddress,16384u);
        return;
    }
    uint fo=p.frictionBias+index*56u;
    uint pair=p.prepBias+pairIndex*64u;
    uint b0=arena.Load(pair),b1=arena.Load(pair+4u);
    // Source DyFrictionCorrelation selected these body-local anchors.
    // Source DyContactPrep selected the two tangents and their order.
    float3 tangent=asfloat(arena.Load3(fo));
    // The original Gu correlation owns one body-local anchor for both
    // tangent rows. PCM retrieves its resident EdgeIndex history;
    // CPU Gu reads its compact patch-local upload without PCM padding.
    uint po=p.patchBias+patchIndex*32u;
    uint first=arena.Load(po);
    if(index<first || ((index-first)>>1u)>=2u) {
        arena.InterlockedOr(errorAddress,16384u);
        return;
    }
    uint slot=(index-first)>>1u;
    uint anchor;
    if(arena.Load(pair+48u)!=0xffffffffu) {
        uint history=p.writebackBias+arena.Load(pair+52u)*64u;
        if(arena.Load(history)<=slot) {
            arena.InterlockedOr(errorAddress,262144u);
            return;
        }
        anchor=history+8u+slot*24u;
    }else {
        uint backward=arena.Load(po+28u);
        if(backward==0u || backward>p.contactBias-p.writebackBias) {
            arena.InterlockedOr(errorAddress,16384u);
            return;
        }
        anchor=p.contactBias-backward+slot*24u;
    }
    float3 p0,p1;
    float4 q0,q1;
    if(!prepBodyPose(b0,p0,q0) || !prepBodyPose(b1,p1,q1)) {
        arena.InterlockedOr(errorAddress,524288u);
        return;
    }
    float3 r0=sourceRigidRotate(q0,asfloat(arena.Load3(anchor)));
    float3 r1=sourceRigidRotate(q1,asfloat(arena.Load3(anchor+12u)));
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
    precise float response=(m0+sourceDyDot3(ja,ja))+(m1+sourceDyDot3(jb,jb));
    precise float velocityMultiplier=response>0.0
        ? rigidBias/response:0.0;
    precise float bias=(sourceDyDot3(tangent,error)*invDt);
    float3 relative=v0-v1;
    precise float targetVelocity=(-(sourceDyDot3(tangent,relative)+
        sourceDyDot3(ax0,w0)-sourceDyDot3(ax1,w1)));
    if(!isfinite(response) || !isfinite(velocityMultiplier) ||
       !isfinite(bias) ||
       !isfinite(targetVelocity) || !all(isfinite(ja)) ||
       !all(isfinite(jb))) {
        arena.InterlockedOr(errorAddress,32768u);
        return;
    }
    arena.Store3(fo+12u,asuint(ja));
    arena.Store3(fo+24u,asuint(jb));
    arena.Store(fo+36u,0u);
    arena.Store(fo+52u,0u);
    prepStoreScalar(fo+40u,velocityMultiplier);
    prepStoreScalar(fo+44u,bias);
    prepStoreScalar(fo+48u,targetVelocity);
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
    uint pair=p.prepBias+pairIndex*64u;
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
    // One Sc-owned EdgeIndex physical mirror feeds both patch growth
    // and Dy tangent preparation. Only a rejected-frame recovery reads
    // the compact host anchor input through PCM's spare Gu offset slot.
    uint history=p.writebackBias+arena.Load(pair+52u)*64u;
    // The source PCM manager has one contiguous correlated contact run.
    // Reuse this resident geometry for both source patch extent and
    // anchor selection, rather than uploading the host-computed diagonal.
    float3 contacts[4];
    float separations[4];
    float3 lower=float3(0.f,0.f,0.f);
    float3 upper=float3(0.f,0.f,0.f);
    [loop]
    for(uint j=0u;j<patch.y;++j) {
        const uint src=stream+20u+j*16u;
        contacts[j]=asfloat(pcmContacts.Load3(src));
        separations[j]=asfloat(pcmContacts.Load(src+12u));
        if(j==0u) {
            lower=contacts[j];
            upper=contacts[j];
        } else {
            lower=min(lower,contacts[j]);
            upper=max(upper,contacts[j]);
        }
    }
    const float3 extent=upper-lower;
    precise float diagonal=(extent.x*extent.x+extent.y*extent.y)+
        extent.z*extent.z;
    uint count=patch.z;
    bool normalChanged=false;
    if((patch.w&0x20000000u)!=0u) {
        float3 position0;
        float4 orientation0;
        if(!prepBodyPose(body0,position0,orientation0)) {
            arena.InterlockedOr(errorAddress,524288u);
            return;
        }
        const uint encoded=arena.Load(po+28u);
        float2 oct=float2(int(encoded<<16u)>>16u,int(encoded)>>16u)*
            (1.0f/32767.0f);
        float3 localNormal=float3(oct,1.0f-abs(oct.x)-abs(oct.y));
        if(localNormal.z<0.0f) {
            const float2 xy=localNormal.xy;
            localNormal.xy=float2(
                (1.0f-abs(xy.y))*(xy.x>=0.0f?1.0f:-1.0f),
                (1.0f-abs(xy.x))*(xy.y>=0.0f?1.0f:-1.0f));
        }
        const float3 previous=sourceRigidRotate(
            orientation0,normalize(localNormal));
        const float3 current=asfloat(pcmContacts.Load3(stream+8u));
        if(!all(isfinite(previous)) || !all(isfinite(current))) {
            arena.InterlockedOr(errorAddress,262144u);
            return;
        }
        normalChanged=sourceDyDot3(previous,current)<0.999f;
        if(normalChanged)count=0u;
    }
    if(count>0u) {
        if((patch.w&0x40000000u)!=0u) {
            uint backward=arena.Load(pair+60u);
            if(backward==0xffffffffu || backward==0u ||
               backward>p.contactBias-p.writebackBias) {
                arena.InterlockedOr(errorAddress,262144u);
                return;
            }
            uint source=p.contactBias-backward;
            [unroll]
            for(uint k=0u;k<2u;++k) {
                if(k>=count)break;
                arena.Store3(history+8u+k*24u,
                             arena.Load3(source+k*24u));
                arena.Store3(history+20u+k*24u,
                             arena.Load3(source+k*24u+12u));
            }
        }else if(arena.Load(history)!=count) {
            arena.InterlockedOr(errorAddress,262144u);
            return;
        }
    }
    float3 local0=count>0u?asfloat(arena.Load3(history+8u)):
                            float3(0.f,0.f,0.f);
    float3 local1=count>1u?asfloat(arena.Load3(history+32u)):
                            float3(0.f,0.f,0.f);
    bool keepPair=false;
    if(count==2u) {
        float3 delta=local0-local1;
        float distance=delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
        keepPair=distance*4.0f>=diagonal;
        if(!keepPair)count=0u;
    }
    float3 anchors[2];
    uint selected[2]={0u,0u};
    bool changed[2]={false,false};
    float pointDistance=0.0f;
    if(!keepPair) {
        float3 position0,position1;
        float4 orientation0,orientation1;
        if(!prepBodyPose(body0,position0,orientation0) ||
           !prepBodyPose(body1,position1,orientation1)) {
            arena.InterlockedOr(errorAddress,524288u);
            return;
        }
        if(count==1u)
            anchors[0]=sourceRigidRotate(orientation0,local0)+position0;
        float limit=asfloat(arena.Load(pair+8u))+0.04f;
        [loop]
        for(uint j=0u;j<patch.y;++j) {
            if(separations[j]>=limit)continue;
            float3 world=contacts[j];
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
            float3 a=position0,b=position1;
            float4 qa=orientation0,qb=orientation1;
            float3 da=w-a,db=w-b;
            float3 ta=cross(-qa.xyz,da)*2.0f;
            float3 tb=cross(-qb.xyz,db)*2.0f;
            float3 la=da+qa.w*ta+cross(-qa.xyz,ta);
            float3 lb=db+qb.w*tb+cross(-qb.xyz,tb);
            arena.Store3(history+8u+k*24u,asuint(la));
            arena.Store3(history+20u+k*24u,asuint(lb));
        }
    }
    arena.Store(history,count);
    arena.Store(po+8u,count);
    uint report=(count<<1u) |
        ((changed[0]?(selected[0]+1u):0u)<<3u) |
        ((changed[1]?(selected[1]+1u):0u)<<6u) |
        (normalChanged?0x200u:0u);
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
    if(patch.y==0u || !sourceDyContactOrderValid(orderOffset,patch.y) ||
       patch.x>p.frictionCount || patch.z>2u ||
       patch.z*2u>p.frictionCount-patch.x) {
        arena.InterlockedOr(errorAddress,16384u);
        return;
    }
    // setupFinalizeSolverConstraints chooses the first normal from the
    // source's reverse correlated contact-run list for this patch.
    uint first=sourceDyContactOrderPoint(orderOffset,0u);
    if(first>=p.contactCount) {
        arena.InterlockedOr(errorAddress,16384u);
        return;
    }
    float3 n=asfloat(arena.Load3(p.contactBias+first*64u));
    uint pair=p.prepBias+pairIndex*64u;
    float3 a=prepBody3(arena.Load(pair),12u);
    float3 b=prepBody3(arena.Load(pair+4u),12u);
    float3 relative=a-b;
    float projection=sourceDyDot3(relative,n);
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
        arena.Store3(p.frictionBias+row*56u,asuint(tangent0));
        arena.Store3(p.frictionBias+(row+1u)*56u,asuint(tangent1));
        prepareFrictionRow(row,patchIndex,pairIndex,errorAddress);
        prepareFrictionRow(row+1u,patchIndex,pairIndex,errorAddress);
    }
}
