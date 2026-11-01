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
    if(finalPass)
        arena.Store2(p.writebackBias+p.resultFrictionOffset+
            rowIndex*8u,packed);
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
            arena.Store(po+12u,broken?1u:0u);
            arena.Store(p.writebackBias+p.resultPatchOffset+
                patchIndex*4u,broken?1u:0u);
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
float3 prepInertia(uint body,float3 v) {
    float3 c0=asfloat(arena.Load3(body+36u));
    float3 c1=asfloat(arena.Load3(body+48u));
    float3 c2=asfloat(arena.Load3(body+60u));
    return c0*v.x+c1*v.y+c2*v.z;
}
void prepStoreDouble(uint address,double value) {
    uint lo,hi;
    asuint(value,lo,hi);
    arena.Store2(address,uint2(lo,hi));
}
void prepareContactRow(uint index,uint pairIndex,uint errorAddress) {
    if(index>=p.contactCount || pairIndex>=p.prepCount) {
        arena.InterlockedOr(errorAddress,4096u);
        return;
    }
    uint co=p.contactBias+index*88u;
    uint pair=p.prepBias+pairIndex*200u;
    uint b0=pair,b1=pair+80u;
    // The source correlation pass has already chosen this row's header
    // normal. The other raw inputs occupy its future Jacobian fields.
    float3 n=asfloat(arena.Load3(co));
    float3 pos=asfloat(arena.Load3(co+12u));
    float separation=asfloat(arena.Load(co+24u));
    float3 p0=asfloat(arena.Load3(b0));
    float3 p1=asfloat(arena.Load3(b1));
    float3 v0=asfloat(arena.Load3(b0+12u));
    float3 v1=asfloat(arena.Load3(b1+12u));
    float3 w0=asfloat(arena.Load3(b0+24u));
    float3 w1=asfloat(arena.Load3(b1+24u));
    float m0=asfloat(arena.Load(b0+72u));
    float m1=asfloat(arena.Load(b1+72u));
    float3 r0=cross(pos-p0,n);
    float3 r1=cross(pos-p1,n);
    float linearRel=dot(n,v0-v1);
    float pairSlop=asfloat(arena.Load(pair+164u));
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
    float restDistance=asfloat(arena.Load(pair+160u));
    float restitution=asfloat(arena.Load(pair+168u));
    float damping=asfloat(arena.Load(pair+172u));
    float maxPenBias=asfloat(arena.Load(pair+176u));
    float invDt=asfloat(arena.Load(pair+180u));
    float seconds=asfloat(arena.Load(pair+184u));
    float rigidBias=asfloat(arena.Load(pair+188u));
    float bounceThreshold=asfloat(arena.Load(pair+192u));
    bool acceleration=arena.Load(pair+196u)!=0u;
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
    arena.Store3(co+12u,asuint(ja));
    arena.Store3(co+24u,asuint(jb));
    prepStoreDouble(co+40u,multiplier);
    prepStoreDouble(co+48u,impulseMultiplier);
    prepStoreDouble(co+56u,biasedError);
    prepStoreDouble(co+64u,unbiasedError);
}
