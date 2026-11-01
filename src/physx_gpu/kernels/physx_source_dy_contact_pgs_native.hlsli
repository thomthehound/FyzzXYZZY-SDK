// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Source DySolverConstraintsShared.h contact and friction PGS, executing
// correlated Gu contact runs in source DyConstraintPartition order.
double contactDouble(uint addr) {
    uint2 v=contactPoints.Load2(addr);
    return asdouble(v.x,v.y);
}
void storeContactDouble(uint addr,double value,uint pointIndex,bool finalPass) {
    uint lo,hi;asuint(value,lo,hi);
    uint2 packed=uint2(lo,hi);
    contactPoints.Store2(addr,packed);
    if(finalPass)
        writeback.Store2(p.writebackBias+p.resultContactOffset+
            pointIndex*8u,packed);
}
double frictionDouble(uint addr) {
    uint2 v=frictionRows.Load2(addr);
    return asdouble(v.x,v.y);
}
void storeFrictionDouble(uint addr,double value,uint rowIndex,bool finalPass) {
    uint lo,hi;asuint(value,lo,hi);
    uint2 packed=uint2(lo,hi);
    frictionRows.Store2(addr,packed);
    if(finalPass)
        writeback.Store2(p.writebackBias+p.resultFrictionOffset+
            rowIndex*8u,packed);
}
bool contactImpulse(uint a,uint b,float3 n,float3 ja,float3 jb,
                    double delta,float ma,float mb,uint errorAddress) {
    uint ao=p.bodyBias+a*52u,bo=p.bodyBias+b*52u;
    float3 va=float3(0,0,0),vb=float3(0,0,0);
    float3 aa=float3(0,0,0),ab=float3(0,0,0);
    if(a!=invalidBody) {
        va=asfloat(bodies.Load3(ao));
        aa=asfloat(bodies.Load3(ao+12u));
    }
    if(b!=invalidBody) {
        vb=asfloat(bodies.Load3(bo));
        ab=asfloat(bodies.Load3(bo+12u));
    }
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
        tasks.InterlockedOr(errorAddress,16u);
        return false;
    }
    if(a!=invalidBody && ma>0.f) {
        bodies.Store3(ao,asuint(v0));
        bodies.Store3(ao+12u,asuint(w0));
    }
    if(b!=invalidBody && mb>0.f) {
        bodies.Store3(bo,asuint(v1));
        bodies.Store3(bo+12u,asuint(w1));
    }
    return true;
}
void solveContact(uint4 descriptor,uint errorAddress,bool biased,
                  bool frictionEnabled,bool finalPass) {
    if(descriptor.x>=p.manifoldCount ||
       (descriptor.y!=invalidBody && descriptor.y>=p.bodyCount) ||
       (descriptor.z!=invalidBody && descriptor.z>=p.bodyCount)) {
        tasks.InterlockedOr(errorAddress,32u);
        return;
    }
    uint4 manager=contactManifolds.Load4(
        p.manifoldBias+descriptor.x*16u);
    if(manager.x>p.patchCount || manager.y>p.patchCount-manager.x ||
       manager.z>p.orderCount || manager.w>p.orderCount-manager.z) {
        tasks.InterlockedOr(errorAddress,64u);
        return;
    }
    if(manager.y==0u)return;
    uint a=descriptor.y,b=descriptor.z;
    uint ao=p.bodyBias+a*52u,bo=p.bodyBias+b*52u;
    float ma=0.f,mb=0.f;
    if(a!=invalidBody)ma=asfloat(bodies.Load(ao+48u));
    if(b!=invalidBody)mb=asfloat(bodies.Load(bo+48u));
    uint cursor=manager.z;
    for(uint patchIndex=manager.x;patchIndex<manager.x+manager.y;++patchIndex) {
        uint po=p.patchBias+patchIndex*32u;
        uint4 patch=contactPatches.Load4(po);
        uint4 settings=contactPatches.Load4(po+16u);
        if(patch.x>p.frictionCount || patch.z>2u ||
           patch.z*2u>p.frictionCount-patch.x ||
           cursor>p.orderCount || patch.y>p.orderCount-cursor ||
           settings.z!=cursor) {
            tasks.InterlockedOr(errorAddress,128u);
            return;
        }
        double normalImpulse=0.0;
        for(uint i=0;i<patch.y;++i) {
            uint pointIndex=contactOrder.Load(
                p.orderBias+(cursor++)*4u);
            if(pointIndex>=p.contactCount) {
                tasks.InterlockedOr(errorAddress,256u);
                return;
            }
            uint co=p.contactBias+pointIndex*88u;
            float3 n=asfloat(contactPoints.Load3(co));
            float3 ja=asfloat(contactPoints.Load3(co+12u));
            float3 jb=asfloat(contactPoints.Load3(co+24u));
            float3 va=float3(0,0,0),vb=float3(0,0,0);
            float3 aa=float3(0,0,0),ab=float3(0,0,0);
            if(a!=invalidBody) {
                va=asfloat(bodies.Load3(ao));
                aa=asfloat(bodies.Load3(ao+12u));
            }
            if(b!=invalidBody) {
                vb=asfloat(bodies.Load3(bo));
                ab=asfloat(bodies.Load3(bo+12u));
            }
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
                tasks.InterlockedOr(errorAddress,1024u);
                return;
            }
            if(!contactImpulse(a,b,n,ja,jb,nextForce-oldForce,
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
            float3 n=asfloat(frictionRows.Load3(fo));
            float3 ja=asfloat(frictionRows.Load3(fo+12u));
            float3 jb=asfloat(frictionRows.Load3(fo+24u));
            float3 va=float3(0,0,0),vb=float3(0,0,0);
            float3 aa=float3(0,0,0),ab=float3(0,0,0);
            if(a!=invalidBody) {
                va=asfloat(bodies.Load3(ao));
                aa=asfloat(bodies.Load3(ao+12u));
            }
            if(b!=invalidBody) {
                vb=asfloat(bodies.Load3(bo));
                ab=asfloat(bodies.Load3(bo+12u));
            }
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
                tasks.InterlockedOr(errorAddress,2048u);
                return;
            }
            if(!contactImpulse(a,b,n,ja,jb,nextForce-oldForce,
                               ma,mb,errorAddress))return;
            storeFrictionDouble(fo+64u,nextForce,k,finalPass);
            broken=broken||saturated;
        }
        if(finalPass) {
            contactPatches.Store(po+12u,broken?1u:0u);
            writeback.Store(p.writebackBias+p.resultPatchOffset+
                patchIndex*4u,broken?1u:0u);
        }
    }
    if(cursor!=manager.z+manager.w)
        tasks.InterlockedOr(errorAddress,512u);
}
