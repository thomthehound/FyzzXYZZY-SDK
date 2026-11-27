// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_source_box_bounds_math_native.hlsli"

// DyConstraintPartition supplies disjoint writes within each partition.
// Wide tasks use device dispatch boundaries between partitions. Compact
// tasks keep those dependencies inside their owning workgroup.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer arena : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer pcmContacts : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer scTransforms : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer scBounds : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer scGeometry : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer scActorLocals : register(u5,space0);
#ifdef FYZZXYZZY_CONVEX
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer scHullVertices : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer scIslandWork : register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer jointSources : register(u8,space0);
#ifdef __spirv__
[[vk::binding(9,0)]]
#endif
RWByteAddressBuffer retainedSources : register(u9,space0);
#else
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer scIslandWork : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer jointSources : register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer retainedSources : register(u8,space0);
#endif

struct Parameters {
    uint bodyBias, distanceBias, descriptorBias, partitionBias;
    uint ordinalBias, taskBias;
    uint bodyCount, rowCount, launchBegin, partitionCount;
    uint launchControl, integrationBias;
    uint contactBias,frictionBias,patchBias,manifoldBias,orderBias;
    uint contactCount,frictionCount,patchCount,manifoldCount,orderCount;
    uint resultBodyOffset,resultTaskOffset,resultContactOffset;
    uint resultFrictionOffset,resultPatchOffset,reservedOutput;
    uint writebackBias;
    uint prepBias,prepCount,pcmBias;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif
static const uint invalidBody = 0xffffffffu;
uint sourceDyInertiaAddress(uint ordinal) {
    return arena.Load(p.integrationBias+40u)+ordinal*36u;
}
static const uint sourceDyPcmExtentValid = 1u<<30u;
static const uint sourceDyPcmExtentShift = 16u;
static const uint sourceDyPcmCompacted = 1u<<28u;
static const float maxImpulse = 3.4028234663852886e+38;

// Source PxReal dot3 evaluates three rounded products followed by two
// ordered sums. Preserve that arithmetic at the iterative PGS boundary.
float sourceDyDot3(float3 a,float3 b) {
    precise float x=a.x*b.x;
    precise float y=a.y*b.y;
    precise float z=a.z*b.z;
    precise float xy=x+y;
    precise float result=xy+z;
    return result;
}

float loadScalar(uint byteOffset) {
    return asfloat(arena.Load(byteOffset));
}
uint sourceDyBodyOrdinal(uint index) {
    const uint address=p.ordinalBias+index*4u;
    if((p.launchControl&0x08000000u)==0u)
        return arena.Load(address);
    const uint source=scIslandWork.Load(address);
    const uint scratch=arena.Load(p.integrationBias+28u);
    const uint remap=arena.Load(scratch+48u);
    return remap!=0u && source!=invalidBody
        ?scIslandWork.Load(p.ordinalBias+remap+source*4u):source;
}
uint sourceDyResponseToken(uint scratch,uint colors,uint index) {
    if((p.launchControl&0x08000000u)!=0u) {
        const uint offset=arena.Load(scratch+20u);
        const uint token=scIslandWork.Load(p.ordinalBias+offset+index*4u);
        if((token&0xc0000000u)==0x80000000u) {
            const uint remap=arena.Load(scratch+52u);
            if(remap!=0u) {
                const uint ordinal=token&0x3fffffffu;
                const uint count=arena.Load(scratch+56u);
                const uint row=ordinal<count
                    ?scIslandWork.Load(p.ordinalBias+remap+ordinal*4u)
                    :invalidBody;
                return 0x80000000u|(row&0x3fffffffu);
            }
        }
        return token;
    }
    return arena.Load(colors+index*4u);
}
uint sourceDyResponseExtent(uint scratch,uint index) {
    if((p.launchControl&0x08000000u)==0u)return 0u;
    const uint offset=arena.Load(scratch+28u);
    return scIslandWork.Load(p.ordinalBias+offset+index*4u);
}
uint sourceDyContactPrefix(uint scratch,uint response) {
    const uint offset=arena.Load(scratch+44u);
    return scIslandWork.Load(p.ordinalBias+offset+response*4u);
}
uint sourceDyTaskLoad(uint address) {
    return (p.launchControl&0x08000000u)!=0u
        ?scIslandWork.Load(address):arena.Load(address);
}
uint2 sourceDyTaskLoad2(uint address) {
    return (p.launchControl&0x08000000u)!=0u
        ?scIslandWork.Load2(address):arena.Load2(address);
}
uint4 sourceDyTaskLoad4(uint address) {
    return (p.launchControl&0x08000000u)!=0u
        ?scIslandWork.Load4(address):arena.Load4(address);
}
void sourceDyTaskStore(uint address,uint value) {
    if((p.launchControl&0x08000000u)!=0u)
        scIslandWork.Store(address,value);
    else arena.Store(address,value);
}
void sourceDyTaskErrorOr(uint address,uint value) {
    if((p.launchControl&0x08000000u)!=0u)
        scIslandWork.InterlockedOr(address,value);
    else arena.InterlockedOr(address,value);
}
struct SourceDySolverPair {
    float3 va,vb,aa,ab;
    float ma,mb;
};

void sourceDyLoadEndpoint(uint ordinal,uint2 retained,SourceDySolverPair prior,
                          out float3 linearDelta,out float3 angular) {
    linearDelta=0.0f;
    angular=0.0f;
    if(ordinal==invalidBody)return;
    if(ordinal==retained.y) {
        linearDelta=prior.vb;
        angular=prior.ab;
    } else if(ordinal==retained.x) {
        linearDelta=prior.va;
        angular=prior.aa;
    } else {
        const uint address=p.bodyBias+ordinal*52u;
        linearDelta=asfloat(arena.Load3(address));
        angular=asfloat(arena.Load3(address+12u));
    }
}

// Nonresponding endpoints are read-only, including shared kinematics.
bool sourceDyLoadSolverPair(uint4 desc,uint errorAddress,bool response,
                            uint2 retained,SourceDySolverPair prior,
                            out SourceDySolverPair state) {
    state=(SourceDySolverPair)0;
    const bool joint=desc.w!=0u;
    if((desc.y!=invalidBody && desc.y>=p.bodyCount) ||
       (desc.z!=invalidBody && desc.z>=p.bodyCount) ||
       (joint && (desc.x>=p.rowCount || desc.w>p.rowCount-desc.x)) ||
       (!joint && desc.x>=p.manifoldCount)) {
        sourceDyTaskErrorOr(errorAddress,joint?1u:32u);
        return false;
    }
    if(!response)return true;
    const uint ao=p.bodyBias+desc.y*52u;
    const uint bo=p.bodyBias+desc.z*52u;
    sourceDyLoadEndpoint(desc.y,retained,prior,state.va,state.aa);
    sourceDyLoadEndpoint(desc.z,retained,prior,state.vb,state.ab);
    if(joint) {
        const uint first=p.distanceBias+desc.x*64u;
        state.ma=loadScalar(first+36u);
        state.mb=loadScalar(first+40u);
    } else {
        if(desc.y!=invalidBody)state.ma=loadScalar(ao+48u);
        if(desc.z!=invalidBody)state.mb=loadScalar(bo+48u);
    }
    return true;
}

bool sourceDyLoadSolverPair(uint4 desc,uint errorAddress,bool response,
                            out SourceDySolverPair state) {
    return sourceDyLoadSolverPair(desc,errorAddress,response,
        uint2(invalidBody,invalidBody),(SourceDySolverPair)0,state);
}

// Ordered biased-phase writers also own the motion snapshot in resident
// wide tasks. Untouched bodies retain their initialized zero deltas.
void sourceDyStoreSolverPair(uint4 desc,SourceDySolverPair state,bool captureMotion) {
    if(desc.y!=invalidBody && state.ma>0.0f) {
        const uint ao=p.bodyBias+desc.y*52u;
        arena.Store3(ao,asuint(state.va));
        arena.Store3(ao+12u,asuint(state.aa));
        if(captureMotion) {
            arena.Store3(ao+24u,asuint(state.va));
            arena.Store3(ao+36u,asuint(state.aa));
        }
    }
    if(desc.z!=invalidBody && state.mb>0.0f) {
        const uint bo=p.bodyBias+desc.z*52u;
        arena.Store3(bo,asuint(state.vb));
        arena.Store3(bo+12u,asuint(state.ab));
        if(captureMotion) {
            arena.Store3(bo+24u,asuint(state.vb));
            arena.Store3(bo+36u,asuint(state.ab));
        }
    }
}

struct SourceDyJointRow {
    float3 axis,ja,jb;
    float multiplier;
};

SourceDyJointRow sourceDyLoadJointRow(uint offset) {
    SourceDyJointRow row;
    row.axis=asfloat(arena.Load3(offset));
    row.ja=asfloat(arena.Load3(offset+12u));
    row.jb=asfloat(arena.Load3(offset+24u));
    row.multiplier=loadScalar(offset+56u);
    return row;
}

bool sourceDySolveJointRow(SourceDyJointRow row,uint coefficient,
                          uint errorAddress,inout SourceDySolverPair local,
                          inout float previous) {
    precise float normalDelta=
        sourceDyDot3(row.axis,local.va-local.vb)+
        sourceDyDot3(row.ja,local.aa)-sourceDyDot3(row.jb,local.ab);
    if((coefficient&0x7f800000u)==0x7f800000u) {
        sourceDyTaskErrorOr(errorAddress,2u);
        return false;
    }
    precise float velocityTerm=normalDelta*row.multiplier;
    precise float force=previous+velocityTerm;
    precise float candidate=force+asfloat(coefficient);
    if(!isfinite(candidate) || !isfinite(normalDelta) ||
       !isfinite(previous)) {
        sourceDyTaskErrorOr(errorAddress,2u);
        return false;
    }
    precise float next=clamp(candidate,-maxImpulse,maxImpulse);
    precise float impulse=next-previous;
    precise float3 va=local.va+row.axis*(impulse*local.ma);
    precise float3 vb=local.vb-row.axis*(impulse*local.mb);
    precise float3 aa=local.aa+row.ja*impulse;
    precise float3 ab=local.ab-row.jb*impulse;
    if(!isfinite(next) || !all(isfinite(va)) || !all(isfinite(vb)) ||
       !all(isfinite(aa)) || !all(isfinite(ab))) {
        sourceDyTaskErrorOr(errorAddress,2u);
        return false;
    }
    local.va=va;
    local.vb=vb;
    local.aa=aa;
    local.ab=ab;
    previous=next;
    return true;
}

// Only responding endpoints carry joint deltas into the next descriptor pass.
void sourceDyCommitJointPair(uint4 desc,SourceDySolverPair local,
                             inout SourceDySolverPair state) {
    if(desc.y!=invalidBody && local.ma>0.0f) {
        state.va=local.va;
        state.aa=local.aa;
    }
    if(desc.z!=invalidBody && local.mb>0.0f) {
        state.vb=local.vb;
        state.ab=local.ab;
    }
}

bool solveJointRows(uint4 desc,uint errorAddress,bool biased,
                    inout SourceDySolverPair state) {
    SourceDySolverPair local=state;
    const uint first=p.distanceBias+desc.x*64u;
    for(uint row=0u;row<desc.w;++row) {
        const uint offset=first+row*64u;
        const SourceDyJointRow data=sourceDyLoadJointRow(offset);
        const uint coefficient=arena.Load(offset+(biased?48u:52u));
        float previous=loadScalar(offset+60u);
        if(!sourceDySolveJointRow(data,coefficient,errorAddress,local,previous))
            return false;
        arena.Store(offset+60u,asuint(previous));
    }
    sourceDyCommitJointPair(desc,local,state);
    return true;
}

// Native Dy reconstructs the mass frame from resident actor motion.
// Standalone inputs retain their cached-shape or explicit pose source.
bool sourceDyMassPose(uint input,out float3 position,out float4 quaternion) {
    const uint pose=arena.Load(input+20u);
    if((arena.Load(input+8u)&128u)!=0u) {
        const uint local=arena.Load(input+24u);
        const float3 actorPosition=asfloat(scActorLocals.Load3(local+116u));
        const float4 actorAxes=asfloat(scActorLocals.Load4(local+128u));
        const float3 massLocal=asfloat(scActorLocals.Load3(local));
        const float4 massAxes=asfloat(scActorLocals.Load4(local+12u));
        position=actorPosition+sourceRigidRotate(actorAxes,massLocal);
        quaternion=sourceRigidMultiply(actorAxes,massAxes);
    }else if(pose==invalidBody) {
        const uint4 views=arena.Load4(p.integrationBias);
        const uint element=arena.Load(input+12u);
        if(element>=views.w)return false;
        const uint geo=views.z+element*64u;
        const uint kind=scGeometry.Load(geo+28u);
#ifdef FYZZXYZZY_CONVEX
        if(kind>2u)return false;
#else
        if(kind>1u)return false;
#endif
        const float3 shapeLocal=asfloat(scGeometry.Load3(geo+16u));
        const float4 shapeLocalAxes=asfloat(scGeometry.Load4(geo+32u));
        const uint shape=views.x+element*32u;
        const float4 shapeAxes=asfloat(scTransforms.Load4(shape));
        const float3 shapeCenter=asfloat(scTransforms.Load3(shape+16u));
        const float4 actorAxes=sourceRigidMultiply(
            shapeAxes,float4(-shapeLocalAxes.xyz,shapeLocalAxes.w));
        const float3 actorPosition=shapeCenter-
            sourceRigidRotate(actorAxes,shapeLocal);
        const uint local=arena.Load(input+24u);
        const float3 centerLocal=asfloat(scActorLocals.Load3(local));
        const float4 massLocalAxes=asfloat(scActorLocals.Load4(local+12u));
        quaternion=sourceRigidMultiply(actorAxes,massLocalAxes);
        position=actorPosition+sourceRigidRotate(actorAxes,centerLocal);
    }else {
        position=asfloat(arena.Load3(pose));
        quaternion=asfloat(arena.Load4(pose+12u));
    }
    return all(isfinite(position)) && all(isfinite(quaternion));
}

// Source-authored velocities update the previous GPU result only when
// an FP32 component has actually changed. Dy preintegration consumes
// this exact input and commits its prepared velocities in the actor pool.
float3 sourceDyAuthoredVelocity(uint input,bool angular) {
    const uint flags=arena.Load(input+8u);
    const uint changed=(flags>>8u)&63u;
    const uint component=angular?12u:0u;
    if(changed==63u)
        return asfloat(arena.Load3(arena.Load(input+28u)+component));
    const uint local=arena.Load(input+24u);
    float3 velocity=asfloat(scActorLocals.Load3(local+64u+component));
    if(changed==0u)return velocity;
    const uint source=arena.Load(input+28u);
    [unroll]
    for(uint axis=0u;axis<3u;++axis) {
        const uint bit=1u<<((angular?3u:0u)+axis);
        if((changed&bit)!=0u)
            velocity[axis]=asfloat(arena.Load(
                source+countbits(changed&(bit-1u))*4u));
    }
    return velocity;
}
float3 sourceDyPreparedVelocity(uint input,bool angular) {
    if((arena.Load(input+8u)&32u)!=0u) {
        const uint local=arena.Load(input+24u);
        return asfloat(scActorLocals.Load3(local+64u+(angular?12u:0u)));
    }
    return sourceDyAuthoredVelocity(input,angular);
}
// Sparse per-step velocity and acceleration modifiers retain their
// original Sc ordering: velocity delta, then acceleration * dt.
float3 sourceDyMotionMod(uint input,uint stream) {
    const uint flags=arena.Load(input+8u);
    const uint mask=(flags>>16u)&4095u;
    float3 result=0.0f;
    if(mask==0u)return result;
    const uint source=arena.Load(input+32u);
    [unroll]
    for(uint axis=0u;axis<3u;++axis) {
        const uint bit=1u<<(stream*3u+axis);
        if((mask&bit)!=0u)
            result[axis]=asfloat(arena.Load(
                source+countbits(mask&(bit-1u))*4u));
    }
    return result;
}
// DyRigidBodyToSolverBody::copyToSolverBodyData /
// bodyCoreComputeUnconstrainedVelocity. Every awake rigid is prepared once
// before contact/joint PGS; free flight performs it immediately before
// integrateCore. No host solver-state shadow participates.
bool sourceDyPreIntegrateBody(uint ordinal) {
    const uint input=p.integrationBias+44u+ordinal*40u;
    if((arena.Load(input+8u)&32u)==0u)return true;
    const uint local=arena.Load(input+24u);
    if(local==invalidBody)return false;
    const float dt=asfloat(arena.Load(input+4u));
    const uint flags=scActorLocals.Load(local+60u);
    const float3 rawLinear=sourceDyAuthoredVelocity(input,false);
    const float3 rawAngular=sourceDyAuthoredVelocity(input,true);
    const float3 linearDelta=sourceDyMotionMod(input,0u);
    const float3 linearAcceleration=sourceDyMotionMod(input,1u);
    const float3 angularDelta=sourceDyMotionMod(input,2u);
    const float3 angularAcceleration=sourceDyMotionMod(input,3u);
    precise float3 velocityLinear=rawLinear+linearDelta+linearAcceleration*dt;
    velocityLinear.y+=(flags&1u)!=0u?-9.81f*dt:0.0f;
    const float linearDamping=asfloat(scActorLocals.Load(local+32u));
    velocityLinear*=max(1.0f-linearDamping*dt,0.0f);
    precise float3 angular=rawAngular+angularDelta+angularAcceleration*dt;
    if(!all(isfinite(angular)))return false;
    const float angularDamping=asfloat(scActorLocals.Load(local+36u));
    angular*=max(1.0f-angularDamping*dt,0.0f);
    precise float linearSpeed=velocityLinear.x*velocityLinear.x+velocityLinear.y*velocityLinear.y+velocityLinear.z*velocityLinear.z;
    const float maxLinearSq=asfloat(scActorLocals.Load(local+40u));
    if(linearSpeed>maxLinearSq)
        velocityLinear*=sqrt(maxLinearSq/linearSpeed);
    precise float angularSpeed=
        angular.x*angular.x+angular.y*angular.y+angular.z*angular.z;
    const float maxAngularSq=asfloat(scActorLocals.Load(local+44u));
    if(angularSpeed>maxAngularSq)
        angular*=sqrt(maxAngularSq/angularSpeed);
    if((flags&2u)!=0u) {
        float3 pose;
        float4 q;
        if(!sourceDyMassPose(input,pose,q))return false;
        const float3 inverseInertia=asfloat(scActorLocals.Load3(local+48u));
        const float3 inertia=float3(
            inverseInertia.x>0.0f?1.0f/inverseInertia.x:0.0f,
            inverseInertia.y>0.0f?1.0f/inverseInertia.y:0.0f,
            inverseInertia.z>0.0f?1.0f/inverseInertia.z:0.0f);
        const float3 localAngular=sourceRigidRotate(float4(-q.xyz,q.w),angular);
        const float3 original=inertia*localAngular;
        const float3 torque=cross(localAngular,original);
        float3 next=original-torque*dt;
        const float oldMagnitude=sqrt(sourceDyDot3(original,original));
        const float newMagnitude=sqrt(sourceDyDot3(next,next));
        next*=newMagnitude>0.0f?oldMagnitude/newMagnitude:0.0f;
        const float3 delta=sourceRigidRotate(
            q,inverseInertia*next-localAngular);
        angular+=delta;
    }
    const uint locks=arena.Load(input+0u);
    if(locks&8u)angular.x=0.0f;
    if(locks&16u)angular.y=0.0f;
    if(locks&32u)angular.z=0.0f;
    if(!all(isfinite(velocityLinear)) || !all(isfinite(angular)))return false;
    scActorLocals.Store3(local+64u,asuint(velocityLinear));
    scActorLocals.Store3(local+76u,asuint(angular));
    return true;
}
void sourceDyInvalidatePreparedVelocity(uint ordinal) {
    const uint input=p.integrationBias+44u+ordinal*40u;
    const uint local=arena.Load(input+24u);
    if(local==invalidBody)return;
    const uint3 invalid=uint3(0x7fc00000u,0x7fc00000u,0x7fc00000u);
    scActorLocals.Store3(local+64u,invalid);
    scActorLocals.Store3(local+76u,invalid);
}

// DySleep::updateWakeCounter consumes position-phase motion after
// integrateCore has produced the actor quaternion. IslandSim remains
// the owner of actual deactivation and reactivation.
bool sourceDyPrepareSleep(uint input,uint local,
                          float3 motionLinear,float3 motionAngular,
                          float4 actorOrientation,
                          out float3 linearAcc,out float3 angularAcc,
                          out float wakeCounter) {
    wakeCounter=asfloat(scActorLocals.Load(local+88u));
    linearAcc=asfloat(scActorLocals.Load3(local+92u));
    angularAcc=asfloat(scActorLocals.Load3(local+104u));
    // Only a device-admitted dormant body reaches sleepCheck.
    if((arena.Load(input+8u)&0x10000000u)!=0u) {
        wakeCounter=max(wakeCounter,0.4f);
        linearAcc=0.0f;
        angularAcc=0.0f;
    }
    const float dt=asfloat(arena.Load(input+4u));
    if(wakeCounter<0.2f || wakeCounter<dt) {
        const float4 inverseOrientation=
            float4(-actorOrientation.xyz,actorOrientation.w);
        const float3 localAngular=
            sourceRigidRotate(inverseOrientation,motionAngular);
        linearAcc+=motionLinear;
        angularAcc+=localAngular;
        if(!all(isfinite(linearAcc)) || !all(isfinite(angularAcc)))
            return false;
        const float3 inverseInertia=asfloat(scActorLocals.Load3(local+48u));
        const float3 inertia=float3(
            inverseInertia.x>0.0f?1.0f/inverseInertia.x:1.0f,
            inverseInertia.y>0.0f?1.0f/inverseInertia.y:1.0f,
            inverseInertia.z>0.0f?1.0f/inverseInertia.z:1.0f);
        const float inverseMass=asfloat(scActorLocals.Load(local+28u));
        precise float angularEnergy=(
            angularAcc.x*angularAcc.x*inertia.x+
            angularAcc.y*angularAcc.y*inertia.y+
            angularAcc.z*angularAcc.z*inertia.z)*inverseMass;
        precise float accumulatedLinear=
            linearAcc.x*linearAcc.x+
            linearAcc.y*linearAcc.y+
            linearAcc.z*linearAcc.z;
        const float normalizedEnergy=0.5f*(angularEnergy+accumulatedLinear);
        const float cluster=1.0f+float(arena.Load(input+36u));
        const float threshold=cluster*0.005f;
        if(normalizedEnergy>=threshold) {
            linearAcc=0.0f;
            angularAcc=0.0f;
            const float factor=threshold==0.0f
                ?2.0f:min(normalizedEnergy/threshold,2.0f);
            wakeCounter=factor*0.5f*0.4f+dt*(cluster-1.0f);
        }else
            wakeCounter=max(wakeCounter-dt,0.0f);
    }else wakeCounter=max(wakeCounter-dt,0.0f);
    if(wakeCounter==0.0f) {
        linearAcc=0.0f;
        angularAcc=0.0f;
    }
    return all(isfinite(linearAcc)) && all(isfinite(angularAcc)) &&
           isfinite(wakeCounter) && wakeCounter>=0.0f;
}

uint sourceDyPrepAddress(uint index) {
    return p.prepBias+16u+index*48u;
}
#include "physx_source_dy_contact_pgs_native.hlsli"

// Sc assigns each response its exact numerical contact window.
bool sourceDyRelocateContact(uint4 desc,uint scratch,uint responseIndex,
                             uint errorAddress) {
    if((desc.w&255u)!=0u)return true;
    if(desc.x>=p.manifoldCount) {
        sourceDyTaskErrorOr(errorAddress,65536u);
        return false;
    }
    const uint manager=p.manifoldBias+desc.x*24u;
    const uint patchFirst=arena.Load(manager);
    const uint patchCount=arena.Load(manager+4u);
    if(patchCount==0u)return true;
    const uint prepIndex=arena.Load(manager+16u);
    if(prepIndex>=p.prepCount ||
       patchFirst>p.patchCount ||
       patchCount>p.patchCount-patchFirst) {
        sourceDyTaskErrorOr(errorAddress,65536u);
        return false;
    }
    const bool pcm=arena.Load(
        sourceDyPrepAddress(prepIndex)+32u)!=invalidBody;
    const uint source=arena.Load(manager+20u);
    const uint exact=arena.Load(manager+12u);
    if(source>p.contactCount || exact>p.contactCount-source ||
       (pcm && patchCount!=1u)) {
        sourceDyTaskErrorOr(errorAddress,65536u);
        return false;
    }
    const uint firstRow=sourceDyContactPrefix(scratch,responseIndex);
    if(firstRow>p.contactCount || exact>p.contactCount-firstRow) {
        sourceDyTaskErrorOr(errorAddress,65536u);
        return false;
    }
    arena.Store(manager+8u,firstRow);
    if(pcm) {
        arena.Store(p.patchBias+patchFirst*32u+24u,
                    0x80000000u|firstRow);
        if(firstRow!=source) {
            uint previous;
            arena.InterlockedOr(
                p.writebackBias+p.resultPatchOffset+patchFirst*4u,
                sourceDyPcmCompacted,previous);
        }
    }else {
        for(uint patch=patchFirst;patch<patchFirst+patchCount;++patch) {
            const uint po=p.patchBias+patch*32u;
            const uint order=arena.Load(po+24u);
            const uint points=arena.Load(po+4u);
            if((order&0x80000000u)!=0u ||
               order>p.orderCount ||
               points>p.orderCount-order) {
                sourceDyTaskErrorOr(errorAddress,65536u);
                return false;
            }
            for(uint j=0u;j<points;++j) {
                const uint address=p.orderBias+(order+j)*4u;
                const uint old=arena.Load(address);
                if(old<source || old-source>=exact) {
                    sourceDyTaskErrorOr(errorAddress,65536u);
                    return false;
                }
                arena.Store(address,firstRow+(old-source));
            }
        }
    }
    return true;
}

// Sc owns task membership. One workgroup colors independent source
// constraints concurrently while preserving per-body greedy precedence.
// Static reservations follow the source descriptor encounter order.
groupshared uint dyNeeded,dyAssigned,dyWave,dyProgress,dyHighest;
groupshared uint dyNextWave,dySerial;
groupshared uint dyReuseColors,dyRetainedPartitionEnd;
groupshared uint dySingletonPartitions,dyPreparationError;
groupshared uint dyPartitionTileCount,dyPartitionTileBase;
groupshared uint dyPackedTaskCount,dyCompactTaskCount;
groupshared uint dyHubA,dyHubB,dyHubMask;
groupshared uint dyCapacityScan[128];

// The resident source token has one authoritative decoding rule for
// colored and scalar tasks.
bool sourceDyExpandDescriptor(uint token,uint extent,
                              uint sourceCount,uint sourceTable,
                              uint errorAddress,out uint4 desc) {
    const uint kind=token>>30u;
    const uint index=token&0x3fffffffu;
    if(kind==3u) {
        if(index>=p.manifoldCount) {
            sourceDyTaskErrorOr(errorAddress,1048576u);
            return false;
        }
        const uint manager=p.manifoldBias+index*24u;
        if(arena.Load(manager+4u)==0u) {
            desc=uint4(index,arena.Load(manager),
                       arena.Load(manager+8u),arena.Load(manager+16u));
        }else {
            const uint prepIndex=arena.Load(manager+16u);
            if(prepIndex>=p.prepCount) {
                sourceDyTaskErrorOr(errorAddress,1048576u);
                return false;
            }
            const uint prep=sourceDyPrepAddress(prepIndex);
            const uint pcmIndex=arena.Load(prep+32u);
            if(pcmIndex!=invalidBody) {
                const uint source=p.pcmBias+pcmIndex*84u;
                const uint packed=pcmContacts.Load(source+4u);
                const uint emitted=(packed>>10u)&7u;
                const uint sourceBegin=arena.Load(manager+20u);
                const uint numericalBegin=arena.Load(manager+8u);
                const uint admitted=arena.Load(manager+12u);
                const uint patchIndex=arena.Load(manager);
                const bool singlePatch=arena.Load(manager+4u)==1u &&
                    patchIndex<p.patchCount;
                if(!singlePatch) {
                    sourceDyTaskErrorOr(errorAddress,65536u);
                    return false;
                }
                const uint patch=p.patchBias+patchIndex*32u;
                const uint capacity=arena.Load(patch+4u);
                if(pcmContacts.Load(source)!=arena.Load(prep+36u) ||
                   ((packed>>3u)&7u)!=0u ||
                   ((p.launchControl&0x08000000u)!=0u &&
                    (extent&0x80000000u)==0u) ||
                   ((extent&0x80000000u)!=0u && (extent&7u)!=emitted) ||
                   emitted>capacity || emitted>4u ||
                   sourceBegin>p.contactCount ||
                   capacity>p.contactCount-sourceBegin ||
                   (numericalBegin!=invalidBody &&
                    numericalBegin!=sourceBegin) ||
                   (admitted!=0u && admitted!=emitted)) {
                    sourceDyTaskErrorOr(errorAddress,65536u);
                    return false;
                }
                if(numericalBegin==invalidBody)
                    arena.Store(manager+8u,sourceBegin);
                if(admitted==0u) {
                    arena.Store(manager+12u,emitted);
                    arena.Store(patch+4u,emitted);
                    for(uint i=emitted;i<capacity;++i)
                        arena.Store(p.writebackBias+p.resultContactOffset+
                                    (sourceBegin+i)*4u,0u);
                }
                arena.Store(patch+24u,0x80000000u|sourceBegin);
                arena.Store(p.writebackBias+p.resultPatchOffset+
                            patchIndex*4u,
                            sourceDyPcmExtentValid|(emitted<<sourceDyPcmExtentShift));
            } else if((p.launchControl&0x08000000u)!=0u) {
                const uint first=arena.Load(manager);
                const uint count=arena.Load(manager+4u);
                const uint contacts=arena.Load(manager+12u);
                if((extent&0x80000000u)==0u ||
                   (extent&255u)!=contacts ||
                   first>p.patchCount || count>p.patchCount-first) {
                    sourceDyTaskErrorOr(errorAddress,65536u);
                    return false;
                }
                for(uint i=0u;i<count;++i)
                    arena.Store(p.writebackBias+p.resultPatchOffset+
                                (first+i)*4u,0u);
            }
            const uint refA=arena.Load(prep);
            const uint refB=arena.Load(prep+4u);
            const bool dynamicA=(refA&3u)==1u;
            const bool dynamicB=(refB&3u)==1u;
            const uint ordinalA=dynamicA?(refA>>2u):
                (refA&3u)==3u?invalidBody:
                arena.Load(p.prepBias+refA+76u);
            const uint ordinalB=dynamicB?(refB>>2u):
                (refB&3u)==3u?invalidBody:
                arena.Load(p.prepBias+refB+76u);

            desc=uint4(index,ordinalA,ordinalB,
                       (dynamicA?0x100u:0u)|
                       (dynamicB?0x200u:0u));
        }
    }else if(kind==2u) {
        if(index>=p.rowCount) {
            sourceDyTaskErrorOr(errorAddress,1048576u);
            return false;
        }
        const uint row=p.distanceBias+index*64u;
        desc=uint4(index,arena.Load(row),
                   arena.Load(row+4u),arena.Load(row+8u));
    }else if(kind==0u) {
        if(token>=sourceCount) {
            sourceDyTaskErrorOr(errorAddress,1048576u);
            return false;
        }
        desc=arena.Load4(sourceTable+4u+token*16u);
    }else {
        sourceDyTaskErrorOr(errorAddress,1048576u);
        return false;
    }
    return true;
}

bool sourceDyResidentColor() {
    return (p.launchControl&0x0c00000fu)==0x0c000008u;
}
uint sourceDyRetainedTask(uint taskId) {
    const uint taskCapacity=arena.Load(p.integrationBias+24u);
    return p.resultTaskOffset+taskCapacity*4u+taskId*20u;
}
void sourceDyRejectScheduleReuse(uint scratch) {
    uint unused;
    arena.InterlockedExchange(scratch+32u,invalidBody,unused);
}
void sourceDyInitializeColorState(uint4 desc,uint bodyState,uint colors,uint i) {
    const bool da=(desc.w&0x100u)!=0u;
    const bool db=(desc.w&0x200u)!=0u;
    // Constraint endpoints are authoritative even when a solver
    // task omits a body from its integration writeback ordinal list.
    uint unused;
    if(da) {
        const uint state=bodyState+desc.y*12u;
        arena.InterlockedExchange(state,0u,unused);
        arena.InterlockedExchange(state+4u,0u,unused);
    }
    if(db) {
        const uint state=bodyState+desc.z*12u;
        arena.InterlockedExchange(state,0u,unused);
        arena.InterlockedExchange(state+4u,0u,unused);
    }
    arena.Store(colors+i*4u,0xffffffffu);
}
void sourceDyPartitionTask(uint taskId,uint scratch,uint errorAddress,
                           uint lane) {
    const uint4 views=arena.Load4(scratch);
    const bool sourceDescriptors=views.x==24u || views.x==64u;
    const uint descriptorCapacity=arena.Load(p.integrationBias+20u);
    const uint colors=scratch+views.y;
    const uint bodyState=scratch+views.w;
    const uint raw=sourceDescriptors
        ?bodyState-descriptorCapacity*16u:scratch+views.x;
    const uint task=p.taskBias+taskId*36u;
    const uint4 ranges=sourceDyTaskLoad4(task);
    const uint first=ranges.x,last=ranges.y;
    const uint partitionFirst=ranges.z,partitionCapacity=ranges.w;
    if(lane==0u &&
       (last<first || last>descriptorCapacity ||
        partitionCapacity<partitionFirst ||
        partitionCapacity-partitionFirst!=last-first ||
        partitionCapacity>p.partitionCount))
        sourceDyTaskErrorOr(errorAddress,1048576u);
    AllMemoryBarrierWithGroupSync();
    if(sourceDyTaskLoad(errorAddress)!=0u)return;
    if(views.x==64u && arena.Load(scratch+48u)!=0u) {
        const uint2 bodyRange=sourceDyTaskLoad2(task+16u);
        if(lane==0u &&
           (bodyRange.x>bodyRange.y || bodyRange.y>p.bodyCount))
            sourceDyTaskErrorOr(errorAddress,8u);
        for(uint i=bodyRange.x+lane;
            i<bodyRange.y && i<p.bodyCount;i+=128u)
            if(sourceDyBodyOrdinal(i)>=p.bodyCount)
                sourceDyTaskErrorOr(errorAddress,8u);
        AllMemoryBarrierWithGroupSync();
        if(sourceDyTaskLoad(errorAddress)!=0u)return;
    }
    const uint count=last-first;
    const bool residentColor=sourceDyResidentColor();
    if(lane==0u) {
        dyReuseColors=0u;
        dyRetainedPartitionEnd=0u;
        if(residentColor) {
            const uint header=p.taskBias-arena.Load(scratch+24u);
            uint previousCount;
            arena.InterlockedOr(scratch+32u,0u,previousCount);
            if(previousCount==scIslandWork.Load(header+4u) &&
               taskId<previousCount && partitionFirst==first) {
                const uint record=sourceDyRetainedTask(taskId);
                const uint4 old=arena.Load4(record);
                const uint2 bodyRange=sourceDyTaskLoad2(task+16u);
                const uint end=arena.Load(record+16u);
                if(all(old==uint4(first,last,bodyRange)) &&
                   end>=partitionFirst && end<=partitionCapacity &&
                   (count==0u || end>partitionFirst)) {
                    dyReuseColors=1u;
                    dyRetainedPartitionEnd=end;
                }
            }
        }
        dyNeeded=0u;dyAssigned=0u;dyWave=0u;dyHighest=0u;
        dyNextWave=0u;dySerial=0u;
    }
    AllMemoryBarrierWithGroupSync();
    // The native Sc path transfers only encounter-order manager/joint
    // identities. Expand source descriptors from the existing resident
    // contact prep and joint-row records before partition coloring.
    const uint sourceTable=scratch+views.x+
        (views.x==64u?0u:descriptorCapacity*4u);
    const uint sourceCount=sourceDescriptors?arena.Load(sourceTable):0u;
    const bool compareEndpoints=dyReuseColors!=0u;
    if(residentColor)GroupMemoryBarrierWithGroupSync();
    const bool deviceAdmission=(p.launchControl&0x08000000u)!=0u;
    uint needed=0u;
    for(uint i=first+lane;i<last;i+=128u) {
        uint4 desc;
        uint retainedDestination=invalidBody;
        if(sourceDescriptors) {
            const uint token=sourceDyResponseToken(scratch,colors,i);
            if(!sourceDyExpandDescriptor(
                    token,sourceDyResponseExtent(scratch,i),
                    sourceCount,sourceTable,errorAddress,desc))
                continue;
            if(compareEndpoints) {
                const uint4 prior=arena.Load4(raw+i*16u);
                const uint destination=arena.Load(colors+i*4u);
                if(any(prior.yz!=desc.yz) || ((prior.w^desc.w)&0x300u)!=0u ||
                   destination<first || destination>=last) {
                    uint unused;
                    InterlockedAnd(dyReuseColors,0u,unused);
                } else if(any(prior!=desc)) {
                    retainedDestination=destination;
                }
                if(any(prior!=desc))arena.Store4(raw+i*16u,desc);
            } else {
                arena.Store4(raw+i*16u,desc);
            }
        }else desc=arena.Load4(raw+i*16u);
        const bool da=(desc.w&0x100u)!=0u;
        const bool db=(desc.w&0x200u)!=0u;
        if((desc.w&~0x3ffu)!=0u || (!da && !db) ||
           (da && desc.y>=p.bodyCount) ||
           (db && desc.z>=p.bodyCount)) {
            sourceDyTaskErrorOr(errorAddress,1048576u);
            continue;
        }
        const uint rowCount=desc.w&255u;
        if(rowCount!=0u) {
            if(desc.x>=p.rowCount || rowCount>p.rowCount-desc.x) {
                sourceDyTaskErrorOr(errorAddress,1u);
                continue;
            }
            bool invalidCoefficient=false;
            for(uint row=0u;row<rowCount;++row) {
                const uint address=p.distanceBias+(desc.x+row)*64u;
                // Device-authored rows acquire their coefficients during
                // numerical preparation. Their solver validates the result.
                if(arena.Load(address+44u)==2u)continue;
                const uint biased=arena.Load(address+48u);
                const uint unbiased=arena.Load(address+52u);
                if((biased&0x7f800000u)==0x7f800000u ||
                   (unbiased&0x7f800000u)==0x7f800000u) {
                    invalidCoefficient=true;
                    break;
                }
            }
            if(invalidCoefficient) {
                sourceDyTaskErrorOr(errorAddress,2u);
                continue;
            }
        }
        if(deviceAdmission &&
           !sourceDyRelocateContact(desc,scratch,i,errorAddress))
            continue;
        // Unchanged descriptors retain their numerical slots. Changed
        // topology triggers a complete recolor of the task.
        if(retainedDestination!=invalidBody)
            arena.Store4(p.descriptorBias+retainedDestination*16u,
                         uint4(desc.xyz,rowCount));
        if(!compareEndpoints && count>1u) {
            sourceDyInitializeColorState(desc,bodyState,colors,i);
            needed+=uint(da && db);
        }
    }
    if(needed!=0u)InterlockedAdd(dyNeeded,needed);
    AllMemoryBarrierWithGroupSync();
    if(sourceDyTaskLoad(errorAddress)!=0u)return;
    if(dyReuseColors!=0u) {
        if(lane==0u)sourceDyTaskStore(task+12u,dyRetainedPartitionEnd);
        return;
    }
    if(residentColor && lane==0u)sourceDyRejectScheduleReuse(scratch);
    if(count==1u) {
        // A single constraint owns its entire partition and requires
        // no conflict-mask coloring or cross-lane prefix scan.
        if(lane==0u) {
            uint4 desc=arena.Load4(raw+first*16u);
            desc.w&=255u;
            arena.Store4(p.descriptorBias+first*16u,desc);
            arena.Store(p.partitionBias+partitionFirst*4u,last);
            arena.Store(colors+first*4u,residentColor?first:0u);
            sourceDyTaskStore(task+12u,partitionFirst+1u);
        }
        return;
    }

    // Only a rejected retained schedule still needs an initialization pass.
    if(compareEndpoints) {
        uint needed=0u;
        for(uint i=first+lane;i<last;i+=128u) {
            const uint4 desc=arena.Load4(raw+i*16u);
            sourceDyInitializeColorState(desc,bodyState,colors,i);
            needed+=uint((desc.w&0x300u)==0x300u);
        }
        if(needed!=0u)InterlockedAdd(dyNeeded,needed);
        AllMemoryBarrierWithGroupSync();
    }

    // A hub's dynamic/dynamic color is its encounter rank.
    bool mixedHub=false;
    if(dyNeeded>=32u) {
        if(lane==0u) {
            dyHubA=0xffffffffu;
            dyHubB=0xffffffffu;
            dyHubMask=3u;
            for(uint i=first;i<last;++i) {
                const uint4 desc=arena.Load4(raw+i*16u);
                if((desc.w&0x300u)==0x300u) {
                    dyHubA=desc.y;
                    dyHubB=desc.z;
                    break;
                }
            }
        }
        AllMemoryBarrierWithGroupSync();
        // Contiguous lane ranges admit one encounter-order prefix per
        // lane, independent of the native subgroup width.
        const uint chunk=(count+127u)/128u;
        const uint begin=first+min(count,lane*chunk);
        const uint end=begin+min(chunk,last-begin);
        uint common=3u,dynamicCount=0u;
        for(uint i=begin;i<end && common!=0u;++i) {
            const uint4 desc=arena.Load4(raw+i*16u);
            if((desc.w&0x300u)!=0x300u)continue;
            ++dynamicCount;
            if(desc.y!=dyHubA && desc.z!=dyHubA)common&=~1u;
            if(desc.y!=dyHubB && desc.z!=dyHubB)common&=~2u;
        }
        dyCapacityScan[lane]=dynamicCount;
        InterlockedAnd(dyHubMask,common);
        AllMemoryBarrierWithGroupSync();
        if(dyHubMask!=0u) {
            if(dyNeeded==count) {
                // All edges conflict at the common endpoint. Encounter
                // order already gives the final singleton partitions.
                for(uint i=first+lane;i<last;i+=128u) {
                    uint4 desc=arena.Load4(raw+i*16u);
                    desc.w&=255u;
                    arena.Store4(p.descriptorBias+i*16u,desc);
                    arena.Store(p.partitionBias+(partitionFirst+i-first)*4u,i+1u);
                    if(residentColor)arena.Store(colors+i*4u,i);
                }
                if(lane==0u)
                    sourceDyTaskStore(task+12u,partitionCapacity);
                return;
            } else {
                mixedHub=true;
                // Prefix dynamic ranks; the complement is the static
                // encounter rank. Partition slots hold the static list
                // until reservations have been assigned.
                [unroll]
                for(uint step=1u;step<128u;step<<=1u) {
                    const uint prior=lane>=step?dyCapacityScan[lane-step]:0u;
                    GroupMemoryBarrierWithGroupSync();
                    dyCapacityScan[lane]+=prior;
                    GroupMemoryBarrierWithGroupSync();
                }
                uint color=dyCapacityScan[lane]-dynamicCount;
                uint staticRank=begin-first-color;
                const uint hub=(dyHubMask&1u)!=0u?dyHubA:dyHubB;
                for(uint i=begin;i<end;++i) {
                    const uint4 desc=arena.Load4(raw+i*16u);
                    if((desc.w&0x300u)!=0x300u) {
                        arena.Store(p.partitionBias+(partitionFirst+staticRank)*4u,i);
                        ++staticRank;
                        continue;
                    }
                    arena.Store(colors+i*4u,color);
                    const uint spoke=desc.y==hub?desc.z:desc.y;
                    if(spoke!=hub) {
                        uint unused;
                        arena.InterlockedMax(bodyState+spoke*12u+4u,
                                              color+1u,unused);
                    }
                    ++color;
                }
                if(lane==0u) {
                    arena.Store(bodyState+hub*12u+4u,dyNeeded);
                    dyHighest=dyNeeded;
                    dyAssigned=dyNeeded;
                }
            }
        }
    }
    AllMemoryBarrierWithGroupSync();
    while(dyAssigned<dyNeeded) {
        // Rollover retires the old masks during priority initialization.
        const bool resetMask=dyNextWave!=0u;
        if(lane==0u)dyProgress=0u;
        // A source-index priority preserves greedy partition ordering.
        for(uint i=first+lane;i<last;i+=128u) {
            if(arena.Load(colors+i*4u)!=0xffffffffu)continue;
            const uint4 desc=arena.Load4(raw+i*16u);
            if((desc.w&0x300u)!=0x300u)continue;
            uint unused;
            arena.InterlockedExchange(bodyState+desc.y*12u+8u,
                                       0xffffffffu,unused);
            arena.InterlockedExchange(bodyState+desc.z*12u+8u,
                                       0xffffffffu,unused);
            if(resetMask) {
                arena.InterlockedExchange(bodyState+desc.y*12u,0u,unused);
                arena.InterlockedExchange(bodyState+desc.z*12u,0u,unused);
            }
        }
        AllMemoryBarrierWithGroupSync();
        for(uint i=first+lane;i<last;i+=128u) {
            if(arena.Load(colors+i*4u)!=0xffffffffu)continue;
            const uint4 desc=arena.Load4(raw+i*16u);
            if((desc.w&0x300u)!=0x300u)continue;
            const uint a=bodyState+desc.y*12u;
            const uint b=bodyState+desc.z*12u;
            if((arena.Load(a)|arena.Load(b))==0xffffffffu)continue;
            const uint priority=i;
            uint unused;
            arena.InterlockedMin(a+8u,priority,unused);
            arena.InterlockedMin(b+8u,priority,unused);
        }
        AllMemoryBarrierWithGroupSync();
        uint progress=0u,highest=0u;
        for(uint i=first+lane;i<last;i+=128u) {
            if(arena.Load(colors+i*4u)!=0xffffffffu)continue;
            const uint4 desc=arena.Load4(raw+i*16u);
            if((desc.w&0x300u)!=0x300u)continue;
            const uint a=bodyState+desc.y*12u;
            const uint b=bodyState+desc.z*12u;
            const uint priority=i;
            if(arena.Load(a+8u)!=priority ||
               arena.Load(b+8u)!=priority)continue;
            const uint maskA=arena.Load(a),maskB=arena.Load(b);
            const uint available=~(maskA|maskB);
            if(available==0u)continue;
            const uint bit=firstbitlow(available);
            const uint value=dyWave+bit;
            if(value>=count) {
                sourceDyTaskErrorOr(errorAddress,1048576u);
                continue;
            }
            const uint mask=1u<<bit;
            // Both priorities name this edge; no other winner owns
            // either endpoint during this round.
            arena.Store(a,maskA|mask);
            arena.Store(b,maskB|mask);
            arena.Store(a+4u,max(arena.Load(a+4u),value+1u));
            arena.Store(b+4u,max(arena.Load(b+4u),value+1u));
            arena.Store(colors+i*4u,value);
            highest=max(highest,value+1u);
            ++progress;
        }
        if(progress!=0u) {
            InterlockedMax(dyHighest,highest);
            InterlockedAdd(dyProgress,progress);
        }
        AllMemoryBarrierWithGroupSync();
        if(lane==0u) {
            if(dyProgress==0u) {
                dyWave+=32u;
                dyNextWave=1u;
            } else {
                dyNextWave=0u;
                dyAssigned+=dyProgress;
                // A source-ordered chain exposes little independent
                // work. Continue its exact greedy prefix in one lane
                // instead of imposing a barrier for every next edge.
                if(dyNeeded>=256u && dyProgress*16u<dyNeeded)
                    dySerial=1u;
            }
            if(dyWave>=count && dyAssigned<dyNeeded)
                sourceDyTaskErrorOr(errorAddress,1048576u);
        }
        AllMemoryBarrierWithGroupSync();
        if(sourceDyTaskLoad(errorAddress)!=0u)return;
        if(dySerial!=0u)break;
    }
    if(dySerial!=0u) {
        if(lane==0u) {
            while(dyAssigned<dyNeeded) {
                for(uint i=first;i<last;++i) {
                    if(arena.Load(colors+i*4u)!=0xffffffffu)continue;
                    const uint4 desc=arena.Load4(raw+i*16u);
                    if((desc.w&0x300u)!=0x300u)continue;
                    const uint a=bodyState+desc.y*12u;
                    const uint b=bodyState+desc.z*12u;
                    const uint available=~(arena.Load(a)|arena.Load(b));
                    if(available==0u)continue;
                    const uint bit=firstbitlow(available);
                    const uint value=dyWave+bit;
                    if(value>=count) {
                        sourceDyTaskErrorOr(errorAddress,1048576u);
                        break;
                    }
                    const uint mask=1u<<bit;
                    arena.Store(a,arena.Load(a)|mask);
                    arena.Store(b,arena.Load(b)|mask);
                    arena.Store(a+4u,max(arena.Load(a+4u),value+1u));
                    arena.Store(b+4u,max(arena.Load(b+4u),value+1u));
                    arena.Store(colors+i*4u,value);
                    dyHighest=max(dyHighest,value+1u);
                    ++dyAssigned;
                }
                if(sourceDyTaskLoad(errorAddress)!=0u)break;
                if(dyAssigned==dyNeeded)break;
                dyWave+=32u;
                if(dyWave>=count) {
                    sourceDyTaskErrorOr(errorAddress,1048576u);
                    break;
                }
                for(uint i=first;i<last;++i) {
                    if(arena.Load(colors+i*4u)!=0xffffffffu)continue;
                    const uint4 desc=arena.Load4(raw+i*16u);
                    if((desc.w&0x300u)!=0x300u)continue;
                    arena.Store(bodyState+desc.y*12u,0u);
                    arena.Store(bodyState+desc.z*12u,0u);
                }
            }
        }
        AllMemoryBarrierWithGroupSync();
        if(sourceDyTaskLoad(errorAddress)!=0u)return;
    }
    if(dyNeeded!=count) {
        const uint staticCount=count-dyNeeded;
        const uint scanCount=mixedHub?staticCount:count;
        const bool parallelReservations=staticCount>=128u;
        if(lane==0u)dyProgress=0u;
        // The dynamic masks are dead; count static reservations per body
        // without disturbing its highest dynamic color.
        for(uint j=lane;j<scanCount;j+=128u) {
            const uint i=mixedHub?arena.Load(p.partitionBias+(partitionFirst+j)*4u):first+j;
            if(arena.Load(colors+i*4u)!=invalidBody)continue;
            const uint4 desc=arena.Load4(raw+i*16u);
            const uint body=(desc.w&0x100u)!=0u?desc.y:desc.z;
            const uint state=bodyState+body*12u;
            uint unused;
            arena.InterlockedExchange(state+8u,0u,unused);
            if(parallelReservations && dyNeeded!=0u)
                arena.InterlockedExchange(state,0u,unused);
        }
        AllMemoryBarrierWithGroupSync();
        if(parallelReservations) {
            uint repeated=0u;
            for(uint j=lane;j<scanCount;j+=128u) {
                const uint i=mixedHub?arena.Load(p.partitionBias+(partitionFirst+j)*4u):first+j;
                if(arena.Load(colors+i*4u)!=invalidBody)continue;
                const uint4 desc=arena.Load4(raw+i*16u);
                const uint body=(desc.w&0x100u)!=0u?desc.y:desc.z;
                uint prior;
                arena.InterlockedAdd(bodyState+body*12u,1u,prior);
                repeated+=uint(prior==1u);
            }
            if(repeated!=0u)InterlockedAdd(dyProgress,repeated);
            AllMemoryBarrierWithGroupSync();
            uint highest=0u;
            for(uint j=lane;j<scanCount;j+=128u) {
                const uint i=mixedHub?arena.Load(p.partitionBias+(partitionFirst+j)*4u):first+j;
                if(arena.Load(colors+i*4u)!=invalidBody)continue;
                const uint4 desc=arena.Load4(raw+i*16u);
                const uint body=(desc.w&0x100u)!=0u?desc.y:desc.z;
                const uint state=bodyState+body*12u;
                if(arena.Load(state)!=1u)continue;
                const uint value=arena.Load(state+4u);
                if(value>=count) {
                    sourceDyTaskErrorOr(errorAddress,1048576u);
                    continue;
                }
                arena.Store(colors+i*4u,value);
                highest=max(highest,value+1u);
            }
            if(highest!=0u)InterlockedMax(dyHighest,highest);
            AllMemoryBarrierWithGroupSync();
            if(sourceDyTaskLoad(errorAddress)!=0u)return;
        }
        if(!parallelReservations || dyProgress!=0u) {
            if(lane==0u) {
                uint highest=dyHighest;
                // Repeated endpoints reserve colors in encounter order.
                for(uint j=0u;j<scanCount;++j) {
                    const uint i=mixedHub?arena.Load(p.partitionBias+(partitionFirst+j)*4u):first+j;
                    if(arena.Load(colors+i*4u)!=invalidBody)continue;
                    const uint4 desc=arena.Load4(raw+i*16u);
                    const uint state=bodyState+
                        (((desc.w&0x100u)!=0u)?desc.y:desc.z)*12u;
                    const uint rank=arena.Load(state+8u);
                    const uint value=arena.Load(state+4u)+rank;
                    if(value>=count) {
                        sourceDyTaskErrorOr(errorAddress,1048576u);
                        break;
                    }
                    arena.Store(state+8u,rank+1u);
                    arena.Store(colors+i*4u,value);
                    highest=max(highest,value+1u);
                }
                dyHighest=highest;
            }
            AllMemoryBarrierWithGroupSync();
            if(sourceDyTaskLoad(errorAddress)!=0u)return;
        }
    }
    if(dyHighest==1u) {
        // A single color is already conflict-free in encounter order.
        for(uint i=first+lane;i<last;i+=128u) {
            uint4 desc=arena.Load4(raw+i*16u);
            desc.w&=255u;
            arena.Store4(p.descriptorBias+i*16u,desc);
            if(residentColor)arena.Store(colors+i*4u,i);
        }
        if(lane==0u) {
            arena.Store(p.partitionBias+partitionFirst*4u,last);
            sourceDyTaskStore(task+12u,partitionFirst+1u);
        }
        return;
    }
    // Only the admitted colors have histogram entries.
    if(dyHighest>partitionCapacity-partitionFirst) {
        if(lane==0u)sourceDyTaskErrorOr(errorAddress,1048576u);
        return;
    }
    for(uint i=lane;i<dyHighest;i+=128u)
        arena.Store(p.partitionBias+(partitionFirst+i)*4u,0u);
    AllMemoryBarrierWithGroupSync();
    for(uint i=first+lane;i<last;i+=128u) {
        const uint color=arena.Load(colors+i*4u);
        if(color>=dyHighest) {
            sourceDyTaskErrorOr(errorAddress,1048576u);
            continue;
        }
        const uint target=p.partitionBias+(partitionFirst+color)*4u;
        uint unused;
        arena.InterlockedAdd(target,1u,unused);
    }
    AllMemoryBarrierWithGroupSync();
    if(sourceDyTaskLoad(errorAddress)!=0u)return;
    if(lane==0u) {
        uint end=first;
        // Convert counts into scatter starts in the same partition slots.
        for(uint i=partitionFirst;i<partitionFirst+dyHighest;++i) {
            const uint address=p.partitionBias+i*4u;
            const uint n=arena.Load(address);
            arena.Store(address,end);
            end+=n;
        }
        if(end!=last)
            sourceDyTaskErrorOr(errorAddress,1048576u);
        else
            sourceDyTaskStore(task+12u,partitionFirst+dyHighest);
    }
    AllMemoryBarrierWithGroupSync();
    if(sourceDyTaskLoad(errorAddress)!=0u)return;
    // Each scatter counter finishes at its partition's exact end.
    // Descriptors within a color own distinct responding bodies.
    for(uint i=first+lane;i<last;i+=128u) {
        uint4 desc=arena.Load4(raw+i*16u);
        const uint target=p.partitionBias+
            (partitionFirst+arena.Load(colors+i*4u))*4u;
        uint offset;
        arena.InterlockedAdd(target,1u,offset);
        desc.w&=255u;
        arena.Store4(p.descriptorBias+offset*16u,desc);
        if(residentColor)arena.Store(colors+i*4u,offset);
    }
}

// Sc admits immutable joint local frames and non-dynamic authored poses.
// Active rigid endpoints read the same BodyId-resident actor state as PCM
// contact preparation and integrateCore. Dy constructs every 1D Jacobian
// before source-order PGS, with no host world-frame joint row dependency.
bool sourceDyJointActorFrame(uint ordinal,uint fallback,uint centerFallback,
                             out float3 actorPosition,out float4 actorAxes,
                             out float3 massCenter) {
    actorPosition=asfloat(jointSources.Load3(fallback));
    actorAxes=asfloat(jointSources.Load4(fallback+12u));
    massCenter=asfloat(jointSources.Load3(centerFallback));
    if(ordinal!=invalidBody) {
        const uint input=p.integrationBias+44u+ordinal*40u;
        if((arena.Load(input+8u)&128u)!=0u) {
            const uint local=arena.Load(input+24u);
            if(local==invalidBody)return false;
            actorPosition=asfloat(scActorLocals.Load3(local+116u));
            actorAxes=asfloat(scActorLocals.Load4(local+128u));
            const float3 localCenter=asfloat(scActorLocals.Load3(local));
            massCenter=actorPosition+sourceRigidRotate(actorAxes,localCenter);
        }
    }
    return all(isfinite(actorPosition)) && all(isfinite(actorAxes)) &&
           all(isfinite(massCenter));
}
float3 sourceDyJointInertia(uint ordinal,float3 angular) {
    if(ordinal==invalidBody)return 0.0f;
    const uint inertia=sourceDyInertiaAddress(ordinal);
    return asfloat(arena.Load3(inertia))*angular.x+
           asfloat(arena.Load3(inertia+12u))*angular.y+
           asfloat(arena.Load3(inertia+24u))*angular.z;
}
bool sourceDyPrepareJointGeometry(uint4 desc,uint endError) {
    const uint first=p.distanceBias+desc.x*64u;
    const uint source=arena.Load(first+60u);
    if(jointSources.Load(source+144u)!=arena.Load(first+48u) ||
       jointSources.Load(source+148u)!=arena.Load(first+52u)) {
        sourceDyTaskErrorOr(endError,8192u);
        return false;
    }
    const uint kind=jointSources.Load(source+140u);
    const uint count=kind==0u?1u:kind==2u?3u:kind==1u?6u:5u;
    if(kind>4u || desc.w!=count) {
        sourceDyTaskErrorOr(endError,8192u);
        return false;
    }
    const float3 localA=asfloat(jointSources.Load3(source));
    const float4 localQA=asfloat(jointSources.Load4(source+12u));
    const float3 localB=asfloat(jointSources.Load3(source+28u));
    const float4 localQB=asfloat(jointSources.Load4(source+40u));
    // Fallback actor frames and mass centers are borrowed authored state.
    // Dynamic endpoints consume the resident actor frame instead.
    float3 pA,pB,cA,cB;
    float4 qA,qB;
    if(!sourceDyJointActorFrame(desc.y,source+56u,source+112u,pA,qA,cA) ||
       !sourceDyJointActorFrame(desc.z,source+84u,source+124u,pB,qB,cB)) {
        sourceDyTaskErrorOr(endError,8192u);
        return false;
    }
     const float3 wA=pA+sourceRigidRotate(qA,localA);
    const float3 wB=pB+sourceRigidRotate(qB,localB);
    const float3 rA=wA-cA,rB=wB-cB;
    const float4 qa=sourceRigidMultiply(qA,localQA);
    const float4 qbRaw=sourceRigidMultiply(qB,localQB);
    const float4 qb=dot(qa,qbRaw)<0.0f?-qbRaw:qbRaw;
    const float3 localError=sourceRigidRotate(float4(-qa.xyz,qa.w),wA-wB);
    const float4 errorQ=sourceRigidMultiply(float4(-qa.xyz,qa.w),qb);
    const float3 va=qa.xyz,vb=qb.xyz;
    const float d=qa.w*qb.w-sourceDyDot3(va,vb);
    const float overlap=qa.w*qb.w+sourceDyDot3(va,vb);
    const float3 c=qa.w*vb+qb.w*va;
    const float3 diagonal[3]={
        float3(d,c.z,-c.y),float3(-c.z,d,c.x),float3(c.y,-c.x,d)};
    const float3 basis[3]={
        float3(1.0f,0.0f,0.0f),float3(0.0f,1.0f,0.0f),float3(0.0f,0.0f,1.0f)};
    const float mass0=asfloat(arena.Load(first+36u));
    const float mass1=asfloat(arena.Load(first+40u));
    const uint inputA=p.integrationBias+44u+desc.y*40u;
    const uint inputB=p.integrationBias+44u+desc.z*40u;
    const float3 vA=desc.y==invalidBody?0.0f:sourceDyPreparedVelocity(inputA,false);
    const float3 vB=desc.z==invalidBody?0.0f:sourceDyPreparedVelocity(inputB,false);
    const float3 omegaA=desc.y==invalidBody?0.0f:sourceDyPreparedVelocity(inputA,true);
    const float3 omegaB=desc.z==invalidBody?0.0f:sourceDyPreparedVelocity(inputB,true);
    const uint dtInput=mass0>0.0f?inputA:inputB;
    const float dt=asfloat(arena.Load(dtInput+4u));
    if(!(dt>0.0f) || !isfinite(dt)) {
        sourceDyTaskErrorOr(endError,8192u);
        return false;
    }
    const float invDt=1.0f/dt;
    float3 linearRows[6],angularA[6],angularB[6],jacA[6],jacB[6];
    float invResponse[6],projectedError[6];
    const float requiredDistance=asfloat(jointSources.Load(source+136u));
    for(uint i=0u;i<count;++i) {
        float3 axis=0.0f,angA=0.0f,angB=0.0f;
        float error=0.0f;
        if(kind==0u) {
            precise float3 offset=((cA+rA)-cB)-rB;
            const float distance=sqrt(sourceDyDot3(offset,offset));
            if(!isfinite(distance)) {
                sourceDyTaskErrorOr(endError,8192u);
                return false;
            }
            axis=distance<1.192092896e-07f
                ?float3(1.0f,0.0f,0.0f):offset/distance;
            angA=cross(rA,axis);
            angB=cross(rB,axis);
            error=distance-requiredDistance;
        }else {
            const uint component=i+(kind==4u?1u:0u)+
                (kind==3u && i>=3u?1u:0u);
            error=component<3u?localError[component]:
                -errorQ[component-3u];
            if(component<3u) {
                axis=sourceRigidRotate(qa,basis[component]);
                const float3 arm=kind==4u?wB-cA:rA;
                angA=cross(arm,axis);
                angB=cross(rB,axis);
            }else {
                const uint k=component-3u;
                const float3 v=diagonal[k];
                angA=0.5f*(va*vb[k]+vb*va[k]+v);
                if(overlap==0.0f)
                    angA+=basis[k]*1.1920928955078125e-7f;
                angB=angA;
            }
        }
        float3 ja=mass0>0.0f?sourceDyJointInertia(desc.y,angA):0.0f;
        float3 jb=mass1>0.0f?sourceDyJointInertia(desc.z,angB):0.0f;
        for(uint j=0u;j<i;++j) {
            const float factor=((mass0+mass1)*sourceDyDot3(axis,linearRows[j])+
                sourceDyDot3(ja,jacA[j])+sourceDyDot3(jb,jacB[j]))*
                invResponse[j];
            axis-=linearRows[j]*factor;
            angA-=angularA[j]*factor;
            angB-=angularB[j]*factor;
            ja-=jacA[j]*factor;
            jb-=jacB[j]*factor;
            // Geometry and its error must share the same source projection.
            error-=projectedError[j]*factor;
        }
        const float response=(mass0+mass1)*sourceDyDot3(axis,axis)+
            sourceDyDot3(ja,ja)+sourceDyDot3(jb,jb);
        if(!(response>0.0f) || !isfinite(response)) {
            sourceDyTaskErrorOr(endError,8192u);
            return false;
        }
        invResponse[i]=1.0f/response;
        projectedError[i]=error;
        linearRows[i]=axis;angularA[i]=angA;angularB[i]=angB;
        jacA[i]=ja;jacB[i]=jb;
        const float velocity=sourceDyDot3(axis,vA-vB)+
            sourceDyDot3(angA,omegaA)-sourceDyDot3(angB,omegaB);
        const float multiplier=-invResponse[i];
        const float constant=multiplier*(error*invDt+velocity);
        const float unbiased=multiplier*velocity;
        if(!all(isfinite(axis)) || !all(isfinite(ja)) ||
           !all(isfinite(jb)) || !isfinite(constant) ||
           !isfinite(unbiased)) {
            sourceDyTaskErrorOr(endError,8192u);
            return false;
        }
        const uint row=first+i*64u;
        arena.Store3(row,asuint(axis));
        arena.Store3(row+12u,asuint(ja));
        arena.Store3(row+24u,asuint(jb));
        arena.Store(row+48u,asuint(constant));
        arena.Store(row+52u,asuint(unbiased));
        arena.Store(row+56u,asuint(multiplier));
        arena.Store(row+60u,asuint(0.0f));
        // Reserved 2 distinguishes device geometry admission. All rows
        // remain unavailable to the ordinary host-geometry branch.
        arena.Store(row+44u,0u);
    }
    return true;
}

// Host-geometry rows are retained for standalone conformance. Native Sc
// joint rows instead use the source-authored frame payload above.
void prepareJointRows(uint4 desc,uint lane,uint lanes,uint endError) {
    if(desc.x>=p.rowCount || desc.w>p.rowCount-desc.x ||
       (desc.y!=invalidBody && desc.y>=p.bodyCount) ||
       (desc.z!=invalidBody && desc.z>=p.bodyCount)) {
        sourceDyTaskErrorOr(endError,4096u);
        return;
    }
    const uint massRow=p.distanceBias+desc.x*64u;
    if(arena.Load(massRow+44u)==2u) {
        if(lane==0u)sourceDyPrepareJointGeometry(desc,endError);
        return;
    }
    const float inverseMassA=asfloat(arena.Load(massRow+36u));
    const float inverseMassB=asfloat(arena.Load(massRow+40u));
    const uint inputA=p.integrationBias+44u+desc.y*40u;
    const uint inputB=p.integrationBias+44u+desc.z*40u;
    float3 va=0.0f,vb=0.0f,wa=0.0f,wb=0.0f;
    if(desc.y!=invalidBody) {
        va=sourceDyPreparedVelocity(inputA,false);
        wa=sourceDyPreparedVelocity(inputA,true);
    }
    if(desc.z!=invalidBody) {
        vb=sourceDyPreparedVelocity(inputB,false);
        wb=sourceDyPreparedVelocity(inputB,true);
    }
    for(uint i=lane;i<desc.w;i+=lanes) {
        const uint row=p.distanceBias+(desc.x+i)*64u;
        if(arena.Load(row+44u)!=1u)continue;
        const float3 axis=asfloat(arena.Load3(row));
        const float3 angularA=asfloat(arena.Load3(row+12u));
        const float3 angularB=asfloat(arena.Load3(row+24u));
        float3 ja=0.0f,jb=0.0f;
        if(inverseMassA>0.0f) {
            const uint inertia=sourceDyInertiaAddress(desc.y);
            ja=asfloat(arena.Load3(inertia))*angularA.x+
               asfloat(arena.Load3(inertia+12u))*angularA.y+
               asfloat(arena.Load3(inertia+24u))*angularA.z;
        }
        if(inverseMassB>0.0f) {
            const uint inertia=sourceDyInertiaAddress(desc.z);
            jb=asfloat(arena.Load3(inertia))*angularB.x+
               asfloat(arena.Load3(inertia+12u))*angularB.y+
               asfloat(arena.Load3(inertia+24u))*angularB.z;
        }
        precise float speed=sourceDyDot3(axis,va-vb)+
            sourceDyDot3(angularA,wa)-sourceDyDot3(angularB,wb);
        const float multiplier=asfloat(arena.Load(row+56u));
        const float errorRate=asfloat(arena.Load(row+48u));
        const float constant=multiplier*(errorRate+speed);
        const float unbiased=multiplier*speed;
        if(!isfinite(constant) || !isfinite(unbiased) ||
           !all(isfinite(ja)) || !all(isfinite(jb))) {
            sourceDyTaskErrorOr(endError,8192u);
            continue;
        }
        arena.Store3(row+12u,asuint(ja));
        arena.Store3(row+24u,asuint(jb));
        arena.Store(row+48u,asuint(constant));
        arena.Store(row+52u,asuint(unbiased));
        arena.Store(row+44u,0u);
    }
}

// PCM determines active numerical rows even for Sc-packed Dy tasks.
// The existing device-admitted descriptor path applies the same extent.
void admitPcmContactExtents(uint begin,uint end,uint lane,uint endError) {
    for(uint i=begin+lane;i<end;i+=128u) {
        const uint4 desc=arena.Load4(p.descriptorBias+i*16u);
        if(desc.w!=0u || desc.x>=p.manifoldCount)continue;
        const uint manager=p.manifoldBias+desc.x*24u;
        if(arena.Load(manager+4u)==0u)continue;
        const uint prepIndex=arena.Load(manager+16u);
        if(prepIndex>=p.prepCount) {
            sourceDyTaskErrorOr(endError,65536u);
            continue;
        }
        if(arena.Load(sourceDyPrepAddress(prepIndex)+32u)==invalidBody)
            continue;
        uint4 resolved;
        sourceDyExpandDescriptor(0xc0000000u|desc.x,0u,0u,0u,
                                 endError,resolved);
    }
}

void prepareConstraint(uint4 desc,uint lane,uint lanes,uint endError) {
    if(desc.w!=0u) {
        prepareJointRows(desc,lane,lanes,endError);
        return;
    }
    if(desc.x>=p.manifoldCount) {
        sourceDyTaskErrorOr(endError,4096u);
        return;
    }
    uint mo=p.manifoldBias+desc.x*24u;
    uint4 manager=arena.Load4(mo);
    if(manager.y==0u)return;
    uint pairIndex=arena.Load(mo+16u);
    if(pairIndex>=p.prepCount ||
       manager.x>p.patchCount ||
       manager.y>p.patchCount-manager.x ||
       manager.z>p.contactCount ||
       manager.w>p.contactCount-manager.z) {
        sourceDyTaskErrorOr(endError,4096u);
        return;
    }
    const uint pair=sourceDyPrepAddress(pairIndex);
    SourceDyContactFrame frame=(SourceDyContactFrame)0;
    bool ready=false;
    uint covered=0u;
    for(uint patchIndex=manager.x;patchIndex<manager.x+manager.y;
        ++patchIndex) {
        const uint po=p.patchBias+patchIndex*32u;
        const uint count=arena.Load(po+4u);
        const uint order=arena.Load(po+24u);
        if(covered>manager.w || count>manager.w-covered ||
           !sourceDyContactOrderValid(order,count)) {
            sourceDyTaskErrorOr(endError,4096u);
            break;
        }
        if(lane<count) {
            if(!sourceDyEnsureContactFrame(pair,endError,frame,ready))return;
            for(uint c=lane;c<count;c+=lanes) {
                const uint index=sourceDyContactOrderPoint(order,c);
                prepareContactRow(index,pair,manager.z,manager.w,frame,endError);
            }
        }
        // Each patch has one friction writer. Its source geometry
        // is independent of the normal rows written by other lanes.
        if((patchIndex-manager.x)%lanes==lane) {
            const uint4 patch=arena.Load4(po);
            if(patch.x>p.frictionCount || patch.z>2u ||
               patch.z*2u>p.frictionCount-patch.x)
                sourceDyTaskErrorOr(endError,16384u);
            else
                prepareFrictionPatch(patchIndex,pairIndex,
                                     manager.z,manager.w,endError,frame,ready);
        }
        covered+=count;
    }
    if(covered!=manager.w)
        sourceDyTaskErrorOr(endError,4096u);
}

void prepareConstraints(uint begin,uint end,uint lane,uint endError) {
    for(uint i=begin+(lane>>2u);i<end;i+=32u)
        prepareConstraint(arena.Load4(p.descriptorBias+i*16u),lane&3u,4u,endError);
}

bool sourceDySolveDescriptor(uint4 desc,uint errorAddress,bool biased,
                             bool friction,bool finalPass,uint2 retained,
                             SourceDySolverPair prior,
                             out SourceDySolverPair state) {
    state=(SourceDySolverPair)0;
    if(desc.w!=0u)
        return sourceDyLoadSolverPair(desc,errorAddress,true,retained,prior,state) &&
               solveJointRows(desc,errorAddress,biased,state);
    SourceDyContactSpan span;
    if(!sourceDyLoadContactSpan(desc,errorAddress,span))return false;
    if(span.manager.y==0u)return true;
    return sourceDyLoadSolverPair(desc,errorAddress,true,retained,prior,state) &&
           solveContactRows(desc,span,errorAddress,biased,friction,finalPass,state);
}

void solvePartition(uint begin,uint end,uint lane,uint2 iters,
                    uint iter,uint endError,bool captureMotion) {
    const uint total=iters.x+max(iters.y,1u);
    const bool biased=iter<iters.x;
    const bool finalPass=iter+1u==total;
    const bool friction=!biased || iters.x-iter<=3u;
    for(uint i=begin+lane;i<end;i+=128u) {
        const uint4 desc=arena.Load4(p.descriptorBias+i*16u);
        SourceDySolverPair state;
        if(sourceDySolveDescriptor(desc,endError,biased,friction,finalPass,
                uint2(invalidBody,invalidBody),(SourceDySolverPair)0,state))
            sourceDyStoreSolverPair(desc,state,captureMotion);
    }
}

// The serial owner retains responding endpoints until eviction or the
// numerical phase boundary, including across consecutive iterations.
void solveSerialRun(uint begin,uint end,uint2 iters,uint2 iterations,
                    uint endError,bool captureMotion) {
    const uint total=iters.x+max(iters.y,1u);
    uint2 retained=uint2(invalidBody,invalidBody);
    SourceDySolverPair prior=(SourceDySolverPair)0;
    for(uint iter=iterations.x;iter<iterations.y;++iter) {
        const bool biased=iter<iters.x;
        const bool finalPass=iter+1u==total;
        const bool friction=!biased || iters.x-iter<=3u;
        for(uint i=begin;i<end;++i) {
            const uint4 desc=arena.Load4(p.descriptorBias+i*16u);
            SourceDySolverPair state;
            if(!sourceDySolveDescriptor(desc,endError,biased,friction,finalPass,
                                        retained,prior,state)) {
                sourceDyStoreSolverPair(uint4(0u,retained,0u),prior,captureMotion);
                return;
            }
            const uint2 next=uint2(state.ma>0.0f?desc.y:invalidBody,
                                   state.mb>0.0f?desc.z:invalidBody);
            uint2 evicted=retained;
            if(any(next==retained.x))evicted.x=invalidBody;
            if(any(next==retained.y))evicted.y=invalidBody;
            sourceDyStoreSolverPair(uint4(0u,evicted,0u),prior,captureMotion);
            retained=next;
            prior=state;
        }
    }
    sourceDyStoreSolverPair(uint4(0u,retained,0u),prior,captureMotion);
}

void saveMotion(uint index,uint endError) {
    const uint ordinal=sourceDyBodyOrdinal(index);
    if(ordinal>=p.bodyCount) {
        sourceDyTaskErrorOr(endError,8u);
        return;
    }
    const uint base=p.bodyBias+ordinal*52u;
    arena.Store3(base+24u,arena.Load3(base));
    arena.Store3(base+36u,arena.Load3(base+12u));
}

void integrateBody(uint ordinal,bool freeSource) {
    float3 linearDelta=0.0f;
    float3 angularState=0.0f;
    const uint input=p.integrationBias+44u+ordinal*40u;
    const uint inputFlags=arena.Load(input+8u);
    [branch]
    if(ordinal<p.bodyCount && !freeSource) {
        const uint from=p.bodyBias+ordinal*52u;
        const uint to=p.writebackBias+p.resultBodyOffset+
            ordinal*((inputFlags&64u)!=0u?52u:48u);
        linearDelta=asfloat(arena.Load3(from+24u));
        angularState=asfloat(arena.Load3(from+36u));
        if((inputFlags&8u)==0u) {
            // Standalone Dy returns the caller's original four deltas.
            arena.Store4(to,arena.Load4(from));
            arena.Store4(to+16u,arena.Load4(from+16u));
            arena.Store4(to+32u,arena.Load4(from+32u));
        }
    }
    // Position integration consumes the saved position-iteration state.
    if((inputFlags&1u)==0u)return;
    const uint destination=p.writebackBias+p.resultPatchOffset+
        p.patchCount*4u+ordinal*32u;
    arena.Store(destination+28u,0u);
    if(freeSource && (inputFlags&32u)!=0u &&
       !sourceDyPreIntegrateBody(ordinal)) {
        sourceDyInvalidatePreparedVelocity(ordinal);
        return;
    }
    const float3 baseLinear=sourceDyPreparedVelocity(input,false);
    const float3 baseAngular=sourceDyPreparedVelocity(input,true);
    if(!all(isfinite(baseLinear)) || !all(isfinite(baseAngular)))return;
    const uint locks=arena.Load(input+0u);
    const float dt=asfloat(arena.Load(input+4u));
    float3 position;
    float4 quaternion;
    if(!sourceDyMassPose(input,position,quaternion))return;
    const uint local=arena.Load(input+24u);
    const float4 massLocalAxes=asfloat(scActorLocals.Load4(local+12u));
    // World angular motion integrates the actor frame directly; the
    // mass-frame origin still supplies the center-of-mass translation.
    quaternion=(inputFlags&128u)!=0u
        ?asfloat(scActorLocals.Load4(local+128u))
        :sourceRigidMultiply(quaternion,
            float4(-massLocalAxes.xyz,massLocalAxes.w));
    float3 motionLinear=baseLinear+linearDelta;
    if(locks&1u)motionLinear.x=0.0f;
    if(locks&2u)motionLinear.y=0.0f;
    if(locks&4u)motionLinear.z=0.0f;
    if(locks&8u)angularState.x=0.0f;
    if(locks&16u)angularState.y=0.0f;
    if(locks&32u)angularState.z=0.0f;
    float3 inertiaX=0.0f,inertiaY=0.0f,inertiaZ=0.0f;
    if(ordinal<p.bodyCount && !freeSource) {
        const uint inertia=sourceDyInertiaAddress(ordinal);
        inertiaX=asfloat(arena.Load3(inertia));
        inertiaY=asfloat(arena.Load3(inertia+12u));
        inertiaZ=asfloat(arena.Load3(inertia+24u));
    }
    const float3 angular=baseAngular+float3(
        sourceDyDot3(inertiaX,angularState),
        sourceDyDot3(inertiaY,angularState),
        sourceDyDot3(inertiaZ,angularState));
    float3 finalLinear=motionLinear;
    float3 finalAngular=baseAngular;
    if(ordinal<p.bodyCount && (inputFlags&8u)!=0u && !freeSource) {
        const uint from=p.bodyBias+ordinal*52u;
        const uint to=p.writebackBias+p.resultBodyOffset+
            ordinal*((inputFlags&64u)!=0u?52u:48u);
        finalLinear=baseLinear+asfloat(arena.Load3(from));
        if(locks&1u)finalLinear.x=0.0f;
        if(locks&2u)finalLinear.y=0.0f;
        if(locks&4u)finalLinear.z=0.0f;
        float3 finalState=asfloat(arena.Load3(from+12u));
        if(locks&8u)finalState.x=0.0f;
        if(locks&16u)finalState.y=0.0f;
        if(locks&32u)finalState.z=0.0f;
        finalAngular=baseAngular+float3(
            sourceDyDot3(inertiaX,finalState),
            sourceDyDot3(inertiaY,finalState),
            sourceDyDot3(inertiaZ,finalState));
        arena.Store3(to,asuint(finalLinear));
        arena.Store3(to+12u,asuint(finalAngular));
        if((inputFlags&64u)==0u) {
            arena.Store3(to+24u,asuint(motionLinear));
            arena.Store3(to+36u,asuint(angular));
        }
    }
    if(freeSource && ordinal<p.bodyCount) {
        const uint to=p.writebackBias+p.resultBodyOffset+ordinal*52u;
        arena.Store3(to,asuint(finalLinear));
        arena.Store3(to+12u,asuint(finalAngular));
    }
    // The axis lock was applied in solver space before the world
    // inverse-inertia transform; the resulting motion is not masked
    // again by DyBodyCoreIntegrator::integrateCore.
    precise float3 motionAngular=angular;
    precise float3 nextPosition=position+motionLinear*dt;
    precise float4 nextQuaternion=quaternion;
    precise float w=sourceDyDot3(motionAngular,motionAngular);
    if(w!=0.0f) {
        w=sqrt(w);
        if(w>1.0e7f) {
            motionAngular*=1.0e7f/w;
            w=1.0e7f;
        }
        const float angle=dt*w*0.5f;
        precise float3 omega=motionAngular*(sin(angle)/w);
        const float4 dq=float4(
            quaternion.w*omega.x+omega.y*quaternion.z-omega.z*quaternion.y,
            quaternion.w*omega.y+omega.z*quaternion.x-omega.x*quaternion.z,
            quaternion.w*omega.z+omega.x*quaternion.y-omega.y*quaternion.x,
            -(omega.x*quaternion.x)-omega.y*quaternion.y-
                omega.z*quaternion.z);
        nextQuaternion=dq+quaternion*cos(angle);
        // Source integrate_quaternion rounds each square and sum in order.
        precise float qx=nextQuaternion.x*nextQuaternion.x;
        precise float qy=nextQuaternion.y*nextQuaternion.y;
        precise float qz=nextQuaternion.z*nextQuaternion.z;
        precise float qw=nextQuaternion.w*nextQuaternion.w;
        precise float length=sqrt(((qx+qy)+qz)+qw);
        if(!(length>0.0f) || !isfinite(length))return;
        nextQuaternion/=length;
    }
    // The integrated actor rotation orients every attached shape.
    const float3 localCenter=asfloat(scActorLocals.Load3(local));
    nextPosition-=sourceRigidRotate(nextQuaternion,localCenter);
    if(!all(isfinite(nextPosition)) || !all(isfinite(nextQuaternion)))return;
    float3 sleepLinear=0.0f,sleepAngular=0.0f;
    float wakeCounter=0.0f;
    if((inputFlags&64u)!=0u &&
       !sourceDyPrepareSleep(input,local,motionLinear,angular,
                             nextQuaternion,sleepLinear,sleepAngular,
                             wakeCounter))return;
    if(ordinal<p.bodyCount && (inputFlags&64u)!=0u) {
        const uint result=p.writebackBias+p.resultBodyOffset+ordinal*52u;
        arena.Store3(result+24u,asuint(sleepLinear));
        arena.Store3(result+36u,asuint(sleepAngular));
        arena.Store(result+48u,asuint(wakeCounter));
    }
    if(ordinal>=p.bodyCount && (inputFlags&32u)!=0u) {
        const uint count=arena.Load(p.integrationBias+32u);
        const uint output=p.writebackBias+p.resultPatchOffset+
            p.patchCount*4u+count*32u+
            (ordinal-p.bodyCount)*((inputFlags&64u)!=0u?52u:24u);
        arena.Store3(output,asuint(finalLinear));
        arena.Store3(output+12u,asuint(finalAngular));
        if((inputFlags&64u)!=0u) {
            arena.Store3(output+24u,asuint(sleepLinear));
            arena.Store3(output+36u,asuint(sleepAngular));
            arena.Store(output+48u,asuint(wakeCounter));
        }
    }
    if((inputFlags&8u)!=0u) {
        scActorLocals.Store3(local+64u,asuint(finalLinear));
        scActorLocals.Store3(local+76u,asuint(finalAngular));
        if((inputFlags&64u)!=0u) {
            scActorLocals.Store(local+88u,asuint(wakeCounter));
            scActorLocals.Store3(local+92u,asuint(sleepLinear));
            scActorLocals.Store3(local+104u,asuint(sleepAngular));
        }
    }
    arena.Store3(destination,asuint(nextPosition));
    arena.Store4(destination+12u,asuint(nextQuaternion));
    uint element=arena.Load(input+12u);
    const uint shapeCount=arena.Load(input+16u);
    const uint4 views=arena.Load4(p.integrationBias);
    for(uint shape=0u;shape<shapeCount;++shape) {
        if(element>=views.w)return;
        const uint geometryIndex=views.z+element*64u;
        const float3 halfExtents=asfloat(scGeometry.Load3(geometryIndex));
        const float3 localPosition=asfloat(scGeometry.Load3(geometryIndex+16u));
        const uint kind=scGeometry.Load(geometryIndex+28u);
        const float4 localQuaternion=asfloat(scGeometry.Load4(geometryIndex+32u));
        const float3 center=nextPosition+sourceRigidRotate(nextQuaternion,localPosition);
        const float4 shapeQuaternion=sourceRigidMultiply(
            nextQuaternion,localQuaternion);
#ifdef FYZZXYZZY_CONVEX
        float3 lower,upper;
        if(kind==2u) {
            const uint vertexBias=p.reservedOutput;
            const uint first=scGeometry.Load(geometryIndex+52u);
            const uint vertexCount=scGeometry.Load(geometryIndex+56u);
            if(vertexCount<4u || first>0xffffffffu/12u ||
               vertexCount>(0xffffffffu/12u)-first)return;
            lower=float3(3.402823466e38f,3.402823466e38f,3.402823466e38f);
            upper=-lower;
            for(uint j=0u;j<vertexCount;++j) {
                const float3 vertex=asfloat(scHullVertices.Load3(
                    vertexBias+(first+j)*12u));
                const float3 vertexWorld=center+
                    sourceRigidRotate(shapeQuaternion,vertex);
                lower=min(lower,vertexWorld);
                upper=max(upper,vertexWorld);
            }
        }else {
            const float3 worldExtents=kind==1u?halfExtents:
                sourceBoxHalfBounds(shapeQuaternion,halfExtents);
            lower=center-worldExtents;
            upper=center+worldExtents;
        }
        if(kind>2u || !all(isfinite(lower)) || !all(isfinite(upper)) ||
           !all(isfinite(shapeQuaternion)))return;
#else
        const float3 worldExtents=kind==1u?halfExtents:
            sourceBoxHalfBounds(shapeQuaternion,halfExtents);
        const float3 lower=center-worldExtents;
        const float3 upper=center+worldExtents;
        if(kind>1u || !all(isfinite(lower)) || !all(isfinite(upper)) ||
           !all(isfinite(shapeQuaternion)))return;
#endif
        const uint cache=views.x+element*32u;
        scTransforms.Store4(cache,asuint(shapeQuaternion));
        scTransforms.Store3(cache+16u,asuint(center));
        scTransforms.Store(cache+28u,0u);
        const uint bound=views.y+element*24u;
        scBounds.Store3(bound,asuint(lower));
        scBounds.Store3(bound+12u,asuint(upper));
        element=scGeometry.Load(geometryIndex+48u);
    }
    if(shapeCount==0u || element==0xffffffffu) {
        if((inputFlags&128u)!=0u) {
            scActorLocals.Store3(local+116u,asuint(nextPosition));
            scActorLocals.Store4(local+128u,asuint(nextQuaternion));
        }
        arena.Store(destination+28u,1u);
    }
}

static const uint fusedTask=0u, prepareTile=1u;
static const uint partitionTile=3u, motionTile=4u, integrationTile=5u;
static const uint publishErrors=6u, freeBodyTile=7u, colorTile=8u;
static const uint initializeBodiesTile=9u, buildPartitionTiles=10u;

// Native Sc resolves inverse mass through the persistent actor-local
// pool. Standalone Dy callers retain their explicit inverse-mass stream.
// CmUtils::transformInertiaTensor(sqrt(invI), massPose.q) is prepared
// directly from native Dy's resident actor mass frame for contact-only islands.
// Source joint-island rows continue consuming their CPU-prepared tensor.
// Preserve the ordered PxQuat and PxQuat::rotate products used by
// CmUtils::transformInertiaTensor at the iterative contact boundary.
float4 sourceDyMassOrientationOrdered(float4 a,float4 b) {
    precise float x=((a.w*b.x+b.w*a.x)+a.y*b.z)-a.z*b.y;
    precise float y=((a.w*b.y+b.w*a.y)+a.z*b.x)-a.x*b.z;
    precise float z=((a.w*b.z+b.w*a.z)+a.x*b.y)-a.y*b.x;
    precise float w=((a.w*b.w-a.x*b.x)-a.y*b.y)-a.z*b.z;
    return float4(x,y,z,w);
}
float3 sourceDyRotateOrdered(float4 q,float3 v) {
    precise float cx=q.y*v.z-q.z*v.y;
    precise float cy=q.z*v.x-q.x*v.z;
    precise float cz=q.x*v.y-q.y*v.x;
    precise float tx=cx*2.0f,ty=cy*2.0f,tz=cz*2.0f;
    precise float dx=q.y*tz-q.z*ty;
    precise float dy=q.z*tx-q.x*tz;
    precise float dz=q.x*ty-q.y*tx;
    precise float x=(v.x+q.w*tx)+dx;
    precise float y=(v.y+q.w*ty)+dy;
    precise float z=(v.z+q.w*tz)+dz;
    return float3(x,y,z);
}
bool sourceDyPrepareWorldInertia(uint ordinal) {
    const uint input=p.integrationBias+44u+ordinal*40u;
    const uint inertia=sourceDyInertiaAddress(ordinal);
    // Nonresponding kinematic rows have no numerical inertia.
    const uint flags=arena.Load(input+8u);
    if((flags&0x4000u)==0u) {
        if((flags&1u)==0u) {
            const uint3 zero=uint3(0u,0u,0u);
            arena.Store3(inertia,zero);
            arena.Store3(inertia+12u,zero);
            arena.Store3(inertia+24u,zero);
        }
        return true;
    }
    const uint local=arena.Load(input+24u);
    if(local==invalidBody)return false;
    const float3 inverseI=asfloat(scActorLocals.Load3(local+48u));
    if(!all(isfinite(inverseI)) || any(inverseI<0.0f))return false;
    const float3 lengths=sqrt(inverseI);
    const float4 actorAxes=asfloat(scActorLocals.Load4(local+128u));
    const float4 massAxes=asfloat(scActorLocals.Load4(local+12u));
    const float4 q=sourceDyMassOrientationOrdered(actorAxes,massAxes);
    if(!all(isfinite(q)))return false;
    float3 c0,c1,c2;
    if(q.x==0.0f && q.y==0.0f && q.z==0.0f && q.w==1.0f) {
        c0=float3(lengths.x,0.0f,0.0f);
        c1=float3(0.0f,lengths.y,0.0f);
        c2=float3(0.0f,0.0f,lengths.z);
    }else {
        const float3 rx=sourceDyRotateOrdered(q,float3(1.0f,0.0f,0.0f));
        const float3 ry=sourceDyRotateOrdered(q,float3(0.0f,1.0f,0.0f));
        const float3 rz=sourceDyRotateOrdered(q,float3(0.0f,0.0f,1.0f));
        precise float xx=(lengths.x*rx.x*rx.x+lengths.y*ry.x*ry.x)+lengths.z*rz.x*rz.x;
        precise float yy=(lengths.x*rx.y*rx.y+lengths.y*ry.y*ry.y)+lengths.z*rz.y*rz.y;
        precise float zz=(lengths.x*rx.z*rx.z+lengths.y*ry.z*ry.z)+lengths.z*rz.z*rz.z;
        precise float xy=(lengths.x*rx.x*rx.y+lengths.y*ry.x*ry.y)+lengths.z*rz.x*rz.y;
        precise float xz=(lengths.x*rx.x*rx.z+lengths.y*ry.x*ry.z)+lengths.z*rz.x*rz.z;
        precise float yz=(lengths.x*rx.y*rx.z+lengths.y*ry.y*ry.z)+lengths.z*rz.y*rz.z;
        c0=float3(xx,xy,xz);
        c1=float3(xy,yy,yz);
        c2=float3(xz,yz,zz);
    }
    if(!all(isfinite(c0)) || !all(isfinite(c1)) ||
       !all(isfinite(c2)))return false;
    arena.Store3(inertia,asuint(c0));
    arena.Store3(inertia+12u,asuint(c1));
    arena.Store3(inertia+24u,asuint(c2));
    return true;
}

void initializeSolverBody(uint ordinal,uint source) {
    const uint target=p.bodyBias+ordinal*52u;
    const uint4 zero=uint4(0u,0u,0u,0u);
    arena.Store4(target,zero);
    arena.Store4(target+16u,zero);
    arena.Store4(target+32u,zero);
    uint massBits=0u;
    if(source==invalidBody) {
        const uint input=p.integrationBias+44u+ordinal*40u;
        const uint local=arena.Load(input+24u);
        if(local!=invalidBody)
            massBits=scActorLocals.Load(local+28u);
    }else
        massBits=arena.Load(source+ordinal*4u);
    arena.Store(target+48u,massBits);
}

void publishTaskError(uint task,uint error) {
    if((p.launchControl&0x08000000u)!=0u) {
        if(error!=0u) {
            const uint scratch=arena.Load(p.integrationBias+28u);
            arena.InterlockedOr(scratch+36u,error);
            arena.InterlockedMin(scratch+40u,task);
        }
    }else
        arena.Store(p.writebackBias+p.resultTaskOffset+task*4u,error);
}

// Each lane owns one independent constraint and up to two integrated bodies.
// The descriptor owns its responding endpoints for this iteration range.
void sourceDySolveExclusive(uint4 desc,uint2 iters,uint2 iterations,
                            uint2 bodies,uint errorAddress) {
    SourceDyContactSpan span=(SourceDyContactSpan)0;
    SourceDySolverPair state;
    const bool joint=desc.w!=0u;
    if((!joint && !sourceDyLoadContactSpan(desc,errorAddress,span)) ||
       !sourceDyLoadSolverPair(desc,errorAddress,joint || span.manager.y!=0u,
                              state)) {
        return;
    }
    // One exclusively owned joint row has a bounded live working set.
    const bool localJoint=desc.w==1u;
    const uint jointOffset=p.distanceBias+desc.x*64u;
    SourceDyJointRow jointRow=(SourceDyJointRow)0;
    uint2 jointCoefficients=uint2(0u,0u);
    float jointImpulse=0.0f;
    if(localJoint) {
        jointRow=sourceDyLoadJointRow(jointOffset);
        jointCoefficients=arena.Load2(jointOffset+48u);
        jointImpulse=loadScalar(jointOffset+60u);
    }
    const uint total=iters.x+max(iters.y,1u);
    for(uint iteration=iterations.x;iteration<iterations.y;++iteration) {
        const bool biased=iteration<iters.x;
        const bool friction=!biased || iters.x-iteration<=3u;
        const bool finalPass=iteration+1u==total;
        bool solved;
        if(localJoint) {
            SourceDySolverPair local=state;
            solved=sourceDySolveJointRow(jointRow,
                biased?jointCoefficients.x:jointCoefficients.y,
                errorAddress,local,jointImpulse);
            if(solved)sourceDyCommitJointPair(desc,local,state);
        } else if(joint)
            solved=solveJointRows(desc,errorAddress,biased,state);
        else
            solved=solveContactRows(desc,span,errorAddress,biased,friction,
                                    finalPass,state);
        if(!solved)return;
        if(iteration+1u==iters.x) {
            // Position integration consumes the biased-phase snapshot,
            // without publishing the live velocity deltas.
            for(uint body=bodies.x;body<bodies.y;++body) {
                const uint ordinal=sourceDyBodyOrdinal(body);
                if(ordinal>=p.bodyCount) {
                    sourceDyTaskErrorOr(errorAddress,8u);
                    continue;
                }
                const uint base=p.bodyBias+ordinal*52u;
                if(ordinal==desc.z && state.mb>0.0f) {
                    arena.Store3(base+24u,asuint(state.vb));
                    arena.Store3(base+36u,asuint(state.ab));
                } else if(ordinal==desc.y && state.ma>0.0f) {
                    arena.Store3(base+24u,asuint(state.va));
                    arena.Store3(base+36u,asuint(state.aa));
                } else {
                    arena.Store3(base+24u,arena.Load3(base));
                    arena.Store3(base+36u,arena.Load3(base+12u));
                }
            }
        }
        if(desc.y==desc.z && desc.y!=invalidBody) {
            if(state.mb>0.0f) {
                state.va=state.vb;
                state.aa=state.ab;
            } else if(state.ma>0.0f) {
                state.vb=state.va;
                state.ab=state.aa;
            }
        }
    }
    if(localJoint)arena.Store(jointOffset+60u,asuint(jointImpulse));
    sourceDyStoreSolverPair(desc,state,false);
}

bool sourceDyPackedTaskShape(uint4 ranges,uint2 bodies) {
    return ranges.y==ranges.x+1u && ranges.w==ranges.z+1u &&
           bodies.y>bodies.x && bodies.y-bodies.x<=2u;
}

void sourceDySolvePackedTask(uint taskId,bool precolored) {
    const uint task=p.taskBias+taskId*36u;
    const uint4 ranges=sourceDyTaskLoad4(task);
    const uint2 bodies=sourceDyTaskLoad2(task+16u);
    const uint2 iters=sourceDyTaskLoad2(task+24u);
    const uint errorAddress=task+32u;
    const uint4 schedule=arena.Load4(p.integrationBias+16u);
    if(!sourceDyPackedTaskShape(ranges,bodies) ||
       ranges.y>schedule.y || ranges.w>p.partitionCount ||
       bodies.y>schedule.x ||
       !iters.x || iters.x>255u || iters.y>255u) {
        sourceDyTaskErrorOr(errorAddress,1048576u);
        publishTaskError(taskId,sourceDyTaskLoad(errorAddress));
        return;
    }
    const uint scratch=schedule.w;
    if(arena.Load(scratch)==64u && arena.Load(scratch+48u)!=0u) {
        for(uint body=bodies.x;body<bodies.y;++body)
            if(sourceDyBodyOrdinal(body)>=p.bodyCount)
                sourceDyTaskErrorOr(errorAddress,8u);
        if(sourceDyTaskLoad(errorAddress)!=0u) {
            publishTaskError(taskId,sourceDyTaskLoad(errorAddress));
            return;
        }
    }
    uint4 desc;
    if(precolored) {
        desc=arena.Load4(p.descriptorBias+ranges.x*16u);
    }else {
        const uint4 views=arena.Load4(scratch);
        const uint colors=scratch+views.y;
        const uint sourceTable=scratch+views.x+
            (views.x==64u?0u:schedule.y*4u);
        if(!sourceDyExpandDescriptor(
                sourceDyResponseToken(scratch,colors,ranges.x),
                sourceDyResponseExtent(scratch,ranges.x),
                arena.Load(sourceTable),sourceTable,
                                     errorAddress,desc)) {
            publishTaskError(taskId,sourceDyTaskLoad(errorAddress));
            return;
        }
        const bool da=(desc.w&0x100u)!=0u;
        const bool db=(desc.w&0x200u)!=0u;
        if((desc.w&~0x3ffu)!=0u || (!da && !db) ||
           (da && desc.y>=p.bodyCount) ||
           (db && desc.z>=p.bodyCount)) {
            sourceDyTaskErrorOr(errorAddress,1048576u);
            publishTaskError(taskId,sourceDyTaskLoad(errorAddress));
            return;
        }
        if((p.launchControl&0x08000000u)!=0u &&
           !sourceDyRelocateContact(
                desc,scratch,ranges.x,errorAddress)) {
            publishTaskError(taskId,sourceDyTaskLoad(errorAddress));
            return;
        }
    }
    // This invocation owns the descriptor and its complete numerical solve.
    desc.w&=255u;
    prepareConstraint(desc,0u,1u,errorAddress);
    if(sourceDyTaskLoad(errorAddress)!=0u) {
        publishTaskError(taskId,sourceDyTaskLoad(errorAddress));
        return;
    }

    const uint total=iters.x+max(iters.y,1u);
    sourceDySolveExclusive(desc,iters,uint2(0u,total),bodies,errorAddress);
    if(sourceDyTaskLoad(errorAddress)!=0u) {
        publishTaskError(taskId,sourceDyTaskLoad(errorAddress));
        return;
    }
    for(uint body=bodies.x;body<bodies.y;++body) {
        const uint ordinal=sourceDyBodyOrdinal(body);
        if(ordinal>=p.bodyCount)
            sourceDyTaskErrorOr(errorAddress,8u);
        else
            integrateBody(ordinal,false);
    }
    publishTaskError(taskId,sourceDyTaskLoad(errorAddress));
}

// The accurate component root owns the source body's numerical admission.
bool sourceDyRespondingSource(uint ordinal,uint counts,uint header) {
    const uint root=scIslandWork.Load(header+48u+ordinal*4u);
    return root<p.bodyCount &&
        scIslandWork.Load(header+counts+root*24u+20u)!=invalidBody;
}

// A color owns disjoint task intervals. Emit only tiles that contain
// actual descriptors and publish one indirect count per reserved chunk.
void sourceDyBuildPartitionTiles(uint color,uint lane) {
    const uint scratch=arena.Load(p.integrationBias+28u);
    const uint metadata=p.writebackBias+arena.Load(scratch+16u);
    const uint4 reservation=arena.Load4(metadata+(p.contactCount+color)*16u);
    if(lane==0u)dyPartitionTileCount=0u;
    GroupMemoryBarrierWithGroupSync();
    for(uint item=lane;item<p.patchCount;item+=128u) {
        const uint4 bound=arena.Load4(metadata+(p.frictionCount+item)*16u);
        const uint task=bound.x;
        const uint address=p.taskBias+task*36u;
        if(sourceDyTaskLoad(address+32u)!=0u)continue;
        const uint4 ranges=sourceDyTaskLoad4(address);
        if(ranges.w<ranges.z || ranges.w-ranges.z>bound.y) {
            sourceDyTaskErrorOr(address+32u,0x02000000u);
            continue;
        }
        if(color>=ranges.w-ranges.z)continue;
        const uint partition=ranges.z+color;
        const uint first=color==0u?ranges.x:
            arena.Load(p.partitionBias+(partition-1u)*4u);
        const uint last=arena.Load(p.partitionBias+partition*4u);
        if(last<first || last>ranges.y) {
            sourceDyTaskErrorOr(address+32u,0x02000000u);
            continue;
        }
        const uint width=last-first;
        const uint groups=(width>>7u)+uint((width&127u)!=0u);
        if(groups>bound.z) {
            sourceDyTaskErrorOr(address+32u,0x02000000u);
            continue;
        }
        uint offset;
        InterlockedAdd(dyPartitionTileCount,groups,offset);
        if(offset>reservation.y || groups>reservation.y-offset) {
            sourceDyTaskErrorOr(address+32u,0x02000000u);
            continue;
        }
        for(uint tile=0u;tile<groups;++tile) {
            const uint begin=tile*128u;
            arena.Store4(p.rowCount+(reservation.x+offset+tile)*16u,
                uint4(task,begin,begin+min(128u,width-begin),color));
        }
    }
    AllMemoryBarrierWithGroupSync();
    const uint count=dyPartitionTileCount<=reservation.y?dyPartitionTileCount:0u;
    for(uint chunk=lane;chunk<reservation.w;chunk+=128u) {
        const uint begin=chunk*p.partitionCount;
        const uint groups=begin<count?min(p.partitionCount,count-begin):0u;
        retainedSources.Store3((reservation.z+chunk)*16u,uint3(groups,1u,1u));
    }
}

// Disjoint descriptor intervals own disjoint lists of full 128-row colors.
uint sourceDyWideColorList(uint scratch,uint first) {
    return scratch+arena.Load(scratch+8u)+(first/128u)*4u;
}

bool sourceDyResidentTileRange(uint task,uint kind,
        out uint begin,out uint end,out uint step,out uint color) {
    begin=0u;end=0u;step=1u;color=0u;
    const uint address=p.taskBias+task*36u;
    // Color mode is immutable across the count and fill passes.
    // Errors from other color builders cannot remove a reserved tile.
    if(kind<3u && sourceDyTaskLoad(address+32u)!=0u)return false;
    const uint wideCount=arena.Load(p.resultTaskOffset+task*4u);
    if((wideCount!=0u)==(kind==0u))return false;
    const uint4 ranges=sourceDyTaskLoad4(address);
    const uint2 bodies=sourceDyTaskLoad2(address+16u);
    begin=0u;end=1u;step=1u;color=0u;
    if(kind==1u) {
        begin=ranges.x;end=ranges.y;step=32u;
    }else if(kind==2u) {
        begin=bodies.x;end=bodies.y;step=128u;
    }else if(kind>=3u) {
        const uint stage=kind-3u;
        const uint ordinal=stage/2u;
        if(ranges.w<ranges.z || ranges.y<ranges.x ||
           ranges.y>arena.Load(p.integrationBias+20u) ||
           wideCount>(ranges.y-ranges.x)/128u ||
           wideCount>(p.frictionCount-1u)/2u) {
            sourceDyTaskErrorOr(address+32u,0x02000000u);
            return false;
        }
        if(ordinal>wideCount || ((stage&1u)!=0u && ordinal==wideCount))
            return false;
        // The retained task window lists its wide partition indices.
        const uint scratch=arena.Load(p.integrationBias+28u);
        const uint list=sourceDyWideColorList(scratch,ranges.x);
        if((stage&1u)==0u) {
            begin=ordinal==0u?ranges.z:arena.Load(list+(ordinal-1u)*4u)+1u;
            end=ordinal==wideCount?ranges.w:arena.Load(list+ordinal*4u);
            if(begin<ranges.z || end>ranges.w || end<begin) {
                sourceDyTaskErrorOr(address+32u,0x02000000u);
                return false;
            }
            if(end==begin)return false;
            // One workgroup executes the complete narrow color run.
            step=end-begin;
            color=0x80000000u;
        }else {
            const uint partition=arena.Load(list+ordinal*4u);
            if(partition<ranges.z || partition>=ranges.w) {
                sourceDyTaskErrorOr(address+32u,0x02000000u);
                return false;
            }
            color=partition-ranges.z;
            const uint first=color==0u?ranges.x:
                arena.Load(p.partitionBias+(partition-1u)*4u);
            const uint last=arena.Load(p.partitionBias+partition*4u);
            if(last<first || last>ranges.y) {
                sourceDyTaskErrorOr(address+32u,0x02000000u);
                return false;
            }
            end=last-first;step=128u;
        }
    }
    if(end<begin) {
        sourceDyTaskErrorOr(address+32u,0x02000000u);
        return false;
    }
    return true;
}

// A packed tile covers a block of independent one-descriptor tasks.
bool sourceDyResidentPackedTask(uint task) {
    const uint address=p.taskBias+task*36u;
    return sourceDyTaskLoad(address+32u)==0u &&
           arena.Load(p.resultTaskOffset+task*4u)==0u &&
           sourceDyPackedTaskShape(sourceDyTaskLoad4(address),
                                  sourceDyTaskLoad2(address+16u));
}

void sourceDyBuildResidentFusedTiles(uint4 reservation,uint taskCount,uint lane) {
    if(lane==0u) {
        dyPartitionTileCount=0u;
        dyCompactTaskCount=0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint compact=0u;
    for(uint first=0u;first<taskCount;first+=128u) {
        if(lane==0u)dyPackedTaskCount=0u;
        GroupMemoryBarrierWithGroupSync();
        const uint task=first+lane;
        if(task<taskCount) {
            uint begin,end,step,color;
            if(sourceDyResidentTileRange(task,0u,begin,end,step,color)) {
                ++compact;
                if(sourceDyResidentPackedTask(task)) {
                    InterlockedAdd(dyPackedTaskCount,1u);
                }else {
                    uint offset;
                    InterlockedAdd(dyPartitionTileCount,1u,offset);
                    if(offset<reservation.y)
                        arena.Store4(p.rowCount+(reservation.x+offset)*16u,
                                    uint4(task,0u,1u,0u));
                    else sourceDyTaskErrorOr(p.taskBias+task*36u+32u,0x02000000u);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if(lane==0u && dyPackedTaskCount!=0u) {
            uint offset;
            InterlockedAdd(dyPartitionTileCount,1u,offset);
            if(offset<reservation.y)
                arena.Store4(p.rowCount+(reservation.x+offset)*16u,
                            uint4(first,first,first+min(128u,taskCount-first),
                                  0x80000000u));
            else sourceDyTaskErrorOr(p.taskBias+first*36u+32u,0x02000000u);
        }
    }
    if(compact!=0u)InterlockedAdd(dyCompactTaskCount,compact);
    AllMemoryBarrierWithGroupSync();
    const uint count=dyPartitionTileCount<=reservation.y?dyPartitionTileCount:0u;
    for(uint chunk=lane;chunk<reservation.w;chunk+=128u) {
        const uint begin=chunk*p.partitionCount;
        const uint groups=begin<count?min(p.partitionCount,count-begin):0u;
        retainedSources.Store3((reservation.z+chunk)*16u,uint3(groups,1u,1u));
    }
    if(lane==0u) {
        const uint wide=taskCount-dyCompactTaskCount;
        retainedSources.Store(reservation.z*16u+12u,wide);
        arena.Store2(p.writebackBias+p.resultContactOffset,uint2(wide,count));
    }
}

// Wide colors and narrow runs consume a shared response-bounded tile pool.
void sourceDyBuildResidentTiles(uint kind,uint lane) {
    const uint scratch=arena.Load(p.integrationBias+28u);
    const uint metadata=p.writebackBias+arena.Load(scratch+16u);
    const uint4 reservation=arena.Load4(metadata+(p.contactCount+kind)*16u);
    const uint sourceHeader=p.taskBias-arena.Load(scratch+24u);
    const uint4 header=scIslandWork.Load4(sourceHeader);
    const bool valid=header.x==0u && header.y<=p.patchCount &&
        header.z<=p.bodyCount && header.w<=p.orderCount;
    if(valid && arena.Load(scratch+32u)==header.y) {
        if(lane==0u) {
            if(kind>=3u)
                arena.Store(metadata+(p.contactCount+kind)*16u,
                            retainedSources.Load(reservation.z*16u+12u));
            if(kind==0u) {
                uint count=0u;
                for(uint chunk=0u;chunk<reservation.w;++chunk)
                    count+=retainedSources.Load((reservation.z+chunk)*16u);
                arena.Store2(p.writebackBias+p.resultContactOffset,
                    uint2(0x80000000u|retainedSources.Load(reservation.z*16u+12u),count));
            }
        }
        return;
    }
    const uint taskCount=valid?header.y:0u;
    if(kind==0u) {
        sourceDyBuildResidentFusedTiles(reservation,taskCount,lane);
        return;
    }
    if(lane==0u)dyPartitionTileCount=0u;
    GroupMemoryBarrierWithGroupSync();
    uint laneGroups=0u;
    for(uint task=lane;task<taskCount;task+=128u) {
        uint begin,end,step,color;
        if(!sourceDyResidentTileRange(task,kind,begin,end,step,color))continue;
        const uint width=end-begin;
        const uint groups=width/step+uint(width%step!=0u);
        if(kind>=3u) {
            laneGroups+=groups;
            continue;
        }
        uint offset;
        InterlockedAdd(dyPartitionTileCount,groups,offset);
        if(offset>reservation.y || groups>reservation.y-offset) {
            sourceDyTaskErrorOr(p.taskBias+task*36u+32u,0x02000000u);
            continue;
        }
        for(uint tile=0u;tile<groups;++tile) {
            const uint first=begin+tile*step;
            arena.Store4(p.rowCount+(reservation.x+offset+tile)*16u,
                uint4(task,first,first+min(step,end-first),color));
        }
    }
    if(kind>=3u) {
        uint laneOffset=0u;
        if(laneGroups!=0u)
            InterlockedAdd(dyPartitionTileCount,laneGroups,laneOffset);
        GroupMemoryBarrierWithGroupSync();
        if(lane==0u) {
            const uint poolAddress=metadata+(p.contactCount+3u+p.frictionCount)*16u;
            const uint2 pool=arena.Load2(poolAddress);
            const uint count=dyPartitionTileCount;
            uint offset=0u;
            arena.InterlockedAdd(poolAddress+8u,count,offset);
            if(count>reservation.y || offset>pool.y || count>pool.y-offset) {
                uint unused;
                arena.InterlockedOr(scratch+36u,0x02000000u,unused);
                arena.InterlockedMin(scratch+40u,0u,unused);
                dyPartitionTileCount=0u;
                dyPartitionTileBase=0u;
            }else {
                dyPartitionTileBase=p.rowCount+(pool.x+offset)*16u;
                arena.Store(metadata+(p.contactCount+kind)*16u,
                            dyPartitionTileBase);
                retainedSources.Store(reservation.z*16u+12u,
                                      dyPartitionTileBase);
            }
        }
        AllMemoryBarrierWithGroupSync();
        if(dyPartitionTileCount!=0u && laneGroups!=0u)
            for(uint task=lane;task<taskCount;task+=128u) {
                uint begin,end,step,color;
                if(!sourceDyResidentTileRange(task,kind,begin,end,step,color))continue;
                const uint width=end-begin;
                const uint groups=width/step+uint(width%step!=0u);
                for(uint tile=0u;tile<groups;++tile) {
                    const uint first=begin+tile*step;
                    arena.Store4(dyPartitionTileBase+(laneOffset+tile)*16u,
                        uint4(task,first,first+min(step,end-first),color));
                }
                laneOffset+=groups;
            }
    }
    AllMemoryBarrierWithGroupSync();
    const uint count=dyPartitionTileCount<=reservation.y?dyPartitionTileCount:0u;
    for(uint chunk=lane;chunk<reservation.w;chunk+=128u) {
        const uint begin=chunk*p.partitionCount;
        const uint groups=begin<count?min(p.partitionCount,count-begin):0u;
        retainedSources.Store3((reservation.z+chunk)*16u,uint3(groups,1u,1u));
    }
}

// Other tiles of a wide task may report errors concurrently. All lanes
// must take the same preparation branch around workgroup barriers.
uint preparationError(uint errorAddress,uint lane) {
    if(lane==0u)dyPreparationError=sourceDyTaskLoad(errorAddress);
    GroupMemoryBarrierWithGroupSync();
    return dyPreparationError;
}

[numthreads(128,1,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint phase=p.launchControl&15u;
    const bool residentWide=phase!=freeBodyTile &&
        (p.launchControl&0x0c000000u)==0x0c000000u;
    if(phase==buildPartitionTiles) {
        if(residentWide)
            sourceDyBuildResidentTiles(p.launchBegin+group.x,lane.x);
        else
            sourceDyBuildPartitionTiles(p.launchBegin+group.x,lane.x);
        return;
    }
    const bool devicePartitions=(p.launchControl&0x80000000u)!=0u;
    const bool precolored=(p.launchControl&0x40000000u)!=0u;
    const bool indirectPartition=phase==partitionTile &&
        (p.launchControl&0x10000000u)!=0u;
    const uint iter=phase==partitionTile || phase==motionTile
        ?(p.launchControl>>4u)&511u:(p.launchControl&0x1fffffffu)>>4u;
    uint index=p.launchBegin+group.x;
    uint admittedTasks=0xffffffffu;
    uint admittedBodies=0xffffffffu;
    bool deviceBodyOrder=false;
    uint admittedDispatch=0u;
    uint admittedDispatchX=0u;
    uint sourceCounts=0u,sourceHeaderBias=0u;
    // A submitted Sc island graph can differ from CPU-predicted counts
    // because contact activation is owned by resident Gu history. Reject
    // the entire numerical epoch before initializing any solver body.
    // The task error report still runs, including compact-only scenes.
    if(devicePartitions) {
        const uint4 schedule=arena.Load4(p.integrationBias+16u);
        if(schedule.w!=0u && arena.Load(schedule.w)==64u) {
            const uint taskOffset=arena.Load(schedule.w+24u);
            const uint sourceHeader=p.taskBias-taskOffset;
            const uint4 header=scIslandWork.Load4(sourceHeader);
            if(header.x!=0u || header.y>schedule.z ||
               header.z>schedule.x || header.w>schedule.y) {
                if(phase==publishErrors) {
                    const uint task=index*128u+lane.x;
                    if(task<schedule.z)
                        publishTaskError(task,0x01000000u);
                }
                return;
            }
            admittedTasks=header.y;
            admittedBodies=header.z;
            sourceCounts=arena.Load(schedule.w+60u);
            sourceHeaderBias=sourceHeader;
            deviceBodyOrder=arena.Load(schedule.w+56u)==0u;
            admittedDispatchX=scIslandWork.Load(sourceHeader+16u);
            admittedDispatch=admittedDispatchX*
                scIslandWork.Load(sourceHeader+20u);
        }
    }
    if(phase==fusedTask && admittedTasks!=0xffffffffu)
        index+=group.y*admittedDispatchX;
    // All lane-owned task launches enter the same numerical routine.
    uint4 residentFusedTile=uint4(0u,0u,0u,0u);
    if(phase==fusedTask && devicePartitions) {
        bool packed=false,eligible=false;
        uint task=0u;
        if(residentWide) {
            residentFusedTile=arena.Load4(p.launchBegin+group.x*16u);
            packed=(residentFusedTile.w&0x80000000u)!=0u;
            if(packed) {
                task=residentFusedTile.y+lane.x;
                eligible=task<residentFusedTile.z && sourceDyResidentPackedTask(task);
            }
        }else if(!precolored && (p.launchControl&0x10000000u)!=0u) {
            packed=true;
            const uint within=group.x*128u+lane.x;
            const uint count=(p.launchControl&0x0ffffff0u)>>4u;
            task=p.launchBegin+within;
            eligible=within<count;
        }else if(!precolored && admittedTasks!=0xffffffffu &&
                 admittedTasks>1u && admittedDispatch<admittedTasks) {
            packed=true;
            task=index*128u+lane.x;
            eligible=task<admittedTasks;
        }
        [branch]
        if(packed) {
            if(eligible)sourceDySolvePackedTask(task,residentWide);
            return;
        }
    }
    if(phase==initializeBodiesTile) {
        // Device Sc has already selected the responding body ordinals.
        // Joint graphs retain their nonresponding kinematic row inputs.
        const uint item=index*128u+lane.x;
        const uint count=deviceBodyOrder?admittedBodies:p.bodyCount;
        if(item<count) {
            const uint ordinal=deviceBodyOrder
                ?sourceDyBodyOrdinal(item):item;
            if(ordinal<p.bodyCount) {
                const uint flags=arena.Load(p.integrationBias+52u+ordinal*40u);
                if(sourceCounts!=0u && (flags&1u)!=0u &&
                   !sourceDyRespondingSource(ordinal,sourceCounts,sourceHeaderBias))
                    return;
                initializeSolverBody(ordinal,arena.Load(p.integrationBias+36u));
                if(!sourceDyPreIntegrateBody(ordinal) ||
                   !sourceDyPrepareWorldInertia(ordinal))
                    sourceDyInvalidatePreparedVelocity(ordinal);
            }
        }
        return;
    }
    const bool freeBody=phase==freeBodyTile;
    const bool fused=phase==fusedTask;
    uint taskId=0u,endError=0u;
    uint2 bodies;
    [branch]
    if(freeBody) {
        const uint begin=(sourceCounts!=0u?0u:p.bodyCount)+index*128u;
        bodies=uint2(begin,min(begin+128u,iter));
    } else {
        const uint4 schedule=arena.Load4(p.integrationBias+16u);
        // Color groups consume [0, taskCount). Additional groups in that
        // SAME dispatch initialize distinct body blocks in parallel.
        // No solver reads body state until the next ordered dispatch.
        if(phase==colorTile && index>=schedule.z) {
            const uint ordinal=(index-schedule.z)*128u+lane.x;
            const uint source=arena.Load(p.integrationBias+36u);
            if(source!=0u && ordinal<p.bodyCount) {
                initializeSolverBody(ordinal,source);
                if(!sourceDyPreIntegrateBody(ordinal) ||
                   !sourceDyPrepareWorldInertia(ordinal))
                    sourceDyInvalidatePreparedVelocity(ordinal);
            }
            return;
        }
        // Compact single-task scenes can initialize up to one workgroup
        // of bodies before their fused PGS. Larger or independent tasks
        // use a separate initialization dispatch for global ordering.
        if(fused && !precolored && schedule.z==1u &&
           p.bodyCount<=128u) {
            const uint source=arena.Load(p.integrationBias+36u);
            if(source!=0u) {
                const uint flags=lane.x<p.bodyCount
                    ?arena.Load(p.integrationBias+52u+lane.x*40u):0u;
                if(lane.x<p.bodyCount &&
                   (sourceCounts==0u || (flags&1u)==0u ||
                    sourceDyRespondingSource(lane.x,sourceCounts,sourceHeaderBias))) {
                    initializeSolverBody(lane.x,source);
                    if(!sourceDyPreIntegrateBody(lane.x) ||
                       !sourceDyPrepareWorldInertia(lane.x))
                        sourceDyInvalidatePreparedVelocity(lane.x);
                }
                AllMemoryBarrierWithGroupSync();
            }
        }
        if(phase==publishErrors) {
            const uint task=index*128u+lane.x;
            if(task<schedule.z)
                publishTaskError(task,task<admittedTasks
                    ?sourceDyTaskLoad(p.taskBias+task*36u+32u):0u);
            return;
        }
        const bool residentTile=residentWide && phase!=colorTile;
        if(!residentTile && index>=admittedTasks)return;

        // A frame containing only compact tasks uses direct task indices.
        uint4 tile=uint4(index,0u,0u,0u);
        [branch]
        if(devicePartitions) {
            const uint tileOffset=arena.Load(schedule.w+16u);
            if(residentWide && indirectPartition)
                tile=arena.Load4(arena.Load(p.launchBegin)+
                                 p.resultTaskOffset+group.x*16u);
            else if(residentTile && fused)
                tile=residentFusedTile;
            else if(indirectPartition || residentTile)
                tile=arena.Load4(p.launchBegin+group.x*16u);
            else if(phase!=colorTile && tileOffset!=0u)
                tile=arena.Load4(p.writebackBias+tileOffset+index*16u);
        } else if(schedule.w!=0u)
            tile=arena.Load4(schedule.w+index*16u);
        const uint t=p.taskBias+tile.x*36u;
        taskId=tile.x;
        endError=t+32u;
        if(devicePartitions && (phase==colorTile ||
                                (fused && !precolored))) {
            sourceDyPartitionTask(taskId,schedule.w,endError,lane.x);
            AllMemoryBarrierWithGroupSync();
            if(sourceDyTaskLoad(endError)!=0u) {
                if(lane.x==0u) {
                    if(residentWide && phase==colorTile) {
                        sourceDyRejectScheduleReuse(schedule.w);
                        arena.Store(p.resultTaskOffset+taskId*4u,0u);
                    }
                    publishTaskError(taskId,sourceDyTaskLoad(endError));
                }
                return;
            }
            if(phase==colorTile) {
                // A retained task already owns its exact partition ends,
                // wide-color list and execution mode, even if another
                // task requires the global tile pool to be rebuilt.
                if(residentWide && lane.x==0u && dyReuseColors==0u) {
                    const uint4 ranges=sourceDyTaskLoad4(t);
                    const uint2 bodyRange=sourceDyTaskLoad2(t+16u);
                    const uint record=sourceDyRetainedTask(taskId);
                    arena.Store4(record,uint4(ranges.xy,bodyRange));
                    arena.Store(record+16u,ranges.w);
                    uint wideCount=0u;
                    if(ranges.y-ranges.x>=512u &&
                       bodyRange.y-bodyRange.x>=512u) {
                        const uint scratch=schedule.w;
                        const uint list=sourceDyWideColorList(scratch,ranges.x);
                        const uint capacity=(ranges.y-ranges.x)/128u;
                        uint first=ranges.x;
                        for(uint color=ranges.z;color<ranges.w;++color) {
                            const uint last=arena.Load(p.partitionBias+color*4u);
                            if(last<first || last>ranges.y) {
                                sourceDyTaskErrorOr(endError,0x02000000u);
                                break;
                            }
                            if(last-first>=128u) {
                                if(wideCount>=capacity) {
                                    sourceDyTaskErrorOr(endError,0x02000000u);
                                    break;
                                }
                                arena.Store(list+4u*wideCount++,color);
                            }
                            first=last;
                        }
                        if(first!=ranges.y)
                            sourceDyTaskErrorOr(endError,0x02000000u);
                        if(sourceDyTaskLoad(endError)!=0u) {
                            sourceDyRejectScheduleReuse(scratch);
                            publishTaskError(taskId,sourceDyTaskLoad(endError));
                            wideCount=0u;
                        }
                    }
                    arena.Store(p.resultTaskOffset+taskId*4u,wideCount);
                }
                return;
            }
        }
        const uint2 iters=sourceDyTaskLoad2(t+24u);
        uint2 descriptors=tile.yz;
        bodies=tile.yz;
        uint2 partitions=uint2(0u,1u);
        const bool localRun=residentWide && phase==partitionTile &&
            (tile.w&0x80000000u)!=0u;
        if(localRun) {
            const uint4 ranges=sourceDyTaskLoad4(t);
            partitions=tile.yz;
            if(partitions.x<ranges.z || partitions.y>ranges.w ||
               partitions.x>=partitions.y) {
                if(lane.x==0u)sourceDyTaskErrorOr(endError,0x02000000u);
                return;
            }
            descriptors=uint2(partitions.x==ranges.z?ranges.x:
                arena.Load(p.partitionBias+(partitions.x-1u)*4u),
                arena.Load(p.partitionBias+(partitions.y-1u)*4u));
        }else if(devicePartitions && phase==partitionTile) {
            const uint4 ranges=sourceDyTaskLoad4(t);
            if(tile.w>=ranges.w-ranges.z)return;
            const uint partition=ranges.z+tile.w;
            const uint start=tile.w==0u?ranges.x:
                arena.Load(p.partitionBias+(partition-1u)*4u);
            const uint finish=arena.Load(p.partitionBias+partition*4u);
            descriptors=uint2(min(finish,start+tile.y),
                              min(finish,start+tile.z));
            if(descriptors.x>=descriptors.y)return;
        }
        if(fused) {
            const uint4 ranges=sourceDyTaskLoad4(t);
            descriptors=ranges.xy;
            partitions=ranges.zw;
            bodies=sourceDyTaskLoad2(t+16u);
            if(descriptors.x>descriptors.y || descriptors.y>schedule.y ||
               partitions.x>partitions.y || partitions.y>p.partitionCount ||
               bodies.x>bodies.y || bodies.y>schedule.x) {
                if(lane.x==0u) {
                    sourceDyTaskErrorOr(endError,4u);
                    publishTaskError(tile.x,4u);
                }
                return;
            }
        }
        // A preparation tile owns complete contact managers. Normal
        // and friction rows consume their shared source geometry.
        if(!fused) {
            if(phase==prepareTile) {
                if(preparationError(endError,lane.x)!=0u)return;
            }else if(sourceDyTaskLoad(endError)!=0u)return;
        }
        if(fused || phase==prepareTile) {
            // Source response-token expansion already applies the PCM
            // extent before coloring. Other Dy callers use this path.
            const uint scratch=arena.Load(p.integrationBias+28u);
            const uint recordBytes=devicePartitions && scratch!=0u
                ?arena.Load(scratch):0u;
            if(recordBytes!=24u && recordBytes!=64u) {
                admitPcmContactExtents(
                    descriptors.x,descriptors.y,lane.x,endError);
                AllMemoryBarrierWithGroupSync();
                const uint error=preparationError(endError,lane.x);
                if(error!=0u) {
                    if(fused && lane.x==0u)publishTaskError(taskId,error);
                    return;
                }
            }
            prepareConstraints(descriptors.x,descriptors.y,lane.x,endError);
            if(!fused)return;
            AllMemoryBarrierWithGroupSync();
        }
        // Preparation failures belong to this task, not its solver or
        // integration output. Tiled phases observe the preceding dispatch.
        if(sourceDyTaskLoad(endError)!=0u) {
            if(fused && lane.x==0u)
                publishTaskError(taskId,sourceDyTaskLoad(endError));
            return;
        }

        const bool singleLaneTask=fused &&
            descriptors.y-descriptors.x==1u &&
            partitions.y-partitions.x==1u &&
            bodies.y-bodies.x<=2u;
        bool serialPartitions=false;
        if((fused || localRun) && partitions.y-partitions.x>=32u) {
            // A narrow, deeply dependent island may have thousands of
            // singleton colors. All of those writes occur in the same
            // lane, so per-color group barriers have no consumer.
            if(lane.x==0u) {
                uint begin=descriptors.x;
                dySingletonPartitions=1u;
                for(uint color=partitions.x;color<partitions.y;++color) {
                    const uint end=arena.Load(p.partitionBias+color*4u);
                    if(end-begin!=1u) {
                        dySingletonPartitions=0u;
                        break;
                    }
                    begin=end;
                }
            }
            AllMemoryBarrierWithGroupSync();
            serialPartitions=dySingletonPartitions!=0u;
        }
        const bool independent=fused && devicePartitions &&
            partitions.y-partitions.x==1u && descriptors.y>descriptors.x;
        if(independent || (fused && serialPartitions)) {
            if(independent &&
               arena.Load(p.partitionBias+partitions.x*4u)!=descriptors.y) {
                if(lane.x==0u) {
                    sourceDyTaskErrorOr(endError,4u);
                    publishTaskError(taskId,4u);
                }
                return;
            }
            // A fixed owner can finish a numerical phase before the
            // cooperative motion capture. Single colors own disjoint
            // endpoints per lane; singleton colors share lane zero.
            const uint total=iters.x+max(iters.y,1u);
            for(uint pass=0u;pass<2u;++pass) {
                const uint2 iterations=pass==0u?uint2(0u,iters.x):
                    uint2(iters.x,total);
                if(serialPartitions) {
                    if(lane.x==0u)
                        solveSerialRun(descriptors.x,descriptors.y,iters,
                                       iterations,endError,false);
                } else {
                    for(uint i=descriptors.x+lane.x;i<descriptors.y;i+=128u)
                        sourceDySolveExclusive(
                            arena.Load4(p.descriptorBias+i*16u),iters,iterations,
                            uint2(0u,0u),endError);
                }
                if(pass==0u) {
                    AllMemoryBarrierWithGroupSync();
                    for(uint i=bodies.x+lane.x;i<bodies.y;i+=128u)
                        saveMotion(i,endError);
                    AllMemoryBarrierWithGroupSync();
                }
            }
        } else if(fused || phase==partitionTile || phase==motionTile) {
            const uint total=iters.x+max(iters.y,1u);
            if(!fused && iter>=total)return;
            const uint endIteration=fused ? total : min(iter+1u,total);
            for(uint iteration=fused?0u:iter;iteration<endIteration;++iteration) {
                if(fused || phase==partitionTile) {
                    const bool captureMotion=residentWide && phase==partitionTile &&
                        iteration+1u==iters.x;
                    if(serialPartitions) {
                        if(lane.x==0u)
                            solveSerialRun(descriptors.x,descriptors.y,iters,
                                           uint2(iteration,iteration+1u),endError,captureMotion);
                    } else {
                        uint begin=descriptors.x;
                        uint partition=partitions.x;
                        uint end=descriptors.y;
                        if((fused || localRun) && partition<partitions.y)
                            end=arena.Load(p.partitionBias+partition*4u);
                        while(partition<partitions.y) {
                            // Consecutive singleton colors have the same
                            // lane owner and can stream in source order.
                            const bool singleton=end-begin==1u;
                            uint nextEnd=end;
                            ++partition;
                            if(fused || localRun) {
                                while(partition<partitions.y) {
                                    nextEnd=arena.Load(
                                        p.partitionBias+partition*4u);
                                    if(!singleton || nextEnd-end!=1u)break;
                                    end=nextEnd;
                                    ++partition;
                                }
                            }
                            if(singleton) {
                                if(lane.x==0u)
                                    solveSerialRun(begin,end,iters,
                                                   uint2(iteration,iteration+1u),
                                                   endError,captureMotion);
                            } else {
                                solvePartition(begin,end,lane.x,iters,
                                               iteration,endError,captureMotion);
                            }
                            // The final range hands off at integration
                            // or the dispatch boundary.
                            const bool nextPartition=partition<partitions.y;
                            const bool nextIteration=fused &&
                                iteration+1u<endIteration;
                            if(!singleLaneTask && (fused || localRun) &&
                               (nextPartition || nextIteration))
                                AllMemoryBarrierWithGroupSync();
                            begin=end;
                            end=nextEnd;
                        }
                    }
                }
                if((fused || phase==motionTile) && iteration+1u==iters.x) {
                    if(singleLaneTask) {
                        // Keep both snapshots with the numerical writer.
                        if(lane.x==0u)
                            for(uint i=bodies.x;i<bodies.y;++i)
                                saveMotion(i,endError);
                    } else {
                        for(uint i=bodies.x+lane.x;i<bodies.y;i+=128u)
                            saveMotion(i,endError);
                        if(fused)AllMemoryBarrierWithGroupSync();
                    }
                }
            }
            if(!fused)return;
        }
    }
    if(fused) {
        AllMemoryBarrierWithGroupSync();
        if(sourceDyTaskLoad(endError)!=0u) {
            if(lane.x==0u)
                publishTaskError(taskId,sourceDyTaskLoad(endError));
            return;
        }
    }
    if(fused || phase==integrationTile || freeBody) {
        for(uint i=bodies.x+lane.x;i<bodies.y;i+=128u) {
            uint ordinal=i;
            [branch]
            if(!freeBody) {
                ordinal=sourceDyBodyOrdinal(i);
                if(ordinal>=p.bodyCount) {
                    sourceDyTaskErrorOr(endError,8u);
                    continue;
                }
            }
            if(freeBody && ordinal<p.bodyCount) {
                const uint flags=arena.Load(p.integrationBias+52u+ordinal*40u);
                if((flags&1u)==0u ||
                   sourceDyRespondingSource(ordinal,sourceCounts,sourceHeaderBias))
                    continue;
                // Sleep-ready sources require an active accurate component.
                if((flags&0x8000u)!=0u) {
                    const uint root=scIslandWork.Load(
                        sourceHeaderBias+48u+ordinal*4u);
                    if(root>=p.bodyCount ||
                       scIslandWork.Load(sourceHeaderBias+sourceCounts+
                                         root*24u+8u)==0u) {
                        const uint destination=p.writebackBias+p.resultPatchOffset+
                            p.patchCount*4u+ordinal*32u;
                        arena.Store(destination+28u,root<p.bodyCount?2u:0u);
                        continue;
                    }
                }
            }
            integrateBody(ordinal,freeBody);
        }
    }
    if(fused) {
        AllMemoryBarrierWithGroupSync();
        if(lane.x==0u)publishTaskError(taskId,sourceDyTaskLoad(endError));
    }
}
