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
void solveJoint(uint4 desc,uint errorAddress,bool biased) {
    if(desc.x>=p.rowCount || desc.w>p.rowCount-desc.x ||
       (desc.y!=invalidBody && desc.y>=p.bodyCount) ||
       (desc.z!=invalidBody && desc.z>=p.bodyCount)) {
        arena.InterlockedOr(errorAddress,1u);
        return;
    }
    const uint ao=p.bodyBias+desc.y*52u;
    const uint bo=p.bodyBias+desc.z*52u;
    precise float3 va=0.0f,vb=0.0f,aa=0.0f,ab=0.0f;
    if(desc.y!=invalidBody) {
        va=asfloat(arena.Load3(ao));
        aa=asfloat(arena.Load3(ao+12u));
    }
    if(desc.z!=invalidBody) {
        vb=asfloat(arena.Load3(bo));
        ab=asfloat(arena.Load3(bo+12u));
    }
    const uint first=p.distanceBias+desc.x*64u;
    const float massA=loadScalar(first+36u);
    const float massB=loadScalar(first+40u);
    // One lane owns both endpoints for every row of this joint.
    for(uint row=0u;row<desc.w;++row) {
        const uint offset=first+row*64u;
        const float3 axis=asfloat(arena.Load3(offset));
        const float3 ja=asfloat(arena.Load3(offset+12u));
        const float3 jb=asfloat(arena.Load3(offset+24u));
        precise float normalDelta=
            sourceDyDot3(axis,va-vb)+sourceDyDot3(ja,aa)-
            sourceDyDot3(jb,ab);
        const float previous=loadScalar(offset+60u);
        precise float velocityTerm=normalDelta*loadScalar(offset+56u);
        precise float force=previous+velocityTerm;
        precise float candidate=force+
            loadScalar(offset+(biased?48u:52u));
        precise float next=clamp(candidate,-maxImpulse,maxImpulse);
        precise float impulse=next-previous;
        va+=axis*(impulse*massA);
        vb-=axis*(impulse*massB);
        aa+=ja*impulse;
        ab-=jb*impulse;
        if(!isfinite(next) || !all(isfinite(va)) || !all(isfinite(vb)) ||
           !all(isfinite(aa)) || !all(isfinite(ab))) {
            arena.InterlockedOr(errorAddress,2u);
            return;
        }
        arena.Store(offset+60u,asuint(next));
    }
    // Kinematic rows are shared read-only by disjoint solver tasks.
    if(desc.y!=invalidBody && massA>0.f) {
        arena.Store3(ao,asuint(va));
        arena.Store3(ao+12u,asuint(aa));
    }
    if(desc.z!=invalidBody && massB>0.f) {
        arena.Store3(bo,asuint(vb));
        arena.Store3(bo+12u,asuint(ab));
    }
}

// Native Dy reconstructs the mass frame from resident actor motion.
// Standalone inputs retain their cached-shape or explicit pose source.
bool sourceDyMassPose(uint input,out float3 position,out float4 quaternion) {
    const uint pose=arena.Load(input+56u);
    if((arena.Load(input+44u)&128u)!=0u) {
        const uint local=arena.Load(input+60u);
        const float3 actorPosition=asfloat(scActorLocals.Load3(local+116u));
        const float4 actorAxes=asfloat(scActorLocals.Load4(local+128u));
        const float3 massLocal=asfloat(scActorLocals.Load3(local));
        const float4 massAxes=asfloat(scActorLocals.Load4(local+12u));
        position=actorPosition+sourceRigidRotate(actorAxes,massLocal);
        quaternion=sourceRigidMultiply(actorAxes,massAxes);
    }else if(pose==invalidBody) {
        const uint4 views=arena.Load4(p.integrationBias);
        const uint element=arena.Load(input+48u);
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
        const uint local=arena.Load(input+60u);
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
    const uint flags=arena.Load(input+44u);
    const uint changed=(flags>>8u)&63u;
    const uint component=angular?12u:0u;
    if(changed==63u)
        return asfloat(arena.Load3(arena.Load(input+64u)+component));
    const uint local=arena.Load(input+60u);
    float3 velocity=asfloat(scActorLocals.Load3(local+64u+component));
    if(changed==0u)return velocity;
    const uint source=arena.Load(input+64u);
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
    if((arena.Load(input+44u)&32u)!=0u) {
        const uint local=arena.Load(input+60u);
        return asfloat(scActorLocals.Load3(local+64u+(angular?12u:0u)));
    }
    return sourceDyAuthoredVelocity(input,angular);
}
// Sparse per-step velocity and acceleration modifiers retain their
// original Sc ordering: velocity delta, then acceleration * dt.
float3 sourceDyMotionMod(uint input,uint stream) {
    const uint flags=arena.Load(input+44u);
    const uint mask=(flags>>16u)&4095u;
    float3 result=0.0f;
    if(mask==0u)return result;
    const uint source=arena.Load(input+68u);
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
    const uint input=p.integrationBias+40u+ordinal*76u;
    if((arena.Load(input+44u)&32u)==0u)return true;
    const uint local=arena.Load(input+60u);
    if(local==invalidBody)return false;
    const float dt=asfloat(arena.Load(input+40u));
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
    const uint locks=arena.Load(input+36u);
    if(locks&8u)angular.x=0.0f;
    if(locks&16u)angular.y=0.0f;
    if(locks&32u)angular.z=0.0f;
    if(!all(isfinite(velocityLinear)) || !all(isfinite(angular)))return false;
    scActorLocals.Store3(local+64u,asuint(velocityLinear));
    scActorLocals.Store3(local+76u,asuint(angular));
    return true;
}
void sourceDyInvalidatePreparedVelocity(uint ordinal) {
    const uint input=p.integrationBias+40u+ordinal*76u;
    const uint local=arena.Load(input+60u);
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
    const float dt=asfloat(arena.Load(input+40u));
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
        const float cluster=1.0f+float(arena.Load(input+72u));
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

#include "physx_source_dy_contact_pgs_native.hlsli"

// Sc owns task membership. One workgroup colors independent source
// constraints concurrently while preserving per-body greedy precedence.
// Static reservations follow the source descriptor encounter order.
groupshared uint dyNeeded,dyAssigned,dyWave,dyProgress,dyHighest;
groupshared uint dyNextWave,dySerial;
groupshared uint dySingletonPartitions;
groupshared uint dyHubA,dyHubB,dyHubMask;

// The resident source token has one authoritative decoding rule for
// colored and scalar tasks.
bool sourceDyExpandDescriptor(uint token,uint sourceCount,uint sourceTable,
                              uint errorAddress,out uint4 desc) {
    const uint kind=token>>30u;
    const uint index=token&0x3fffffffu;
    if(kind==3u) {
        if(index>=p.manifoldCount) {
            arena.InterlockedOr(errorAddress,1048576u);
            return false;
        }
        const uint manager=p.manifoldBias+index*20u;
        if(arena.Load(manager+4u)==0u) {
            desc=uint4(index,arena.Load(manager),
                       arena.Load(manager+8u),arena.Load(manager+16u));
        }else {
            const uint prepIndex=arena.Load(manager+16u);
            if(prepIndex>=p.prepCount) {
                arena.InterlockedOr(errorAddress,1048576u);
                return false;
            }
            const uint prep=p.prepBias+prepIndex*64u;
            const uint refA=arena.Load(prep);
            const uint refB=arena.Load(prep+4u);
            const bool dynamicA=(refA&1u)!=0u;
            const bool dynamicB=(refB&1u)!=0u;
            const uint ordinalA=dynamicA?(refA>>1u):
                arena.Load(p.prepBias+refA+76u);
            const uint ordinalB=dynamicB?(refB>>1u):
                arena.Load(p.prepBias+refB+76u);
            desc=uint4(index,ordinalA,ordinalB,
                       (dynamicA?0x100u:0u)|
                       (dynamicB?0x200u:0u));
        }
    }else if(kind==2u) {
        if(index>=p.rowCount) {
            arena.InterlockedOr(errorAddress,1048576u);
            return false;
        }
        const uint row=p.distanceBias+index*64u;
        desc=uint4(index,arena.Load(row),
                   arena.Load(row+4u),arena.Load(row+8u));
    }else if(kind==0u) {
        if(token>=sourceCount) {
            arena.InterlockedOr(errorAddress,1048576u);
            return false;
        }
        desc=arena.Load4(sourceTable+4u+token*16u);
    }else {
        arena.InterlockedOr(errorAddress,1048576u);
        return false;
    }
    return true;
}

void sourceDyPartitionTask(uint taskId,uint scratch,uint errorAddress,
                           uint lane) {
    const uint4 views=arena.Load4(scratch);
    const uint raw=scratch+views.x;
    const uint colors=scratch+views.y;
    const uint cursors=scratch+views.z;
    const uint bodyState=scratch+views.w;
    const uint task=p.taskBias+taskId*36u;
    const uint4 ranges=arena.Load4(task);
    const uint first=ranges.x,last=ranges.y;
    const uint partitionFirst=ranges.z,partitionCapacity=ranges.w;
    if(lane==0u &&
       (last<first || last>arena.Load(p.integrationBias+20u) ||
        partitionCapacity<partitionFirst ||
        partitionCapacity-partitionFirst!=last-first ||
        partitionCapacity>p.partitionCount))
        arena.InterlockedOr(errorAddress,1048576u);
    AllMemoryBarrierWithGroupSync();
    if(arena.Load(errorAddress)!=0u)return;
    const uint count=last-first;
    if(lane==0u) {
        dyNeeded=0u;dyAssigned=0u;dyWave=0u;dyHighest=0u;
        dyNextWave=0u;dySerial=0u;
    }
    AllMemoryBarrierWithGroupSync();
    // The native Sc path transfers only encounter-order manager/joint
    // identities. Expand source descriptors from the existing resident
    // contact prep and joint-row records before partition coloring.
    const bool sourceDescriptors=views.x==24u || views.x==44u;
    const uint sourceTable=bodyState+p.bodyCount*12u;
    const uint sourceCount=sourceDescriptors?arena.Load(sourceTable):0u;
    for(uint i=first+lane;i<last;i+=128u) {
        uint4 desc;
        if(sourceDescriptors) {
            if(!sourceDyExpandDescriptor(
                   arena.Load(colors+i*4u),sourceCount,sourceTable,
                   errorAddress,desc))
                continue;
            arena.Store4(raw+i*16u,desc);
        }else desc=arena.Load4(raw+i*16u);
        const bool da=(desc.w&0x100u)!=0u;
        const bool db=(desc.w&0x200u)!=0u;
        if((desc.w&~0x3ffu)!=0u || (!da && !db) ||
           (da && desc.y>=p.bodyCount) ||
           (db && desc.z>=p.bodyCount)) {
            arena.InterlockedOr(errorAddress,1048576u);
            continue;
        }
        // Constraint endpoints are authoritative even when a solver
        // task omits a body from its integration writeback ordinal list.
        uint unused;
        if(da) {
            const uint state=bodyState+desc.y*12u;
            arena.InterlockedExchange(state,0u,unused);
            arena.InterlockedExchange(state+4u,0u,unused);
            arena.InterlockedExchange(state+8u,0xffffffffu,unused);
        }
        if(db) {
            const uint state=bodyState+desc.z*12u;
            arena.InterlockedExchange(state,0u,unused);
            arena.InterlockedExchange(state+4u,0u,unused);
            arena.InterlockedExchange(state+8u,0xffffffffu,unused);
        }
        arena.Store(colors+i*4u,0xffffffffu);
        if(da && db)InterlockedAdd(dyNeeded,1u);
    }
    AllMemoryBarrierWithGroupSync();
    if(arena.Load(errorAddress)!=0u)return;
    if(count==1u) {
        // A single constraint owns its entire partition and requires
        // no conflict-mask coloring or cross-lane prefix scan.
        if(lane==0u) {
            uint4 desc=arena.Load4(raw+first*16u);
            desc.w&=255u;
            arena.Store4(p.descriptorBias+first*16u,desc);
            arena.Store(p.partitionBias+partitionFirst*4u,last);
            arena.Store(colors+first*4u,0u);
            arena.Store(cursors+partitionFirst*4u,last);
            arena.Store(task+12u,partitionFirst+1u);
        }
        return;
    }

    // A hub's dynamic/dynamic color is its encounter rank. Recognize
    // the common endpoint in parallel; when every descriptor is dynamic,
    // the rank is simply its source ordinal and needs no serial scan.
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
        uint common=3u;
        for(uint i=first+lane;i<last && common!=0u;i+=128u) {
            const uint4 desc=arena.Load4(raw+i*16u);
            if((desc.w&0x300u)!=0x300u)continue;
            if(desc.y!=dyHubA && desc.z!=dyHubA)common&=~1u;
            if(desc.y!=dyHubB && desc.z!=dyHubB)common&=~2u;
        }
        InterlockedAnd(dyHubMask,common);
        AllMemoryBarrierWithGroupSync();
        if(dyHubMask!=0u) {
            if(dyNeeded==count) {
                // Every source ordinal is one conflicting dynamic edge.
                // Each writes a unique color; body maxima are atomic
                // because the hub and spokes can be shared across lanes.
                for(uint i=first+lane;i<last;i+=128u) {
                    const uint4 desc=arena.Load4(raw+i*16u);
                    const uint color=i-first;
                    arena.Store(colors+i*4u,color);
                    uint unused;
                    arena.InterlockedMax(bodyState+desc.y*12u+4u,
                                         color+1u,unused);
                    arena.InterlockedMax(bodyState+desc.z*12u+4u,
                                         color+1u,unused);
                }
                if(lane==0u) {
                    dyHighest=count;
                    dyAssigned=count;
                }
            } else if(lane==0u) {
                // Static reservations follow the dynamic colors. Mixed
                // hubs retain their exact dynamic encounter ranks.
                uint color=0u;
                for(uint i=first;i<last;++i) {
                    const uint4 desc=arena.Load4(raw+i*16u);
                    if((desc.w&0x300u)!=0x300u)continue;
                    const uint a=bodyState+desc.y*12u;
                    const uint b=bodyState+desc.z*12u;
                    arena.Store(colors+i*4u,color);
                    arena.Store(a+4u,max(arena.Load(a+4u),color+1u));
                    arena.Store(b+4u,max(arena.Load(b+4u),color+1u));
                    ++color;
                }
                dyHighest=color;
                dyAssigned=color;
            }
        }
    }
    AllMemoryBarrierWithGroupSync();
    while(dyAssigned<dyNeeded) {
        // Masks are wave-local. A 32-color overflow retires the prior
        // mask before new absolute partition indices are assigned.
        if(dyNextWave!=0u) {
            for(uint i=first+lane;i<last;i+=128u) {
                if(arena.Load(colors+i*4u)!=0xffffffffu)continue;
                const uint4 desc=arena.Load4(raw+i*16u);
                if((desc.w&0x300u)!=0x300u)continue;
                uint unused;
                arena.InterlockedExchange(
                    bodyState+desc.y*12u,0u,unused);
                arena.InterlockedExchange(
                    bodyState+desc.z*12u,0u,unused);
            }
            AllMemoryBarrierWithGroupSync();
            if(lane==0u)dyNextWave=0u;
            AllMemoryBarrierWithGroupSync();
        }
        // A source-index priority preserves the original greedy PGS
        // partition order wherever two edges share a dynamic body.
        // Independent source edges can still color concurrently.
        for(uint i=first+lane;i<last;i+=128u) {
            if(arena.Load(colors+i*4u)!=0xffffffffu)continue;
            const uint4 desc=arena.Load4(raw+i*16u);
            if((desc.w&0x300u)!=0x300u)continue;
            uint unused;
            arena.InterlockedExchange(bodyState+desc.y*12u+8u,
                                       0xffffffffu,unused);
            arena.InterlockedExchange(bodyState+desc.z*12u+8u,
                                       0xffffffffu,unused);
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
        if(lane==0u)dyProgress=0u;
        AllMemoryBarrierWithGroupSync();
        for(uint i=first+lane;i<last;i+=128u) {
            if(arena.Load(colors+i*4u)!=0xffffffffu)continue;
            const uint4 desc=arena.Load4(raw+i*16u);
            if((desc.w&0x300u)!=0x300u)continue;
            const uint a=bodyState+desc.y*12u;
            const uint b=bodyState+desc.z*12u;
            const uint priority=i;
            if(arena.Load(a+8u)!=priority ||
               arena.Load(b+8u)!=priority)continue;
            const uint available=~(arena.Load(a)|arena.Load(b));
            if(available==0u)continue;
            const uint bit=firstbitlow(available);
            const uint value=dyWave+bit;
            if(value>=count) {
                arena.InterlockedOr(errorAddress,1048576u);
                continue;
            }
            const uint mask=1u<<bit;
            uint unused;
            arena.InterlockedOr(a,mask,unused);
            arena.InterlockedOr(b,mask,unused);
            arena.InterlockedMax(a+4u,value+1u,unused);
            arena.InterlockedMax(b+4u,value+1u,unused);
            arena.Store(colors+i*4u,value);
            InterlockedMax(dyHighest,value+1u);
            InterlockedAdd(dyProgress,1u);
        }
        AllMemoryBarrierWithGroupSync();
        if(lane==0u) {
            if(dyProgress==0u) {
                dyWave+=32u;
                dyNextWave=1u;
            } else {
                dyAssigned+=dyProgress;
                // A source-ordered chain exposes little independent
                // work. Continue its exact greedy prefix in one lane
                // instead of imposing a barrier for every next edge.
                if(dyNeeded>=256u && dyProgress*16u<dyNeeded)
                    dySerial=1u;
            }
            if(dyWave>=count && dyAssigned<dyNeeded)
                arena.InterlockedOr(errorAddress,1048576u);
        }
        AllMemoryBarrierWithGroupSync();
        if(arena.Load(errorAddress)!=0u)return;
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
                        arena.InterlockedOr(errorAddress,1048576u);
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
                if(arena.Load(errorAddress)!=0u)break;
                if(dyAssigned==dyNeeded)break;
                dyWave+=32u;
                if(dyWave>=count) {
                    arena.InterlockedOr(errorAddress,1048576u);
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
        if(arena.Load(errorAddress)!=0u)return;
    }
    for(uint i=first+lane;i<last;i+=128u) {
        const uint4 desc=arena.Load4(raw+i*16u);
        uint unused;
        if((desc.w&0x100u)!=0u)
            arena.InterlockedExchange(bodyState+desc.y*12u+8u,0u,unused);
        if((desc.w&0x200u)!=0u)
            arena.InterlockedExchange(bodyState+desc.z*12u+8u,0u,unused);
    }
    AllMemoryBarrierWithGroupSync();
    if(lane==0u) {
        uint highest=dyHighest;
        // Static/kinematic reservations retain source encounter order
        // after all dynamic/dynamic partitions have been committed.
        for(uint i=first;i<last;++i) {
            if(arena.Load(colors+i*4u)!=0xffffffffu)continue;
            const uint4 desc=arena.Load4(raw+i*16u);
            const uint state=bodyState+
                (((desc.w&0x100u)!=0u)?desc.y:desc.z)*12u;
            const uint value=arena.Load(state+4u)+arena.Load(state+8u);
            if(value>=count) {
                arena.InterlockedOr(errorAddress,1048576u);
                break;
            }
            arena.Store(state+8u,arena.Load(state+8u)+1u);
            arena.Store(colors+i*4u,value);
            highest=max(highest,value+1u);
        }
        dyHighest=highest;
    }
    AllMemoryBarrierWithGroupSync();
    if(arena.Load(errorAddress)!=0u)return;
    for(uint i=partitionFirst+lane;i<partitionCapacity;i+=128u)
        arena.Store(cursors+i*4u,0u);
    AllMemoryBarrierWithGroupSync();
    for(uint i=first+lane;i<last;i+=128u) {
        const uint color=arena.Load(colors+i*4u);
        const uint target=cursors+(partitionFirst+color)*4u;
        uint unused;
        arena.InterlockedAdd(target,1u,unused);
    }
    AllMemoryBarrierWithGroupSync();
    if(lane==0u) {
        uint end=first;
        for(uint i=partitionFirst;i<partitionFirst+dyHighest;++i) {
            const uint n=arena.Load(cursors+i*4u);
            arena.Store(cursors+i*4u,end);
            end+=n;
            arena.Store(p.partitionBias+i*4u,end);
        }
        if(end!=last)
            arena.InterlockedOr(errorAddress,1048576u);
        else
            arena.Store(task+12u,partitionFirst+dyHighest);
    }
    AllMemoryBarrierWithGroupSync();
    if(arena.Load(errorAddress)!=0u)return;
    // Within a color every descriptor owns distinct writable bodies.
    // Atomic scatter does not change solver dependencies or Gu indices.
    for(uint i=first+lane;i<last;i+=128u) {
        uint4 desc=arena.Load4(raw+i*16u);
        const uint target=cursors+
            (partitionFirst+arena.Load(colors+i*4u))*4u;
        uint offset;
        arena.InterlockedAdd(target,1u,offset);
        desc.w&=255u;
        arena.Store4(p.descriptorBias+offset*16u,desc);
    }
}

// Sc admits immutable joint local frames and non-dynamic authored poses.
// Active rigid endpoints read the same BodyId-resident actor state as PCM
// contact preparation and integrateCore. Dy constructs every 1D Jacobian
// before source-order PGS, with no host world-frame joint row dependency.
bool sourceDyJointActorFrame(uint ordinal,uint fallback,uint centerFallback,
                             out float3 actorPosition,out float4 actorAxes,
                             out float3 massCenter) {
    actorPosition=asfloat(arena.Load3(fallback));
    actorAxes=asfloat(arena.Load4(fallback+12u));
    massCenter=asfloat(arena.Load3(centerFallback));
    if(ordinal!=invalidBody) {
        const uint input=p.integrationBias+40u+ordinal*76u;
        if((arena.Load(input+44u)&128u)!=0u) {
            const uint local=arena.Load(input+60u);
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
    const uint input=p.integrationBias+40u+ordinal*76u;
    return asfloat(arena.Load3(input))*angular.x+
           asfloat(arena.Load3(input+12u))*angular.y+
           asfloat(arena.Load3(input+24u))*angular.z;
}
bool sourceDyPrepareJointGeometry(uint4 desc,uint endError) {
    const uint first=p.distanceBias+desc.x*64u;
    const uint source=arena.Load(first+60u);
    if(arena.Load(source+144u)!=arena.Load(first+48u) ||
       arena.Load(source+148u)!=arena.Load(first+52u)) {
        arena.InterlockedOr(endError,8192u);
        return false;
    }
    const uint kind=arena.Load(source+140u);
    const uint count=kind==0u?1u:kind==2u?3u:kind==1u?6u:5u;
    if(kind>4u || desc.w!=count) {
        arena.InterlockedOr(endError,8192u);
        return false;
    }
    const float3 localA=asfloat(arena.Load3(source));
    const float4 localQA=asfloat(arena.Load4(source+12u));
    const float3 localB=asfloat(arena.Load3(source+28u));
    const float4 localQB=asfloat(arena.Load4(source+40u));
    // Fallback actor frames and mass centers are borrowed authored state.
    // Dynamic endpoints consume the resident actor frame instead.
    float3 pA,pB,cA,cB;
    float4 qA,qB;
    if(!sourceDyJointActorFrame(desc.y,source+56u,source+112u,pA,qA,cA) ||
       !sourceDyJointActorFrame(desc.z,source+84u,source+124u,pB,qB,cB)) {
        arena.InterlockedOr(endError,8192u);
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
    const uint inputA=p.integrationBias+40u+desc.y*76u;
    const uint inputB=p.integrationBias+40u+desc.z*76u;
    const float3 vA=desc.y==invalidBody?0.0f:sourceDyPreparedVelocity(inputA,false);
    const float3 vB=desc.z==invalidBody?0.0f:sourceDyPreparedVelocity(inputB,false);
    const float3 omegaA=desc.y==invalidBody?0.0f:sourceDyPreparedVelocity(inputA,true);
    const float3 omegaB=desc.z==invalidBody?0.0f:sourceDyPreparedVelocity(inputB,true);
    const uint dtInput=mass0>0.0f?inputA:inputB;
    const float dt=asfloat(arena.Load(dtInput+40u));
    if(!(dt>0.0f) || !isfinite(dt)) {
        arena.InterlockedOr(endError,8192u);
        return false;
    }
    const float invDt=1.0f/dt;
    float3 linearRows[6],angularA[6],angularB[6],jacA[6],jacB[6];
    float invResponse[6],projectedError[6];
    const float requiredDistance=asfloat(arena.Load(source+136u));
    for(uint i=0u;i<count;++i) {
        float3 axis=0.0f,angA=0.0f,angB=0.0f;
        float error=0.0f;
        if(kind==0u) {
            precise float3 offset=((cA+rA)-cB)-rB;
            const float distance=sqrt(sourceDyDot3(offset,offset));
            if(!isfinite(distance)) {
                arena.InterlockedOr(endError,8192u);
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
            arena.InterlockedOr(endError,8192u);
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
            arena.InterlockedOr(endError,8192u);
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
void prepareJointRows(uint4 desc,uint lane,uint endError) {
    if(desc.x>=p.rowCount || desc.w>p.rowCount-desc.x ||
       (desc.y!=invalidBody && desc.y>=p.bodyCount) ||
       (desc.z!=invalidBody && desc.z>=p.bodyCount)) {
        arena.InterlockedOr(endError,4096u);
        return;
    }
    const uint massRow=p.distanceBias+desc.x*64u;
    if(arena.Load(massRow+44u)==2u) {
        if(lane==0u)sourceDyPrepareJointGeometry(desc,endError);
        return;
    }
    const float inverseMassA=asfloat(arena.Load(massRow+36u));
    const float inverseMassB=asfloat(arena.Load(massRow+40u));
    const uint inputA=p.integrationBias+40u+desc.y*76u;
    const uint inputB=p.integrationBias+40u+desc.z*76u;
    float3 va=0.0f,vb=0.0f,wa=0.0f,wb=0.0f;
    if(desc.y!=invalidBody) {
        va=sourceDyPreparedVelocity(inputA,false);
        wa=sourceDyPreparedVelocity(inputA,true);
    }
    if(desc.z!=invalidBody) {
        vb=sourceDyPreparedVelocity(inputB,false);
        wb=sourceDyPreparedVelocity(inputB,true);
    }
    for(uint i=lane;i<desc.w;i+=4u) {
        const uint row=p.distanceBias+(desc.x+i)*64u;
        if(arena.Load(row+44u)!=1u)continue;
        const float3 axis=asfloat(arena.Load3(row));
        const float3 angularA=asfloat(arena.Load3(row+12u));
        const float3 angularB=asfloat(arena.Load3(row+24u));
        float3 ja=0.0f,jb=0.0f;
        if(inverseMassA>0.0f) {
            ja=asfloat(arena.Load3(inputA))*angularA.x+
               asfloat(arena.Load3(inputA+12u))*angularA.y+
               asfloat(arena.Load3(inputA+24u))*angularA.z;
        }
        if(inverseMassB>0.0f) {
            jb=asfloat(arena.Load3(inputB))*angularB.x+
               asfloat(arena.Load3(inputB+12u))*angularB.y+
               asfloat(arena.Load3(inputB+24u))*angularB.z;
        }
        precise float speed=sourceDyDot3(axis,va-vb)+
            sourceDyDot3(angularA,wa)-sourceDyDot3(angularB,wb);
        const float multiplier=asfloat(arena.Load(row+56u));
        const float errorRate=asfloat(arena.Load(row+48u));
        const float constant=multiplier*(errorRate+speed);
        const float unbiased=multiplier*speed;
        if(!isfinite(constant) || !isfinite(unbiased) ||
           !all(isfinite(ja)) || !all(isfinite(jb))) {
            arena.InterlockedOr(endError,8192u);
            continue;
        }
        arena.Store3(row+12u,asuint(ja));
        arena.Store3(row+24u,asuint(jb));
        arena.Store(row+48u,asuint(constant));
        arena.Store(row+52u,asuint(unbiased));
        arena.Store(row+44u,0u);
    }
}

void prepareNormals(uint begin,uint end,uint lane,uint endError) {
    for(uint i=begin+(lane>>2u);i<end;i+=32u) {
        uint4 desc=arena.Load4(p.descriptorBias+i*16u);
        if(desc.w!=0u) {
            prepareJointRows(desc,lane&3u,endError);
            continue;
        }
        if(desc.x>=p.manifoldCount) {
            arena.InterlockedOr(endError,4096u);
            continue;
        }
        uint mo=p.manifoldBias+desc.x*20u;
        uint4 manager=arena.Load4(mo);
        if(manager.y==0u)continue;
        uint pairIndex=arena.Load(mo+16u);
        if(pairIndex>=p.prepCount ||
           manager.x>p.patchCount ||
           manager.y>p.patchCount-manager.x ||
           manager.z>p.contactCount ||
           manager.w>p.contactCount-manager.z) {
            arena.InterlockedOr(endError,4096u);
            continue;
        }
        uint covered=0u;
        for(uint patchIndex=manager.x;patchIndex<manager.x+manager.y;
            ++patchIndex) {
            const uint po=p.patchBias+patchIndex*32u;
            const uint count=arena.Load(po+4u);
            const uint order=arena.Load(po+24u);
            if(covered>manager.w || count>manager.w-covered ||
               !sourceDyContactOrderValid(order,count)) {
                arena.InterlockedOr(endError,4096u);
                break;
            }
            for(uint c=lane&3u;c<count;c+=4u) {
                const uint index=sourceDyContactOrderPoint(order,c);
                prepareContactRow(index,pairIndex,manager.z,manager.w,endError);
            }
            covered+=count;
        }
        if(covered!=manager.w)
            arena.InterlockedOr(endError,4096u);
    }
}

void prepareFriction(uint begin,uint end,uint lane,uint endError) {
    for(uint i=begin+(lane>>2u);i<end;i+=32u) {
        uint4 desc=arena.Load4(p.descriptorBias+i*16u);
        if(desc.w!=0u)continue;
        if(desc.x>=p.manifoldCount)continue;
        uint mo=p.manifoldBias+desc.x*20u;
        uint4 manager=arena.Load4(mo);
        if(manager.y==0u)continue;
        uint pairIndex=arena.Load(mo+16u);
        if(pairIndex>=p.prepCount ||
           manager.x>p.patchCount ||
           manager.y>p.patchCount-manager.x ||
           manager.z>p.contactCount ||
           manager.w>p.contactCount-manager.z)
            continue;
        for(uint patchIndex=manager.x+(lane&3u);
            patchIndex<manager.x+manager.y;patchIndex+=4u) {
            uint4 patch=arena.Load4(p.patchBias+patchIndex*32u);
            if(patch.x>p.frictionCount || patch.z>2u ||
               patch.z*2u>p.frictionCount-patch.x) {
                arena.InterlockedOr(endError,16384u);
                continue;
            }
            prepareFrictionPatch(patchIndex,pairIndex,endError);
        }
    }
}

void solvePartition(uint begin,uint end,uint lane,uint2 iters,
                    uint iter,uint endError) {
    const uint total=iters.x+max(iters.y,1u);
    const bool biased=iter<iters.x;
    const bool finalPass=iter+1u==total;
    const bool friction=!biased || iters.x-iter<=3u;
    for(uint i=begin+lane;i<end;i+=128u) {
        const uint4 desc=arena.Load4(p.descriptorBias+i*16u);
        if(desc.w!=0u)
            solveJoint(desc,endError,biased);
        else
            solveContact(desc,endError,biased,friction,finalPass);
    }
}

void saveMotion(uint index,uint endError) {
    const uint ordinal=arena.Load(p.ordinalBias+index*4u);
    if(ordinal>=p.bodyCount) {
        arena.InterlockedOr(endError,8u);
        return;
    }
    const uint base=p.bodyBias+ordinal*52u;
    arena.Store3(base+24u,arena.Load3(base));
    arena.Store3(base+36u,arena.Load3(base+12u));
}

void integrateBody(uint ordinal) {
    float3 linearDelta=0.0f;
    float3 angularState=0.0f;
    const uint input=p.integrationBias+40u+ordinal*76u;
    const uint inputFlags=arena.Load(input+44u);
    [branch]
    if(ordinal<p.bodyCount) {
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
    if(ordinal>=p.bodyCount && (inputFlags&32u)!=0u &&
       !sourceDyPreIntegrateBody(ordinal)) {
        sourceDyInvalidatePreparedVelocity(ordinal);
        return;
    }
    const float3 baseLinear=sourceDyPreparedVelocity(input,false);
    const float3 baseAngular=sourceDyPreparedVelocity(input,true);
    if(!all(isfinite(baseLinear)) || !all(isfinite(baseAngular)))return;
    const uint locks=arena.Load(input+36u);
    const float dt=asfloat(arena.Load(input+40u));
    float3 position;
    float4 quaternion;
    if(!sourceDyMassPose(input,position,quaternion))return;
    float3 motionLinear=baseLinear+linearDelta;
    if(locks&1u)motionLinear.x=0.0f;
    if(locks&2u)motionLinear.y=0.0f;
    if(locks&4u)motionLinear.z=0.0f;
    if(locks&8u)angularState.x=0.0f;
    if(locks&16u)angularState.y=0.0f;
    if(locks&32u)angularState.z=0.0f;
    const float3 inertiaX=asfloat(arena.Load3(input));
    const float3 inertiaY=asfloat(arena.Load3(input+12u));
    const float3 inertiaZ=asfloat(arena.Load3(input+24u));
    const float3 angular=baseAngular+float3(
        sourceDyDot3(inertiaX,angularState),
        sourceDyDot3(inertiaY,angularState),
        sourceDyDot3(inertiaZ,angularState));
    float3 finalLinear=motionLinear;
    float3 finalAngular=baseAngular;
    if(ordinal<p.bodyCount && (inputFlags&8u)!=0u) {
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
    // integrateCore advances the principal mass frame. Recover the
    // actor frame before composing each independent local shape pose.
    const uint local=arena.Load(input+60u);
    const float3 localCenter=asfloat(scActorLocals.Load3(local));
    const float4 localAxes=asfloat(scActorLocals.Load4(local+12u));
    nextQuaternion=sourceRigidMultiply(nextQuaternion,float4(-localAxes.xyz,localAxes.w));
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
    uint element=arena.Load(input+48u);
    const uint shapeCount=arena.Load(input+52u);
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

static const uint fusedTask=0u, normalTile=1u, frictionTile=2u;
static const uint partitionTile=3u, motionTile=4u, integrationTile=5u;
static const uint publishErrors=6u, freeBodyTile=7u, colorTile=8u;
static const uint initializeBodiesTile=9u;

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
    const uint input=p.integrationBias+40u+ordinal*76u;
    // Bit 14 does not overlap velocity changes [8:13] or force mods [16:27].
    if((arena.Load(input+44u)&0x4000u)==0u)return true;
    const uint local=arena.Load(input+60u);
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
    arena.Store3(input,asuint(c0));
    arena.Store3(input+12u,asuint(c1));
    arena.Store3(input+24u,asuint(c2));
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
        const uint input=p.integrationBias+40u+ordinal*76u;
        const uint local=arena.Load(input+60u);
        if(local!=invalidBody)
            massBits=scActorLocals.Load(local+28u);
    }else
        massBits=arena.Load(source+ordinal*4u);
    arena.Store(target+48u,massBits);
}

void publishTaskError(uint task,uint error) {
    arena.Store(p.writebackBias+p.resultTaskOffset+task*4u,error);
}

// A packed lane owns one independent body/constraint pair. Source Sc
// selects this dispatch only when every admitted task has this shape.
void sourceDySolvePackedTask(uint taskId) {
    const uint task=p.taskBias+taskId*36u;
    const uint4 ranges=arena.Load4(task);
    const uint2 bodies=arena.Load2(task+16u);
    const uint2 iters=arena.Load2(task+24u);
    const uint errorAddress=task+32u;
    const uint4 schedule=arena.Load4(p.integrationBias+16u);
    if(ranges.y!=ranges.x+1u || ranges.w!=ranges.z+1u ||
       bodies.y!=bodies.x+1u ||
       ranges.y>schedule.y || ranges.w>p.partitionCount ||
       bodies.y>schedule.x ||
       !iters.x || iters.x>255u || iters.y>255u) {
        arena.InterlockedOr(errorAddress,1048576u);
        publishTaskError(taskId,arena.Load(errorAddress));
        return;
    }
    const uint scratch=schedule.w;
    const uint4 views=arena.Load4(scratch);
    const uint colors=scratch+views.y;
    const uint bodyState=scratch+views.w;
    const uint sourceTable=bodyState+p.bodyCount*12u;
    uint4 desc;
    if(!sourceDyExpandDescriptor(arena.Load(colors+ranges.x*4u),
                                 arena.Load(sourceTable),sourceTable,
                                 errorAddress,desc)) {
        publishTaskError(taskId,arena.Load(errorAddress));
        return;
    }
    const bool da=(desc.w&0x100u)!=0u;
    const bool db=(desc.w&0x200u)!=0u;
    if((desc.w&~0x3ffu)!=0u || (!da && !db) ||
       (da && desc.y>=p.bodyCount) ||
       (db && desc.z>=p.bodyCount)) {
        arena.InterlockedOr(errorAddress,1048576u);
        publishTaskError(taskId,arena.Load(errorAddress));
        return;
    }
    // The singleton's ranges already define its only partition.
    // No conflict-color state is needed for its exclusive solver row.
    desc.w&=255u;
    arena.Store4(p.descriptorBias+ranges.x*16u,desc);

    // Four scalar visits preserve the source contact/joint row
    // assignment of the existing four lanes per descriptor.
    [unroll]
    for(uint lane=0u;lane<4u;++lane)
        prepareNormals(ranges.x,ranges.y,lane,errorAddress);
    [unroll]
    for(uint lane=0u;lane<4u;++lane)
        prepareFriction(ranges.x,ranges.y,lane,errorAddress);

    const uint total=iters.x+max(iters.y,1u);
    for(uint iteration=0u;iteration<total;++iteration) {
        solvePartition(ranges.x,ranges.y,0u,iters,iteration,errorAddress);
        if(iteration+1u==iters.x)
            saveMotion(bodies.x,errorAddress);
    }
    const uint ordinal=arena.Load(p.ordinalBias+bodies.x*4u);
    if(ordinal>=p.bodyCount)
        arena.InterlockedOr(errorAddress,8u);
    else
        integrateBody(ordinal);
    publishTaskError(taskId,arena.Load(errorAddress));
}

[numthreads(128,1,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint phase=p.launchControl&15u;
    const bool devicePartitions=(p.launchControl&0x80000000u)!=0u;
    const bool precolored=(p.launchControl&0x40000000u)!=0u;
    const uint iter=(p.launchControl&0x1fffffffu)>>4u;
    const uint index=p.launchBegin+group.x;
    uint admittedTasks=0xffffffffu;
    // A submitted Sc island graph can differ from CPU-predicted counts
    // because contact activation is owned by resident Gu history. Reject
    // the entire numerical epoch before initializing any solver body.
    // The task error report still runs, including compact-only scenes.
    if(devicePartitions) {
        const uint4 schedule=arena.Load4(p.integrationBias+16u);
        if(schedule.w!=0u && arena.Load(schedule.w)==44u) {
            const uint4 header=arena.Load4(schedule.w+24u);
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
        }
    }
    // The Sc prefix reduces the indirect launch only when every
    // resident task is a singleton. One GPU lane then owns its
    // complete PGS epoch without workgroup-wide barriers.
    if(phase==fusedTask && devicePartitions && !precolored &&
       admittedTasks>1u &&
       arena.Load(p.integrationBias+28u)!=0u &&
       arena.Load(arena.Load(p.integrationBias+28u))==44u &&
       arena.Load(arena.Load(p.integrationBias+28u)+40u)<admittedTasks) {
        const uint task=index*128u+lane.x;
        if(task<admittedTasks)sourceDySolvePackedTask(task);
        return;
    }
    if(phase==initializeBodiesTile) {
        // Joint participants include nonresponding kinematic solver
        // rows. They require prepared velocities even though no task
        // integrates them or writes back their actor pose.
        const uint ordinal=index*128u+lane.x;
        if(ordinal<p.bodyCount) {
            initializeSolverBody(ordinal,arena.Load(p.integrationBias+36u));
            if(!sourceDyPreIntegrateBody(ordinal) ||
               !sourceDyPrepareWorldInertia(ordinal))
                sourceDyInvalidatePreparedVelocity(ordinal);
        }
        return;
    }
    const bool freeBody=phase==freeBodyTile;
    const bool fused=phase==fusedTask;
    uint taskId=0u,endError=0u;
    uint2 bodies;
    [branch]
    if(freeBody) {
        const uint begin=p.bodyCount+index*128u;
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
                if(lane.x<p.bodyCount) {
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
                    ?arena.Load(p.taskBias+task*36u+32u):0u);
            return;
        }
        if(index>=admittedTasks)return;

        // A frame containing only compact tasks uses direct task indices.
        uint4 tile=uint4(index,0u,0u,0u);
        [branch]
        if(devicePartitions) {
            const uint tileOffset=arena.Load(schedule.w+16u);
            if(phase!=colorTile && tileOffset!=0u)
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
            if(arena.Load(endError)!=0u) {
                if(lane.x==0u)
                    publishTaskError(taskId,arena.Load(endError));
                return;
            }
            if(phase==colorTile)return;
        }
        const uint2 iters=arena.Load2(t+24u);
        uint2 descriptors=tile.yz;
        bodies=tile.yz;
        uint2 partitions=uint2(0u,1u);
        if(devicePartitions && phase==partitionTile) {
            const uint4 ranges=arena.Load4(t);
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
            const uint4 ranges=arena.Load4(t);
            descriptors=ranges.xy;
            partitions=ranges.zw;
            bodies=arena.Load2(t+16u);
            if(descriptors.x>descriptors.y || descriptors.y>schedule.y ||
               partitions.x>partitions.y || partitions.y>p.partitionCount ||
               bodies.x>bodies.y || bodies.y>schedule.x) {
                if(lane.x==0u) {
                    arena.InterlockedOr(endError,4u);
                    publishTaskError(tile.x,4u);
                }
                return;
            }
        }
        // Each phase has one shader call site. Fused groups use workgroup
        // dependencies, while tiled launches use the command stream's
        // device-wide dependencies between the same phases.
        if(fused || phase==normalTile) {
            prepareNormals(descriptors.x,descriptors.y,lane.x,endError);
            if(!fused)return;
            AllMemoryBarrierWithGroupSync();
        }
        if(fused || phase==frictionTile) {
            prepareFriction(descriptors.x,descriptors.y,lane.x,endError);
            if(!fused)return;
            AllMemoryBarrierWithGroupSync();
        }

        const bool singleLaneTask=fused &&
            descriptors.y-descriptors.x==1u &&
            partitions.y-partitions.x==1u &&
            bodies.y-bodies.x==1u;
        bool serialPartitions=false;
        if(fused && partitions.y-partitions.x>=32u) {
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
        if(fused || phase==partitionTile || phase==motionTile) {
            const uint total=iters.x+max(iters.y,1u);
            if(!fused && iter>=total)return;
            const uint endIteration=fused ? total : min(iter+1u,total);
            for(uint iteration=fused?0u:iter;iteration<endIteration;++iteration) {
                if(fused || phase==partitionTile) {
                    if(serialPartitions) {
                        if(lane.x==0u) {
                            uint begin=descriptors.x;
                            for(uint partition=partitions.x;
                                partition<partitions.y;++partition) {
                                const uint end=arena.Load(
                                    p.partitionBias+partition*4u);
                                solvePartition(begin,end,0u,iters,
                                               iteration,endError);
                                begin=end;
                            }
                        }
                        AllMemoryBarrierWithGroupSync();
                    } else {
                        uint begin=descriptors.x;
                        for(uint partition=partitions.x;
                            partition<partitions.y;++partition) {
                            uint end=descriptors.y;
                            if(fused)
                                end=arena.Load(p.partitionBias+partition*4u);
                            solvePartition(begin,end,lane.x,iters,
                                           iteration,endError);
                            if(fused && !singleLaneTask)
                                AllMemoryBarrierWithGroupSync();
                            begin=end;
                        }
                    }
                }
                if((fused || phase==motionTile) && iteration+1u==iters.x) {
                    for(uint i=bodies.x+lane.x;i<bodies.y;i+=128u)
                        saveMotion(i,endError);
                    if(fused && !singleLaneTask)
                        AllMemoryBarrierWithGroupSync();
                }
            }
            if(!fused)return;
        }
    }
    if(fused || phase==integrationTile || freeBody) {
        for(uint i=bodies.x+lane.x;i<bodies.y;i+=128u) {
            uint ordinal=i;
            [branch]
            if(!freeBody) {
                ordinal=arena.Load(p.ordinalBias+i*4u);
                if(ordinal>=p.bodyCount) {
                    arena.InterlockedOr(endError,8u);
                    continue;
                }
            }
            integrateBody(ordinal);
        }
    }
    if(fused) {
        AllMemoryBarrierWithGroupSync();
        if(lane.x==0u)publishTaskError(taskId,arena.Load(endError));
    }
}
