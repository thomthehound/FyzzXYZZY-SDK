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
#ifdef FYZZXYZZY_CONVEX
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer scHullVertices : register(u5,space0);
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
    float3 va=0.0f,vb=0.0f,aa=0.0f,ab=0.0f;
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
        const float normalDelta=dot(axis,va-vb)+dot(ja,aa)-dot(jb,ab);
        const float previous=loadScalar(offset+60u);
        const float next=clamp(previous+
            normalDelta*loadScalar(offset+56u)+
            loadScalar(offset+(biased?48u:52u)),-maxImpulse,maxImpulse);
        const float impulse=next-previous;
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

// Sc's cached ElementSim pose is the common mass-frame source for
// device contact preparation, friction correlation, and integrateCore.
bool sourceDyMassPose(uint input,out float3 position,out float4 quaternion) {
    const uint poseIndex=arena.Load(input+108u);
    if(poseIndex==invalidBody) {
        const uint4 views=arena.Load4(p.integrationBias);
        const uint element=arena.Load(input+72u);
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
        const float3 centerLocal=asfloat(arena.Load3(input+76u));
        const float4 massLocalAxes=asfloat(arena.Load4(input+88u));
        quaternion=sourceRigidMultiply(actorAxes,massLocalAxes);
        position=actorPosition+sourceRigidRotate(actorAxes,centerLocal);
    }else {
        const uint count=arena.Load(p.integrationBias+32u);
        if(poseIndex>=count)return false;
        const uint pose=p.integrationBias+36u+count*112u+poseIndex*28u;
        position=asfloat(arena.Load3(pose));
        quaternion=asfloat(arena.Load4(pose+12u));
    }
    return all(isfinite(position)) && all(isfinite(quaternion));
}

#include "physx_source_dy_contact_pgs_native.hlsli"

// Each original Sc solver task is isolated by accurate-island body
// ownership. A single lane assigns source-order, conflict-free
// dynamic partitions, followed by static-contact reservations. The
// same group consumes the compacted partitions in Dy PGS immediately.
// Scratch is transient to this solve; Sc remains the only graph owner.
void sourceDyPartitionTask(uint taskId,uint scratch,uint errorAddress) {
    const uint4 views=arena.Load4(scratch);
    const uint raw=scratch+views.x;
    const uint colors=scratch+views.y;
    const uint cursors=scratch+views.z;
    const uint bodyState=scratch+views.w;
    const uint task=p.taskBias+taskId*36u;
    const uint4 ranges=arena.Load4(task);
    const uint first=ranges.x,last=ranges.y;
    const uint partitionFirst=ranges.z,partitionCapacity=ranges.w;
    if(last<first || last>arena.Load(p.integrationBias+20u) ||
       partitionCapacity<partitionFirst ||
       partitionCapacity-partitionFirst!=last-first ||
       partitionCapacity>p.partitionCount) {
        arena.InterlockedOr(errorAddress,1048576u);
        return;
    }
    const uint count=last-first;
    for(uint i=first;i<last;++i) {
        const uint4 desc=arena.Load4(raw+i*16u);
        const bool dynamicA=(desc.w&0x100u)!=0u;
        const bool dynamicB=(desc.w&0x200u)!=0u;
        if((desc.w&~0x3ffu)!=0u ||
           (!dynamicA && !dynamicB) ||
           (dynamicA && desc.y>=p.bodyCount) ||
           (dynamicB && desc.z>=p.bodyCount)) {
            arena.InterlockedOr(errorAddress,1048576u);
            return;
        }
        if(dynamicA)arena.Store3(bodyState+desc.y*12u,uint3(0u,0u,0u));
        if(dynamicB)arena.Store3(bodyState+desc.z*12u,uint3(0u,0u,0u));
        arena.Store(colors+i*4u,0xffffffffu);
    }
    uint assigned=0u,wave=0u,highest=0u;
    uint needed=0u;
    for(uint i=first;i<last;++i) {
        const uint flags=arena.Load(raw+i*16u+12u)&0x300u;
        if(flags==0x300u)++needed;
    }
    while(assigned<needed) {
        for(uint i=first;i<last;++i) {
            const uint4 desc=arena.Load4(raw+i*16u);
            if((desc.w&0x300u)==0x300u) {
                arena.Store(bodyState+desc.y*12u,0u);
                arena.Store(bodyState+desc.z*12u,0u);
            }
        }
        const uint old=assigned;
        for(uint i=first;i<last;++i) {
            if(arena.Load(colors+i*4u)!=0xffffffffu)continue;
            const uint4 desc=arena.Load4(raw+i*16u);
            if((desc.w&0x300u)!=0x300u)continue;
            const uint a=bodyState+desc.y*12u;
            const uint b=bodyState+desc.z*12u;
            const uint available=~(arena.Load(a)|arena.Load(b));
            if(available==0u)continue;
            const uint bit=firstbitlow(available);
            const uint value=wave+bit;
            if(value>=count) {
                arena.InterlockedOr(errorAddress,1048576u);
                return;
            }
            const uint mask=1u<<bit;
            arena.Store(a,arena.Load(a)|mask);
            arena.Store(b,arena.Load(b)|mask);
            arena.Store(a+4u,max(arena.Load(a+4u),value+1u));
            arena.Store(b+4u,max(arena.Load(b+4u),value+1u));
            arena.Store(colors+i*4u,value);
            highest=max(highest,value+1u);
            ++assigned;
        }
        if(assigned==old || wave>count) {
            arena.InterlockedOr(errorAddress,1048576u);
            return;
        }
        wave+=32u;
    }
    // Static and kinematic endpoints never contribute a conflicting
    // dynamic write. Each dynamic body reserves its static contacts
    // after its last dynamic/dynamic color, in source encounter order.
    for(uint i=first;i<last;++i) {
        if(arena.Load(colors+i*4u)!=0xffffffffu)continue;
        const uint4 desc=arena.Load4(raw+i*16u);
        const bool dynamicA=(desc.w&0x100u)!=0u;
        const uint ordinal=dynamicA?desc.y:desc.z;
        const uint state=bodyState+ordinal*12u;
        const uint value=arena.Load(state+4u)+arena.Load(state+8u);
        if(value>=count) {
            arena.InterlockedOr(errorAddress,1048576u);
            return;
        }
        arena.Store(state+8u,arena.Load(state+8u)+1u);
        arena.Store(colors+i*4u,value);
        highest=max(highest,value+1u);
    }
    for(uint i=partitionFirst;i<partitionCapacity;++i)
        arena.Store(cursors+i*4u,0u);
    for(uint i=first;i<last;++i) {
        const uint color=arena.Load(colors+i*4u);
        const uint target=cursors+(partitionFirst+color)*4u;
        arena.Store(target,arena.Load(target)+1u);
    }
    uint end=first;
    for(uint i=partitionFirst;i<partitionFirst+highest;++i) {
        const uint n=arena.Load(cursors+i*4u);
        arena.Store(cursors+i*4u,end);
        end+=n;
        arena.Store(p.partitionBias+i*4u,end);
    }
    if(end!=last) {
        arena.InterlockedOr(errorAddress,1048576u);
        return;
    }
    for(uint i=first;i<last;++i) {
        uint4 desc=arena.Load4(raw+i*16u);
        const uint target=cursors+
            (partitionFirst+arena.Load(colors+i*4u))*4u;
        const uint offset=arena.Load(target);
        arena.Store(target,offset+1u);
        desc.w&=255u;
        arena.Store4(p.descriptorBias+offset*16u,desc);
    }
    arena.Store(task+12u,partitionFirst+highest);
}

void prepareNormals(uint begin,uint end,uint lane,uint endError) {
    for(uint i=begin+(lane>>2u);i<end;i+=32u) {
        uint4 desc=arena.Load4(p.descriptorBias+i*16u);
        if(desc.w!=0u)continue;
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
           manager.z>p.orderCount ||
           manager.w>p.orderCount-manager.z) {
            arena.InterlockedOr(endError,4096u);
            continue;
        }
        for(uint c=lane&3u;c<manager.w;c+=4u) {
            uint index=arena.Load(p.orderBias+(manager.z+c)*4u);
            prepareContactRow(index,pairIndex,manager.z,manager.w,endError);
        }
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
           manager.z>p.orderCount ||
           manager.w>p.orderCount-manager.z)
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
    [branch]
    if(ordinal<p.bodyCount) {
        const uint from=p.bodyBias+ordinal*52u;
        const uint to=p.writebackBias+p.resultBodyOffset+ordinal*48u;
        arena.Store4(to,arena.Load4(from));
        arena.Store4(to+16u,arena.Load4(from+16u));
        arena.Store4(to+32u,arena.Load4(from+32u));
        linearDelta=asfloat(arena.Load3(from+24u));
        angularState=asfloat(arena.Load3(from+36u));
    }
    // Source DyBodyCoreIntegrator::integrateCore uses saved
    // position-iteration deltas, not the final velocity deltas.
    const uint input=p.integrationBias+36u+ordinal*112u;
    if(arena.Load(input+68u)==0u)return;
    const float3 baseLinear=asfloat(arena.Load3(input));
    const float3 baseAngular=asfloat(arena.Load3(input+12u));
    const uint locks=arena.Load(input+60u);
    const float dt=asfloat(arena.Load(input+64u));
    float3 position;
    float4 quaternion;
    if(!sourceDyMassPose(input,position,quaternion))return;
    const uint destination=p.writebackBias+p.resultPatchOffset+
        p.patchCount*4u+ordinal*32u;
    arena.Store(destination+28u,0u);
    float3 motionLinear=baseLinear+linearDelta;
    if(locks&1u)motionLinear.x=0.0f;
    if(locks&2u)motionLinear.y=0.0f;
    if(locks&4u)motionLinear.z=0.0f;
    if(locks&8u)angularState.x=0.0f;
    if(locks&16u)angularState.y=0.0f;
    if(locks&32u)angularState.z=0.0f;
    const float3 angular=baseAngular+float3(
        dot(asfloat(arena.Load3(input+24u)),angularState),
        dot(asfloat(arena.Load3(input+36u)),angularState),
        dot(asfloat(arena.Load3(input+48u)),angularState));
    // The axis lock was applied in solver space before the world
    // inverse-inertia transform; the resulting motion is not masked
    // again by DyBodyCoreIntegrator::integrateCore.
    float3 motionAngular=angular;
    float3 nextPosition=position+motionLinear*dt;
    float4 nextQuaternion=quaternion;
    float w=dot(motionAngular,motionAngular);
    if(w!=0.0f) {
        w=sqrt(w);
        if(w>1.0e7f) {
            motionAngular*=1.0e7f/w;
            w=1.0e7f;
        }
        const float angle=dt*w*0.5f;
        const float3 omega=motionAngular*(sin(angle)/w);
        const float4 dq=float4(
            quaternion.w*omega.x+omega.y*quaternion.z-omega.z*quaternion.y,
            quaternion.w*omega.y+omega.z*quaternion.x-omega.x*quaternion.z,
            quaternion.w*omega.z+omega.x*quaternion.y-omega.y*quaternion.x,
            -dot(omega,quaternion.xyz));
        nextQuaternion=dq+quaternion*cos(angle);
        const float length=sqrt(dot(nextQuaternion,nextQuaternion));
        if(!(length>0.0f) || !isfinite(length))return;
        nextQuaternion/=length;
    }
    // integrateCore advances the principal mass frame. Recover the
    // actor frame before composing each independent local shape pose.
    const float3 localCenter=asfloat(arena.Load3(input+76u));
    const float4 localAxes=asfloat(arena.Load4(input+88u));
    nextQuaternion=sourceRigidMultiply(nextQuaternion,float4(-localAxes.xyz,localAxes.w));
    nextPosition-=sourceRigidRotate(nextQuaternion,localCenter);
    if(!all(isfinite(nextPosition)) || !all(isfinite(nextQuaternion)))return;
    arena.Store3(destination,asuint(nextPosition));
    arena.Store4(destination+12u,asuint(nextQuaternion));
    uint element=arena.Load(input+72u);
    const uint shapeCount=arena.Load(input+104u);
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
    if(shapeCount==0u || element==0xffffffffu)
        arena.Store(destination+28u,1u);
}

static const uint fusedTask=0u, normalTile=1u, frictionTile=2u;
static const uint partitionTile=3u, motionTile=4u, integrationTile=5u;
static const uint publishErrors=6u, freeBodyTile=7u, colorTile=8u;

void publishTaskError(uint task,uint error) {
    arena.Store(p.writebackBias+p.resultTaskOffset+task*4u,error);
}

[numthreads(128,1,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint phase=p.launchControl&15u;
    const bool devicePartitions=(p.launchControl&0x80000000u)!=0u;
    const bool precolored=(p.launchControl&0x40000000u)!=0u;
    const uint iter=(p.launchControl&0x3fffffffu)>>4u;
    const uint index=p.launchBegin+group.x;
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
        if(phase==publishErrors) {
            const uint task=index*128u+lane.x;
            if(task<schedule.z)
                publishTaskError(task,arena.Load(p.taskBias+task*36u+32u));
            return;
        }

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
            if(lane.x==0u)
                sourceDyPartitionTask(taskId,schedule.w,endError);
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

        if(fused || phase==partitionTile || phase==motionTile) {
            const uint total=iters.x+max(iters.y,1u);
            const uint endIteration=fused ? total : min(iter+1u,total);
            for(uint iteration=fused?0u:iter;iteration<endIteration;++iteration) {
                if(fused || phase==partitionTile) {
                    uint begin=descriptors.x;
                    for(uint partition=partitions.x;partition<partitions.y;++partition) {
                        uint end=descriptors.y;
                        if(fused)end=arena.Load(p.partitionBias+partition*4u);
                        solvePartition(begin,end,lane.x,iters,iteration,endError);
                        if(fused)AllMemoryBarrierWithGroupSync();
                        begin=end;
                    }
                }
                if((fused || phase==motionTile) && iteration+1u==iters.x) {
                    for(uint i=bodies.x+lane.x;i<bodies.y;i+=128u)
                        saveMotion(i,endError);
                    if(fused)AllMemoryBarrierWithGroupSync();
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
