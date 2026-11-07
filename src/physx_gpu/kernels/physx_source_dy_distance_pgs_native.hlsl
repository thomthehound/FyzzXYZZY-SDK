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
        const uint pose=p.integrationBias+40u+count*112u+poseIndex*28u;
        position=asfloat(arena.Load3(pose));
        quaternion=asfloat(arena.Load4(pose+12u));
    }
    return all(isfinite(position)) && all(isfinite(quaternion));
}

#include "physx_source_dy_contact_pgs_native.hlsli"

// Sc owns task membership. One workgroup colors independent source
// constraints concurrently while preserving per-body greedy precedence.
// Static reservations follow the source descriptor encounter order.
groupshared uint dyNeeded,dyAssigned,dyWave,dyProgress,dyHighest;
groupshared uint dyNextWave,dySerial;
groupshared uint dySingletonPartitions;

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
    for(uint i=first+lane;i<last;i+=128u) {
        const uint4 desc=arena.Load4(raw+i*16u);
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

    // For a hub graph, every dynamic edge shares one writable body.
    // Its exact source greedy coloring is the encounter ordinal of
    // that edge: no conflict search or 32-color wave scans are needed.
    // Static reservations still use the ordinary post-dynamic phase.
    if(lane==0u && dyNeeded>=32u) {
        uint candidateA=0xffffffffu,candidateB=0xffffffffu;
        for(uint i=first;i<last;++i) {
            const uint4 desc=arena.Load4(raw+i*16u);
            if((desc.w&0x300u)==0x300u) {
                candidateA=desc.y;
                candidateB=desc.z;
                break;
            }
        }
        bool sharedA=candidateA!=0xffffffffu;
        bool sharedB=sharedA;
        for(uint i=first;i<last && (sharedA||sharedB);++i) {
            const uint4 desc=arena.Load4(raw+i*16u);
            if((desc.w&0x300u)!=0x300u)continue;
            sharedA=sharedA && (desc.y==candidateA||desc.z==candidateA);
            sharedB=sharedB && (desc.y==candidateB||desc.z==candidateB);
        }
        if(sharedA || sharedB) {
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
    const uint input=p.integrationBias+40u+ordinal*112u;
    const uint inputFlags=arena.Load(input+68u);
    [branch]
    if(ordinal<p.bodyCount) {
        const uint from=p.bodyBias+ordinal*52u;
        const uint to=p.writebackBias+p.resultBodyOffset+ordinal*48u;
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
    const float3 inertiaX=asfloat(arena.Load3(input+24u));
    const float3 inertiaY=asfloat(arena.Load3(input+36u));
    const float3 inertiaZ=asfloat(arena.Load3(input+48u));
    const float3 angular=baseAngular+float3(
        sourceDyDot3(inertiaX,angularState),
        sourceDyDot3(inertiaY,angularState),
        sourceDyDot3(inertiaZ,angularState));
    if(ordinal<p.bodyCount && (inputFlags&8u)!=0u) {
        const uint from=p.bodyBias+ordinal*52u;
        const uint to=p.writebackBias+p.resultBodyOffset+ordinal*48u;
        float3 finalLinear=baseLinear+asfloat(arena.Load3(from));
        if(locks&1u)finalLinear.x=0.0f;
        if(locks&2u)finalLinear.y=0.0f;
        if(locks&4u)finalLinear.z=0.0f;
        float3 finalState=asfloat(arena.Load3(from+12u));
        if(locks&8u)finalState.x=0.0f;
        if(locks&16u)finalState.y=0.0f;
        if(locks&32u)finalState.z=0.0f;
        const float3 finalAngular=baseAngular+float3(
            sourceDyDot3(inertiaX,finalState),
            sourceDyDot3(inertiaY,finalState),
            sourceDyDot3(inertiaZ,finalState));
        arena.Store3(to,asuint(finalLinear));
        arena.Store3(to+12u,asuint(finalAngular));
        arena.Store3(to+24u,asuint(motionLinear));
        arena.Store3(to+36u,asuint(angular));
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
static const uint initializeBodiesTile=9u;

// Only the initial inverse mass enters from Sc. Each workgroup owns
// disjoint body ordinals before any solver consumer reads the records.
void initializeSolverBody(uint ordinal,uint source) {
    const uint target=p.bodyBias+ordinal*52u;
    const uint4 zero=uint4(0u,0u,0u,0u);
    arena.Store4(target,zero);
    arena.Store4(target+16u,zero);
    arena.Store4(target+32u,zero);
    arena.Store(target+48u,arena.Load(source+ordinal*4u));
}

void publishTaskError(uint task,uint error) {
    arena.Store(p.writebackBias+p.resultTaskOffset+task*4u,error);
}

[numthreads(128,1,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint phase=p.launchControl&15u;
    const bool devicePartitions=(p.launchControl&0x80000000u)!=0u;
    const bool precolored=(p.launchControl&0x40000000u)!=0u;
    const uint iter=(p.launchControl&0x1fffffffu)>>4u;
    const uint index=p.launchBegin+group.x;
    if(phase==initializeBodiesTile) {
        const uint ordinal=index*128u+lane.x;
        if(ordinal<p.bodyCount)
            initializeSolverBody(ordinal,arena.Load(p.integrationBias+36u));
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
            if(source!=0u && ordinal<p.bodyCount)
                initializeSolverBody(ordinal,source);
            return;
        }
        // Compact single-task scenes can initialize up to one workgroup
        // of bodies before their fused PGS. Larger or independent tasks
        // use a separate initialization dispatch for global ordering.
        if(fused && !precolored && schedule.z==1u &&
           p.bodyCount<=128u) {
            const uint source=arena.Load(p.integrationBias+36u);
            if(source!=0u) {
                if(lane.x<p.bodyCount)
                    initializeSolverBody(lane.x,source);
                AllMemoryBarrierWithGroupSync();
            }
        }
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
                            if(fused)AllMemoryBarrierWithGroupSync();
                            begin=end;
                        }
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
