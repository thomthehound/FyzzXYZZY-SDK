// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// DyConstraintPartition's independent rows execute concurrently within
// a source conflict partition. Each workgroup owns one disjoint Dy task
// and preserves the source position/conclude/velocity iteration sequence.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer arena : register(u0,space0);

struct Parameters {
    uint bodyBias, distanceBias, descriptorBias, partitionBias;
    uint ordinalBias, taskBias;
    uint bodyCount, rowCount, descriptorCount, partitionCount;
    uint ordinalCount, taskCount;
    uint contactBias,frictionBias,patchBias,manifoldBias,orderBias;
    uint contactCount,frictionCount,patchCount,manifoldCount,orderCount;
    uint resultBodyOffset,resultRowOffset,resultTaskOffset;
    uint resultContactOffset,resultFrictionOffset,resultPatchOffset;
    uint writebackBias;
    uint prepBias,prepCount;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif
static const uint invalidBody = 0xffffffffu;
static const double maxImpulse = 3.4028234663852886e+38;

double loadDouble(uint byteOffset) {
    uint2 words=arena.Load2(byteOffset);
    return asdouble(words.x,words.y);
}
void storeDouble(uint byteOffset,double value,uint rowIndex,bool finalPass) {
    uint lo,hi;
    asuint(value,lo,hi);
    uint2 packed=uint2(lo,hi);
    arena.Store2(byteOffset,packed);
    if(finalPass)
        arena.Store2(p.writebackBias+p.resultRowOffset+
            rowIndex*8u,packed);
}
void solveRow(uint index,uint errorAddress,bool biased,bool finalPass) {
    uint d=p.descriptorBias+index*16u;
    uint4 desc=arena.Load4(d);
    if(desc.w!=1u || desc.x>=p.rowCount ||
       (desc.y!=invalidBody && desc.y>=p.bodyCount) ||
       (desc.z!=invalidBody && desc.z>=p.bodyCount)) {
        arena.InterlockedOr(errorAddress,1u);
        return;
    }
    uint offset=p.distanceBias+desc.x*80u;
    float3 axis=asfloat(arena.Load3(offset));
    float3 ja=asfloat(arena.Load3(offset+12u));
    float3 jb=asfloat(arena.Load3(offset+24u));
    float massA=asfloat(arena.Load(offset+36u));
    float massB=asfloat(arena.Load(offset+40u));
    uint ao=p.bodyBias+desc.y*52u;
    uint bo=p.bodyBias+desc.z*52u;
    float3 va=float3(0,0,0),vb=float3(0,0,0);
    float3 aa=float3(0,0,0),ab=float3(0,0,0);
    if(desc.y!=invalidBody) {
        va=asfloat(arena.Load3(ao));
        aa=asfloat(arena.Load3(ao+12u));
    }
    if(desc.z!=invalidBody) {
        vb=asfloat(arena.Load3(bo));
        ab=asfloat(arena.Load3(bo+12u));
    }

    // DySolverConstraints::solveExt1D and source native 1D PGS:
    // float Jacobian dot operations followed by wide accumulated impulse.
    double normalDelta=
        (double)axis.x*(double)(va.x-vb.x)+
        (double)axis.y*(double)(va.y-vb.y)+
        (double)axis.z*(double)(va.z-vb.z)+
        (double)dot(ja,aa)-(double)dot(jb,ab);
    double previous=loadDouble(offset+72u);
    double next=clamp(previous+
        normalDelta*loadDouble(offset+64u)+
        loadDouble(offset+(biased?48u:56u)),
        -maxImpulse,maxImpulse);
    double impulse=next-previous;
    double ia=impulse*(double)massA;
    double ib=impulse*(double)massB;
    float3 newVa=float3(
        (float)((double)va.x+ia*(double)axis.x),
        (float)((double)va.y+ia*(double)axis.y),
        (float)((double)va.z+ia*(double)axis.z));
    float3 newVb=float3(
        (float)((double)vb.x-ib*(double)axis.x),
        (float)((double)vb.y-ib*(double)axis.y),
        (float)((double)vb.z-ib*(double)axis.z));
    float3 newAa=float3(
        (float)((double)aa.x+impulse*(double)ja.x),
        (float)((double)aa.y+impulse*(double)ja.y),
        (float)((double)aa.z+impulse*(double)ja.z));
    float3 newAb=float3(
        (float)((double)ab.x-impulse*(double)jb.x),
        (float)((double)ab.y-impulse*(double)jb.y),
        (float)((double)ab.z-impulse*(double)jb.z));
    if(!isfinite(next) || !all(isfinite(newVa)) ||
       !all(isfinite(newVb)) || !all(isfinite(newAa)) ||
       !all(isfinite(newAb))) {
        arena.InterlockedOr(errorAddress,2u);
        return;
    }
    // Only finite-mass dynamic bodies can be updated. Kinematic rows
    // retain their original zero-response state across disjoint groups.
    if(desc.y!=invalidBody && massA>0.f) {
        arena.Store3(ao,asuint(newVa));
        arena.Store3(ao+12u,asuint(newAa));
    }
    if(desc.z!=invalidBody && massB>0.f) {
        arena.Store3(bo,asuint(newVb));
        arena.Store3(bo+12u,asuint(newAb));
    }
    storeDouble(offset+72u,next,desc.x,finalPass);
}

#include "physx_source_dy_contact_pgs_native.hlsli"

[numthreads(128,1,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    if(group.x>=p.taskCount)return;
    uint t=p.taskBias+group.x*36u;
    uint4 ranges=arena.Load4(t);
    uint2 bodyRange=arena.Load2(t+16u);
    uint2 iters=arena.Load2(t+24u);
    uint endError=t+32u;
    if(ranges.x>ranges.y || ranges.y>p.descriptorCount ||
       ranges.z>ranges.w || ranges.w>p.partitionCount ||
       bodyRange.x>bodyRange.y || bodyRange.y>p.ordinalCount) {
        if(lane.x==0u) {
            arena.InterlockedOr(endError,4u);
            arena.Store(p.writebackBias+p.resultTaskOffset+
                            group.x*4u,4u);
        }
        return;
    }
    // Independent Gu normal rows are prepared in parallel after the
    // source Sc/Dy friction correlation has fixed their order. The same
    // task group then executes the original PGS partitions and iterations.
    for(uint i=ranges.x+lane.x;i<ranges.y;i+=128u) {
        uint4 desc=arena.Load4(p.descriptorBias+i*16u);
        if(desc.w==1u)continue;
        if(desc.x>=p.manifoldCount) {
            arena.InterlockedOr(endError,4096u);
            continue;
        }
        uint mo=p.manifoldBias+desc.x*20u;
        uint4 manager=arena.Load4(mo);
        if(manager.y==0u)continue;
        uint pairIndex=arena.Load(mo+16u);
        if(pairIndex>=p.prepCount ||
           manager.z>p.orderCount ||
           manager.w>p.orderCount-manager.z) {
            arena.InterlockedOr(endError,4096u);
            continue;
        }
        for(uint c=0u;c<manager.w;++c) {
            uint index=arena.Load(p.orderBias+(manager.z+c)*4u);
            prepareContactRow(index,pairIndex,endError);
        }
    }
    AllMemoryBarrierWithGroupSync();
    uint velocityIterations=max(iters.y,1u);
    uint firstPartition=ranges.z;
    for(uint iter=0u;iter<iters.x+velocityIterations;++iter) {
        for(uint partition=firstPartition;partition<ranges.w;++partition) {
            uint begin=partition==firstPartition ? ranges.x:
                arena.Load(p.partitionBias+(partition-1u)*4u);
            uint end=arena.Load(p.partitionBias+partition*4u);
            for(uint i=begin+lane.x;i<end;i+=128u) {
                uint4 desc=arena.Load4(p.descriptorBias+i*16u);
                bool finalPass=iter+1u==iters.x+velocityIterations;
                if(desc.w==1u)
                    solveRow(i,endError,iter<iters.x,finalPass);
                else
                    solveContact(desc,endError,iter<iters.x,
                        iter>=iters.x || iters.x-iter<=3u,finalPass);
            }
            // The original conflict partition guarantees no body writes
            // overlap inside this barrier; later partitions depend on it.
            AllMemoryBarrierWithGroupSync();
        }
        if(iter+1u==iters.x) {
            for(uint i=bodyRange.x+lane.x;i<bodyRange.y;i+=128u) {
                uint ordinal=arena.Load(p.ordinalBias+i*4u);
                if(ordinal>=p.bodyCount) {
                    arena.InterlockedOr(endError,8u);
                    continue;
                }
                uint base=p.bodyBias+ordinal*52u;
                arena.Store3(base+24u,arena.Load3(base));
                arena.Store3(base+36u,arena.Load3(base+12u));
            }
            AllMemoryBarrierWithGroupSync();
        }
    }
    // A Dy task owns exclusive solver-body ordinals and descriptor ranges.
    // Publish only the fields consumed by the source PxsSolverEndTask and
    // Gu writeback; the original prepared inputs remain device-local.
    AllMemoryBarrierWithGroupSync();
    for(uint i=bodyRange.x+lane.x;i<bodyRange.y;i+=128u) {
        uint ordinal=arena.Load(p.ordinalBias+i*4u);
        if(ordinal>=p.bodyCount) {
            arena.InterlockedOr(endError,8u);
            continue;
        }
        uint from=p.bodyBias+ordinal*52u;
        uint to=p.writebackBias+p.resultBodyOffset+ordinal*48u;
        arena.Store4(to,arena.Load4(from));
        arena.Store4(to+16u,arena.Load4(from+16u));
        arena.Store4(to+32u,arena.Load4(from+32u));
    }
    AllMemoryBarrierWithGroupSync();
    if(lane.x==0u)
        arena.Store(p.writebackBias+p.resultTaskOffset+
            group.x*4u,arena.Load(endError));
}
