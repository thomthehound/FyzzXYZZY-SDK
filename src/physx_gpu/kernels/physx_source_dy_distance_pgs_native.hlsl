// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// DyConstraintPartition's independent rows execute concurrently within
// a source conflict partition. Each workgroup owns one disjoint Dy task
// and preserves the source position/conclude/velocity iteration sequence.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer bodies : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer distances : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer descriptors : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer partitionEnds : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer bodyOrdinals : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer tasks : register(u5,space0);

struct Parameters {
    uint bodyBias, distanceBias, descriptorBias, partitionBias;
    uint ordinalBias, taskBias;
    uint bodyCount, rowCount, descriptorCount, partitionCount;
    uint ordinalCount, taskCount;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif
static const uint invalidBody = 0xffffffffu;
static const double maxImpulse = 3.4028234663852886e+38;

double loadDouble(uint byteOffset) {
    uint2 words=distances.Load2(byteOffset);
    return asdouble(words.x,words.y);
}
void storeDouble(uint byteOffset,double value) {
    uint lo,hi;
    asuint(value,lo,hi);
    distances.Store2(byteOffset,uint2(lo,hi));
}
void solveRow(uint index,uint errorAddress,bool biased) {
    uint d=p.descriptorBias+index*16u;
    uint4 desc=descriptors.Load4(d);
    if(desc.w!=1u || desc.x>=p.rowCount ||
       (desc.y!=invalidBody && desc.y>=p.bodyCount) ||
       (desc.z!=invalidBody && desc.z>=p.bodyCount)) {
        tasks.InterlockedOr(errorAddress,1u);
        return;
    }
    uint offset=p.distanceBias+desc.x*80u;
    float3 axis=asfloat(distances.Load3(offset));
    float3 ja=asfloat(distances.Load3(offset+12u));
    float3 jb=asfloat(distances.Load3(offset+24u));
    float massA=asfloat(distances.Load(offset+36u));
    float massB=asfloat(distances.Load(offset+40u));
    uint ao=p.bodyBias+desc.y*48u;
    uint bo=p.bodyBias+desc.z*48u;
    float3 va=float3(0,0,0),vb=float3(0,0,0);
    float3 aa=float3(0,0,0),ab=float3(0,0,0);
    if(desc.y!=invalidBody) {
        va=asfloat(bodies.Load3(ao));
        aa=asfloat(bodies.Load3(ao+12u));
    }
    if(desc.z!=invalidBody) {
        vb=asfloat(bodies.Load3(bo));
        ab=asfloat(bodies.Load3(bo+12u));
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
        tasks.InterlockedOr(errorAddress,2u);
        return;
    }
    // Only finite-mass dynamic bodies can be updated. Kinematic rows
    // retain their original zero-response state across disjoint groups.
    if(desc.y!=invalidBody && massA>0.f) {
        bodies.Store3(ao,asuint(newVa));
        bodies.Store3(ao+12u,asuint(newAa));
    }
    if(desc.z!=invalidBody && massB>0.f) {
        bodies.Store3(bo,asuint(newVb));
        bodies.Store3(bo+12u,asuint(newAb));
    }
    storeDouble(offset+72u,next);
}

[numthreads(128,1,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    if(group.x>=p.taskCount)return;
    uint t=p.taskBias+group.x*36u;
    uint4 ranges=tasks.Load4(t);
    uint2 bodyRange=tasks.Load2(t+16u);
    uint2 iters=tasks.Load2(t+24u);
    uint endError=t+32u;
    if(ranges.x>ranges.y || ranges.y>p.descriptorCount ||
       ranges.z>ranges.w || ranges.w>p.partitionCount ||
       bodyRange.x>bodyRange.y || bodyRange.y>p.ordinalCount) {
        if(lane.x==0u)tasks.InterlockedOr(endError,4u);
        return;
    }
    uint velocityIterations=max(iters.y,1u);
    uint firstPartition=ranges.z;
    for(uint iter=0u;iter<iters.x+velocityIterations;++iter) {
        for(uint partition=firstPartition;partition<ranges.w;++partition) {
            uint begin=partition==firstPartition ? ranges.x:
                partitionEnds.Load(p.partitionBias+(partition-1u)*4u);
            uint end=partitionEnds.Load(p.partitionBias+partition*4u);
            for(uint i=begin+lane.x;i<end;i+=128u)
                solveRow(i,endError,iter<iters.x);
            // The original conflict partition guarantees no body writes
            // overlap inside this barrier; later partitions depend on it.
            AllMemoryBarrierWithGroupSync();
        }
        if(iter+1u==iters.x) {
            for(uint i=bodyRange.x+lane.x;i<bodyRange.y;i+=128u) {
                uint ordinal=bodyOrdinals.Load(p.ordinalBias+i*4u);
                if(ordinal>=p.bodyCount) {
                    tasks.InterlockedOr(endError,8u);
                    continue;
                }
                uint base=p.bodyBias+ordinal*48u;
                bodies.Store3(base+24u,bodies.Load3(base));
                bodies.Store3(base+36u,bodies.Load3(base+12u));
            }
            AllMemoryBarrierWithGroupSync();
        }
    }
}
