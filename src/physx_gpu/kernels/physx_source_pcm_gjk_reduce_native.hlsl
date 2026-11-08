// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// The closest-simplex reduction is isolated from support mapping and
// contact publication so DXIL lowers the higher-dimensional Voronoi math
// without the entire GuGJK continuation in the same shader.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer transforms : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer geometry : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer pairs : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer results : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer candidates : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer edgeCaches : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer work : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer streamContacts : register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer hullVertices : register(u8,space0);
struct Parameters {
    uint pairCount,transformBias,geometryBias,pairBias,resultBias,
         candidateBias,edgeCacheBias,workBias,shapeCapacity,
         candidateCapacity,edgeCapacity,outputBias,hullBias,hullCount;
#ifdef FYZZXYZZY_GJK_STAGED_REDUCE
    uint gjkPass;
#endif
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

#include "physx_source_pcm_box_math_native.hlsli"
#include "physx_source_pcm_box_gjk_native.hlsli"
#include "physx_source_pcm_gjk_state_native.hlsli"

[numthreads(32,1,1)]
void main(uint3 thread:SV_DispatchThreadID) {
#ifdef FYZZXYZZY_GJK_STAGED_REDUCE
    if(thread.x>=work.Load(p.workBias))return;
    const uint i=work.Load(p.workBias+(thread.x+1u)*4u);
#else
    const uint i=thread.x;
#endif
    if(i>=p.pairCount)return;
    const uint result=p.resultBias+i*32u;
    if(results.Load(result+8u)!=3u)return;
    const uint source=p.pairBias+i*20u;
    const uint edge=pairs.Load(source+8u);
    if(edge>=p.edgeCapacity)return;
    const uint cache=p.edgeCacheBias+edge*240u;
    GjkGpuState state=gjkLoadGpuState(cache);
    if(state.phase!=1u)return;

    const GjkClosest next=gjkClosestSimplex(state.simplex);
    if(next.mask==0u) {
        state.phase=2u;
        state.outcome=0u;
        gjkStoreGpuState(cache,state);
        return;
    }
    const float nextDist=sqrt(next.dist2);
    state.closest=next.closestPoint;
    state.dist2=next.dist2;
    state.weights=next.weights;
    state.mask=next.mask;
    if(nextDist>=state.previous) {
        state.phase=2u;
        state.outcome=state.pointCore!=0u
            ?(state.projection>0.f
                ?(nextDist>0.f && isfinite(nextDist) &&
                  nextDist<=state.contactLimit?1u:0u)
                :3u)
            :0u;
        gjkStoreGpuState(cache,state);
        return;
    }

    float4 packedWeights=0.f;
    uint packed=0u;
    if((next.mask&1u)!=0u && next.weights.x>1.e-6f)
        packedWeights[packed++]=next.weights.x;
    if((next.mask&2u)!=0u && next.weights.y>1.e-6f)
        packedWeights[packed++]=next.weights.y;
    if((next.mask&4u)!=0u && next.weights.z>1.e-6f)
        packedWeights[packed++]=next.weights.z;
    if((next.mask&8u)!=0u && next.weights.w>1.e-6f)
        packedWeights[packed++]=next.weights.w;
    gjkCompact(state.simplex,next);
    if(packed==0u) {
        packedWeights=float4(1.f,0.f,0.f,0.f);
        packed=1u;
    }
    state.weights=packedWeights;
    state.mask=(1u<<packed)-1u;
    state.previous=nextDist;
    state.phase=0u;
    gjkStoreGpuState(cache,state);
}
