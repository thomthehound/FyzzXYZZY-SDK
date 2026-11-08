// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Compact source Sc EPA continuation work after the candidate-owned
// indirect argument initialization. The Sc pair results own admission.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer results : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer work : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer dispatchArgs : register(u2,space0);

struct Parameters {
    uint pairCount;
    uint resultBias;
    uint workBias;
    uint argsBias;
    uint requiredStatus;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0,space0);
#endif
[numthreads(128,1,1)]
void compact(uint3 thread : SV_DispatchThreadID) {
    uint i=thread.x;
    if(i>=params.pairCount)return;
    // Select the GPU result state required by this worklist.
    if(results.Load(params.resultBias+i*32u+8u)!=params.requiredStatus)return;
    uint destination;
    if(params.requiredStatus==3u) {
        // GJK uses one 32-lane workgroup per compact group.
        work.InterlockedAdd(params.workBias,1u,destination);
        if((destination&31u)==0u) {
            uint ignored;
            dispatchArgs.InterlockedAdd(params.argsBias,1u,ignored);
        }
        work.Store(params.workBias+(destination+1u)*4u,i);
    }else {
        dispatchArgs.InterlockedAdd(params.argsBias,1u,destination);
        work.Store(params.workBias+destination*4u,i);
    }
}

[numthreads(1,1,1)]
void reset(uint3 thread : SV_DispatchThreadID) {
    dispatchArgs.Store3(params.argsBias,uint3(0u,1u,1u));
    if(params.requiredStatus==3u)work.Store(params.workBias,0u);
}
