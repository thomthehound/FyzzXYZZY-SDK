// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Compact Sc edge managers admitted to Gu EPA without host-visible counts.
// The same source stream resets, collects, then consumes the indirect work.
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
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0,space0);
#endif

[numthreads(1,1,1)]
void reset(uint3 thread : SV_DispatchThreadID) {
    if(thread.x!=0u)return;
    dispatchArgs.Store3(params.argsBias,uint3(0u,1u,1u));
}

[numthreads(128,1,1)]
void compact(uint3 thread : SV_DispatchThreadID) {
    uint i=thread.x;
    if(i>=params.pairCount)return;
    // SourcePcmBoxPairResultNative.status: 3 requires EPA continuation.
    if(results.Load(params.resultBias+i*32u+8u)!=3u)return;
    uint destination;
    dispatchArgs.InterlockedAdd(params.argsBias,1u,destination);
    work.Store(params.workBias+destination*4u,i);
}
