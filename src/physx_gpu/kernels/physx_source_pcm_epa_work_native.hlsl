// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Compact source Sc EPA continuation work after the candidate-owned
// indirect argument initialization. The Sc pair results own admission.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
ByteAddressBuffer results : register(t0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer work : register(u0,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer dispatchArgs : register(u1,space0);

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
