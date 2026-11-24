// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer dyArena : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer scWork : register(u1,space0);

struct Parameters {
    uint manifoldBias,prepBias,scHeaderBias,endpointBias;
    uint admissionsBias,managerCount,prepCount,bodyCount;
    uint first;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p:register(b0,space0);
#endif

[numthreads(128,1,1)]
void main(uint3 thread:SV_DispatchThreadID) {
    const uint source=p.first+thread.x;
    if(source>=p.managerCount || scWork.Load(p.scHeaderBias)!=0u)return;
    const uint word=scWork.Load(p.admissionsBias+(source>>5u)*4u);
    if((word&(1u<<(source&31u)))==0u)return;
    const uint manager=p.manifoldBias+source*24u;
    if(dyArena.Load(manager+4u)==0u)return;
    const uint prepIndex=dyArena.Load(manager+16u);
    if(prepIndex>=p.prepCount) {
        uint unused;
        scWork.InterlockedOr(p.scHeaderBias,512u,unused);
        return;
    }
    const uint prep=p.prepBias+16u+prepIndex*48u;
    const uint2 refs=dyArena.Load2(prep);
    // Only unresolved dynamic references need Sc's device-owned rows.
    const bool pendingA=refs.x==0xfffffffdu;
    const bool pendingB=refs.y==0xfffffffdu;
    if(!pendingA && !pendingB)return;
    const uint2 endpoints=scWork.Load2(p.endpointBias+source*8u);
    if((pendingA && endpoints.x>=p.bodyCount) ||
       (pendingB && endpoints.y>=p.bodyCount)) {
        uint unused;
        scWork.InterlockedOr(p.scHeaderBias,512u,unused);
        return;
    }
    dyArena.Store2(prep,uint2(
        pendingA?(endpoints.x<<2u)|1u:refs.x,
        pendingB?(endpoints.y<<2u)|1u:refs.y));
}
