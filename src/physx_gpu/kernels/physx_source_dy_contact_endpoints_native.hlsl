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
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p:register(b0,space0);
#endif

[numthreads(128,1,1)]
void main(uint3 thread:SV_DispatchThreadID) {
    const uint source=thread.x;
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
    const bool dynamicA=(refs.x&3u)==1u;
    const bool dynamicB=(refs.y&3u)==1u;
    if(!dynamicA && !dynamicB)return;
    const uint2 endpoints=scWork.Load2(p.endpointBias+source*8u);
    if((dynamicA && endpoints.x>=p.bodyCount) ||
       (dynamicB && endpoints.y>=p.bodyCount)) {
        uint unused;
        scWork.InterlockedOr(p.scHeaderBias,512u,unused);
        return;
    }
    dyArena.Store2(prep,uint2(
        dynamicA?(endpoints.x<<2u)|1u:refs.x,
        dynamicB?(endpoints.y<<2u)|1u:refs.y));
}
