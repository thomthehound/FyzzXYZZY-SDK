// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Sc contact-manager history has one committed GPU epoch. A frame takes
// its own device journal before Gu mutates the persistent EdgeIndex cache.
// Accepted frames discard that journal; rejected frames restore it before
// the next SAP/PCM submission. No contact history crosses the CPU.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer pairInputs:register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer edgeCaches:register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer checkpoint:register(u2,space0);

struct Parameters {
    uint pairBias,edgeBias,checkpointBias,phase;
    uint wordStart,wordCount,edgeCapacity,clearBegin;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p:register(b0,space0);
#endif

[numthreads(128,1,1)]
void main(uint3 dispatch:SV_DispatchThreadID) {
    const uint word=p.wordStart+dispatch.x;
    if(word>=p.wordCount)return;
    if(p.phase==3u) {
        // Every Gu producer shares the same resident EdgeIndex touch.
        const uint edge=pairInputs.Load(p.pairBias+word*84u);
        if(edge>=p.edgeCapacity)return;
        const uint packed=pairInputs.Load(p.pairBias+word*84u+4u);
        const uint transition=(packed>>13u)&3u;
        edgeCaches.Store(p.edgeBias+edge*240u+72u,
            transition==1u || transition==2u?1u:0u);
        edgeCaches.Store(p.edgeBias+edge*240u+76u,
            transition==2u || transition==3u?1u:0u);
        return;
    }
    const uint record=word/60u;
    const uint lane=word%60u;
    if(p.phase==0u) {
        const uint edge=p.clearBegin+record;
        if(edge<p.edgeCapacity)
            edgeCaches.Store(p.edgeBias+edge*240u+lane*4u,0u);
        return;
    }
    const bool host_stream=p.phase==4u || p.phase==5u;
    const uint edge=host_stream
        ?pairInputs.Load(p.pairBias+record*84u)
        :pairInputs.Load(p.pairBias+record*20u+8u);
    if(edge>=p.edgeCapacity)return;
    const uint cache=p.edgeBias+edge*240u+lane*4u;
    const uint saved=p.checkpointBias+word*4u;
    if(p.phase==1u || p.phase==4u)
        checkpoint.Store(saved,edgeCaches.Load(cache));
    else
        edgeCaches.Store(cache,checkpoint.Load(saved));
}
