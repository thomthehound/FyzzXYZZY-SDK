// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// Source broadphase.cu::clearNewFlagLaunch (32 x 256).
// Direct source allocation views, never a copied packed arena.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer created : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer sapX : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer sapY : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer sapZ : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer handlesX : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer handlesY : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer handlesZ : register(u6,space0);
struct Parameters {
    uint createdBias;
    uint sapXBias,sapYBias,sapZBias;
    uint handlesXBias,handlesYBias,handlesZBias;
    uint createdCount;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    const uint threadId=group.x*256u+thread.x;
    for(uint i=threadId;i<p.createdCount;i+=32u*256u) {
        const uint handle=created.Load(p.createdBias+i*4u);
        const uint startTag=(handle<<3u)|1u;
        const uint endTag=handle<<3u;
        const uint x0=sapX.Load(p.sapXBias+handle*8u);
        const uint x1=sapX.Load(p.sapXBias+handle*8u+4u);
        const uint y0=sapY.Load(p.sapYBias+handle*8u);
        const uint y1=sapY.Load(p.sapYBias+handle*8u+4u);
        const uint z0=sapZ.Load(p.sapZBias+handle*8u);
        const uint z1=sapZ.Load(p.sapZBias+handle*8u+4u);
        handlesX.Store(p.handlesXBias+x0*4u,startTag);
        handlesX.Store(p.handlesXBias+x1*4u,endTag);
        handlesY.Store(p.handlesYBias+y0*4u,startTag);
        handlesY.Store(p.handlesYBias+y1*4u,endTag);
        handlesZ.Store(p.handlesZBias+z0*4u,startTag);
        handlesZ.Store(p.handlesZBias+z1*4u,endTag);
    }
}
