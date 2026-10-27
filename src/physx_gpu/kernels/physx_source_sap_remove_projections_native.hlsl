// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// PhysX broadphase.cu::markRemovedPairsProjectionsLaunch.
// After incremental SAP, before created pairs and WOR radix sorting.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer removed : register(u0,space0);
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
RWByteAddressBuffer projX : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer projY : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer projZ : register(u6,space0);
struct Parameters {
    uint removedBias, sapXBias, sapYBias, sapZBias;
    uint projXBias, projYBias, projZBias, removedCount;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif
[numthreads(256,1,1)]
void main(uint3 thread:SV_DispatchThreadID) {
    for(uint i=thread.x; i<p.removedCount; i+=8192u) {
        const uint handle=removed.Load(p.removedBias+i*4u);
        const uint sx=p.sapXBias+handle*8u;
        const uint sy=p.sapYBias+handle*8u;
        const uint sz=p.sapZBias+handle*8u;
        const uint minX=sapX.Load(sx), maxX=sapX.Load(sx+4u);
        const uint minY=sapY.Load(sy), maxY=sapY.Load(sy+4u);
        const uint minZ=sapZ.Load(sz), maxZ=sapZ.Load(sz+4u);
        projX.Store(p.projXBias+minX*4u,0xffffffffu);
        projX.Store(p.projXBias+maxX*4u,0xffffffffu);
        projY.Store(p.projYBias+minY*4u,0xffffffffu);
        projY.Store(p.projYBias+maxY*4u,0xffffffffu);
        projZ.Store(p.projZBias+minZ*4u,0xffffffffu);
        projZ.Store(p.projZBias+maxZ*4u,0xffffffffu);
        sapX.Store(sx,0xffffffffu);
        sapX.Store(sx+4u,0xffffffffu);
        sapY.Store(sy,0xffffffffu);
        sapY.Store(sy+4u,0xffffffffu);
        sapZ.Store(sz,0xffffffffu);
        sapZ.Store(sz+4u,0xffffffffu);
    }
}
