// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// PhysX broadphase.cu::markUpdatedPairsLaunch / BP_UPDATE_UPDATEDPAIRS.
// Normal GPU AABB-manager path (changed, added and removed source bitmaps).
// 32 workgroups x 256 threads, one bit per lane and grid-stride over volumes.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer changed : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer added : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer removed : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer sapX : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer sapY : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer sapZ : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer projX : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer projY : register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer projZ : register(u8,space0);
#ifdef __spirv__
[[vk::binding(9,0)]]
#endif
RWByteAddressBuffer newBounds : register(u9,space0);
struct Parameters {
    uint changedBias, addedBias, removedBias, sapXBias;
    uint sapYBias, sapZBias, projXBias, projYBias;
    uint projZBias, boundsBias, bitmapWords, boxCapacity;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif
[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    const uint count=p.bitmapWords*32u;
    const uint lane=tid.x&31u;
    for(uint handle=tid.x;handle<count;handle+=8192u) {
        const uint word=handle>>5u;
        const uint changedMask=changed.Load(p.changedBias+word*4u);
        const uint excluded=added.Load(p.addedBias+word*4u) |
                            removed.Load(p.removedBias+word*4u);
        if(handle>=p.boxCapacity ||
           ((changedMask & ~excluded) & (1u<<lane))==0u)
            continue;
        const uint b=p.boundsBias+handle*24u;
        const uint sx=p.sapXBias+handle*8u;
        const uint sy=p.sapYBias+handle*8u;
        const uint sz=p.sapZBias+handle*8u;
        const uint ix=sapX.Load(sx), iy=sapY.Load(sy), iz=sapZ.Load(sz);
        const uint jx=sapX.Load(sx+4u), jy=sapY.Load(sy+4u), jz=sapZ.Load(sz+4u);
        projX.Store(p.projXBias+ix*4u,newBounds.Load(b));
        projX.Store(p.projXBias+jx*4u,newBounds.Load(b+12u));
        projY.Store(p.projYBias+iy*4u,newBounds.Load(b+4u));
        projY.Store(p.projYBias+jy*4u,newBounds.Load(b+16u));
        projZ.Store(p.projZBias+iz*4u,newBounds.Load(b+8u));
        projZ.Store(p.projZBias+jz*4u,newBounds.Load(b+20u));
    }
}
