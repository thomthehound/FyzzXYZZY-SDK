// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// PhysX broadphase.cu::markRemovedPairsLaunch, original 32x256 grid.
// Mark deleted in PREVIOUS handles, then empty OLD integer bounds.
// Eight original PxgBroadPhaseDesc pointers, no packed arena.
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
RWByteAddressBuffer handlesX : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer handlesY : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer handlesZ : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer oldBounds : register(u7,space0);
struct Parameters {
    uint removedBias, sapXBias, sapYBias, sapZBias;
    uint handlesXBias, handlesYBias, handlesZBias, boundsBias;
    uint removedCount, reserved0, reserved1, reserved2;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif
[numthreads(256,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    for(uint i=thread.x; i<p.removedCount; i+=8192u) {
        const uint handle=removed.Load(p.removedBias+i*4u);
        const uint sx=p.sapXBias+handle*8u;
        const uint sy=p.sapYBias+handle*8u;
        const uint sz=p.sapZBias+handle*8u;
        const uint minX=sapX.Load(sx), maxX=sapX.Load(sx+4u);
        const uint minY=sapY.Load(sy), maxY=sapY.Load(sy+4u);
        const uint minZ=sapZ.Load(sz), maxZ=sapZ.Load(sz+4u);
        const uint hx=p.handlesXBias, hy=p.handlesYBias, hz=p.handlesZBias;
        const uint x0=hx+minX*4u, x1=hx+maxX*4u;
        const uint y0=hy+minY*4u, y1=hy+maxY*4u;
        const uint z0=hz+minZ*4u, z1=hz+maxZ*4u;
        handlesX.Store(x0,handlesX.Load(x0)|4u);
        handlesX.Store(x1,handlesX.Load(x1)|4u);
        handlesY.Store(y0,handlesY.Load(y0)|4u);
        handlesY.Store(y1,handlesY.Load(y1)|4u);
        handlesZ.Store(z0,handlesZ.Load(z0)|4u);
        handlesZ.Store(z1,handlesZ.Load(z1)|4u);
        const uint b=p.boundsBias+handle*24u;
        [unroll] for(uint axis=0;axis<3u;++axis) {
            oldBounds.Store(b+axis*4u,0xff7fffffu);
            oldBounds.Store(b+12u+axis*4u,0x00800000u);
        }
    }
}
