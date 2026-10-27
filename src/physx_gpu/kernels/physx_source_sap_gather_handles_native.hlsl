// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// PhysX BP_UDPATE_HANDLES 64x256: gather by final WR ranks from
// ORIGINAL previous-frame handles into current-frame handle allocations.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer rankX:register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer rankY:register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer rankZ:register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer prevX:register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer prevY:register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer prevZ:register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer curX:register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer curY:register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer curZ:register(u8,space0);
struct Params {
    uint rx;
    uint ry;
    uint rz;
    uint px;
    uint py;
    uint pz;
    uint cx;
    uint cy;
    uint cz;
    uint boxes;
    uint pad0;
    uint pad1;
};
#ifdef __spirv__
[[vk::push_constant]] Params p;
#else
ConstantBuffer<Params> p:register(b0,space0);
#endif

[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    const uint count=2u*p.boxes;
    for(uint i=tid.x;i<count;i+=16384u) {
        curX.Store(p.cx+i*4u,prevX.Load(p.px+rankX.Load(p.rx+i*4u)*4u));
        curY.Store(p.cy+i*4u,prevY.Load(p.py+rankY.Load(p.ry+i*4u)*4u));
        curZ.Store(p.cz+i*4u,prevZ.Load(p.pz+rankZ.Load(p.rz+i*4u)*4u));
    }
}
