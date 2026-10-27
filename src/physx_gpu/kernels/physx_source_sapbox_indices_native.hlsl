// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// Source broadphase.cu::initializeSapBox1DLaunch (BP_INITIALIZE_SAPBOX).
// Caller selects the original CURRENT or NEW source PxgSapBox1D pointers.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer handlesX : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer handlesY : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer handlesZ : register(u2,space0);
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
struct Parameters {
    uint bias[6];
    uint numProjections;
    uint reserved;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    const uint index=group.x*256u+thread.x;
    if(index>=p.numProjections)return;
    const uint hx=handlesX.Load(p.bias[0]+index*4u);
    const uint hy=handlesY.Load(p.bias[1]+index*4u);
    const uint hz=handlesZ.Load(p.bias[2]+index*4u);
    // Source getHandle removes the low three endpoint/new flags.
    const uint ix=(hx>>3u)*8u+((hx&1u)==0u?4u:0u);
    const uint iy=(hy>>3u)*8u+((hy&1u)==0u?4u:0u);
    const uint iz=(hz>>3u)*8u+((hz&1u)==0u?4u:0u);
    sapX.Store(p.bias[3]+ix,index);
    sapY.Store(p.bias[4]+iy,index);
    sapZ.Store(p.bias[5]+iz,index);
}
