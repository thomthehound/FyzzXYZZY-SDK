// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Rigid-box specialization of updateCacheAndBound: the Sc absolute shape
// pose is already composed, and original element indices address both
// the source bounds array and the shared transform cache.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer shapes : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer bounds : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer transforms : register(u2,space0);

struct Parameters {
    uint shapeCount;
    uint shapesBias;
    uint boundsBias;
    uint transformsBias;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

[numthreads(128,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    if(thread.x>=p.shapeCount)return;

    const uint base=p.shapesBias+thread.x*48u;
    const uint index=shapes.Load(base);
    const float3 center=asfloat(shapes.Load3(base+4u));
    const float4 q=asfloat(shapes.Load4(base+16u));
    const float3 e=asfloat(shapes.Load3(base+32u));

    // PxMat33(quaternion), three column vectors. updateCacheAndBound's
    // eBOX branch uses basisExtent, then center +/- world-space extents.
    const float xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
    const float xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
    const float wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
    const float3 c0=float3(1.0f-2.0f*(yy+zz),
                           2.0f*(xy+wz),2.0f*(xz-wy));
    const float3 c1=float3(2.0f*(xy-wz),
                           1.0f-2.0f*(xx+zz),2.0f*(yz+wx));
    const float3 c2=float3(2.0f*(xz+wy),
                           2.0f*(yz-wx),1.0f-2.0f*(xx+yy));
    const float3 extents=abs(c0)*e.x+abs(c1)*e.y+abs(c2)*e.z;

    const uint bound=p.boundsBias+index*24u;
    bounds.Store3(bound,asuint(center-extents));
    bounds.Store3(bound+12u,asuint(center+extents));

    // PxsCachedTransform: PxTransform (q[4], p[3]) and flags, 32 bytes.
    // The existing source kernel initializes transform flags to zero.
    const uint cache=p.transformsBias+index*32u;
    transforms.Store4(cache,asuint(q));
    transforms.Store3(cache+16u,asuint(center));
    transforms.Store(cache+28u,0u);
}
