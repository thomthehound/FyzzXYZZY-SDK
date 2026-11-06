// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_source_box_bounds_math_native.hlsli"

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
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer geometry : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer hullVertices : register(u4,space0);

struct Parameters {
    uint shapeCount;
    uint shapesBias;
    uint boundsBias;
    uint transformsBias;
    uint geometryBias;
    uint verticesBias;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

[numthreads(128,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    if(thread.x>=p.shapeCount)return;

    const uint base=p.shapesBias+thread.x*96u;
    const uint index=shapes.Load(base);
    const float3 center=asfloat(shapes.Load3(base+4u));
    const float4 q=asfloat(shapes.Load4(base+16u));
    const float3 e=asfloat(shapes.Load3(base+32u));
    const float contactDistance=asfloat(shapes.Load(base+44u));

    // PxMat33(quaternion), three column vectors. updateCacheAndBound's
    // eBOX branch uses basisExtent, then center +/- world-space extents.
    float3 lower,upper;
    const uint kind=shapes.Load(base+60u);
    if(kind==2u) {
        const uint first=shapes.Load(base+84u);
        const uint count=shapes.Load(base+88u);
        lower=float3(3.402823466e38f,3.402823466e38f,3.402823466e38f);
        upper=-lower;
        for(uint j=0u;j<count;++j) {
            const float3 vertex=asfloat(hullVertices.Load3(
                p.verticesBias+(first+j)*12u));
            const float3 position=center+sourceRigidRotate(q,vertex);
            lower=min(lower,position);
            upper=max(upper,position);
        }
    }else {
        const float3 extents=kind==1u?e:sourceBoxHalfBounds(q,e);
        lower=center-extents;
        upper=center+extents;
    }
    const uint bound=p.boundsBias+index*24u;
    bounds.Store3(bound,asuint(lower));
    bounds.Store3(bound+12u,asuint(upper));

    // PxsCachedTransform: PxTransform (q[4], p[3]) and flags, 32 bytes.
    // The existing source kernel initializes transform flags to zero.
    const uint cache=p.transformsBias+index*32u;
    transforms.Store4(cache,asuint(q));
    transforms.Store3(cache+16u,asuint(center));
    transforms.Store(cache+28u,0u);

    const uint geometryBase=p.geometryBias+index*64u;
    geometry.Store3(geometryBase,asuint(e));
    geometry.Store(geometryBase+12u,asuint(contactDistance));
    geometry.Store4(geometryBase+16u,shapes.Load4(base+48u));
    geometry.Store4(geometryBase+32u,shapes.Load4(base+64u));
    geometry.Store4(geometryBase+48u,shapes.Load4(base+80u));
}
