// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// GuContactBoxBox face-axis selection and optional cross-axis separation.
// Shape indices address the resident Sc element transform/geometry caches.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer transforms : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer geometry : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer pairs : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer axes : register(u3,space0);

struct Parameters {
    uint pairCount;
    uint transformBias;
    uint geometryBias;
    uint pairBias;
    uint axisBias;
    uint shapeCapacity;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

float3x3 quaternionBasis(float4 q) {
    // Preserve Gu's PxMat33T construction order, including 1-yy-zz.
    float x2=q.x+q.x, y2=q.y+q.y, z2=q.z+q.z;
    float xx=x2*q.x, yy=y2*q.y, zz=z2*q.z;
    float xy=x2*q.y, xz=x2*q.z, xw=x2*q.w;
    float yz=y2*q.z, yw=y2*q.w, zw=z2*q.w;
    float3 c0=float3(1.f-yy-zz,xy+zw,xz-yw);
    float3 c1=float3(xy-zw,1.f-xx-zz,yz+xw);
    float3 c2=float3(xz+yw,yz-xw,1.f-xx-yy);
    // Rows correspond to the three basis columns.
    return float3x3(c0,c1,c2);
}
uint classify(uint a,uint b,uint previousAxis) {
    if(a>=p.shapeCapacity || b>=p.shapeCapacity || a==b ||
       previousAxis>6u)return 0xffffffffu;

    uint t0=p.transformBias+a*32u;
    uint t1=p.transformBias+b*32u;
    uint g0=p.geometryBias+a*16u;
    uint g1=p.geometryBias+b*16u;
    float4 q0=asfloat(transforms.Load4(t0));
    float4 q1=asfloat(transforms.Load4(t1));
    float3 p0=asfloat(transforms.Load3(t0+16u));
    float3 p1=asfloat(transforms.Load3(t1+16u));
    float3 e0=asfloat(geometry.Load3(g0));
    float3 e1=asfloat(geometry.Load3(g1));
    float contactDistance=asfloat(geometry.Load(g0+12u))+
                          asfloat(geometry.Load(g1+12u));

    float3x3 A=quaternionBasis(q0);
    float3x3 B=quaternionBasis(q1);
    float3 kD=p1-p0;
    float C[3][3], absC[3][3], ad[3], overlap[6];

    // GuContactBoxBox Class I and II tests in the original axis order.
    [unroll]
    for(uint i=0u;i<3u;++i) {
        ad[i]=dot(A[i],kD);
        [unroll]
        for(uint j=0u;j<3u;++j) {
            C[i][j]=dot(A[i],B[j]);
            absC[i][j]=1.0e-6f+abs(C[i][j]);
        }
        overlap[i]=e0[i]+e1.x*absC[i][0]+e1.y*absC[i][1]+
                   e1.z*absC[i][2]-abs(ad[i])+contactDistance;
        if(overlap[i]<0.f)return 0u;
    }
    [unroll]
    for(uint j=0u;j<3u;++j) {
        float d=dot(B[j],kD);
        overlap[j+3u]=e1[j]+e0.x*absC[0][j]+e0.y*absC[1][j]+
                      e0.z*absC[2][j]-abs(d)+contactDistance;
        if(overlap[j+3u]<0.f)return 0u;
    }

    // Class III tests are used only for new or previously separated pairs.
    if(previousAxis==0u) {
        [unroll]
        for(uint i=0u;i<3u;++i) {
            uint i1=(i+1u)%3u, i2=(i+2u)%3u;
            [unroll]
            for(uint j=0u;j<3u;++j) {
                uint j1=(j+1u)%3u,j2=(j+2u)%3u;
                float d=ad[i2]*C[i1][j]-ad[i1]*C[i2][j];
                float extent=contactDistance+
                    e0[i1]*absC[i2][j]+e0[i2]*absC[i1][j]+
                    e1[j1]*absC[i][j2]+e1[j2]*absC[i][j1];
                if(abs(d)>extent)return 0u;
            }
        }
    }

    if(previousAxis!=0u)overlap[previousAxis-1u]*=0.999f;
    float minimum=3.402823466e+38f;
    uint winner=1u;
    [unroll]
    for(uint axis=0u;axis<6u;++axis) {
        if(overlap[axis]>=0.f && overlap[axis]<minimum) {
            minimum=overlap[axis];
            winner=axis+1u;
        }
    }
    return winner;
}

[numthreads(128,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint index=thread.x;
    if(index>=p.pairCount)return;
    uint base=p.pairBias+index*24u;
    uint a=pairs.Load(base);
    uint b=pairs.Load(base+4u);
    uint previousAxis=pairs.Load(base+8u);
    axes.Store(p.axisBias+index*4u,classify(a,b,previousAxis));
}
