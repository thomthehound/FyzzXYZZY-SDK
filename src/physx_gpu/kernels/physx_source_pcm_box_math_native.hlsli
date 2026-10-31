// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

float3 rotateSource(float4 q,float3 v) {
    float3 u=q.xyz;
    precise float3 t=cross(u,v)*2.f;
    precise float3 rotated=v+(t*q.w+cross(u,t));
    return rotated;
}
// Preserve fused float32 source arithmetic without relying on
// backend-specific multiply-add contraction.
float fusedProductSum(float a,float b,float c) {
    precise float product=a*b;
    precise float sa=4097.f*a;
    precise float sb=4097.f*b;
    precise float highA=sa-(sa-a);
    precise float highB=sb-(sb-b);
    precise float lowA=a-highA,lowB=b-highB;
    precise float e0=highA*highB-product;
    precise float e1=e0+highA*lowB;
    precise float e2=e1+lowA*highB;
    precise float productError=e2+lowA*lowB;
    precise float sum=product+c;
    precise float bv=sum-product;
    precise float sumError=(product-(sum-bv))+(c-bv);
    precise float fused=sum+(productError+sumError);
    return fused;
}
float4 quatProduct(float4 a,float4 b) {
    return float4(
        a.w*b.x+b.w*a.x+a.y*b.z-a.z*b.y,
        a.w*b.y+b.w*a.y+a.z*b.x-a.x*b.z,
        a.w*b.z+b.w*a.z+a.x*b.y-a.y*b.x,
        a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z);
}
float4 conjugate(float4 q) {return float4(-q.xyz,q.w);}
