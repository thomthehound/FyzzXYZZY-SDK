// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Gu/PxBounds3::basisExtent for an absolute box pose. Both Sc's
// authored-transform stage and Dy's integrated-pose stage use this
// identical source extent calculation.
float3 sourceBoxHalfBounds(float4 q,float3 e) {
    const float xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
    const float xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
    const float wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
    const float3 c0=float3(1.0f-2.0f*(yy+zz),
                           2.0f*(xy+wz),2.0f*(xz-wy));
    const float3 c1=float3(2.0f*(xy-wz),
                           1.0f-2.0f*(xx+zz),2.0f*(yz+wx));
    const float3 c2=float3(2.0f*(xz+wy),
                           2.0f*(yz-wx),1.0f-2.0f*(xx+yy));
    return abs(c0)*e.x+abs(c1)*e.y+abs(c2)*e.z;
}
