// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// Source aggregate.cu::markAggregateBoundsUpdatedBitmap.
// Each source warp marks the aggregate's root as dirty and removes dirty
// member bits, so normal SAP updates consume the AGGREGATE AABB rather
// than individually inserted members. Original atomic bit operations.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer aggregateSlots : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer memberIndices : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer changedAabbManagerHandles : register(u2,space0);
struct Parameters {
    uint aggregateCount, boundsBias, contactBias, aggregateBias;
    uint memberBias, changedBias, reserved0, reserved1;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif
[numthreads(32,4,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint index=group.x*4u+lane.y;
    if(index>=p.aggregateCount)return;
    const uint4 aggregate=aggregateSlots.Load4(p.aggregateBias+index*16u);
    if(aggregate.z==0u || aggregate.x==0xffffffffu)return;
    if(lane.x==0u) {
        uint ignored;
        changedAabbManagerHandles.InterlockedOr(
            p.changedBias+(aggregate.x>>5u)*4u,1u<<(aggregate.x&31u),ignored);
    }
    for(uint i=lane.x;i<aggregate.z;i+=32u) {
        const uint member=memberIndices.Load(p.memberBias+(aggregate.y+i)*4u);
        uint ignored;
        changedAabbManagerHandles.InterlockedAnd(
            p.changedBias+(member>>5u)*4u,~(1u<<(member&31u)),ignored);
    }
}
