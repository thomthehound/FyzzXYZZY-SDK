// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// aggregate.cu::markAddedAndDeletedAggregatedBounds.
// Newly aggregated shapes have no previous integer AABB. Removed shapes
// have neither a current nor previous AABB during aggregate pair evaluation.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer newBounds : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer oldBounds : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer addedHandles : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer removedHandles : register(u3,space0);

struct Parameters {
    uint numAdded, numRemoved, newBias, oldBias;
    uint addedBias, removedBias, boundsCapacity, reserved;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

void clearAabb(RWByteAddressBuffer bounds,uint address) {
    bounds.Store3(address,uint3(0xff7fffffu,0xff7fffffu,0xff7fffffu));
    bounds.Store3(address+12u,uint3(0x00800000u,0x00800000u,0x00800000u));
}

[numthreads(128,2,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint i=group.x*128u+lane.x;
    const bool removed=lane.y!=0u;
    if(i >= (removed ? p.numRemoved : p.numAdded))return;
    const uint handle=removed
        ? removedHandles.Load(p.removedBias+i*4u)
        : addedHandles.Load(p.addedBias+i*4u);
    // Handles are source AABB-manager admission IDs validated before DMA.
    if(handle>=p.boundsCapacity)return;
    const uint offset=handle*24u;
    if(removed)clearAabb(newBounds,p.newBias+offset);
    clearAabb(oldBounds,p.oldBias+offset);
}
