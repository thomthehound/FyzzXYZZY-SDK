// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// Source aggregate.cu::updateAggregateBounds.
// One 32-lane source warp computes the inflated union of an aggregate's
// admitted member bounds. Four warps share each native workgroup.
// CUDA aggregate device pointers are lowered to dense, source-owned
// descriptor/member storage; no CPU reduction or alternate BP runs.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer bounds : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer contactDistances : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer aggregateSlots : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer memberIndices : register(u3,space0);

// Root handle, first member in the source index array, current member
// count and reserved padding. Validity follows PxgAggregate::isValid.
struct Parameters {
    uint aggregateCount, reserved0, reserved1, reserved2;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

groupshared float3 minOfWarp[4][32];
groupshared float3 maxOfWarp[4][32];

[numthreads(32,4,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    uint aggregate=group.x*4u+lane.y;
    uint4 descriptor=aggregate<p.aggregateCount
        ? aggregateSlots.Load4(aggregate*16u)
        : uint4(0u,0u,0u,0u);
    bool valid=aggregate<p.aggregateCount &&
        descriptor.z!=0u && descriptor.x!=0xffffffffu;
    const float empty=8.5070586659632215e37f; // PX_MAX_REAL * 0.25
    float3 low=float3(empty,empty,empty);
    float3 high=-low;
    if(valid) {
        for(uint i=lane.x;i<descriptor.z;i+=32u) {
            uint handle=memberIndices.Load((descriptor.y+i)*4u);
            float3 memberLow=asfloat(bounds.Load3(handle*24u));
            float3 memberHigh=asfloat(bounds.Load3(handle*24u+12u));
            float inflate=asfloat(contactDistances.Load(handle*4u));
            low=min(low,memberLow-inflate);
            high=max(high,memberHigh+inflate);
        }
    }
    minOfWarp[lane.y][lane.x]=low;
    maxOfWarp[lane.y][lane.x]=high;
    GroupMemoryBarrierWithGroupSync();
    // __shfl_xor_sync source reduction, lowered to shared workgroup
    // storage because native subgroup width need not be 32.
    [unroll] for(uint radius=16u;radius!=0u;radius>>=1u) {
        if(lane.x<radius) {
            minOfWarp[lane.y][lane.x]=min(
                minOfWarp[lane.y][lane.x],
                minOfWarp[lane.y][lane.x+radius]);
            maxOfWarp[lane.y][lane.x]=max(
                maxOfWarp[lane.y][lane.x],
                maxOfWarp[lane.y][lane.x+radius]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if(valid && lane.x==0u) {
        bounds.Store3(descriptor.x*24u,asuint(minOfWarp[lane.y][0]));
        bounds.Store3(descriptor.x*24u+12u,asuint(maxOfWarp[lane.y][0]));
    }
}
