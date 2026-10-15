// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Source broadphase.cu::computeHistogram, 32 x 256.
// Per-group local exclusive prefix with cross-iteration carry. Port of
// the existing arena implementation; now binds source CUDA allocations.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer accum : register(u0, space0);
#ifdef __spirv__
[[vk::binding(1, 0)]]
#endif
RWByteAddressBuffer blocks : register(u1, space0);
#ifdef __spirv__
[[vk::binding(2, 0)]]
#endif
RWByteAddressBuffer sourceDesc : register(u2, space0);
struct Parameters {
    uint boxes; uint accumBias; uint blockBias; uint descriptorBias;
    uint totalFieldOffset; uint r0; uint r1; uint r2;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint scanA[256];
groupshared uint scanB[256];
groupshared uint carry;

[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID, uint3 thread:SV_GroupThreadID) {
    const uint projections = params.boxes*2u;
    const uint iterations = ((projections+255u)/256u+31u)/32u;
    if(thread.x==0u) carry=0u;
    GroupMemoryBarrierWithGroupSync();
    for(uint step=0u;step<iterations;++step) {
        const uint index=(group.x*iterations+step)*256u+thread.x;
        const uint value=index<projections ?
            accum.Load(params.accumBias+index*4u) : 0u;
        scanA[thread.x]=value;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for(uint shift=1u;shift<256u;shift<<=1u) {
            if(shift==1u || shift==4u || shift==16u || shift==64u) {
                uint sum=scanA[thread.x];
                if(thread.x>=shift)sum+=scanA[thread.x-shift];
                scanB[thread.x]=sum;
            } else {
                uint sum=scanB[thread.x];
                if(thread.x>=shift)sum+=scanB[thread.x-shift];
                scanA[thread.x]=sum;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if(index<projections)
            accum.Store(params.accumBias+index*4u,
                        carry+scanA[thread.x]-value);
        GroupMemoryBarrierWithGroupSync();
        if(thread.x==0u)carry+=scanA[255u];
        GroupMemoryBarrierWithGroupSync();
    }
    if(thread.x==0u)blocks.Store(params.blockBias+group.x*4u,carry);
}
