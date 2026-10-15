// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Source broadphase.cu::outputHistogram, 32 x 256. Publishes source's
// total directly into the ORIGINAL GPU-resident PxgBroadPhaseDesc.
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

groupshared uint blockBase;
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID, uint3 thread:SV_GroupThreadID){
    if(thread.x==0u) {
        uint preceding=0u;
        uint total=0u;
        [unroll]
        for(uint i=0u;i<32u;++i){
            const uint count=blocks.Load(params.blockBias+i*4u);
            if(i<group.x)preceding+=count;
            total+=count;
        }
        blockBase=preceding;
        if(group.x==0u)
            sourceDesc.Store(params.descriptorBias+params.totalFieldOffset,total);
    }
    GroupMemoryBarrierWithGroupSync();
    const uint projections=params.boxes*2u;
    const uint iterations=((projections+255u)/256u+31u)/32u;
    for(uint step=0u;step<iterations;++step){
        const uint index=(group.x*iterations+step)*256u+thread.x;
        if(index<projections)
            accum.Store(params.accumBias+index*4u,
                accum.Load(params.accumBias+index*4u)+blockBase);
    }
}
