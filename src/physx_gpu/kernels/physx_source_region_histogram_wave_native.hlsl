// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Source broadphase.cu::computeHistogram, native-wave fast path.
// Wave indices are claimed atomically, independent of physical thread
// assignment. Does not hardcode NVIDIA's 32-lane warp width.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer accum : register(u0, space0);
#ifdef __spirv__
[[vk::binding(1, 0)]]
#endif
RWByteAddressBuffer blocks : register(u1, space0);
struct Parameters {
    uint boxes; uint accumBias; uint blockBias; uint descriptorBias;
    uint totalFieldOffset; uint r0; uint r1; uint r2;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif
groupshared uint nextWave;
groupshared uint waveTotals[64];
groupshared uint waveBases[64];
groupshared uint carry;
groupshared uint iterationCarry;

[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID, uint3 thread:SV_GroupThreadID) {
    const uint projections=params.boxes*2u;
    const uint iterations=((projections+255u)/256u+31u)/32u;
    if(thread.x==0u){nextWave=0u;carry=0u;}
    GroupMemoryBarrierWithGroupSync();
    const uint lanes=WaveGetLaneCount();
    const uint waveCount=256u/lanes;
    uint waveId=0u;
    if(WaveIsFirstLane())InterlockedAdd(nextWave,1u,waveId);
    waveId=WaveReadLaneFirst(waveId);
    for(uint step=0u;step<iterations;++step) {
        const uint index=(group.x*iterations+step)*256u+
            waveId*lanes+WaveGetLaneIndex();
        const uint value=index<projections ?
            accum.Load(params.accumBias+index*4u) : 0u;
        const uint prefix=WavePrefixSum(value);
        const uint total=WaveActiveSum(value);
        if(WaveIsFirstLane())waveTotals[waveId]=total;
        GroupMemoryBarrierWithGroupSync();
        if(thread.x==0u){
            uint running=0u;
            for(uint w=0u;w<waveCount;++w){
                waveBases[w]=running;
                running+=waveTotals[w];
            }
            iterationCarry=carry;
            carry+=running;
        }
        GroupMemoryBarrierWithGroupSync();
        if(index<projections)
            accum.Store(params.accumBias+index*4u,
                iterationCarry+waveBases[waveId]+prefix);
        GroupMemoryBarrierWithGroupSync();
    }
    if(thread.x==0u)blocks.Store(params.blockBias+group.x*4u,carry);
}
