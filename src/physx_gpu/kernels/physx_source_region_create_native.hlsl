// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Source: broadphase.cu::createRegionsKernel / BP_CREATE_REGIONS.
// The source descriptor's original GPU allocations are separately bound
// instead of copied into a synthetic arena. All uint byte-offset biases
// come from resolving the source's original CUdeviceptr addresses.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
ByteAddressBuffer handlesX : register(t0,space0);
#ifdef __spirv__
[[vk::binding(1, 0)]]
#endif
ByteAddressBuffer sapX : register(t1,space0);
#ifdef __spirv__
[[vk::binding(2, 0)]]
#endif
ByteAddressBuffer sapY : register(t2,space0);
#ifdef __spirv__
[[vk::binding(3, 0)]]
#endif
ByteAddressBuffer sapZ : register(t3,space0);
#ifdef __spirv__
[[vk::binding(4, 0)]]
#endif
ByteAddressBuffer endsX : register(t4,space0);
#ifdef __spirv__
[[vk::binding(5, 0)]]
#endif
ByteAddressBuffer endsY : register(t5,space0);
#ifdef __spirv__
[[vk::binding(6, 0)]]
#endif
ByteAddressBuffer endsZ : register(t6,space0);
#ifdef __spirv__
[[vk::binding(7, 0)]]
#endif
RWByteAddressBuffer ranges : register(u0,space0);
#ifdef __spirv__
[[vk::binding(8, 0)]]
#endif
RWByteAddressBuffer startAccum : register(u1,space0);
#ifdef __spirv__
[[vk::binding(9, 0)]]
#endif
RWByteAddressBuffer regionAccum : register(u2,space0);

struct Parameters {
    uint boxes;
    uint handlesBias;
    uint sapXBias;
    uint sapYBias;
    uint sapZBias;
    uint endsXBias;
    uint endsYBias;
    uint endsZBias;
    uint rangesBias;
    uint startAccumBias;
    uint regionAccumBias;
    uint reserved;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0,space0);
#endif

[numthreads(256,1,1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint projections = params.boxes * 2u;
    const uint iterations = ((projections + 255u) / 256u + 31u) / 32u;
    const uint elementsPerRegion = (params.boxes + 3u) / 4u;
    for(uint step=0u; step<iterations; ++step) {
        const uint index = (group.x * iterations + step)*256u + thread.x;
        if(index >= projections) continue;
        const uint tagged=handlesX.Load(params.handlesBias + index*4u);
        uint regionCount=0u;
        if((tagged & 1u)!=0u) {
            const uint handle=tagged>>3u;
            const uint indexX0=sapX.Load(params.sapXBias + handle*8u);
            const uint indexX1=sapX.Load(params.sapXBias + handle*8u + 4u);
            const uint indexY0=sapY.Load(params.sapYBias + handle*8u);
            const uint indexY1=sapY.Load(params.sapYBias + handle*8u + 4u);
            const uint indexZ0=sapZ.Load(params.sapZBias + handle*8u);
            const uint indexZ1=sapZ.Load(params.sapZBias + handle*8u + 4u);
            const uint3 first = uint3(
                endsX.Load(params.endsXBias + indexX0*4u),
                endsY.Load(params.endsYBias + indexY0*4u),
                endsZ.Load(params.endsZBias + indexZ0*4u)) / elementsPerRegion;
            const uint3 last = uint3(
                endsX.Load(params.endsXBias + indexX1*4u),
                endsY.Load(params.endsYBias + indexY1*4u),
                endsZ.Load(params.endsZBias + indexZ1*4u)) / elementsPerRegion;

            ranges.Store4(params.rangesBias + handle*32u,
                          uint4(first.x,first.y,first.z,0u));
            ranges.Store4(params.rangesBias + handle*32u + 16u,
                          uint4(last.x,last.y,last.z,0u));
            const uint yz=(last.y-first.y+1u)*(last.z-first.z+1u);
            startAccum.Store(params.startAccumBias + indexX0*4u,yz);
            startAccum.Store(params.startAccumBias + indexX1*4u,0u);
            if(last.x>=first.x)
                regionCount=(last.x-first.x+1u)*yz;
        }
        regionAccum.Store(params.regionAccumBias + index*4u,regionCount);
    }
}
