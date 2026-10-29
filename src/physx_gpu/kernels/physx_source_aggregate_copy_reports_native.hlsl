// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// aggregate.cu::aggCopyReports.
// Copy original GPU pair reports to the source mapped-report heap. The
// existing GPU counts remain unsaturated for original overflow detection.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer foundSource : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer lostSource : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer foundMapped : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer lostMapped : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer pairCounts : register(u4,space0);

struct Parameters {
    uint capacity, sourceFoundBias, sourceLostBias;
    uint mappedFoundBias, mappedLostBias, countBias;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID) {
    const uint index=group.x*256u+thread.x;
    const uint stride=256u*min(64u,(p.capacity*2u+255u)/256u);
    const uint foundCount=min(pairCounts.Load(p.countBias),p.capacity);
    const uint lostCount=min(pairCounts.Load(p.countBias+4u),p.capacity);
    for(uint i=index;i<foundCount*2u;i+=stride) {
        foundMapped.Store(p.mappedFoundBias+i*4u,
                          foundSource.Load(p.sourceFoundBias+i*4u));
    }
    for(uint i=index;i<lostCount*2u;i+=stride) {
        lostMapped.Store(p.mappedLostBias+i*4u,
                         lostSource.Load(p.sourceLostBias+i*4u));
    }
}
