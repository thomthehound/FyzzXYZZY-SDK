// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::outputHistogram, invoked by
// BP_OUTPUT_START_REGION_HISTOGRAM and BP_OUTPUT_REGION_HISTOGRAM.
// Adds preceding block totals to per-block exclusive sums and writes
// the source's grand total to the descriptor once, from group 0.
#ifndef FYZZXYZZY_REGION_START
#define FYZZXYZZY_REGION_START 1
#endif
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);
struct Parameters {
    uint descriptor_offset;
    uint boxes;
    uint arena_bytes;
    uint r0; uint r1; uint r2; uint r3; uint r4;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint blockBase;
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint desc = params.descriptor_offset;
#if FYZZXYZZY_REGION_START
    const uint data = arena.Load(desc + 0u);
    const uint blocks = arena.Load(desc + 8u);
    const uint total = arena.Load(desc + 16u);
#else
    const uint data = arena.Load(desc + 4u);
    const uint blocks = arena.Load(desc + 12u);
    const uint total = arena.Load(desc + 20u);
#endif
    if (thread.x == 0u) {
        uint preceding = 0u;
        uint all = 0u;
        [unroll]
        for (uint i = 0u; i < 32u; ++i) {
            const uint count = arena.Load(blocks + i * 4u);
            if (i < group.x) preceding += count;
            all += count;
        }
        blockBase = preceding;
        if (group.x == 0u) arena.Store(total, all);
    }
    GroupMemoryBarrierWithGroupSync();
    const uint projections = params.boxes * 2u;
    const uint iterations = ((projections + 255u) / 256u + 31u) / 32u;
    for (uint step = 0u; step < iterations; ++step) {
        const uint index = (group.x * iterations + step) * 256u + thread.x;
        if (index < projections)
            arena.Store(data + index * 4u,
                        arena.Load(data + index * 4u) + blockBase);
    }
}
