// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::accumulateReportsStage_1 / BP_ACCUMULATE_REPORT_STAGE_1.
// blockIdx.y: aggFound=0, actorFound=1, aggLost=2, actorLost=3.
// Source partitions 32 consecutive blocks, ceil(ceil(raw/256)/32)
// 256-element iterations per block. Only the retained raw report prefix
// is read, even though the iteration count uses full raw counters.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset;
    uint arena_bytes;
    uint r0; uint r1; uint r2; uint r3; uint r4; uint r5;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint partial[256];

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint desc = params.descriptor_offset;
    const uint stream = group.y / 2u;
    const bool aggregate = (group.y & 1u) == 0u;
    const uint counts = arena.Load(desc);
    const uint raw = arena.Load(counts + stream * 4u);
    const uint maxPairs = arena.Load(desc + 56u);
    const uint safeCount = min(raw, maxPairs);
    const uint pairs = arena.Load(desc + (stream == 0u ? 8u : 12u));
    const uint hasVolumes = arena.Load(desc + 64u);
    const uint volumeTags = arena.Load(desc + 48u);
    const uint blocksRequired = (raw + 255u) / 256u;
    const uint iterations = (blocksRequired + 31u) / 32u;

    uint localCount = 0u;
    for (uint step = 0u; step < iterations; ++step) {
        const uint index = (group.x * iterations + step) * 256u + thread.x;
        if (index < safeCount) {
            const uint a = arena.Load(pairs + index * 8u);
            const uint b = arena.Load(pairs + index * 8u + 4u);
            bool singleA = true;
            bool singleB = true;
            if (hasVolumes != 0u) {
                singleA = arena.Load(volumeTags + a * 4u) == 0xffffffffu;
                singleB = arena.Load(volumeTags + b * 4u) == 0xffffffffu;
            }
            if ((!(singleA && singleB)) == aggregate) ++localCount;
        }
    }
    partial[thread.x] = localCount;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (thread.x < stride) partial[thread.x] += partial[thread.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread.x == 0u) {
        const uint field = aggregate
            ? (stream == 0u ? 32u : 40u)
            : (stream == 0u ? 36u : 44u);
        const uint out = arena.Load(desc + field);
        arena.Store(out + group.x * 4u, partial[0]);
    }
}
