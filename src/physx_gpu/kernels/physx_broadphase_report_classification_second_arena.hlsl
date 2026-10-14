// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::accumulateReportsStage_2 / BP_ACCUMULATE_REPORT_STAGE_2.
// Globally stable per-class report compaction using the source 32-block
// partition and independent actor/aggregate found/lost destinations.
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

groupshared uint scanA[256];
groupshared uint scanB[256];
groupshared uint blockBase;
groupshared uint carry;

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
    const uint outputField = aggregate ? (stream == 0u ? 16u : 20u) :
                                         (stream == 0u ? 24u : 28u);
    const uint output = arena.Load(desc + outputField);
    const uint capacity = aggregate ? arena.Load(desc + 60u) : maxPairs;
    const uint blockField = aggregate ? (stream == 0u ? 32u : 40u) :
                                         (stream == 0u ? 36u : 44u);
    const uint blocks = arena.Load(desc + blockField);
    const uint hasVolumes = arena.Load(desc + 64u);
    const uint volumeTags = arena.Load(desc + 48u);
    if (thread.x == 0u) {
        uint exclusive = 0u;
        uint all = 0u;
        [unroll]
        for (uint block = 0u; block < 32u; ++block) {
            const uint value = arena.Load(blocks + block * 4u);
            if (block < group.x) exclusive += value;
            all += value;
        }
        blockBase = exclusive;
        carry = 0u;
        if (aggregate && group.x == 0u) {
            const uint aggregateTotals = arena.Load(desc + 4u);
            arena.Store(aggregateTotals + stream * 4u, all);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint blocksRequired = (raw + 255u) / 256u;
    const uint iterations = (blocksRequired + 31u) / 32u;
    for (uint step = 0u; step < iterations; ++step) {
        const uint index = (group.x * iterations + step) * 256u + thread.x;
        bool eligible = false;
        uint2 pair = uint2(0u, 0u);
        if (index < safeCount) {
            pair = arena.Load2(pairs + index * 8u);
            bool singleA = true;
            bool singleB = true;
            if (hasVolumes != 0u) {
                singleA = arena.Load(volumeTags + pair.x * 4u) == 0xffffffffu;
                singleB = arena.Load(volumeTags + pair.y * 4u) == 0xffffffffu;
            }
            eligible = (!(singleA && singleB)) == aggregate;
        }
        scanA[thread.x] = eligible ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint shift = 1u; shift < 256u; shift <<= 1u) {
            if (shift == 1u || shift == 4u || shift == 16u || shift == 64u) {
                uint value = scanA[thread.x];
                if (thread.x >= shift) value += scanA[thread.x - shift];
                scanB[thread.x] = value;
            } else {
                uint value = scanB[thread.x];
                if (thread.x >= shift) value += scanB[thread.x - shift];
                scanA[thread.x] = value;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (eligible) {
            const uint destination = blockBase + carry + scanA[thread.x] - 1u;
            if (destination < capacity)
                arena.Store2(output + destination * 8u, pair);
        }
        // No wave may read old carry after the leader advances it.
        GroupMemoryBarrierWithGroupSync();
        if (thread.x == 0u) carry += scanA[255u];
        GroupMemoryBarrierWithGroupSync();
    }
}
