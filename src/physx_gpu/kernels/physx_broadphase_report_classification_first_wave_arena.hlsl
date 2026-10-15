// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Pinned PhysX broadphase.cu::accumulateReportsStage_1,
// BP_ACCUMULATE_REPORT_STAGE_1: original 32 x 4 x 256 dispatch.
//
// The reference uses per-warp ballot/popcount counts, then sums 8 warp
// subtotals. Unlike the old portable port's 256-thread shared reduction
// with eight full-group barriers, here every physical wave reduces its
// per-thread eligible count in native registers. One wave leader adds
// the sum to the workgroup total, preserving source output counts.
// Only two group barriers; no per-thread LDS reduction array.
// No matrix accelerator, fixed wave32 assumption, or new dispatch.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);
struct Parameters {
    uint descriptor_offset; uint arena_bytes;
    uint r0; uint r1; uint r2; uint r3; uint r4; uint r5;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint groupTotal;

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

    if (thread.x == 0u) groupTotal = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint localCount = 0u;
    for (uint step = 0u; step < iterations; ++step) {
        const uint index = (group.x * iterations + step) * 256u + thread.x;
        if (index < safeCount) {
            const uint a = arena.Load(pairs + index * 8u);
            const uint b = arena.Load(pairs + index * 8u + 4u);
            bool singleA = true, singleB = true;
            if (hasVolumes != 0u) {
                singleA = arena.Load(volumeTags + a * 4u) == 0xffffffffu;
                singleB = arena.Load(volumeTags + b * 4u) == 0xffffffffu;
            }
            if ((!(singleA && singleB)) == aggregate) ++localCount;
        }
    }
    const uint waveTotal = WaveActiveSum(localCount);
    if (WaveIsFirstLane()) {
        uint unused;
        InterlockedAdd(groupTotal, waveTotal, unused);
    }
    GroupMemoryBarrierWithGroupSync();
    if (thread.x == 0u) {
        const uint field = aggregate
            ? (stream == 0u ? 32u : 40u)
            : (stream == 0u ? 36u : 44u);
        const uint blockOutput = arena.Load(desc + field);
        arena.Store(blockOutput + group.x * 4u, groupTotal);
    }
}
