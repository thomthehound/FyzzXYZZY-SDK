// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Pinned PhysX broadphase.cu::accumulateReportsStage_2,
// BP_ACCUMULATE_REPORT_STAGE_2, original 32 x 4 x 256 dispatch.
//
// Retain the 32-block exclusive base, source stable compaction order,
// per-iteration carry and found/lost/aggregate/actor publication.
// Native WavePrefixSum of 0/1 report eligibility replaces the original
// *port's* 256-thread shared Hillis-Steele scan. Each native subgroup
// claims a contiguous logical 256-entry slice ONCE; wave totals are
// prefixed in logical order with only three group barriers/iteration.
// Two 64-word shared arrays = 512 B, vs two 256-word arrays = 2 KiB.
// No Wave Matrix Multiply Accumulate instruction is required.
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

groupshared uint nextWave;
groupshared uint waveTotals[64];
groupshared uint waveBases[64];
groupshared uint blockBase;
groupshared uint carry;
groupshared uint iterationCarry;
groupshared uint localBlockPrefix;

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

    if (thread.x == 0u) nextWave = 0u;
    GroupMemoryBarrierWithGroupSync();

    const uint laneCount = WaveGetLaneCount();
    const uint laneIndex = WaveGetLaneIndex();
    const uint waveCount = 256u / laneCount;
    uint waveId = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(nextWave, 1u, waveId);
    waveId = WaveReadLaneFirst(waveId);
    const uint logicalThread = waveId * laneCount + laneIndex;

    // Original CUDA prefixes the 32 block totals with one warp scan.
    // Do not serialize 32 device loads through one leader (as the old
    // port did). Coalesced logical lanes load all 32 source block totals;
    // a native wave prefix and a tiny cross-wave carry reconstruct the
    // original exclusive block base for this workgroup.
    const uint blockValue = logicalThread < 32u
        ? arena.Load(blocks + logicalThread * 4u) : 0u;
    const uint blockPrefix = WavePrefixSum(blockValue);
    const uint blockWaveTotal = WaveActiveSum(blockValue);
    if (WaveIsFirstLane()) waveTotals[waveId] = blockWaveTotal;
    if (logicalThread == group.x) localBlockPrefix = blockPrefix;
    GroupMemoryBarrierWithGroupSync();

    if (thread.x == 0u) {
        uint all = 0u;
        uint earlier = 0u;
        const uint selectedWave = group.x / laneCount;
        for (uint w = 0u; w < waveCount; ++w) {
            const uint count = waveTotals[w];
            if (w < selectedWave) earlier += count;
            all += count;
        }
        blockBase = earlier + localBlockPrefix;
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
        const uint index = (group.x * iterations + step) * 256u +
                           waveId * laneCount + laneIndex;
        bool eligible = false;
        uint2 pair = uint2(0u, 0u);
        if (index < safeCount) {
            pair = arena.Load2(pairs + index * 8u);
            bool singleA = true, singleB = true;
            if (hasVolumes != 0u) {
                singleA = arena.Load(volumeTags + pair.x * 4u) == 0xffffffffu;
                singleB = arena.Load(volumeTags + pair.y * 4u) == 0xffffffffu;
            }
            eligible = (!(singleA && singleB)) == aggregate;
        }

        const uint flag = eligible ? 1u : 0u;
        const uint wavePrefix = WavePrefixSum(flag);
        const uint waveTotal = WaveActiveSum(flag);
        if (WaveIsFirstLane()) waveTotals[waveId] = waveTotal;
        GroupMemoryBarrierWithGroupSync();

        if (thread.x == 0u) {
            uint subtotal = 0u;
            for (uint w = 0u; w < waveCount; ++w) {
                const uint count = waveTotals[w];
                waveBases[w] = subtotal;
                subtotal += count;
            }
            iterationCarry = carry;
            carry += subtotal;
        }
        GroupMemoryBarrierWithGroupSync();

        if (eligible) {
            const uint destination = blockBase + iterationCarry +
                                     waveBases[waveId] + wavePrefix;
            if (destination < capacity)
                arena.Store2(output + destination * 8u, pair);
        }
        // Guard shared bases and carry across successive report chunks.
        GroupMemoryBarrierWithGroupSync();
    }
}
