// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::outputOverlapChecksForRegionHistogram
// BP_OUTPUT_OVERLAPCHECKS_HISTOGRAM: source 32 x 256.
// Native subgroup prefix of the 32 block totals. When the actual
// hardware wave is narrower than 32, a leader scans only the handful
// of wave totals; no 32-wide serial device loads or global atomics.

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
uint64_t read64(uint offset) {
    const uint2 words = arena.Load2(offset);
    return uint64_t(words.x) | (uint64_t(words.y) << 32u);
}
void write64(uint offset, uint64_t value) {
    arena.Store2(offset, uint2(uint(value), uint(value >> 32u)));
}

groupshared uint nextWave;
groupshared uint64_t blockBase[32];
groupshared uint64_t waveTotals[64];
groupshared uint64_t waveBase[64];
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint d = params.descriptor_offset;
    const uint counts = arena.Load(d + 0u);
    const uint blocks = arena.Load(d + 4u);
    const uint totalOffset = arena.Load(d + 8u);
    const uint total = arena.Load(arena.Load(d + 12u));
    const uint iterations = ((total + 255u) / 256u + 31u) / 32u;

    if (thread.x == 0u) nextWave = 0u;
    GroupMemoryBarrierWithGroupSync();
    const uint lanes = WaveGetLaneCount();
    const uint lane = WaveGetLaneIndex();
    uint wave = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(nextWave, 1u, wave);
    wave = WaveReadLaneFirst(wave);
    const uint position = wave * lanes + lane;
    uint64_t prefix = 0ull;
    // Only the first ceil(32/waveWidth) physical waves participate.
    // The condition is uniform across each wave and leaves no divergent
    // wave-intrinsic calls.
    if (wave * lanes < 32u) {
        const uint64_t value = position < 32u ?
            read64(blocks + position * 8u) : 0ull;
        prefix = WavePrefixSum(value);
        const uint64_t waveSum = WaveActiveSum(value);
        if (WaveIsFirstLane()) waveTotals[wave] = waveSum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (thread.x == 0u) {
        uint64_t sum = 0ull;
        const uint usedWaves = (32u + lanes - 1u) / lanes;
        for (uint w = 0u; w < usedWaves; ++w) {
            waveBase[w] = sum;
            sum += waveTotals[w];
        }
        if (group.x == 0u) write64(totalOffset, sum);
    }
    GroupMemoryBarrierWithGroupSync();
    if (position < 32u)
        blockBase[position] = waveBase[wave] + prefix;
    GroupMemoryBarrierWithGroupSync();
    const uint64_t preceding = blockBase[group.x];
    for (uint step = 0u; step < iterations; ++step) {
        const uint index = (group.x * iterations + step) * 256u +
                           wave * lanes + lane;
        if (index < total)
            write64(counts + index * 8u, read64(counts + index * 8u) + preceding);
    }
}
