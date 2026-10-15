// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::computeOverlapChecksForRegionsHistogram
// BP_COMPUTE_OVERLAPCHECKS_HISTOGRAM: source 32 x 256, 64-bit
// per-item exclusive prefixes and 32 block counts.
// Native subgroup scans; the tiny wave-total prefix is computed by
// one group leader. No 256-wide Hillis-Steele scan or global atomics.

// PhysX native resource binding for the original overlap histogram scan.
// Input/output allocations are separately owned by the source GPU heap.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer countsBuffer : register(u0, space0);
#ifdef __spirv__
[[vk::binding(1, 0)]]
#endif
RWByteAddressBuffer blockBuffer : register(u1, space0);
#ifdef __spirv__
[[vk::binding(2, 0)]]
#endif
RWByteAddressBuffer regionCounter : register(u2, space0);
#ifdef __spirv__
[[vk::binding(3, 0)]]
#endif
RWByteAddressBuffer overlapTotal : register(u3, space0);

struct Parameters {
    uint counts_offset;
    uint blocks_offset;
    uint region_total_offset;
    uint overlap_total_offset;
    uint r0; uint r1; uint r2; uint r3;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

uint64_t readCounts64(uint at) {
    const uint2 words = countsBuffer.Load2(at);
    return uint64_t(words.x) | (uint64_t(words.y) << 32u);
}
uint64_t readBlocks64(uint at) {
    const uint2 words = blockBuffer.Load2(at);
    return uint64_t(words.x) | (uint64_t(words.y) << 32u);
}
void writeCounts64(uint at, uint64_t val) {
    countsBuffer.Store2(at, uint2(uint(val), uint(val >> 32u)));
}
void writeBlocks64(uint at, uint64_t val) {
    blockBuffer.Store2(at, uint2(uint(val), uint(val >> 32u)));
}
void writeTotal64(uint at, uint64_t val) {
    overlapTotal.Store2(at, uint2(uint(val), uint(val >> 32u)));
}

groupshared uint nextWave;
groupshared uint64_t waveTotals[64];
groupshared uint64_t waveBases[64];
groupshared uint64_t blockCarry;
groupshared uint64_t iterationCarry;
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint counts = params.counts_offset;
    const uint blocks = params.blocks_offset;
    const uint total = regionCounter.Load(params.region_total_offset);
    const uint iterations = ((total + 255u) / 256u + 31u) / 32u;

    if (thread.x == 0u) {
        nextWave = 0u;
        blockCarry = 0ull;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint lanes = WaveGetLaneCount();
    const uint lane = WaveGetLaneIndex();
    const uint waves = 256u / lanes;
    uint wave = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(nextWave, 1u, wave);
    wave = WaveReadLaneFirst(wave);

    for (uint step = 0u; step < iterations; ++step) {
        const uint index = (group.x * iterations + step) * 256u +
                           wave * lanes + lane;
        const uint64_t value = index < total ? readCounts64(counts + index * 8u) : 0ull;
        const uint64_t prefix = WavePrefixSum(value);
        const uint64_t waveSum = WaveActiveSum(value);
        if (WaveIsFirstLane()) waveTotals[wave] = waveSum;
        GroupMemoryBarrierWithGroupSync();
        if (thread.x == 0u) {
            uint64_t sum = 0ull;
            for (uint w = 0u; w < waves; ++w) {
                waveBases[w] = sum;
                sum += waveTotals[w];
            }
            iterationCarry = blockCarry;
            blockCarry += sum;
        }
        GroupMemoryBarrierWithGroupSync();
        if (index < total)
            writeCounts64(counts + index * 8u,
                    iterationCarry + waveBases[wave] + prefix);
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread.x == 0u) writeBlocks64(blocks + group.x * 8u, blockCarry);
}
