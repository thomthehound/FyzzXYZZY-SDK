// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::computeStartAndActiveRegionHistogram
// BP_COMPUTE_ACTIVE_HISTOGRAM: fixed 64 x 512, one region per workgroup.
//
// The original CUDA kernel packs start/active 0-or-1 flags into ONE
// 32-bit value (start << 16 | active). Each 512-element chunk contains
// at most 512 of either flag, so packed addition cannot carry from
// the active half into the start half. Accumulation ACROSS chunks must
// separate the two 32-bit region counters, exactly as source PhysX.
//
// WAVE=1: one packed WavePrefixSum per lane, subgroup subtotal prefix
// in LDS (one subgroup total per wave), just 3 group barriers/chunk.
// Wave IDs are assigned ONCE by subgroup leaders, not inferred from
// an undocumented SV_GroupIndex <-> wave-lane mapping. Wave32/wave64
// both preserve the source's X-projection order.
//
// WAVE=0: correctness fallback for devices lacking subgroup arithmetic.
// One packed Hillis-Steele scan (not two parallel 32-bit scans) and
// half the LDS scratch of the old lowering. Never the shipping fast path.
#ifndef FYZZXYZZY_ACTIVE_REGION_WAVE
#define FYZZXYZZY_ACTIVE_REGION_WAVE 0
#endif
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset; uint boxes; uint arena_bytes;
    uint r0; uint r1; uint r2; uint r3; uint r4;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint startCarry;
groupshared uint activeCarry;

#if FYZZXYZZY_ACTIVE_REGION_WAVE
groupshared uint nextWave;
groupshared uint waveBases[512];
groupshared uint chunkStartBase;
groupshared uint chunkActiveBase;
#else
groupshared uint scanA[512];
groupshared uint scanB[512];
#endif

[numthreads(512, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint d = params.descriptor_offset;
    const uint n = params.boxes * 2u;
    const uint starts = arena.Load(d + 24u) + group.x * n * 4u;
    const uint active = arena.Load(d + 20u) + group.x * n * 4u;
    const uint loops = (n + 511u) / 512u;

    if (thread.x == 0u) {
        startCarry = 0u;
        activeCarry = 0u;
#if FYZZXYZZY_ACTIVE_REGION_WAVE
        nextWave = 0u;
#endif
    }
    GroupMemoryBarrierWithGroupSync();

#if FYZZXYZZY_ACTIVE_REGION_WAVE
    const uint laneCount = WaveGetLaneCount();
    const uint laneIndex = WaveGetLaneIndex();
    // Assign one logical contiguous slice per physical subgroup.
    // Subgroups may be scheduled/mapped in implementation-defined order.
    // This atomically numbered wave index ensures WavePrefixSum's lane
    // order always maps onto contiguous *logical* X projection indices.
    uint waveId = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(nextWave, 1u, waveId);
    waveId = WaveReadLaneFirst(waveId);
    const uint waveCount = 512u / laneCount;
#endif

    for (uint k = 0u; k < loops; ++k) {
#if FYZZXYZZY_ACTIVE_REGION_WAVE
        const uint i = k * 512u + waveId * laneCount + laneIndex;
#else
        const uint i = k * 512u + thread.x;
#endif
        const uint sv = i < n ? arena.Load(starts + i * 4u) : 0u;
        const uint av = i < n ? arena.Load(active + i * 4u) : 0u;
        const uint packed = (sv << 16u) | av;

#if FYZZXYZZY_ACTIVE_REGION_WAVE
        const uint lanePrefix = WavePrefixSum(packed);
        const uint waveTotal = WaveActiveSum(packed);
        if (WaveIsFirstLane()) waveBases[waveId] = waveTotal;
        GroupMemoryBarrierWithGroupSync();

        if (thread.x == 0u) {
            uint base = 0u;
            for (uint w = 0u; w < waveCount; ++w) {
                const uint count = waveBases[w];
                waveBases[w] = base;
                base += count;
            }
            chunkStartBase = startCarry;
            chunkActiveBase = activeCarry;
            startCarry += base >> 16u;
            activeCarry += base & 0xffffu;
        }
        GroupMemoryBarrierWithGroupSync();

        const uint localPrefix = waveBases[waveId] + lanePrefix;
        if (i < n) {
            arena.Store(starts + i * 4u, chunkStartBase + (localPrefix >> 16u));
            arena.Store(active + i * 4u, chunkActiveBase + (localPrefix & 0xffffu));
        }
        // Prevent the next iteration from overwriting wave bases while
        // another subgroup still reads them.
        GroupMemoryBarrierWithGroupSync();
#else
        scanA[thread.x] = packed;
        GroupMemoryBarrierWithGroupSync();

        // Same packed arithmetic as PhysX, lowered to portable
        // shared memory only when native subgroup sums are unavailable.
        [unroll]
        for (uint shift = 1u; shift < 512u; shift <<= 1u) {
            if (shift == 1u || shift == 4u || shift == 16u ||
                shift == 64u || shift == 256u) {
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
        const uint localPrefix = scanB[thread.x] - packed;
        if (i < n) {
            arena.Store(starts + i * 4u, startCarry + (localPrefix >> 16u));
            arena.Store(active + i * 4u, activeCarry + (localPrefix & 0xffffu));
        }
        GroupMemoryBarrierWithGroupSync();
        if (thread.x == 0u) {
            const uint sum = scanB[511u];
            startCarry += sum >> 16u;
            activeCarry += sum & 0xffffu;
        }
        GroupMemoryBarrierWithGroupSync();
#endif
    }
}
