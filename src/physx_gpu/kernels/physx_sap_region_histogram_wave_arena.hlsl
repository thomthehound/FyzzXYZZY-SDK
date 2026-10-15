// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Direct native-wave lowering of pinned PhysX broadphase.cu::computeHistogram.
// BP_COMPUTE_START_REGION_HISTOGRAM and BP_COMPUTE_REGION_HISTOGRAM,
// both 32 groups x 256 threads. Original source processes contiguous
// 256-element iterations per block; each iteration performs warp-local
// exclusive scans, adds preceding warp sums, then adds the block carry.
//
// Use the same hierarchy on wave32/wave64 (or any power-of-two physical
// wave width dividing 256): native WavePrefixSum/WaveActiveSum, followed
// by the tiny wave-total prefix in one workgroup leader. Only THREE
// full-group barriers per iteration, instead of ~11 in the portable
// 2-bank Hillis-Steele fallback. 2x64 u32 LDS slots (512 bytes), versus
// 2x256 u32 (2 KiB) for the fallback. Performance is unmeasured.
//
// Wave-to-SV_GroupIndex assignment is implementation-dependent. Each
// physical wave atomically reserves one logical contiguous projection
// slice exactly ONCE before all iterations, so prefix ordering is
// independent of the scheduler's physical lane mapping.
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

groupshared uint nextWave;
groupshared uint waveTotals[64];
groupshared uint waveBases[64];
groupshared uint carry;
groupshared uint iterationCarry;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint desc = params.descriptor_offset;
#if FYZZXYZZY_REGION_START
    const uint data = arena.Load(desc + 0u);
    const uint blocks = arena.Load(desc + 8u);
#else
    const uint data = arena.Load(desc + 4u);
    const uint blocks = arena.Load(desc + 12u);
#endif
    const uint projections = params.boxes * 2u;
    const uint iterations = ((projections + 255u) / 256u + 31u) / 32u;

    if (thread.x == 0u) {
        nextWave = 0u;
        carry = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint laneCount = WaveGetLaneCount();
    const uint laneIndex = WaveGetLaneIndex();
    const uint waveCount = 256u / laneCount;
    uint waveId = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(nextWave, 1u, waveId);
    waveId = WaveReadLaneFirst(waveId);

    for (uint step = 0u; step < iterations; ++step) {
        const uint index = (group.x * iterations + step) * 256u +
                           waveId * laneCount + laneIndex;
        const uint value =
            index < projections ? arena.Load(data + index * 4u) : 0u;
        const uint wavePrefix = WavePrefixSum(value); // exclusive
        const uint waveTotal = WaveActiveSum(value);
        if (WaveIsFirstLane()) waveTotals[waveId] = waveTotal;
        GroupMemoryBarrierWithGroupSync();

        if (thread.x == 0u) {
            uint blockTotal = 0u;
            for (uint w = 0u; w < waveCount; ++w) {
                const uint count = waveTotals[w];
                waveBases[w] = blockTotal;
                blockTotal += count;
            }
            iterationCarry = carry;
            carry += blockTotal;
        }
        GroupMemoryBarrierWithGroupSync();

        if (index < projections)
            arena.Store(data + index * 4u,
                        iterationCarry + waveBases[waveId] + wavePrefix);
        // Every wave must consume the current totals and carry before
        // the next 256-element iteration overwrites shared storage.
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread.x == 0u) arena.Store(blocks + group.x * 4u, carry);
}
