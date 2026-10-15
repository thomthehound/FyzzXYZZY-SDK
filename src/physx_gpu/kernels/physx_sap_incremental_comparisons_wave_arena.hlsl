// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// PhysX broadphase.cu::computeIncrementalComparisonHistograms_Stage1
// (BP_COMPUTE_INCREMENTAL_CMP_COUNTS1), 32 workgroups x 256 threads.
//
// Source CUDA scans THREE independent 32-bit axis comparison counts
// within native warps, prefixes their warp totals, and accumulates
// independent cross-iteration block carries. Unlike endpoint 0/1
// contributions, these comparison counts can be large; packing them
// into 10- or 16-bit fields would introduce unbounded carry errors.
//
// On a compute-subgroup-arithmetic device, perform exactly three
// wave-local exclusive prefixes and reductions. Prefix only the small
// set of wave totals in shared memory. This needs 3 full-group barriers
// and 2x64xuint3 (1536 B) scratch per 256-projection iteration,
// replacing the port's 2x256xuint3 (6144 B) and 8 Hillis-Steele rounds.
//
// As on prior source-faithful GPU stages, wave leaders claim contiguous
// LOGICAL X-projection ranges once before the loop. The physical
// wave order is not inferred from SV_GroupIndex / WaveGetLaneCount.
// No CUDA wave32, WMMA, or matrix-operation dependency.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset;
    uint previous_num_boxes;
    uint arena_bytes;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    uint reserved3;
    uint reserved4;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint nextWave;
groupshared uint3 waveTotals[64];
groupshared uint3 waveBases[64];
groupshared uint3 carry;
groupshared uint3 iterationCarry;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint projections = params.previous_num_boxes * 2u;
    const uint required = (projections + 255u) / 256u;
    const uint iterations = (required + 31u) / 32u;

    if (thread.x == 0u) {
        nextWave = 0u;
        carry = uint3(0u, 0u, 0u);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint laneCount = WaveGetLaneCount();
    const uint laneIndex = WaveGetLaneIndex();
    const uint waveCount = 256u / laneCount;
    uint waveId = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(nextWave, 1u, waveId);
    waveId = WaveReadLaneFirst(waveId);

    for (uint iteration = 0u; iteration < iterations; ++iteration) {
        const uint index = (group.x * iterations + iteration) * 256u +
                           waveId * laneCount + laneIndex;
        uint3 comparisons = uint3(0u, 0u, 0u);
        if (index < projections) {
            [unroll]
            for (uint axis = 0u; axis < 3u; ++axis) {
                const uint axisDesc = params.descriptor_offset + axis * 32u;
                const uint ranks = arena.Load(axisDesc + 4u);
                const uint handles = arena.Load(axisDesc);
                const uint rank = arena.Load(ranks + index * 4u);
                const uint tagged = arena.Load(handles + index * 4u);

                const bool down = index <= rank;
                const bool startHandle = (tagged & 1u) != 0u;
                const bool deleted = (tagged & 4u) != 0u;
                const uint histOffset = axisDesc +
                    (down ? (startHandle ? 20u : 16u)
                          : (startHandle ? 12u : 8u));
                const uint histogram = arena.Load(histOffset);
                const uint lo = down ? index : rank + 1u;
                const uint hi = down ? min(rank, projections - 1u) : index;
                if (!deleted)
                    comparisons[axis] =
                        arena.Load(histogram + hi * 4u) -
                        arena.Load(histogram + lo * 4u);
            }
        }

        // Source performs independent warp32 scans for X, Y and Z.
        // Keep the 32-bit independence and use the device's native wave.
        const uint3 wavePrefix = uint3(
            WavePrefixSum(comparisons.x),
            WavePrefixSum(comparisons.y),
            WavePrefixSum(comparisons.z));
        const uint3 waveTotal = uint3(
            WaveActiveSum(comparisons.x),
            WaveActiveSum(comparisons.y),
            WaveActiveSum(comparisons.z));
        if (WaveIsFirstLane()) waveTotals[waveId] = waveTotal;
        GroupMemoryBarrierWithGroupSync();

        if (thread.x == 0u) {
            uint3 blockTotal = uint3(0u, 0u, 0u);
            for (uint wave = 0u; wave < waveCount; ++wave) {
                const uint3 count = waveTotals[wave];
                waveBases[wave] = blockTotal;
                blockTotal += count;
            }
            iterationCarry = carry;
            carry += blockTotal;
        }
        GroupMemoryBarrierWithGroupSync();

        if (index < projections) {
            const uint3 localPrefix = iterationCarry +
                                      waveBases[waveId] + wavePrefix;
            [unroll]
            for (uint axis = 0u; axis < 3u; ++axis) {
                const uint output =
                    arena.Load(params.descriptor_offset + axis * 32u + 24u);
                arena.Store(output + index * 4u, localPrefix[axis]);
            }
        }
        // All waves must consume this iteration's wave bases and
        // carries before another iteration changes shared scratch.
        GroupMemoryBarrierWithGroupSync();
    }

    if (thread.x == 0u) {
        [unroll]
        for (uint axis = 0u; axis < 3u; ++axis) {
            const uint output =
                arena.Load(params.descriptor_offset + axis * 32u + 28u);
            arena.Store(output + group.x * 4u, carry[axis]);
        }
    }
}
