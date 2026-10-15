// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Pinned CUDA: gpucommon/src/CUDA/RadixSort.cuh::radixSortCalculateRanks.
// Same 32 blocks * 3 axes * 1024 threads, source radix order, block
// ownership, ping-pong buffers, and stable per-key rank.
// Source CUDA packs counters in registers and scans with warp shuffles.
// This WaveOps lowering scans TWO 16-bit packed bin counts per pass
// (8 pairs rather than 16 full-group digit scans). Per workgroup chunk
// there are at most 1024*4 = 4096 keys, so neither 16-bit field can
// overflow before unpacking into 32-bit per-digit carry accumulators.
//
// No assumption that the hardware maps SV_GroupIndex to wave lane IDs:
// each WaveIsFirstLane atomically claims a contiguous logical 4..128-lane
// range once. The logical range is then used for reading input vectors.
// Stable source order is recovered by prefixing wave totals by claimed ID.
// SM6.6 / Vulkan compute subgroup arithmetic capability is required;
// portable HLSL fallback remains in physx_radix_scatter_arena.hlsl.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);
struct Parameters {
    uint descriptor_offset;
    uint num_keys;
    uint start_bit;
    uint arena_bytes;
    uint r0; uint r1; uint r2; uint r3;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint bucketBase[16];
groupshared uint carriedCount[16];
groupshared uint claimedWaves;
groupshared uint waveCounts[256];
groupshared uint waveBases[256];
groupshared uint chunkPackedTotal;

[numthreads(1024, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint block = group.x;
    const uint axis = group.y;
    const uint desc = params.descriptor_offset + axis * 32u;
    const uint inputKeys = arena.Load(desc + 0u);
    const uint inputRanks = arena.Load(desc + 4u);
    const uint outputKeys = arena.Load(desc + 8u);
    const uint outputRanks = arena.Load(desc + 12u);
    const uint radixCounts = arena.Load(desc + 16u);

    const uint numVectors = (params.num_keys + 3u) / 4u;
    const uint requiredBlocks = (numVectors + 1023u) / 1024u;
    const uint iterationsPerBlock = (requiredBlocks + 31u) / 32u;
    if (iterationsPerBlock * block * 1024u >= numVectors) return;

    if (thread.x < 16u) {
        uint start = 0u;
        for (uint digit = 0u; digit < thread.x; ++digit)
            for (uint b = 0u; b < 32u; ++b)
                start += arena.Load(radixCounts + (digit * 32u + b) * 4u);
        for (uint b = 0u; b < block; ++b)
            start += arena.Load(radixCounts + (thread.x * 32u + b) * 4u);
        bucketBase[thread.x] = start;
        carriedCount[thread.x] = 0u;
    }
    if (thread.x == 0u) claimedWaves = 0u;
    GroupMemoryBarrierWithGroupSync();

    const uint laneCount = WaveGetLaneCount();
    const uint laneIndex = WaveGetLaneIndex();
    uint waveId = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(claimedWaves, 1u, waveId);
    waveId = WaveReadLaneFirst(waveId);
    const uint waveCount = 1024u / laneCount;
    const uint logicalThread = waveId * laneCount + laneIndex;

    for (uint iter = 0u; iter < iterationsPerBlock; ++iter) {
        const uint vectorIndex =
            iterationsPerBlock * block * 1024u + iter * 1024u + logicalThread;
        const bool hasVector = vectorIndex < numVectors;
        uint4 keys = uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
        uint4 ranks = uint4(0u, 0u, 0u, 0u);
        if (hasVector) {
            [unroll]
            for (uint lane = 0u; lane < 4u; ++lane) {
                const uint scalar = vectorIndex * 4u + lane;
                if (scalar < params.num_keys) {
                    keys[lane] = arena.Load(inputKeys + scalar * 4u);
                    ranks[lane] = arena.Load(inputRanks + scalar * 4u);
                }
            }
        }
        const uint4 digits = (keys >> params.start_bit) & 15u;

        [loop]
        for (uint firstDigit = 0u; firstDigit < 16u; firstDigit += 2u) {
            uint firstCount = 0u, secondCount = 0u;
            if (hasVector) {
                [unroll]
                for (uint lane = 0u; lane < 4u; ++lane) {
                    firstCount += digits[lane] == firstDigit ? 1u : 0u;
                    secondCount += digits[lane] == firstDigit + 1u ? 1u : 0u;
                }
            }
            const uint packed = firstCount | (secondCount << 16u);
            const uint wavePrefix = WavePrefixSum(packed);
            const uint waveTotal = WaveActiveSum(packed);
            if (WaveIsFirstLane()) waveCounts[waveId] = waveTotal;
            GroupMemoryBarrierWithGroupSync();

            if (thread.x == 0u) {
                uint total = 0u;
                for (uint wave = 0u; wave < waveCount; ++wave) {
                    const uint count = waveCounts[wave];
                    waveBases[wave] = total;
                    total += count;
                }
                chunkPackedTotal = total;
            }
            GroupMemoryBarrierWithGroupSync();

            const uint prefix = waveBases[waveId] + wavePrefix;
            const uint firstBase =
                bucketBase[firstDigit] + carriedCount[firstDigit] + (prefix & 0xffffu);
            const uint secondBase =
                bucketBase[firstDigit + 1u] + carriedCount[firstDigit + 1u] +
                (prefix >> 16u);
            uint precedingFirst = 0u, precedingSecond = 0u;
            if (hasVector) {
                [unroll]
                for (uint lane = 0u; lane < 4u; ++lane) {
                    const uint scalar = vectorIndex * 4u + lane;
                    if (digits[lane] == firstDigit) {
                        if (scalar < params.num_keys) {
                            const uint destination = firstBase + precedingFirst;
                            arena.Store(outputKeys + destination * 4u, keys[lane]);
                            arena.Store(outputRanks + destination * 4u, ranks[lane]);
                        }
                        ++precedingFirst;
                    } else if (digits[lane] == firstDigit + 1u) {
                        if (scalar < params.num_keys) {
                            const uint destination = secondBase + precedingSecond;
                            arena.Store(outputKeys + destination * 4u, keys[lane]);
                            arena.Store(outputRanks + destination * 4u, ranks[lane]);
                        }
                        ++precedingSecond;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            if (thread.x == 0u) {
                carriedCount[firstDigit] += chunkPackedTotal & 0xffffu;
                carriedCount[firstDigit + 1u] += chunkPackedTotal >> 16u;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
