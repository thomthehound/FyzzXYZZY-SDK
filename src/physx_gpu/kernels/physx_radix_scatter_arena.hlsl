// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// PxgCudaBroadPhaseSap::runRadixSort / RadixSort.cuh::radixSortCalculateRanks.
// The source's 32 blocks x 3 axes x 1024 threads, four scalar keys per
// uint4, 16-digit histogram, block ownership and stable payload permutation
// are preserved. Explicit workgroup scans replace CUDA warp shuffles and
// 8-bit packed accumulator operations without requiring a 32-lane subgroup.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset;
    uint num_keys;
    uint start_bit;
    uint arena_bytes;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    uint reserved3;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint bucketBase[16];
groupshared uint carriedCount[16];
groupshared uint scanValues[2048];

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

    // Every workgroup owns the same consecutive uint4 range as the
    // source kernel. Skip groups with no work before any group barriers.
    const uint numVectors = (params.num_keys + 3u) / 4u;
    const uint requiredBlocks = (numVectors + 1023u) / 1024u;
    const uint iterationsPerBlock = (requiredBlocks + 31u) / 32u;
    if (iterationsPerBlock * block * 1024u >= numVectors) return;

    // The original rank kernel derives these starting offsets by scanning
    // the producer's 16 x 32 per-block counts, then carries bin counts
    // forward between local iterations of the same block.
    if (thread.x < 16u) {
        uint prefix = 0u;
        for (uint digit = 0u; digit < thread.x; ++digit)
            for (uint b = 0u; b < 32u; ++b)
                prefix += arena.Load(radixCounts + (digit * 32u + b) * 4u);
        for (uint b = 0u; b < block; ++b)
            prefix += arena.Load(radixCounts + (thread.x * 32u + b) * 4u);
        bucketBase[thread.x] = prefix;
        carriedCount[thread.x] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint iteration = 0u; iteration < iterationsPerBlock; ++iteration) {
        const uint vectorIndex =
            iterationsPerBlock * block * 1024u + iteration * 1024u + thread.x;
        const bool isVector = vectorIndex < numVectors;
        uint4 keys = uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
        uint4 ranks = uint4(0u, 0u, 0u, 0u);

        if (isVector) {
            [unroll]
            for (uint lane = 0u; lane < 4u; ++lane) {
                const uint scalarIndex = vectorIndex * 4u + lane;
                if (scalarIndex < params.num_keys) {
                    keys[lane] = arena.Load(inputKeys + scalarIndex * 4u);
                    ranks[lane] = arena.Load(inputRanks + scalarIndex * 4u);
                }
            }
        }
        const uint4 digits = (keys >> params.start_bit) & 15u;

        [loop]
        for (uint digit = 0u; digit < 16u; ++digit) {
            uint mine = 0u;
            if (isVector) {
                [unroll]
                for (uint lane = 0u; lane < 4u; ++lane)
                    mine += (digits[lane] == digit) ? 1u : 0u;
            }

            // Subgroup-size-independent inclusive scan of each thread's
            // four consecutive keys. The resulting exclusive count
            // reproduces the source's stable intra-block radix ranks.
            scanValues[thread.x] = mine;
            GroupMemoryBarrierWithGroupSync();
            uint sourceBank = 0u;
            uint targetBank = 1024u;
            [unroll]
            for (uint distance = 1u; distance < 1024u; distance <<= 1u) {
                uint sum = scanValues[sourceBank + thread.x];
                if (thread.x >= distance)
                    sum += scanValues[sourceBank + thread.x - distance];
                scanValues[targetBank + thread.x] = sum;
                GroupMemoryBarrierWithGroupSync();
                const uint swapBank = sourceBank;
                sourceBank = targetBank;
                targetBank = swapBank;
            }

            const uint exclusive = scanValues[sourceBank + thread.x] - mine;
            const uint base = bucketBase[digit] + carriedCount[digit] + exclusive;
            uint precedingInVector = 0u;
            if (isVector) {
                [unroll]
                for (uint lane = 0u; lane < 4u; ++lane) {
                    if (digits[lane] == digit) {
                        const uint scalarIndex = vectorIndex * 4u + lane;
                        if (scalarIndex < params.num_keys) {
                            const uint destination = base + precedingInVector;
                            arena.Store(outputKeys + destination * 4u, keys[lane]);
                            arena.Store(outputRanks + destination * 4u, ranks[lane]);
                        }
                        ++precedingInVector;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            if (thread.x == 0u)
                carriedCount[digit] += scanValues[sourceBank + 1023u];
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
