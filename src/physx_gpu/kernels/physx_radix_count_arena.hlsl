// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// PxgCudaBroadPhaseSap::runRadixSort:
// RS_MULTIBLOCK_COUNT / radixSortMultiBlockLaunchWithCount.
// Fixed 32 blocks x 3 axes x 1024 threads, 4-bit radix, 32-block
// interleaved histogram. The scalar count reduction replaces CUDA's
// packed 16-bit warp reductions; block partitions and output remain
// identical without requiring a physical 32-lane subgroup.
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

groupshared uint radixTotals[16];

[numthreads(1024, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint block = group.x;
    const uint axis = group.y;
    const uint axisDesc = params.descriptor_offset + axis * 16u;
    const uint inputKeys = arena.Load(axisDesc);
    const uint radixCounts = arena.Load(axisDesc + 4u);

    if (thread.x < 16u) radixTotals[thread.x] = 0u;
    GroupMemoryBarrierWithGroupSync();

    // RadixSort.cuh::radixSortSingleBlock partitions in uint4 units:
    // ceil(ceil(numKeys/4)/1024) blocks of uint4, rounded up across 32
    // dispatch blocks. Every block handles a contiguous partition.
    const uint numVectors = (params.num_keys + 3u) / 4u;
    const uint requiredBlocks = (numVectors + 1023u) / 1024u;
    const uint iterationsPerBlock = (requiredBlocks + 31u) / 32u;
    const uint firstVector = min(iterationsPerBlock * block * 1024u, numVectors);
    const uint lastVector = min(firstVector + iterationsPerBlock * 1024u, numVectors);

    for (uint vectorIndex = firstVector + thread.x;
         vectorIndex < lastVector; vectorIndex += 1024u) {
        [unroll]
        for (uint lane = 0; lane < 4u; ++lane) {
            const uint scalarIndex = vectorIndex * 4u + lane;
            // Original sanitizeKeys() fills incomplete uint4 tails with
            // all-one keys before counting.
            const uint key = scalarIndex < params.num_keys
                ? arena.Load(inputKeys + scalarIndex * 4u) : 0xffffffffu;
            const uint digit = (key >> params.start_bit) & 15u;
            uint previous;
            InterlockedAdd(radixTotals[digit], 1u, previous);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (thread.x < 16u)
        arena.Store(radixCounts + (thread.x * 32u + block) * 4u,
                    radixTotals[thread.x]);
}
