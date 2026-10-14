// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::computeIncrementalComparisonHistograms_Stage2.
// Propagate the original 32-block exclusive prefix into each endpoint's
// comparison index and publish the three final global totals.
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

groupshared uint3 blockBase;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    if (thread.x == 0u) {
        uint3 prefix = uint3(0u, 0u, 0u);
        uint3 totals = uint3(0u, 0u, 0u);
        [unroll]
        for (uint axis = 0u; axis < 3u; ++axis) {
            const uint blockOffset = arena.Load(params.descriptor_offset + axis * 32u + 28u);
            uint prior = 0u;
            for (uint b = 0u; b < group.x; ++b)
                prior += arena.Load(blockOffset + b * 4u);
            uint total = prior;
            for (uint b = group.x; b < 32u; ++b)
                total += arena.Load(blockOffset + b * 4u);
            prefix[axis] = prior;
            totals[axis] = total;
        }
        blockBase = prefix;
        if (group.x == 0u) {
            const uint totalOffset = arena.Load(params.descriptor_offset + 96u);
            arena.Store3(totalOffset, totals);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint projections = params.previous_num_boxes * 2u;
    const uint required = (projections + 255u) / 256u;
    const uint iterations = (required + 31u) / 32u;
    for (uint iteration = 0u; iteration < iterations; ++iteration) {
        const uint index = (group.x * iterations + iteration) * 256u + thread.x;
        if (index < projections) {
            [unroll]
            for (uint axis = 0u; axis < 3u; ++axis) {
                const uint output = arena.Load(params.descriptor_offset + axis * 32u + 24u);
                const uint prior = arena.Load(output + index * 4u);
                arena.Store(output + index * 4u, prior + blockBase[axis]);
            }
        }
    }
}
