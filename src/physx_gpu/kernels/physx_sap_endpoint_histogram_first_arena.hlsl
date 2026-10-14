// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::computeEndPtsHistogram / firstPassEndPtsHistogram.
// 32 contiguous original block partitions per axis; 256 threads per group.
// Workgroup scans replace fixed 32-lane CUDA ballot/shuffle accumulators.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset;
    uint previous_boxes;
    uint created_boxes;
    uint removed_boxes;
    uint is_incremental;
    uint arena_bytes;
    uint reserved0;
    uint reserved1;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint3 scanA[256];
groupshared uint3 scanB[256];
groupshared uint3 carry;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint axisDesc = params.descriptor_offset + group.y * 80u;
    const uint currentBoxes =
        (params.is_incremental != 0u)
        ? params.previous_boxes
        : params.previous_boxes + params.created_boxes - params.removed_boxes;

    // Frame one is needed for incremental removed-handle comparisons only.
    [loop]
    for (uint frame = 0u; frame < 2u; ++frame) {
        if (frame == 1u && (params.is_incremental == 0u || params.removed_boxes == 0u))
            break;

        const uint boxes = (frame == 0u) ? currentBoxes : params.previous_boxes;
        const uint frameDesc = axisDesc + frame * 32u;
        const uint handles = arena.Load(frameDesc);
        const uint ends = arena.Load(frameDesc + 4u);
        const uint starts = arena.Load(frameDesc + 8u);
        const uint blockEnds = arena.Load(frameDesc + 20u);
        const uint blockStarts = arena.Load(frameDesc + 24u);
        const uint totalEnds = arena.Load(axisDesc + 64u);
        const uint blockTotal = arena.Load(axisDesc + 68u);
        const uint projections = boxes * 2u;
        const uint required = (projections + 255u) / 256u;
        const uint iterations = (required + 31u) / 32u;

        if (thread.x == 0u) carry = uint3(0u, 0u, 0u);
        GroupMemoryBarrierWithGroupSync();

        for (uint iteration = 0u; iteration < iterations; ++iteration) {
            const uint index = (group.x * iterations + iteration) * 256u + thread.x;
            const bool active = index < projections;
            const uint handle = active ? arena.Load(handles + index * 4u) : 0u;
            const bool isStart = (handle & 1u) != 0u;
            const bool isDeleted = (handle & 4u) != 0u;
            const uint3 contribution = uint3(
                (active && isStart && !isDeleted) ? 1u : 0u,
                (active && !isStart && !isDeleted) ? 1u : 0u,
                (active && !isStart) ? 1u : 0u);
            scanA[thread.x] = contribution;
            GroupMemoryBarrierWithGroupSync();

            [unroll]
            for (uint stride = 1u; stride < 256u; stride <<= 1u) {
                if (stride == 1u || stride == 4u || stride == 16u || stride == 64u) {
                    uint3 value = scanA[thread.x];
                    if (thread.x >= stride) value += scanA[thread.x - stride];
                    scanB[thread.x] = value;
                } else {
                    uint3 value = scanB[thread.x];
                    if (thread.x >= stride) value += scanB[thread.x - stride];
                    scanA[thread.x] = value;
                }
                GroupMemoryBarrierWithGroupSync();
            }

            if (active) {
                const uint3 counts = carry + scanA[thread.x] - contribution;
                arena.Store(ends + index * 4u, counts.y);
                arena.Store(starts + index * 4u, counts.x);
                if (frame == 0u) arena.Store(totalEnds + index * 4u, counts.z);
            }

            // All lanes must consume the previous carry before the
            // leader advances it for the next source block iteration.
            GroupMemoryBarrierWithGroupSync();
            if (thread.x == 0u) carry += scanA[255u];
            GroupMemoryBarrierWithGroupSync();
        }

        if (thread.x == 0u) {
            arena.Store(blockEnds + group.x * 4u, carry.y);
            arena.Store(blockStarts + group.x * 4u, carry.x);
            if (frame == 0u) arena.Store(blockTotal + group.x * 4u, carry.z);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
