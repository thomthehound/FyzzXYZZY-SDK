// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::outputEndPtsHistogram / secondPassEndPtsHistogram.
// Publish globally exclusive endpoint counts and ordered handle lists.
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

groupshared uint3 blockBase;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint axisDesc = params.descriptor_offset + group.y * 80u;
    const uint currentBoxes =
        (params.is_incremental != 0u)
        ? params.previous_boxes
        : params.previous_boxes + params.created_boxes - params.removed_boxes;

    [loop]
    for (uint frame = 0u; frame < 2u; ++frame) {
        if (frame == 1u && (params.is_incremental == 0u || params.removed_boxes == 0u))
            break;
        const uint boxes = (frame == 0u) ? currentBoxes : params.previous_boxes;
        const uint frameDesc = axisDesc + frame * 32u;
        const uint handles = arena.Load(frameDesc);
        const uint ends = arena.Load(frameDesc + 4u);
        const uint starts = arena.Load(frameDesc + 8u);
        const uint orderedEnds = arena.Load(frameDesc + 12u);
        const uint orderedStarts = arena.Load(frameDesc + 16u);
        const uint blockEnds = arena.Load(frameDesc + 20u);
        const uint blockStarts = arena.Load(frameDesc + 24u);
        const uint totalEnds = arena.Load(axisDesc + 64u);
        const uint blockTotal = arena.Load(axisDesc + 68u);

        // Exclusive prefix of the source's 32 block counts.
        if (thread.x == 0u) {
            uint3 prior = uint3(0u, 0u, 0u);
            for (uint b = 0u; b < group.x; ++b) {
                prior.x += arena.Load(blockStarts + b * 4u);
                prior.y += arena.Load(blockEnds + b * 4u);
                if (frame == 0u) prior.z += arena.Load(blockTotal + b * 4u);
            }
            blockBase = prior;
        }
        GroupMemoryBarrierWithGroupSync();

        const uint projections = boxes * 2u;
        const uint required = (projections + 255u) / 256u;
        const uint iterations = (required + 31u) / 32u;
        for (uint iteration = 0u; iteration < iterations; ++iteration) {
            const uint index = (group.x * iterations + iteration) * 256u + thread.x;
            if (index < projections) {
                const uint handle = arena.Load(handles + index * 4u);
                const uint outEnd = arena.Load(ends + index * 4u) + blockBase.y;
                const uint outStart = arena.Load(starts + index * 4u) + blockBase.x;
                arena.Store(ends + index * 4u, outEnd);
                arena.Store(starts + index * 4u, outStart);
                if ((handle & 4u) == 0u) {
                    const uint id = handle >> 3u;
                    if ((handle & 1u) != 0u)
                        arena.Store(orderedStarts + outStart * 4u, id);
                    else
                        arena.Store(orderedEnds + outEnd * 4u, id);
                }
                if (frame == 0u) {
                    const uint allEnd = arena.Load(totalEnds + index * 4u);
                    arena.Store(totalEnds + index * 4u, allEnd + blockBase.z);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
