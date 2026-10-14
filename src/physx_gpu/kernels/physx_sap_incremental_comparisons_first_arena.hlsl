// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::computeIncrementalComparisonHistograms_Stage1.
// One 256-thread workgroup per one of PhysX's 32 contiguous blocks.
// Each endpoint consumes the sorted source rank and the original
// current/previous start-or-end histogram determined by its direction.
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

groupshared uint3 scanA[256];
groupshared uint3 scanB[256];
groupshared uint3 carry;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint projections = params.previous_num_boxes * 2u;
    const uint required = (projections + 255u) / 256u;
    const uint iterations = (required + 31u) / 32u;
    if (thread.x == 0u) carry = uint3(0u, 0u, 0u);
    GroupMemoryBarrierWithGroupSync();

    for (uint iteration = 0u; iteration < iterations; ++iteration) {
        const uint index = (group.x * iterations + iteration) * 256u + thread.x;
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
        scanA[thread.x] = comparisons;
        GroupMemoryBarrierWithGroupSync();

        // Group-shared workgroup scan, independent of AMD/NVIDIA wave width.
        [unroll]
        for (uint shift = 1u; shift < 256u; shift <<= 1u) {
            if (shift == 1u || shift == 4u || shift == 16u || shift == 64u) {
                uint3 value = scanA[thread.x];
                if (thread.x >= shift) value += scanA[thread.x - shift];
                scanB[thread.x] = value;
            } else {
                uint3 value = scanB[thread.x];
                if (thread.x >= shift) value += scanB[thread.x - shift];
                scanA[thread.x] = value;
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (index < projections) {
            [unroll]
            for (uint axis = 0u; axis < 3u; ++axis) {
                const uint output = arena.Load(params.descriptor_offset + axis * 32u + 24u);
                arena.Store(output + index * 4u,
                            carry[axis] + scanA[thread.x][axis] - comparisons[axis]);
            }
        }
        // All hardware waves must finish reading carry before advancing it.
        GroupMemoryBarrierWithGroupSync();
        if (thread.x == 0u) carry += scanA[255u];
        GroupMemoryBarrierWithGroupSync();
    }

    if (thread.x == 0u) {
        [unroll]
        for (uint axis = 0u; axis < 3u; ++axis) {
            const uint output = arena.Load(params.descriptor_offset + axis * 32u + 28u);
            arena.Store(output + group.x * 4u, carry[axis]);
        }
    }
}
