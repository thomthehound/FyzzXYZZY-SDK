// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::updateHandles (BP_UDPATE_HANDLES).
// Source: boxHandles[0][axis][i] = boxHandles[1][axis][boxProjectionRanks[axis][i]].
// The previous-handle buffer must not alias the radix scratch ranks.
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

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint live = params.previous_num_boxes * 2u;
    for (uint i = group.x * 256u + thread.x; i < live; i += 64u * 256u) {
        [unroll]
        for (uint axis = 0u; axis < 3u; ++axis) {
            const uint desc = params.descriptor_offset + axis * 16u;
            const uint rankBuffer = arena.Load(desc + 0u);
            const uint oldHandles = arena.Load(desc + 4u);
            const uint currentHandles = arena.Load(desc + 8u);
            const uint rank = arena.Load(rankBuffer + i * 4u);
            const uint handle = arena.Load(oldHandles + rank * 4u);
            arena.Store(currentHandles + i * 4u, handle);
        }
    }
}
