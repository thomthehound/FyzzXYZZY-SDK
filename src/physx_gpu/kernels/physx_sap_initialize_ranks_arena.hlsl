// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::initializeRadixRanks (BP_INITIALIZE_RANKS).
// Source grid: 64 blocks of 256 threads, each initializes one uint4,
// with grid-stride iteration for large previous-frame projection arrays.
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
    const uint vectorCount = (live + 3u) / 4u;
    for (uint i = group.x * 256u + thread.x; i < vectorCount; i += 64u * 256u) {
        const uint scalar = 4u * i;
        [unroll]
        for (uint axis = 0u; axis < 3u; ++axis) {
            const uint desc = params.descriptor_offset + axis * 16u;
            const uint ranks = arena.Load(desc);
            [unroll]
            for (uint lane = 0u; lane < 4u; ++lane)
                arena.Store(ranks + (scalar + lane) * 4u, scalar + lane);
        }
    }
}
