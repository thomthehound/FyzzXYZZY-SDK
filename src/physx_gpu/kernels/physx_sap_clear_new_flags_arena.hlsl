// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::clearNewFlagLaunch / BP_CLEAR_NEWFLAG.
// Preserve the source: loop over created-handle IDs using the 32x256
// CUDA grid-stride pattern, look up current SAP min/max positions,
// then restore both tagged endpoint handles without the 'new' bit.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset;
    uint arena_bytes;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    uint reserved3;
    uint reserved4;
    uint reserved5;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint desc = params.descriptor_offset;
    const uint createdList = arena.Load(desc + 0u);
    const uint numCreated = arena.Load(desc + 28u);
    const uint threadId = group.x * 256u + thread.x;

    for (uint i = threadId; i < numCreated; i += 32u * 256u) {
        const uint handle = arena.Load(createdList + i * 4u);
        const uint taggedStart = (handle << 3u) | 1u;
        const uint taggedEnd = handle << 3u;
        [unroll]
        for (uint axis = 0u; axis < 3u; ++axis) {
            const uint sap = arena.Load(desc + 4u + axis * 4u);
            const uint sorted = arena.Load(desc + 16u + axis * 4u);
            const uint startIndex = arena.Load(sap + handle * 8u);
            const uint endIndex = arena.Load(sap + handle * 8u + 4u);
            arena.Store(sorted + startIndex * 4u, taggedStart);
            arena.Store(sorted + endIndex * 4u, taggedEnd);
        }
    }
}
