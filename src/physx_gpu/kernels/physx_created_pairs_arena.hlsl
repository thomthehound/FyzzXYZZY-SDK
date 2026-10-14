// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// PhysX broadphase.cu::markCreatedPairsLaunch / BP_UPDATE_CREATEDPAIRS.
// Original physics, fixed 32x256 host dispatch, grid-stride work loop.
// Only the CUDA pointer lowering changes: offsets into one packed arena.
//
// Handle encoding from PhysX createHandle(handle, isStart, isNew):
// (handle << 3) | (isStart ? 1 : 0) | (isNew ? 2 : 0).
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset;
    uint thread_stride;
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
void main(uint3 tid : SV_DispatchThreadID) {
    const uint desc = params.descriptor_offset;
    const uint createdCount = arena.Load(desc + 80u);
    const uint previousCount = arena.Load(desc + 84u);
    const uint createdOffset = arena.Load(desc + 88u);
    const uint newBounds = arena.Load(desc + 92u);
    const uint oldBounds = arena.Load(desc + 44u);

    for (uint i = tid.x; i < createdCount; i += params.thread_stride) {
        const uint handle = arena.Load(createdOffset + i * 4u);
        const uint index = previousCount + i;
        const uint startHandle = (handle << 3u) | 3u;
        const uint endHandle = (handle << 3u) | 2u;

        [unroll]
        for (uint axis = 0; axis < 3u; ++axis) {
            const uint projections = arena.Load(desc + 64u + axis * 4u);
            const uint handles = arena.Load(desc + 20u + axis * 4u); // current [0][axis]
            const uint bounds = newBounds + handle * 24u;
            const uint pairPos = index * 8u;
            arena.Store(projections + pairPos, arena.Load(bounds + axis * 4u));
            arena.Store(projections + pairPos + 4u, arena.Load(bounds + (axis + 3u) * 4u));
            arena.Store(handles + pairPos, startHandle);
            arena.Store(handles + pairPos + 4u, endHandle);
        }

        [unroll]
        for (uint component = 0; component < 6u; ++component)
            arena.Store(oldBounds + handle * 24u + component * 4u,
                        component < 3u ? 0xff7fffffu : 0x00800000u);
    }

    // Preserve PhysX's radix-sort padding branch. Its count is in uint2
    // pairs (one start/end pair per box), not scalar projection endpoints.
    const uint existing = previousCount + createdCount;
    if (tid.x < (existing & 1u)) {
        [unroll]
        for (uint axis = 0; axis < 3u; ++axis) {
            const uint projections = arena.Load(desc + 64u + axis * 4u);
            const uint handles = arena.Load(desc + 20u + axis * 4u);
            arena.Store2(projections + existing * 8u, uint2(0xffffffffu, 0xffffffffu));
            arena.Store2(handles + existing * 8u, uint2(0x3fffffffu, 0x3fffffffu));
        }
    }
}
