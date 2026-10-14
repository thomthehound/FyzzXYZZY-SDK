// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Port of PhysX broadphase.cu::markRemovedPairsProjectionsLaunch and
// clearPairs(), without changing the algorithm. The upstream host launches
// BP_UPDATE_DELETEDPAIRS at grid=32, block=256 after incremental SAP,
// before markCreatedPairs and the second radix sort.
//
// Physical addressing: one packed RWByteAddressBuffer holds the preserved
// 64-byte deletion descriptor prefix and the three added projection offsets.
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
    const uint base = params.descriptor_offset;
    const uint numRemoved = arena.Load(base);
    const uint removed = arena.Load(base + 4u);

    for (uint i = tid.x; i < numRemoved; i += params.thread_stride) {
        const uint handle = arena.Load(removed + i * 4u);
        [unroll]
        for (uint axis = 0; axis < 3u; ++axis) {
            const uint sapArray = arena.Load(base + 8u + axis * 4u);
            const uint projectionArray = arena.Load(base + 64u + axis * 4u);
            const uint sap = sapArray + handle * 8u;
            const uint minIndex = arena.Load(sap);
            const uint maxIndex = arena.Load(sap + 4u);
            arena.Store(projectionArray + minIndex * 4u, 0xffffffffu);
            arena.Store(projectionArray + maxIndex * 4u, 0xffffffffu);
            arena.Store(sap, 0xffffffffu);
            arena.Store(sap + 4u, 0xffffffffu);
        }
    }
}
