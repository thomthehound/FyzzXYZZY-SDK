// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Arena-relative port of PhysX broadphase.cu::markRemovedPairsLaunch.
// Preserves the original SAP previous-frame handle updates, deleted bit,
// and PxgIntegerAABB::setEmpty() values. ONLY address lowering changes.
// Initial experiment deliberately targets one arena: 1 UAV + 8 constants,
// zero GPU pointer-to-descriptor lookups for the hot inner loop.
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

[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Descriptor field byte offsets correspond to MarkRemovedPairsArenaDesc.
    uint base = params.descriptor_offset;
    uint count = arena.Load(base + 0);
    uint removed_array = arena.Load(base + 4);
    uint bounds_array = arena.Load(base + 44);

    for (uint i = tid.x; i < count; i += params.thread_stride) {
        uint handle = arena.Load(removed_array + i * 4);

        [unroll]
        for (uint axis = 0; axis < 3; ++axis) {
            uint sap_array = arena.Load(base + 8 + axis * 4);
            // Source: boxHandles[1][axis], the PREVIOUS frame.
            uint previous_handles = arena.Load(base + 20 + (3 + axis) * 4);
            uint box_base = sap_array + handle * 8;
            uint min_index = arena.Load(box_base + 0);
            uint max_index = arena.Load(box_base + 4);
            uint min_address = previous_handles + min_index * 4;
            uint max_address = previous_handles + max_index * 4;
            arena.Store(min_address, arena.Load(min_address) | 4u);
            arena.Store(max_address, arena.Load(max_address) | 4u);
        }

        uint bounds_base = bounds_array + handle * 24;
        [unroll]
        for (uint component = 0; component < 6; ++component)
            arena.Store(bounds_base + component * 4,
                        component < 3 ? 0xff7fffffu : 0x00800000u);
    }
}
