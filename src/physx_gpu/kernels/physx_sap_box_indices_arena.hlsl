// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::initializeSapBox1DLaunch / BP_INITIALIZE_SAPBOX.
// Original one-thread-per-sorted-endpoint writes to PxgSapBox1D.mMinMax.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset;
    uint num_handles;
    uint is_new;
    uint arena_bytes;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    uint reserved3;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint endpoint = group.x * 256u + thread.x;
    if (endpoint >= params.num_handles * 2u) return;

    [unroll]
    for (uint axis = 0u; axis < 3u; ++axis) {
        const uint desc = params.descriptor_offset + axis * 16u;
        const uint sorted = arena.Load(desc + 0u);
        const uint output = arena.Load(desc + (params.is_new != 0u ? 8u : 4u));
        const uint taggedHandle = arena.Load(sorted + endpoint * 4u);
        const uint handle = taggedHandle >> 3u; // PxgBroadPhaseDesc::getHandle
        const uint isEnd = ((taggedHandle & 1u) == 0u) ? 1u : 0u;
        arena.Store(output + handle * 8u + isEnd * 4u, endpoint);
    }
}
