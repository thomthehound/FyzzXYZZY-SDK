// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::computeHistogram, invoked by
// BP_COMPUTE_START_REGION_HISTOGRAM and BP_COMPUTE_REGION_HISTOGRAM.
// Compiled twice, once for each source-owned array. Each block covers
// a contiguous span of ceil(ceil(2*boxes/256)/32)*256 projections.
// Use workgroup-wide prefix sums; do not assume CUDA warp32.
#ifndef FYZZXYZZY_REGION_START
#define FYZZXYZZY_REGION_START 1
#endif
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);
struct Parameters {
    uint descriptor_offset;
    uint boxes;
    uint arena_bytes;
    uint r0; uint r1; uint r2; uint r3; uint r4;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint scanA[256];
groupshared uint scanB[256];
groupshared uint carry;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint desc = params.descriptor_offset;
#if FYZZXYZZY_REGION_START
    const uint data = arena.Load(desc + 0u);
    const uint blocks = arena.Load(desc + 8u);
#else
    const uint data = arena.Load(desc + 4u);
    const uint blocks = arena.Load(desc + 12u);
#endif
    const uint projections = params.boxes * 2u;
    const uint iterations = ((projections + 255u) / 256u + 31u) / 32u;
    if (thread.x == 0u) carry = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint step = 0u; step < iterations; ++step) {
        const uint index = (group.x * iterations + step) * 256u + thread.x;
        const uint value = index < projections ? arena.Load(data + index * 4u) : 0u;
        scanA[thread.x] = value;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint shift = 1u; shift < 256u; shift <<= 1u) {
            if (shift == 1u || shift == 4u || shift == 16u || shift == 64u) {
                uint sum = scanA[thread.x];
                if (thread.x >= shift) sum += scanA[thread.x - shift];
                scanB[thread.x] = sum;
            } else {
                uint sum = scanB[thread.x];
                if (thread.x >= shift) sum += scanB[thread.x - shift];
                scanA[thread.x] = sum;
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (index < projections)
            arena.Store(data + index * 4u, carry + scanA[thread.x] - value);
        GroupMemoryBarrierWithGroupSync();
        if (thread.x == 0u) carry += scanA[255u];
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread.x == 0u) arena.Store(blocks + group.x * 4u, carry);
}
