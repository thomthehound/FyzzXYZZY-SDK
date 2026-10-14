// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::computeStartAndActiveRegionHistogram,
// BP_COMPUTE_ACTIVE_HISTOGRAM: fixed 64 x 512, one region per block.
// Portable lowering: separate 32-bit start and active counters instead
// of warp-scanned packed 16-bit fields; outputs match representable
// source prefixes without cross-field carry on large input counts.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);
struct Parameters {
    uint descriptor_offset; uint boxes; uint arena_bytes;
    uint r0; uint r1; uint r2; uint r3; uint r4;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint startA[512];
groupshared uint startB[512];
groupshared uint activeA[512];
groupshared uint activeB[512];
groupshared uint startCarry;
groupshared uint activeCarry;

[numthreads(512, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint d = params.descriptor_offset;
    const uint n = params.boxes * 2u;
    const uint starts = arena.Load(d + 24u) + group.x * n * 4u;
    const uint active = arena.Load(d + 20u) + group.x * n * 4u;
    const uint loops = (n + 511u) / 512u;
    if (thread.x == 0u) { startCarry = 0u; activeCarry = 0u; }
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 0u; k < loops; ++k) {
        const uint i = k * 512u + thread.x;
        const uint sv = i < n ? arena.Load(starts + i * 4u) : 0u;
        const uint av = i < n ? arena.Load(active + i * 4u) : 0u;
        startA[thread.x] = sv;
        activeA[thread.x] = av;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint shift = 1u; shift < 512u; shift <<= 1u) {
            if (shift == 1u || shift == 4u || shift == 16u || shift == 64u || shift == 256u) {
                uint st = startA[thread.x], at = activeA[thread.x];
                if (thread.x >= shift) {
                    st += startA[thread.x - shift];
                    at += activeA[thread.x - shift];
                }
                startB[thread.x] = st;
                activeB[thread.x] = at;
            } else {
                uint st = startB[thread.x], at = activeB[thread.x];
                if (thread.x >= shift) {
                    st += startB[thread.x - shift];
                    at += activeB[thread.x - shift];
                }
                startA[thread.x] = st;
                activeA[thread.x] = at;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        // 9 shifts leave the inclusive sums in B.
        if (i < n) {
            arena.Store(starts + i * 4u, startCarry + startB[thread.x] - sv);
            arena.Store(active + i * 4u, activeCarry + activeB[thread.x] - av);
        }
        GroupMemoryBarrierWithGroupSync();
        if (thread.x == 0u) {
            startCarry += startB[511u];
            activeCarry += activeB[511u];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
