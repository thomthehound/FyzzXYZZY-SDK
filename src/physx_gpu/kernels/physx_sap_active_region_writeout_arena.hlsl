// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::writeOutStartAndActiveRegionHistogram
// BP_WRITEOUT_ACTIVE_HISTOGRAM, fixed 32 x 256.
// The caller has zeroed the 64 dense region histograms, as upstream.
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

// Binary search for the final prefix entry <= workIndex. Since the
// prefix represents sparse positive-count start projections, this
// returns the original source X sorted-handle index for each YZ region.
uint prefix_to_x_position(uint base, uint projections, uint workIndex) {
    uint low = 0u, high = projections;
    while (low < high) {
        const uint mid = low + (high - low) / 2u;
        if (arena.Load(base + mid * 4u) <= workIndex) low = mid + 1u;
        else high = mid;
    }
    return low - 1u;
}

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint d = params.descriptor_offset;
    const uint n = params.boxes * 2u;
    const uint prefix = arena.Load(d + 12u);
    const uint total = arena.Load(arena.Load(d + 16u));
    const uint sortedX = arena.Load(d);
    const uint ranges = arena.Load(d + 8u);
    const uint active = arena.Load(d + 20u);
    const uint starts = arena.Load(d + 24u);
    const uint loops = ((total + 255u) / 256u + 31u) / 32u;
    for (uint k = 0u; k < loops; ++k) {
        const uint work = (group.x * loops + k) * 256u + thread.x;
        if (work >= total) continue;
        const uint xPos = prefix_to_x_position(prefix, n, work);
        const uint tagged = arena.Load(sortedX + xPos * 4u);
        const uint handle = tagged >> 3u;
        const uint region = ranges + handle * 32u;
        const uint sy = arena.Load(region + 4u);
        const uint sz = arena.Load(region + 8u);
        const uint ez = arena.Load(region + 24u);
        const uint zWidth = ez - sz + 1u;
        const uint ordinal = work - arena.Load(prefix + xPos * 4u);
        const uint yz = (ordinal / zWidth + sy) * 4u +
                        (ordinal % zWidth + sz) * 16u;
        const uint rx = arena.Load(region);
        const uint regionId = rx + yz;
        const uint index = (regionId * n + xPos) * 4u;
        arena.Store(starts + index, 1u);
        arena.Store(active + index, (tagged >> 1u) & 1u);
    }
}
