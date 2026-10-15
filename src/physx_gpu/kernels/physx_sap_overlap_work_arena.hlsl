// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// PhysX broadphase.cu::writeOutOverlapChecksForInsertedBoundsRegionsHistogram.
// Fixed source dispatch: 32 blocks of 256. One work item per regionAccum entry.
// No subgroup scans: this is the source's independent per-item count producer.
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
uint find_projection(uint base, uint count, uint ordinal) {
    uint lo = 0u, hi = count;
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2u;
        if (arena.Load(base + mid * 4u) <= ordinal) lo = mid + 1u;
        else hi = mid;
    }
    return lo - 1u;
}
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint d = params.descriptor_offset;
    const uint projections = params.boxes * 2u;
    const uint sorted = arena.Load(d + 0u);
    const uint positions = arena.Load(d + 4u);
    const uint ranges = arena.Load(d + 8u);
    const uint prefix = arena.Load(d + 12u);
    const uint total = arena.Load(arena.Load(d + 16u));
    const uint active = arena.Load(d + 20u);
    const uint starts = arena.Load(d + 24u);
    const uint counts = arena.Load(d + 28u);
    const uint handleRegions = arena.Load(d + 32u);
    // Source: ceil(ceil(total / 256) / 32).
    const uint loops = ((total + 255u) / 256u + 31u) / 32u;
    for (uint k = 0u; k < loops; ++k) {
        const uint work = (group.x * loops + k) * 256u + thread.x;
        if (work >= total) continue;
        const uint x = find_projection(prefix, projections, work);
        const uint tagged = arena.Load(sorted + x * 4u);
        const uint handle = tagged >> 3u;
        const uint range = ranges + handle * 32u;
        const uint sy = arena.Load(range + 4u);
        const uint sz = arena.Load(range + 8u);
        const uint ey = arena.Load(range + 20u);
        const uint ez = arena.Load(range + 24u);
        const uint nz = ez - sz + 1u;
        const uint ny = ey - sy + 1u;
        const uint ordinal = work - arena.Load(prefix + x * 4u);
        const uint z = ordinal % nz + sz;
        const uint y = (ordinal / nz) % ny + sy;
        const uint rx = arena.Load(range);
        const uint regionId = rx + y * 4u + z * 16u;
        const uint regionOffset = regionId * projections;
        const uint endX = arena.Load(positions + handle * 8u + 4u);
        const uint histogram = ((tagged & 2u) != 0u) ? starts : active;
        const uint first = arena.Load(histogram + (regionOffset + x + 1u) * 4u);
        const uint last = arena.Load(histogram + (regionOffset + endX) * 4u);
        arena.Store2(handleRegions + work * 8u, uint2(handle, regionOffset));
        // regionOverlapType is 64-bit in the source; the local difference
        // is bounded by the count of sorted projections, fitting uint32.
        arena.Store2(counts + work * 8u, uint2(last - first, 0u));
    }
}
