// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// PhysX broadphase.cu::writeOutOverlapChecksForInsertedBoundsRegionsHistogram
// (BP_WRITEOUT_OVERLAPCHECKS_HISTOGRAM_NEWBOUNDS), original 32 x 256 launch.
// Source allocations are independent native RAW views, not a packed arena.
// UAV views are required for D3D12 when source-read and source-written slices
// coexist in one heap-root buffer. Actual write intent is in the host manifest.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer sortedHandles : register(u0, space0);
#ifdef __spirv__
[[vk::binding(1, 0)]]
#endif
RWByteAddressBuffer positions : register(u1, space0);
#ifdef __spirv__
[[vk::binding(2, 0)]]
#endif
RWByteAddressBuffer regionRanges : register(u2, space0);
#ifdef __spirv__
[[vk::binding(3, 0)]]
#endif
RWByteAddressBuffer regionPrefix : register(u3, space0);
#ifdef __spirv__
[[vk::binding(4, 0)]]
#endif
RWByteAddressBuffer regionTotal : register(u4, space0);
#ifdef __spirv__
[[vk::binding(5, 0)]]
#endif
RWByteAddressBuffer activeHist : register(u5, space0);
#ifdef __spirv__
[[vk::binding(6, 0)]]
#endif
RWByteAddressBuffer startHist : register(u6, space0);
#ifdef __spirv__
[[vk::binding(7, 0)]]
#endif
RWByteAddressBuffer overlapCounts : register(u7, space0);
#ifdef __spirv__
[[vk::binding(8, 0)]]
#endif
RWByteAddressBuffer handleRegions : register(u8, space0);

struct Parameters {
    uint boxes;
    uint sorted_offset;
    uint positions_offset;
    uint ranges_offset;
    uint prefix_offset;
    uint total_offset;
    uint active_offset;
    uint starts_offset;
    uint counts_offset;
    uint handles_offset;
    uint reserved0;
    uint reserved1;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

uint find_projection(uint count, uint ordinal) {
    uint lo = 0u, hi = count;
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2u;
        if (regionPrefix.Load(params.prefix_offset + mid * 4u) <= ordinal)
            lo = mid + 1u;
        else hi = mid;
    }
    return lo - 1u;
}

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint projections = params.boxes * 2u;
    // This is the GPU-resident PxgBroadPhaseDesc::regionAccumTotal field.
    const uint total = regionTotal.Load(params.total_offset);
    const uint loops = ((total + 255u) / 256u + 31u) / 32u;
    for (uint k = 0u; k < loops; ++k) {
        const uint work = (group.x * loops + k) * 256u + thread.x;
        if (work >= total) continue;
        const uint x = find_projection(projections, work);
        const uint tagged = sortedHandles.Load(params.sorted_offset + x * 4u);
        const uint handle = tagged >> 3u;
        const uint range = params.ranges_offset + handle * 32u;
        const uint sy = regionRanges.Load(range + 4u);
        const uint sz = regionRanges.Load(range + 8u);
        const uint ey = regionRanges.Load(range + 20u);
        const uint ez = regionRanges.Load(range + 24u);
        const uint nz = ez - sz + 1u;
        const uint ny = ey - sy + 1u;
        const uint ordinal = work -
            regionPrefix.Load(params.prefix_offset + x * 4u);
        const uint z = ordinal % nz + sz;
        const uint y = (ordinal / nz) % ny + sy;
        const uint rx = regionRanges.Load(range) + ordinal / (ny * nz);
        const uint regionId = rx + y * 4u + z * 16u;
        const uint regionOffset = regionId * projections;
        const uint endX = positions.Load(params.positions_offset + handle * 8u + 4u);
        const bool isNew = (tagged & 2u) != 0u;
        const uint first = isNew
            ? startHist.Load(params.starts_offset + (regionOffset + x + 1u) * 4u)
            : activeHist.Load(params.active_offset + (regionOffset + x + 1u) * 4u);
        const uint last = isNew
            ? startHist.Load(params.starts_offset + (regionOffset + endX) * 4u)
            : activeHist.Load(params.active_offset + (regionOffset + endX) * 4u);
        handleRegions.Store2(params.handles_offset + work * 8u,
                             uint2(handle, regionOffset));
        overlapCounts.Store2(params.counts_offset + work * 8u,
                             uint2(last - first, 0u));
    }
}
