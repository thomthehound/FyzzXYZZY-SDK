// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Source: broadphase.cu::createRegionsKernel / BP_CREATE_REGIONS.
// Source grid: 32 workgroups x 256 threads; contiguous 256-element
// slices per block, with ceil(ceil(2*numHandles/256)/32) iterations.
// Four bins per axis, determined by the count of all end projections.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset;
    uint arena_bytes;
    uint r0; uint r1; uint r2; uint r3; uint r4; uint r5;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint desc = params.descriptor_offset;
    const uint boxes = arena.Load(desc + 40u);
    const uint projections = 2u * boxes;
    const uint elementsPerRegion = (boxes + 3u) / 4u;
    const uint iterations = ((projections + 255u) / 256u + 31u) / 32u;
    const uint sortedX = arena.Load(desc + 0u);
    const uint ranges = arena.Load(desc + 28u);
    const uint startAccum = arena.Load(desc + 32u);
    const uint regionAccum = arena.Load(desc + 36u);
    for (uint step = 0u; step < iterations; ++step) {
        const uint index = (group.x * iterations + step) * 256u + thread.x;
        if (index >= projections) continue;
        const uint tagged = arena.Load(sortedX + index * 4u);
        uint regionCount = 0u;
        if ((tagged & 1u) != 0u) {
            const uint handle = tagged >> 3u;
            uint3 first = uint3(0u, 0u, 0u);
            uint3 last = uint3(0u, 0u, 0u);
            uint startX = 0u, endX = 0u;
            [unroll]
            for (uint axis = 0u; axis < 3u; ++axis) {
                const uint sap = arena.Load(desc + 4u + axis * 4u);
                const uint allEnds = arena.Load(desc + 16u + axis * 4u);
                const uint begin = arena.Load(sap + handle * 8u);
                const uint finish = arena.Load(sap + handle * 8u + 4u);
                if (axis == 0u) {
                    startX = begin;
                    endX = finish;
                }
                first[axis] = arena.Load(allEnds + begin * 4u) / elementsPerRegion;
                last[axis] = arena.Load(allEnds + finish * 4u) / elementsPerRegion;
            }
            arena.Store4(ranges + handle * 32u,
                         uint4(first.x, first.y, first.z, 0u));
            arena.Store4(ranges + handle * 32u + 16u,
                         uint4(last.x, last.y, last.z, 0u));

            const uint yz = (last.y - first.y + 1u) *
                            (last.z - first.z + 1u);
            arena.Store(startAccum + startX * 4u, yz);
            arena.Store(startAccum + endX * 4u, 0u);
            if (last.x >= first.x)
                regionCount = (last.x - first.x + 1u) * yz;
        }
        arena.Store(regionAccum + index * 4u, regionCount);
    }
}
