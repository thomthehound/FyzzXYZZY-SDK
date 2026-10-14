// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// CUDA broadphase.cu::copyReports / BP_COPY_REPORTS.
// Source-grid 64x256. This writes the logical mapped-publication output;
// the context's GPU→CPU transfer and original task continuation remain
// separately owned by PxgCudaBroadPhaseSap::gpuDMABack.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset;
    uint arena_bytes;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    uint reserved3;
    uint reserved4;
    uint reserved5;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint desc = params.descriptor_offset;
    const uint totalCounts = arena.Load(desc);
    const uint aggregateCounts = arena.Load(desc + 4u);
    const uint rawFound = arena.Load(totalCounts);
    const uint rawLost = arena.Load(totalCounts + 4u);
    const uint maxPairs = arena.Load(desc + 28u);
    // Stage 2's aggregate counts describe pairs actually present in
    // capacity-clipped raw reports. Preserve the source's unsigned math.
    const uint found = min(rawFound, maxPairs) -
                       arena.Load(aggregateCounts);
    const uint lost = min(rawLost, maxPairs) -
                      arena.Load(aggregateCounts + 4u);
    const uint srcFound = arena.Load(desc + 8u);
    const uint srcLost = arena.Load(desc + 12u);
    const uint dstFound = arena.Load(desc + 16u);
    const uint dstLost = arena.Load(desc + 20u);

    const uint index = group.x * 256u + thread.x;
    if (index == 0u)
        arena.Store(arena.Load(desc + 24u),
                    (rawFound > maxPairs || rawLost > maxPairs) ? 1u : 0u);
    // PhysX treats both report arrays as contiguous PxU32 data and
    // copies 2 words per canonical PxgBroadPhasePair.
    for (uint i = index; i < found * 2u; i += 64u * 256u)
        arena.Store(dstFound + i * 4u, arena.Load(srcFound + i * 4u));
    for (uint i = index; i < lost * 2u; i += 64u * 256u)
        arena.Store(dstLost + i * 4u, arena.Load(srcLost + i * 4u));
}
