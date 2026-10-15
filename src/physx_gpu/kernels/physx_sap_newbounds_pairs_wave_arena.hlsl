// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::generateFoundPairsForNewBoundsRegion,
// BP_GENERATE_FOUNDPAIR_NEWBOUNDS, original 32 x 256 dispatch.
// Each thread tests one candidate (64-bit global work index), then
// compacts its 0/1 result with native wave operations, reserving output
// slots once per workgroup iteration using the original 32-bit counter.
// No 256-thread Hillis-Steele prefix, no uint64 global atomics.
// The host owns all input/output buffers and the found-count lifecycle.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);
struct Parameters {
    uint descriptor_offset; uint arena_bytes;
    uint r0; uint r1; uint r2; uint r3; uint r4; uint r5;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

uint64_t read64(uint offset) {
    const uint2 pair = arena.Load2(offset);
    return uint64_t(pair.x) | (uint64_t(pair.y) << 32u);
}
uint find_candidate(uint prefix, uint n, uint64_t work) {
    uint low = 0u, high = n;
    while (low < high) {
        const uint mid = low + (high - low) / 2u;
        if (read64(prefix + mid * 8u) <= work) low = mid + 1u;
        else high = mid;
    }
    return low - 1u;
}
bool intersects(uint bounds, uint a, uint b) {
    const uint ap = bounds + a * 24u;
    const uint bp = bounds + b * 24u;
    [unroll]
    for (uint axis = 0u; axis < 3u; ++axis)
        if (arena.Load(bp + axis * 4u) > arena.Load(ap + (axis + 3u) * 4u) ||
            arena.Load(ap + axis * 4u) > arena.Load(bp + (axis + 3u) * 4u))
            return false;
    return true;
}

groupshared uint nextWave;
groupshared uint waveTotals[64];
groupshared uint waveBases[64];
groupshared uint groupBase;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint d = params.descriptor_offset;
    const uint bounds = arena.Load(d + 0u);
    const uint groups = arena.Load(d + 4u);
    const uint envs = arena.Load(d + 8u);
    const uint sorted = arena.Load(d + 12u);
    const uint sap = arena.Load(d + 16u);
    const uint ranges = arena.Load(d + 20u);
    const uint prefix = arena.Load(d + 24u);
    const uint workCount = arena.Load(arena.Load(d + 28u));
    const uint64_t total = read64(arena.Load(d + 32u));
    const uint regions = arena.Load(d + 36u);
    const uint activeHist = arena.Load(d + 40u);
    const uint startHist = arena.Load(d + 44u);
    const uint activeOrder = arena.Load(d + 48u);
    const uint startOrder = arena.Load(d + 52u);
    const uint reports = arena.Load(d + 56u);
    const uint counter = arena.Load(d + 60u);
    const uint capacity = arena.Load(d + 64u);
    const uint projections = 2u * arena.Load(d + 68u);
    const uint boxCapacity = arena.Load(d + 72u);
    const bool useEnv = arena.Load(d + 76u) != 0u;

    // ceil(total/8192) without overflowing on total+8191.
    const uint64_t iterations = total == 0ull ? 0ull :
        ((total - 1ull) / 8192ull + 1ull);
    if (thread.x == 0u) nextWave = 0u;
    GroupMemoryBarrierWithGroupSync();
    const uint lanes = WaveGetLaneCount();
    const uint lane = WaveGetLaneIndex();
    const uint waveCount = 256u / lanes;
    uint wave = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(nextWave, 1u, wave);
    wave = WaveReadLaneFirst(wave);

    for (uint64_t step = 0ull; step < iterations; ++step) {
        const uint64_t work = (uint64_t(group.x) * iterations + step) * 256ull +
                              uint64_t(wave * lanes + lane);
        uint handle = 0u, other = 0u;
        bool report = false;
        if (work < total && workCount != 0u) {
            const uint candidate = find_candidate(prefix, workCount, work);
            handle = arena.Load(regions + candidate * 8u);
            const uint regionOffset = arena.Load(regions + candidate * 8u + 4u);
            if (handle < boxCapacity) {
                const uint minX = arena.Load(sap + handle * 8u);
                const uint maxX = arena.Load(sap + handle * 8u + 4u);
                const bool isNew = (arena.Load(sorted + minX * 4u) & 2u) != 0u;
                const uint hist = isNew ? startHist : activeHist;
                const uint ordered = isNew ? startOrder : activeOrder;
                const uint begin = arena.Load(hist + (regionOffset + minX + 1u) * 4u);
                const uint end = arena.Load(hist + (regionOffset + maxX) * 4u);
                const uint64_t partnerIndex =
                    uint64_t(begin) + work - read64(prefix + candidate * 8u);
                if (end > begin && partnerIndex < uint64_t(end)) {
                    other = arena.Load(ordered +
                            (regionOffset + uint(partnerIndex)) * 4u);
                    if (other < boxCapacity) {
                        const uint ownerRange = ranges + handle * 32u;
                        const uint otherRange = ranges + other * 32u;
                        const uint x = max(arena.Load(ownerRange),
                                           arena.Load(otherRange));
                        const uint y = max(arena.Load(ownerRange + 4u),
                                           arena.Load(otherRange + 4u));
                        const uint z = max(arena.Load(ownerRange + 8u),
                                           arena.Load(otherRange + 8u));
                        const uint canonical =
                            (x + y * 4u + z * 16u) * projections;
                        if (regionOffset == canonical) {
                            const uint groupA = arena.Load(groups + handle * 4u);
                            const uint groupB = arena.Load(groups + other * 4u);
                            bool allowed = groupA != groupB;
                            if (useEnv && allowed) {
                                const uint envA = arena.Load(envs + handle * 4u);
                                const uint envB = arena.Load(envs + other * 4u);
                                allowed = envA == envB || envA == 0xffffffffu ||
                                          envB == 0xffffffffu;
                            }
                            report = allowed && intersects(bounds, handle, other);
                        }
                    }
                }
            }
        }

        const uint bit = report ? 1u : 0u;
        const uint local = WavePrefixSum(bit);
        const uint waveTotal = WaveActiveSum(bit);
        if (WaveIsFirstLane()) waveTotals[wave] = waveTotal;
        GroupMemoryBarrierWithGroupSync();

        if (thread.x == 0u) {
            uint count = 0u;
            for (uint w = 0u; w < waveCount; ++w) {
                waveBases[w] = count;
                count += waveTotals[w];
            }
            uint base = 0u;
            if (count != 0u) arena.InterlockedAdd(counter, count, base);
            groupBase = base;
        }
        GroupMemoryBarrierWithGroupSync();
        if (report) {
            const uint position = groupBase + waveBases[wave] + local;
            if (position < capacity)
                arena.Store2(reports + position * 8u,
                             uint2(min(handle, other), max(handle, other)));
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
