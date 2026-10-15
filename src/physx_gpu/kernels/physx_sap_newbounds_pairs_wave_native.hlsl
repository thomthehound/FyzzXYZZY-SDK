// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::generateFoundPairsForNewBoundsRegion,
// BP_GENERATE_FOUNDPAIR_NEWBOUNDS, original 64 x 256 dispatch.
// Each thread tests one candidate (64-bit global work index), then
// compacts its 0/1 result with native wave operations, reserving output
// slots once per workgroup iteration using the original 32-bit counter.
// No 256-thread Hillis-Steele prefix, no uint64 global atomics.
// The host owns all input/output buffers and the found-count lifecycle.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer boundsBuffer : register(u0, space0);
#ifdef __spirv__
[[vk::binding(1, 0)]]
#endif
RWByteAddressBuffer groupBuffer : register(u1, space0);
#ifdef __spirv__
[[vk::binding(2, 0)]]
#endif
RWByteAddressBuffer envBuffer : register(u2, space0);
#ifdef __spirv__
[[vk::binding(3, 0)]]
#endif
RWByteAddressBuffer sortedBuffer : register(u3, space0);
#ifdef __spirv__
[[vk::binding(4, 0)]]
#endif
RWByteAddressBuffer sapBuffer : register(u4, space0);
#ifdef __spirv__
[[vk::binding(5, 0)]]
#endif
RWByteAddressBuffer rangeBuffer : register(u5, space0);
#ifdef __spirv__
[[vk::binding(6, 0)]]
#endif
RWByteAddressBuffer prefixBuffer : register(u6, space0);
#ifdef __spirv__
[[vk::binding(7, 0)]]
#endif
RWByteAddressBuffer workCounter : register(u7, space0);
#ifdef __spirv__
[[vk::binding(8, 0)]]
#endif
RWByteAddressBuffer overlapTotal : register(u8, space0);
#ifdef __spirv__
[[vk::binding(9, 0)]]
#endif
RWByteAddressBuffer regionBuffer : register(u9, space0);
#ifdef __spirv__
[[vk::binding(10, 0)]]
#endif
RWByteAddressBuffer activeHistBuffer : register(u10, space0);
#ifdef __spirv__
[[vk::binding(11, 0)]]
#endif
RWByteAddressBuffer startHistBuffer : register(u11, space0);
#ifdef __spirv__
[[vk::binding(12, 0)]]
#endif
RWByteAddressBuffer activeOrderBuffer : register(u12, space0);
#ifdef __spirv__
[[vk::binding(13, 0)]]
#endif
RWByteAddressBuffer startOrderBuffer : register(u13, space0);
#ifdef __spirv__
[[vk::binding(14, 0)]]
#endif
RWByteAddressBuffer reportBuffer : register(u14, space0);
#ifdef __spirv__
[[vk::binding(15, 0)]]
#endif
RWByteAddressBuffer foundCounter : register(u15, space0);

struct Parameters {
    uint bounds; uint groups; uint envs; uint sorted;
    uint sap; uint ranges; uint prefix; uint workCount;
    uint total; uint regions; uint activeHist; uint startHist;
    uint activeOrder; uint startOrder; uint reports; uint counter;
    uint currentBoxes; uint capacity; uint boxCapacity; uint useEnv;
    uint r0; uint r1; uint r2; uint r3;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif
uint64_t readPrefix64(uint at) {
    const uint2 words = prefixBuffer.Load2(at);
    return uint64_t(words.x) | (uint64_t(words.y) << 32u);
}
uint64_t readOverlapTotal64(uint at) {
    const uint2 words = overlapTotal.Load2(at);
    return uint64_t(words.x) | (uint64_t(words.y) << 32u);
}

uint find_candidate(uint prefix, uint n, uint64_t work) {
    uint low = 0u, high = n;
    while (low < high) {
        const uint mid = low + (high - low) / 2u;
        if (readPrefix64(prefix + mid * 8u) <= work) low = mid + 1u;
        else high = mid;
    }
    return low - 1u;
}
bool intersects(uint bounds, uint a, uint b) {
    const uint ap = bounds + a * 24u;
    const uint bp = bounds + b * 24u;
    [unroll]
    for (uint axis = 0u; axis < 3u; ++axis)
        if (boundsBuffer.Load(bp + axis * 4u) > boundsBuffer.Load(ap + (axis + 3u) * 4u) ||
            boundsBuffer.Load(ap + axis * 4u) > boundsBuffer.Load(bp + (axis + 3u) * 4u))
            return false;
    return true;
}

groupshared uint nextWave;
groupshared uint waveTotals[64];
groupshared uint waveBases[64];
groupshared uint groupBase;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint bounds = params.bounds;
    const uint groups = params.groups;
    const uint envs = params.envs;
    const uint sorted = params.sorted;
    const uint sap = params.sap;
    const uint ranges = params.ranges;
    const uint prefix = params.prefix;
    const uint workCount = workCounter.Load(params.workCount);
    const uint64_t total = readOverlapTotal64(params.total);
    const uint regions = params.regions;
    const uint activeHist = params.activeHist;
    const uint startHist = params.startHist;
    const uint activeOrder = params.activeOrder;
    const uint startOrder = params.startOrder;
    const uint reports = params.reports;
    const uint counter = params.counter;
    const uint capacity = params.capacity;
    const uint projections = 2u * params.currentBoxes;
    const uint boxCapacity = params.boxCapacity;
    const bool useEnv = params.useEnv != 0u;

    // ceil(total/16384) without overflowing on total+8191.
    const uint64_t iterations = total == 0ull ? 0ull :
        ((total - 1ull) / 16384ull + 1ull);
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
            handle = regionBuffer.Load(regions + candidate * 8u);
            const uint regionOffset = regionBuffer.Load(regions + candidate * 8u + 4u);
            if (handle < boxCapacity) {
                const uint minX = sapBuffer.Load(sap + handle * 8u);
                const uint maxX = sapBuffer.Load(sap + handle * 8u + 4u);
                const bool isNew = (sortedBuffer.Load(sorted + minX * 4u) & 2u) != 0u;
                const uint begin = isNew
                    ? startHistBuffer.Load(startHist + (regionOffset + minX + 1u) * 4u)
                    : activeHistBuffer.Load(activeHist + (regionOffset + minX + 1u) * 4u);
                const uint end = isNew
                    ? startHistBuffer.Load(startHist + (regionOffset + maxX) * 4u)
                    : activeHistBuffer.Load(activeHist + (regionOffset + maxX) * 4u);
                const uint64_t partnerIndex =
                    uint64_t(begin) + work - readPrefix64(prefix + candidate * 8u);
                if (end > begin && partnerIndex < uint64_t(end)) {
                    other = isNew
                        ? startOrderBuffer.Load(startOrder + (regionOffset + uint(partnerIndex)) * 4u)
                        : activeOrderBuffer.Load(activeOrder + (regionOffset + uint(partnerIndex)) * 4u);
                    if (other < boxCapacity) {
                        const uint ownerRange = ranges + handle * 32u;
                        const uint otherRange = ranges + other * 32u;
                        const uint x = max(rangeBuffer.Load(ownerRange),
                                           rangeBuffer.Load(otherRange));
                        const uint y = max(rangeBuffer.Load(ownerRange + 4u),
                                           rangeBuffer.Load(otherRange + 4u));
                        const uint z = max(rangeBuffer.Load(ownerRange + 8u),
                                           rangeBuffer.Load(otherRange + 8u));
                        const uint canonical =
                            (x + y * 4u + z * 16u) * projections;
                        if (regionOffset == canonical) {
                            const uint groupA = groupBuffer.Load(groups + handle * 4u);
                            const uint groupB = groupBuffer.Load(groups + other * 4u);
                            bool allowed = groupA != groupB;
                            if (useEnv && allowed) {
                                const uint envA = envBuffer.Load(envs + handle * 4u);
                                const uint envB = envBuffer.Load(envs + other * 4u);
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
            if (count != 0u) foundCounter.InterlockedAdd(counter, count, base);
            groupBase = base;
        }
        GroupMemoryBarrierWithGroupSync();
        if (report) {
            const uint position = groupBase + waveBases[wave] + local;
            if (position < capacity)
                reportBuffer.Store2(reports + position * 8u,
                             uint2(min(handle, other), max(handle, other)));
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
