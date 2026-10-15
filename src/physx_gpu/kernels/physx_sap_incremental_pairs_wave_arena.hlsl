// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Pinned PhysX broadphase.cu::performIncrementalSAP, BP_INCREMENTAL_SAP.
// The source scans 0/1 found and lost tags separately in native warps,
// then reserves their output space with two 32-bit global atomics.
// A 256-element batch has at most 256 contributions to either stream:
// pack (found << 16) | lost and scan with ONE native wave prefix,
// prefix the small wave-total table, then reserve both original counters.
// Preserve per-batch global atomic reservation and cross-workgroup
// publication semantics; do not imply global output is sorted.
// 3 workgroup barriers / batch rather than a full 8-round
// shared Hillis-Steele scan. No WMMA or fixed CUDA wave32 requirement.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset;
    uint previous_num_boxes;
    uint comparison_totals_offset;
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

groupshared uint nextWave;
groupshared uint waveTotals[64];
groupshared uint waveBases[64];
groupshared uint2 groupBase;

uint load_axis(uint axis, uint field) {
    return arena.Load(params.descriptor_offset + axis * 48u + field);
}

// The source's sorted comparison-prefix binarySearch is upper_bound - 1:
// the last endpoint with prefix <= work id, including repeated prefixes.
uint candidate_endpoint(uint prefixOffset, uint count, uint workId) {
    uint low = 0u;
    uint high = count;
    while (low < high) {
        const uint middle = low + (high - low) / 2u;
        const uint prefix = arena.Load(prefixOffset + middle * 4u);
        if (prefix <= workId) low = middle + 1u;
        else high = middle;
    }
    return low - 1u;
}

uint load_bound(uint bounds, uint handle, uint axis, bool upper) {
    return arena.Load(bounds + (handle * 6u + axis + (upper ? 3u : 0u)) * 4u);
}
bool overlap_axis(uint bounds, uint a, uint b, uint axis) {
    return load_bound(bounds, b, axis, false) <= load_bound(bounds, a, axis, true) &&
           load_bound(bounds, a, axis, false) <= load_bound(bounds, b, axis, true);
}
bool overlap_3d(uint bounds, uint a, uint b) {
    return overlap_axis(bounds, a, b, 0u) &&
           overlap_axis(bounds, a, b, 1u) &&
           overlap_axis(bounds, a, b, 2u);
}
bool separated_before(uint bounds, uint a, uint b, uint axis) {
    for (uint i = 0u; i < axis; ++i)
        if (!overlap_axis(bounds, a, b, i)) return true;
    return false;
}

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint desc = params.descriptor_offset + 144u;
    const uint oldBounds = arena.Load(desc + 0u);
    const uint newBounds = arena.Load(desc + 4u);
    const uint groupIds = arena.Load(desc + 8u);
    const uint envIds = arena.Load(desc + 12u);
    const uint foundReports = arena.Load(desc + 16u);
    const uint lostReports = arena.Load(desc + 20u);
    const uint counters = arena.Load(desc + 24u);
    const uint capacity = arena.Load(desc + 32u);
    const bool useEnv = arena.Load(desc + 36u) != 0u;
    const uint boxCapacity = arena.Load(desc + 28u);

    const uint total = arena.Load(params.comparison_totals_offset + group.y * 4u);
    const uint projections = params.previous_num_boxes * 2u;
    const uint batches = (total + 255u) / 256u;
    const uint prefix = load_axis(group.y, 8u);
    const uint handles = load_axis(group.y, 0u);
    const uint ranks = load_axis(group.y, 4u);

    if (thread.x == 0u) nextWave = 0u;
    GroupMemoryBarrierWithGroupSync();
    const uint laneCount = WaveGetLaneCount();
    const uint laneIndex = WaveGetLaneIndex();
    const uint waveCount = 256u / laneCount;
    uint waveId = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(nextWave, 1u, waveId);
    waveId = WaveReadLaneFirst(waveId);
    // Each physical wave claims a contiguous logical 256-entry slice,
    // rather than assuming physical wave order matches SV_GroupIndex.

    for (uint block = group.x; block < batches; block += 256u) {
        const uint workId = block * 256u +
                            waveId * laneCount + laneIndex;
        uint handle = 0u, other = 0u, reportType = 0u;
        bool report = false;

        if (workId < total) {
            const uint index = candidate_endpoint(prefix, projections, workId);
            const uint tagged = arena.Load(handles + index * 4u);
            handle = tagged >> 3u;
            const bool startHandle = (tagged & 1u) != 0u;
            const uint prevIndex = arena.Load(ranks + index * 4u);
            const bool down = prevIndex > index;
            const uint begin = down ? index : prevIndex + 1u;
            const uint local = workId - arena.Load(prefix + index * 4u);
            const uint histogram = load_axis(group.y,
                down ? (startHandle ? 24u : 20u)
                     : (startHandle ? 16u : 12u));
            const uint ordered = load_axis(group.y,
                down ? (startHandle ? 40u : 36u)
                     : (startHandle ? 32u : 28u));
            const uint partnerIndex = arena.Load(histogram + begin * 4u) + local;
            other = arena.Load(ordered + partnerIndex * 4u);
            const bool willFind = (!startHandle) != down;
            reportType = willFind ? 0u : 1u; // found : lost

            if (handle < boxCapacity && other < boxCapacity) {
                const uint groupA = arena.Load(groupIds + handle * 4u);
                const uint groupB = arena.Load(groupIds + other * 4u);
                bool allowed = groupA != groupB;
                if (useEnv) {
                    const uint envA = arena.Load(envIds + handle * 4u);
                    const uint envB = arena.Load(envIds + other * 4u);
                    allowed = allowed &&
                        (envA == envB || envA == 0xffffffffu ||
                         envB == 0xffffffffu);
                }
                if (allowed) {
                    const uint present = willFind ? newBounds : oldBounds;
                    const uint absent = willFind ? oldBounds : newBounds;
                    report = overlap_3d(present, handle, other) &&
                             !overlap_axis(absent, handle, other, group.y) &&
                             !separated_before(absent, handle, other, group.y);
                }
            }
        }

        const uint2 flags = report
            ? (reportType == 0u ? uint2(1u, 0u) : uint2(0u, 1u))
            : uint2(0u, 0u);
        // Entire source batch fits two 16-bit counters. Low half:
        // lost reports, upper half: found reports.
        const uint packed = (flags.x << 16u) | flags.y;
        const uint wavePrefix = WavePrefixSum(packed);
        const uint waveTotal = WaveActiveSum(packed);
        if (WaveIsFirstLane()) waveTotals[waveId] = waveTotal;
        GroupMemoryBarrierWithGroupSync();

        if (thread.x == 0u) {
            uint totalPacked = 0u;
            for (uint w = 0u; w < waveCount; ++w) {
                const uint count = waveTotals[w];
                waveBases[w] = totalPacked;
                totalPacked += count;
            }
            const uint foundCount = totalPacked >> 16u;
            const uint lostCount = totalPacked & 0xffffu;
            uint2 bases = uint2(0u, 0u);
            if (foundCount != 0u)
                arena.InterlockedAdd(counters, foundCount, bases.x);
            if (lostCount != 0u)
                arena.InterlockedAdd(counters + 4u, lostCount, bases.y);
            groupBase = bases;
        }
        GroupMemoryBarrierWithGroupSync();

        if (report) {
            const uint localPrefix = waveBases[waveId] + wavePrefix;
            const uint rank = reportType == 0u
                ? (localPrefix >> 16u) : (localPrefix & 0xffffu);
            const uint index = groupBase[reportType] + rank;
            if (index < capacity) {
                const uint output = reportType == 0u ? foundReports : lostReports;
                const uint a = min(handle, other);
                const uint b = max(handle, other);
                arena.Store2(output + index * 8u, uint2(a, b));
            }
        }
        // Original source waits before the next atomic reservation
        // because per-block report order follows successive batches.
        GroupMemoryBarrierWithGroupSync();
    }
}
