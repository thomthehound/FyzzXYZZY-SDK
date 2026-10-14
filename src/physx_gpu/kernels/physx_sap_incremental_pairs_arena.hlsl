// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// broadphase.cu::performIncrementalSAP (BP_INCREMENTAL_SAP).
// CUDA's per-warp report scan and atomic block reservation are lowered to
// wave-width-independent workgroup scans; the candidate and overlap rules,
// global counter semantics, and 256x3x256 source dispatch are preserved.
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

groupshared uint2 scanA[256];
groupshared uint2 scanB[256];
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

    for (uint block = group.x; block < batches; block += 256u) {
        const uint workId = block * 256u + thread.x;
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
                down ? (startHandle ? 20u : 24u)
                     : (startHandle ? 12u : 16u));
            const uint ordered = load_axis(group.y,
                down ? (startHandle ? 36u : 40u)
                     : (startHandle ? 28u : 32u));
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

        const uint2 flags = report ? (reportType == 0u ? uint2(1u, 0u) :
                                                       uint2(0u, 1u)) :
                                    uint2(0u, 0u);
        scanA[thread.x] = flags;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint shift = 1u; shift < 256u; shift <<= 1u) {
            if (shift == 1u || shift == 4u || shift == 16u || shift == 64u) {
                uint2 value = scanA[thread.x];
                if (thread.x >= shift) value += scanA[thread.x - shift];
                scanB[thread.x] = value;
            } else {
                uint2 value = scanB[thread.x];
                if (thread.x >= shift) value += scanB[thread.x - shift];
                scanA[thread.x] = value;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (thread.x == 0u) {
            const uint2 sums = scanA[255u];
            uint2 bases = uint2(0u, 0u);
            if (sums.x != 0u)
                arena.InterlockedAdd(counters, sums.x, bases.x);
            if (sums.y != 0u)
                arena.InterlockedAdd(counters + 4u, sums.y, bases.y);
            groupBase = bases;
        }
        GroupMemoryBarrierWithGroupSync();

        if (report) {
            const uint index = groupBase[reportType] +
                scanA[thread.x][reportType] - 1u;
            if (index < capacity) {
                const uint output = reportType == 0u ? foundReports : lostReports;
                const uint a = min(handle, other);
                const uint b = max(handle, other);
                arena.Store2(output + index * 8u, uint2(a, b));
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
