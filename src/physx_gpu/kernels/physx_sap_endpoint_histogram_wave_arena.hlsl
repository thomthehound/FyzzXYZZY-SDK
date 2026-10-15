// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Pinned PhysX broadphase.cu::firstPassEndPtsHistogram:
// BP_COMPUTE_ENDPT_HISTOGRAM, exactly 32x3 groups of 256 threads.
//
// CUDA packs THREE 0/1 counters in one 32-bit integer:
// start << 20 | undeletedEnd << 10 | everyEnd.
// No 10-bit field may overflow: one workgroup chunk contains at most
// 256 projections, each contributing at most one to each field.
//
// Restore source packed arithmetic and the original warp→block
// prefix hierarchy with a hardware-native wave prefix scan.
// One WavePrefixSum and WaveActiveSum per projection, 3 full-group
// barriers per chunk, and 512 B (2x64 u32) of wave subtotal scratch.
// The old portable 3x256 uint3 Hillis-Steele scan is retained solely
// as a correctness fallback on development/software GPU backends.
// Full physical-wave order is not guaranteed to track SV_GroupIndex;
// assign one contiguous LOGICAL 256-entry slice per actual wave
// at workgroup entry, preserving X endpoint ordering.
//
// Requires compute subgroup arithmetic; independently queried by
// D3D12 SM6.6/WaveOps or Vulkan device subgroup-arithmetic capability.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);
struct Parameters {
    uint descriptor_offset;
    uint previous_boxes;
    uint created_boxes;
    uint removed_boxes;
    uint is_incremental;
    uint arena_bytes;
    uint reserved0;
    uint reserved1;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint nextWave;
groupshared uint waveCounts[64];
groupshared uint waveBases[64];
groupshared uint3 carry;
groupshared uint3 iterationCarry;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint axisDesc = params.descriptor_offset + group.y * 80u;
    const uint currentBoxes = (params.is_incremental != 0u)
        ? params.previous_boxes
        : params.previous_boxes + params.created_boxes - params.removed_boxes;

    if (thread.x == 0u) nextWave = 0u;
    GroupMemoryBarrierWithGroupSync();

    const uint laneCount = WaveGetLaneCount();
    const uint laneIndex = WaveGetLaneIndex();
    const uint waveCount = 256u / laneCount;
    uint waveId = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(nextWave, 1u, waveId);
    waveId = WaveReadLaneFirst(waveId);

    [loop]
    for (uint frame = 0u; frame < 2u; ++frame) {
        if (frame == 1u && (params.is_incremental == 0u ||
                            params.removed_boxes == 0u))
            break;

        const uint boxes = (frame == 0u) ? currentBoxes : params.previous_boxes;
        const uint frameDesc = axisDesc + frame * 32u;
        const uint handles = arena.Load(frameDesc);
        const uint ends = arena.Load(frameDesc + 4u);
        const uint starts = arena.Load(frameDesc + 8u);
        const uint blockEnds = arena.Load(frameDesc + 20u);
        const uint blockStarts = arena.Load(frameDesc + 24u);
        const uint totalEnds = arena.Load(axisDesc + 64u);
        const uint blockTotal = arena.Load(axisDesc + 68u);
        const uint projections = boxes * 2u;
        const uint required = (projections + 255u) / 256u;
        const uint iterations = (required + 31u) / 32u;

        if (thread.x == 0u) carry = uint3(0u, 0u, 0u);
        GroupMemoryBarrierWithGroupSync();
        for (uint iteration = 0u; iteration < iterations; ++iteration) {
            const uint index = (group.x * iterations + iteration) * 256u +
                               waveId * laneCount + laneIndex;
            const bool active = index < projections;
            const uint handle = active ? arena.Load(handles + index * 4u) : 0u;
            const bool isStart = (handle & 1u) != 0u;
            const bool isDeleted = (handle & 4u) != 0u;
            const uint packed =
                uint(active && isStart && !isDeleted) << 20u |
                uint(active && !isStart && !isDeleted) << 10u |
                uint(active && !isStart);

            const uint wavePrefix = WavePrefixSum(packed);
            const uint waveTotal = WaveActiveSum(packed);
            if (WaveIsFirstLane()) waveCounts[waveId] = waveTotal;
            GroupMemoryBarrierWithGroupSync();

            if (thread.x == 0u) {
                uint subtotal = 0u;
                for (uint w = 0u; w < waveCount; ++w) {
                    const uint count = waveCounts[w];
                    waveBases[w] = subtotal;
                    subtotal += count;
                }
                iterationCarry = carry;
                carry += uint3(subtotal >> 20u,
                               (subtotal >> 10u) & 1023u,
                               subtotal & 1023u);
            }
            GroupMemoryBarrierWithGroupSync();

            const uint prefix = waveBases[waveId] + wavePrefix;
            if (active) {
                arena.Store(ends + index * 4u,
                            iterationCarry.y + ((prefix >> 10u) & 1023u));
                arena.Store(starts + index * 4u,
                            iterationCarry.x + (prefix >> 20u));
                if (frame == 0u)
                    arena.Store(totalEnds + index * 4u,
                                iterationCarry.z + (prefix & 1023u));
            }
            // Carry and wave totals cannot be overwritten while another
            // subgroup is consuming them for the current iteration.
            GroupMemoryBarrierWithGroupSync();
        }
        if (thread.x == 0u) {
            arena.Store(blockEnds + group.x * 4u, carry.y);
            arena.Store(blockStarts + group.x * 4u, carry.x);
            if (frame == 0u)
                arena.Store(blockTotal + group.x * 4u, carry.z);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
