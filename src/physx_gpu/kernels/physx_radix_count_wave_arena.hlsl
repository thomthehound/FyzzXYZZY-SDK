// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// PhysX gpucommon/src/CUDA/RadixSort.cuh::radixSortSingleBlock:
// accumulate bin counts per thread in eight packed 16-bit pairs,
// reduce counts within a native GPU wave, and only then update the
// 16 block-wide bin totals. This eliminates the fallback's
// per-scalar InterlockedAdd to one of 16 heavily contended LDS bins.
// Each logical source block retains the exact contiguous uint4
// work allocation and 16x32 block histogram layout.
//
// The 32-bit arena offset contract limits any realistic total key
// buffer to well below the per-thread 16-bit packed overflow bound.
// Reductions unpack each half before the wave sum, so the entire wave
// does not inherit the packed 16-bit arithmetic bound.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer arena : register(u0, space0);

struct Parameters {
    uint descriptor_offset;
    uint num_keys;
    uint start_bit;
    uint arena_bytes;
    uint r0; uint r1; uint r2; uint r3;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

groupshared uint radixTotals[16];

[numthreads(1024, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint block = group.x;
    const uint axis = group.y;
    const uint desc = params.descriptor_offset + axis * 16u;
    const uint inputKeys = arena.Load(desc);
    const uint radixCounts = arena.Load(desc + 4u);

    if (thread.x < 16u) radixTotals[thread.x] = 0u;
    GroupMemoryBarrierWithGroupSync();

    const uint numVectors = (params.num_keys + 3u) / 4u;
    const uint requiredBlocks = (numVectors + 1023u) / 1024u;
    const uint iterations = (requiredBlocks + 31u) / 32u;
    const uint first = min(iterations * block * 1024u, numVectors);
    const uint last = min(first + iterations * 1024u, numVectors);

    uint packedBins[8] = {0u,0u,0u,0u,0u,0u,0u,0u};
    for (uint vectorIndex = first + thread.x; vectorIndex < last;
         vectorIndex += 1024u) {
        uint4 digits;
        [unroll]
        for (uint lane = 0u; lane < 4u; ++lane) {
            const uint scalar = vectorIndex * 4u + lane;
            const uint key = scalar < params.num_keys
                ? arena.Load(inputKeys + scalar * 4u) : 0xffffffffu;
            digits[lane] = (key >> params.start_bit) & 15u;
        }
        [unroll]
        for (uint pair = 0u; pair < 8u; ++pair) {
            const uint firstDigit = 2u * pair;
            const uint lo = (digits.x == firstDigit ? 1u : 0u) +
                            (digits.y == firstDigit ? 1u : 0u) +
                            (digits.z == firstDigit ? 1u : 0u) +
                            (digits.w == firstDigit ? 1u : 0u);
            const uint secondDigit = firstDigit + 1u;
            const uint hi = (digits.x == secondDigit ? 1u : 0u) +
                            (digits.y == secondDigit ? 1u : 0u) +
                            (digits.z == secondDigit ? 1u : 0u) +
                            (digits.w == secondDigit ? 1u : 0u);
            packedBins[pair] += lo | (hi << 16u);
        }
    }

    [unroll]
    for (uint pair = 0u; pair < 8u; ++pair) {
        const uint packed = packedBins[pair];
        const uint lo = WaveActiveSum(packed & 0xffffu);
        const uint hi = WaveActiveSum(packed >> 16u);
        if (WaveIsFirstLane()) {
            uint oldValue;
            InterlockedAdd(radixTotals[2u * pair], lo, oldValue);
            InterlockedAdd(radixTotals[2u * pair + 1u], hi, oldValue);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (thread.x < 16u)
        arena.Store(radixCounts + (thread.x * 32u + block) * 4u,
                    radixTotals[thread.x]);
}
