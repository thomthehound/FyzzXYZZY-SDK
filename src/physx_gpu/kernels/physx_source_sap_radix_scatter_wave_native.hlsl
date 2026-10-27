// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// PxgCudaBroadPhaseSap::runRadixSort / RS_CALCULATERANKS_MULTIBLOCK_COUNT.
// Wave-optimized stable 4-bit LSD radix, source 32x3x1024 work distribution. Original
// key/payload ping-pong pointers; histograms stay in the source allocation.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer inKeyX:register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer inRankX:register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer outKeyX:register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer outRankX:register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer countX:register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer inKeyY:register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer inRankY:register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer outKeyY:register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer outRankY:register(u8,space0);
#ifdef __spirv__
[[vk::binding(9,0)]]
#endif
RWByteAddressBuffer countY:register(u9,space0);
#ifdef __spirv__
[[vk::binding(10,0)]]
#endif
RWByteAddressBuffer inKeyZ:register(u10,space0);
#ifdef __spirv__
[[vk::binding(11,0)]]
#endif
RWByteAddressBuffer inRankZ:register(u11,space0);
#ifdef __spirv__
[[vk::binding(12,0)]]
#endif
RWByteAddressBuffer outKeyZ:register(u12,space0);
#ifdef __spirv__
[[vk::binding(13,0)]]
#endif
RWByteAddressBuffer outRankZ:register(u13,space0);
#ifdef __spirv__
[[vk::binding(14,0)]]
#endif
RWByteAddressBuffer countZ:register(u14,space0);
struct Params {
    uint ikx;
    uint irx;
    uint okx;
    uint orx;
    uint cx;
    uint iky;
    uint iry;
    uint oky;
    uint ory;
    uint cy;
    uint ikz;
    uint irz;
    uint okz;
    uint orz;
    uint cz;
    uint numKeys;
    uint startBit;
    uint pad0;
    uint pad1;
    uint pad2;
};
#ifdef __spirv__
[[vk::push_constant]] Params p;
#else
ConstantBuffer<Params> p:register(b0,space0);
#endif

uint load_key(uint axis,uint byteOffset) {if(axis==0u)return inKeyX.Load(byteOffset);if(axis==1u)return inKeyY.Load(byteOffset);return inKeyZ.Load(byteOffset);}
uint load_rank(uint axis,uint byteOffset) {if(axis==0u)return inRankX.Load(byteOffset);if(axis==1u)return inRankY.Load(byteOffset);return inRankZ.Load(byteOffset);}
uint load_count(uint axis,uint byteOffset) {if(axis==0u)return countX.Load(byteOffset);if(axis==1u)return countY.Load(byteOffset);return countZ.Load(byteOffset);}
void store_key(uint axis,uint byteOffset,uint value) {if(axis==0u)outKeyX.Store(byteOffset,value);else if(axis==1u)outKeyY.Store(byteOffset,value);else outKeyZ.Store(byteOffset,value);}
void store_rank(uint axis,uint byteOffset,uint value) {if(axis==0u)outRankX.Store(byteOffset,value);else if(axis==1u)outRankY.Store(byteOffset,value);else outRankZ.Store(byteOffset,value);}
groupshared uint bucketBase[16];
groupshared uint carriedCount[16];
groupshared uint claimedWaves;
groupshared uint waveCounts[256];
groupshared uint waveBases[256];
groupshared uint chunkPackedTotal;

[numthreads(1024, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint block = group.x;
    const uint axis = group.y;
    const uint inputKeys=axis==0u?p.ikx:(axis==1u?p.iky:p.ikz);
    const uint inputRanks=axis==0u?p.irx:(axis==1u?p.iry:p.irz);
    const uint outputKeys=axis==0u?p.okx:(axis==1u?p.oky:p.okz);
    const uint outputRanks=axis==0u?p.orx:(axis==1u?p.ory:p.orz);
    const uint radixCounts=axis==0u?p.cx:(axis==1u?p.cy:p.cz);

    const uint numVectors = (p.numKeys + 3u) / 4u;
    const uint requiredBlocks = (numVectors + 1023u) / 1024u;
    const uint iterationsPerBlock = (requiredBlocks + 31u) / 32u;
    if (iterationsPerBlock * block * 1024u >= numVectors) return;

    if (thread.x < 16u) {
        uint start = 0u;
        for (uint digit = 0u; digit < thread.x; ++digit)
            for (uint b = 0u; b < 32u; ++b)
                start += load_count(axis,radixCounts + (digit * 32u + b) * 4u);
        for (uint b = 0u; b < block; ++b)
            start += load_count(axis,radixCounts + (thread.x * 32u + b) * 4u);
        bucketBase[thread.x] = start;
        carriedCount[thread.x] = 0u;
    }
    if (thread.x == 0u) claimedWaves = 0u;
    GroupMemoryBarrierWithGroupSync();

    const uint laneCount = WaveGetLaneCount();
    const uint laneIndex = WaveGetLaneIndex();
    uint waveId = 0u;
    if (WaveIsFirstLane()) InterlockedAdd(claimedWaves, 1u, waveId);
    waveId = WaveReadLaneFirst(waveId);
    const uint waveCount = 1024u / laneCount;
    const uint logicalThread = waveId * laneCount + laneIndex;

    for (uint iter = 0u; iter < iterationsPerBlock; ++iter) {
        const uint vectorIndex =
            iterationsPerBlock * block * 1024u + iter * 1024u + logicalThread;
        const bool hasVector = vectorIndex < numVectors;
        uint4 keys = uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
        uint4 ranks = uint4(0u, 0u, 0u, 0u);
        if (hasVector) {
            [unroll]
            for (uint lane = 0u; lane < 4u; ++lane) {
                const uint scalar = vectorIndex * 4u + lane;
                if (scalar < p.numKeys) {
                    keys[lane] = load_key(axis,inputKeys + scalar * 4u);
                    ranks[lane] = load_rank(axis,inputRanks + scalar * 4u);
                }
            }
        }
        const uint4 digits = (keys >> p.startBit) & 15u;

        [loop]
        for (uint firstDigit = 0u; firstDigit < 16u; firstDigit += 2u) {
            uint firstCount = 0u, secondCount = 0u;
            if (hasVector) {
                [unroll]
                for (uint lane = 0u; lane < 4u; ++lane) {
                    firstCount += digits[lane] == firstDigit ? 1u : 0u;
                    secondCount += digits[lane] == firstDigit + 1u ? 1u : 0u;
                }
            }
            const uint packed = firstCount | (secondCount << 16u);
            const uint wavePrefix = WavePrefixSum(packed);
            const uint waveTotal = WaveActiveSum(packed);
            if (WaveIsFirstLane()) waveCounts[waveId] = waveTotal;
            GroupMemoryBarrierWithGroupSync();

            if (thread.x == 0u) {
                uint total = 0u;
                for (uint wave = 0u; wave < waveCount; ++wave) {
                    const uint count = waveCounts[wave];
                    waveBases[wave] = total;
                    total += count;
                }
                chunkPackedTotal = total;
            }
            GroupMemoryBarrierWithGroupSync();

            const uint prefix = waveBases[waveId] + wavePrefix;
            const uint firstBase =
                bucketBase[firstDigit] + carriedCount[firstDigit] + (prefix & 0xffffu);
            const uint secondBase =
                bucketBase[firstDigit + 1u] + carriedCount[firstDigit + 1u] +
                (prefix >> 16u);
            uint precedingFirst = 0u, precedingSecond = 0u;
            if (hasVector) {
                [unroll]
                for (uint lane = 0u; lane < 4u; ++lane) {
                    const uint scalar = vectorIndex * 4u + lane;
                    if (digits[lane] == firstDigit) {
                        if (scalar < p.numKeys) {
                            const uint destination = firstBase + precedingFirst;
                            store_key(axis,outputKeys + destination * 4u, keys[lane]);
                            store_rank(axis,outputRanks + destination * 4u, ranks[lane]);
                        }
                        ++precedingFirst;
                    } else if (digits[lane] == firstDigit + 1u) {
                        if (scalar < p.numKeys) {
                            const uint destination = secondBase + precedingSecond;
                            store_key(axis,outputKeys + destination * 4u, keys[lane]);
                            store_rank(axis,outputRanks + destination * 4u, ranks[lane]);
                        }
                        ++precedingSecond;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            if (thread.x == 0u) {
                carriedCount[firstDigit] += chunkPackedTotal & 0xffffu;
                carriedCount[firstDigit + 1u] += chunkPackedTotal >> 16u;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
