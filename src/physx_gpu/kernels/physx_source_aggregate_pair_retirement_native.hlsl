// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// aggregate.cu::removeAggPairsStage1/2/3. Original persistent-pair
// removal bitmap, histogram, dead marker and free-index stack.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer aggregatePairs : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer removeBitmap : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer removeHistogram : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer freeIndices : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer freeList : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer pairCount : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer removedCount : register(u6,space0);

struct Parameters {
    uint capacity, gridCount, pairBias, bitmapBias;
    uint histogramBias, indicesBias, listBias, countBias;
    uint removedBias, reserved0, reserved1, reserved2;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

groupshared uint partial[256];

[numthreads(256,1,1)]
void histogram(uint3 block:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint count=min(pairCount.Load(p.countBias),p.capacity);
    const uint blocks=(count+255u)/256u;
    for(uint chunk=block.x;chunk<blocks;chunk+=p.gridCount) {
        const uint index=chunk*256u+lane.x;
        partial[lane.x]=index<count &&
            removeBitmap.Load(p.bitmapBias+index*4u)!=0u?1u:0u;
        GroupMemoryBarrierWithGroupSync();
        for(uint step=128u;step!=0u;step>>=1u) {
            if(lane.x<step)partial[lane.x]+=partial[lane.x+step];
            GroupMemoryBarrierWithGroupSync();
        }
        if(lane.x==0u)
            removeHistogram.Store(p.histogramBias+chunk*4u,partial[0]);
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(256,1,1)]
void mark(uint3 block:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint count=min(pairCount.Load(p.countBias),p.capacity);
    const uint blocks=(count+255u)/256u;
    const uint numFree=freeList.Load(p.listBias);
    // Original stage 2 publishes nbRemoved for stage 3; every
    // aggregate removal and free-index append remains on the GPU.
    if(block.x==0u && lane.x==0u) {
        uint total=0u;
        for(uint i=0u;i<blocks;++i)
            total+=removeHistogram.Load(p.histogramBias+i*4u);
        removedCount.Store(p.removedBias,total);
    }
    for(uint chunk=block.x;chunk<blocks;chunk+=p.gridCount) {
        // Original histogram runsum: each workgroup reduces all earlier
        // block counts to its global exclusive prefix. This preserves
        // freeIndices order independently of native subgroup width.
        uint before=0u;
        for(uint i=lane.x;i<chunk;i+=256u)
            before+=removeHistogram.Load(p.histogramBias+i*4u);
        partial[lane.x]=before;
        GroupMemoryBarrierWithGroupSync();
        for(uint step=1u;step<256u;step<<=1u) {
            const uint v=partial[lane.x];
            const uint prev=lane.x>=step?partial[lane.x-step]:0u;
            GroupMemoryBarrierWithGroupSync();
            partial[lane.x]=v+prev;
            GroupMemoryBarrierWithGroupSync();
        }
        const uint globalBase=partial[255u];
        const uint index=chunk*256u+lane.x;
        const uint flagged=index<count &&
            removeBitmap.Load(p.bitmapBias+index*4u)!=0u?1u:0u;
        partial[lane.x]=flagged;
        GroupMemoryBarrierWithGroupSync();
        for(uint step=1u;step<256u;step<<=1u) {
            const uint v=partial[lane.x];
            const uint prev=lane.x>=step?partial[lane.x-step]:0u;
            GroupMemoryBarrierWithGroupSync();
            partial[lane.x]=v+prev;
            GroupMemoryBarrierWithGroupSync();
        }
        if(flagged!=0u) {
            const uint position=numFree+globalBase+
                (partial[lane.x]-1u);
            if(position<p.capacity) {
                // PxgAggregatePair bitfield: isNew=bit0, isDead=bit1.
                const uint word=p.pairBias+index*12u+8u;
                aggregatePairs.Store(word,aggregatePairs.Load(word)|2u);
                freeIndices.Store(p.indicesBias+position*4u,index);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(1,1,1)]
void publish(uint3 block:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint oldCount=freeList.Load(p.listBias);
    const uint removed=removedCount.Load(p.removedBias);
    // The source capacity contract ensures all removed slots fit the
    // original free stack. An invalid count stays visible at fetch.
    const uint newCount=oldCount+removed;
    freeList.Store(p.listBias,
        newCount<oldCount?0xffffffffu:newCount);
}
