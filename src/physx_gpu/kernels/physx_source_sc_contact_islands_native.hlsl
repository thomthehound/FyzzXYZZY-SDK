// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Sc accurate rigid connectivity is evaluated from the physical-touch
// stream that GuPCM publishes into the owning EdgeIndex contact manager.
// Dynamic rigid nodes connect; static and kinematic endpoints do not
// merge distinct dynamic islands.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer graph : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer edgeCaches : register(u1,space0);

struct Parameters {
    uint graphBias,contactBias,phase,unused;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p:register(b0,space0);
#endif

// Header: edgeCount, rigidWatermark, pairOffset, rootOffset, errorOffset,
 // contactEdgeCapacity, componentCountsOffset, componentHeadsOffset,
 // linksOffset, edgeRootOffset, graphEdgeCapacity.
uint header(uint word) {return graph.Load(p.graphBias+word*4u);}
uint parentAddress(uint node) {
    return p.graphBias+header(3u)+node*4u;
}
uint componentCountAddress(uint root) {
    return p.graphBias+header(6u)+root*4u;
}
uint componentHeadAddress(uint root) {
    return p.graphBias+header(7u)+root*4u;
}
uint componentNextAddress(uint edgeOrdinal) {
    return p.graphBias+header(8u)+edgeOrdinal*4u;
}
uint edgeRootAddress(uint edge) {
    return p.graphBias+header(9u)+edge*4u;
}
uint rootOf(uint node,uint size) {
    uint cursor=node;
    for(uint hop=0u;hop<size;++hop) {
        const uint parent=graph.Load(parentAddress(cursor));
        if(parent==cursor)return cursor;
        if(parent>cursor || parent>=size)break;
        const uint grand=graph.Load(parentAddress(parent));
        if(grand>parent || grand>=size)break;
        // Every hook points toward a smaller BodyId. Atomic pointer
        // halving preserves that order across concurrent union attempts
        // and avoids walking long serialized contact chains repeatedly.
        uint prior;
        graph.InterlockedMin(parentAddress(cursor),grand,prior);
        cursor=grand;
    }
    return 0xffffffffu;
}
void errorBit(uint bit) {
    uint ignored;
    graph.InterlockedOr(p.graphBias+header(4u),bit,ignored);
}
// Contact and constraint EdgeIndices share one Sc graph, but only
// contacts read physical-touch state from the GuPCM edge cache.
bool connected(uint4 edge) {
    if((edge.z&4u)!=0u)return true;
    if((edge.z&8u)!=0u)return false;
    if(edge.w>=header(5u)) {
        errorBit(2u);
        return false;
    }
    return (edgeCaches.Load(p.contactBias+edge.w*240u+72u)&1u)!=0u;
}

[numthreads(128,1,1)]
void main(uint3 dispatch:SV_DispatchThreadID) {
    const uint i=dispatch.x;
    const uint edgeCount=header(0u);
    const uint rigidCount=header(1u);
    const uint pairBase=p.graphBias+header(2u);
    if(p.phase==0u) {
        if(i==0u)graph.Store(p.graphBias+header(4u),0u);
        if(i<rigidCount) {
            graph.Store(parentAddress(i),i);
            graph.Store(componentCountAddress(i),0u);
            graph.Store(componentHeadAddress(i),0xffffffffu);
        }
        if(i<header(10u))
            graph.Store(edgeRootAddress(i),0xffffffffu);
        return;
    }
    if(p.phase==1u) {
        if(i>=edgeCount)return;
        const uint4 pair=graph.Load4(pairBase+i*16u);
        // The first two fields are Sc BodyIds. Bit 0/1 mark dynamic
        // rigid endpoints; every pair retains its source EdgeIndex.
        if(pair.x>=rigidCount || pair.y>=rigidCount) {
            errorBit(1u);
            return;
        }
        if(!connected(pair) || pair.x==pair.y ||
           (pair.z&3u)!=3u)return;
        // Root hooking always points larger indices toward smaller roots.
        // Concurrent hooks remain acyclic and a completed successful
        // pass resolves every edge, including chains across workgroups.
        for(uint attempt=0u;attempt<rigidCount;++attempt) {
            const uint a=rootOf(pair.x,rigidCount);
            const uint b=rootOf(pair.y,rigidCount);
            if(a==0xffffffffu || b==0xffffffffu) {
                errorBit(8u);
                return;
            }
            if(a==b)return;
            const uint high=max(a,b),low=min(a,b);
            uint previous;
            graph.InterlockedCompareExchange(parentAddress(high),
                                             high,low,previous);
            if(previous==high)return;
        }
        errorBit(16u);
        return;
    }
    if(p.phase==2u && i<rigidCount) {
        const uint root=rootOf(i,rigidCount);
        if(root==0xffffffffu) {
            errorBit(32u);
            return;
        }
        graph.Store(parentAddress(i),root);
        return;
    }
    if(p.phase==3u && i<edgeCount) {
        const uint4 edge=graph.Load4(pairBase+i*16u);
        if(edge.x>=rigidCount || edge.y>=rigidCount ||
           (edge.z&3u)==0u || !connected(edge))
            return;
        // The same source contact or joint EdgeIndex contributes to
        // exactly one accurate dynamic island. Static and kinematic
        // endpoints are terminals, not members of another island.
        const uint node=(edge.z&1u)!=0u?edge.x:edge.y;
        const uint root=graph.Load(parentAddress(node));
        if(root>=rigidCount ||
           ((edge.z&3u)==3u &&
            root!=graph.Load(parentAddress(edge.y)))) {
            errorBit(64u);
            return;
        }
        if(edge.w>=header(10u)) {
            errorBit(128u);
            return;
        }
        graph.Store(edgeRootAddress(edge.w),root);
        uint previous;
        graph.InterlockedExchange(componentHeadAddress(root),i,previous);
        graph.Store(componentNextAddress(i),previous);
        graph.InterlockedAdd(componentCountAddress(root),1u,previous);
    }
}
