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

// Header: pairCount, rigidWatermark, pairOffset, rootOffset, errorOffset.
uint header(uint word) {return graph.Load(p.graphBias+word*4u);}
uint parentAddress(uint node) {
    return p.graphBias+header(3u)+node*4u;
}
uint rootOf(uint node,uint size) {
    uint cursor=node;
    for(uint hop=0u;hop<size;++hop) {
        const uint next=graph.Load(parentAddress(cursor));
        if(next==cursor)return cursor;
        if(next>cursor || next>=size)break;
        cursor=next;
    }
    return 0xffffffffu;
}
void errorBit(uint bit) {
    uint ignored;
    graph.InterlockedOr(p.graphBias+header(4u),bit,ignored);
}

[numthreads(128,1,1)]
void main(uint3 dispatch:SV_DispatchThreadID) {
    const uint i=dispatch.x;
    const uint edgeCount=header(0u);
    const uint rigidCount=header(1u);
    const uint pairBase=p.graphBias+header(2u);
    if(p.phase==0u) {
        if(i==0u)graph.Store(p.graphBias+header(4u),0u);
        if(i<rigidCount)graph.Store(parentAddress(i),i);
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
        // Authored joints remain connected until their Sc edge retires.
        // Dormant Gu managers retain the device's last physical touch.
        bool touching=(pair.z&4u)!=0u;
        [branch] if(!touching) {
            if(pair.w>=header(5u)) {
                errorBit(2u);
                return;
            }
            touching=(edgeCaches.Load(p.contactBias+pair.w*240u+72u)&1u)!=0u;
        }
        if(!touching || pair.x==pair.y || (pair.z&3u)!=3u)return;
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
    }
}
