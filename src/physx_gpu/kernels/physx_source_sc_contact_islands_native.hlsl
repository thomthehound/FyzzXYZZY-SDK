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

// The original accurate graph and its speculative counterpart share
// one Sc allocation and EdgeIndex input stream. Only accurate contact
// connectivity consults Gu's physical-touch bit.
uint header(uint word) {return graph.Load(p.graphBias+word*4u);}
uint parentAddress(uint node,bool speculative) {
    return p.graphBias+header(speculative?11u:3u)+node*4u;
}
uint componentCountAddress(uint root,bool speculative) {
    return p.graphBias+header(speculative?12u:6u)+root*4u;
}
uint componentHeadAddress(uint root,bool speculative) {
    return p.graphBias+header(speculative?13u:7u)+root*4u;
}
uint componentNextAddress(uint ordinal,bool speculative) {
    return p.graphBias+header(speculative?14u:8u)+ordinal*4u;
}
uint edgeRootAddress(uint edge,bool speculative) {
    return p.graphBias+header(speculative?15u:9u)+edge*4u;
}
uint rootOf(uint node,uint size,bool speculative) {
    uint cursor=node;
    for(uint hop=0u;hop<size;++hop) {
        const uint parent=graph.Load(parentAddress(cursor,speculative));
        if(parent==cursor)return cursor;
        if(parent>cursor || parent>=size)break;
        const uint grand=graph.Load(parentAddress(parent,speculative));
        if(grand>parent || grand>=size)break;
        uint prior;
        graph.InterlockedMin(parentAddress(cursor,speculative),grand,prior);
        cursor=grand;
    }
    return 0xffffffffu;
}
void errorBit(uint bit) {
    uint ignored;
    graph.InterlockedOr(p.graphBias+header(4u),bit,ignored);
}
bool physicallyConnected(uint4 edge) {
    if((edge.z&4u)!=0u)return true;
    if((edge.z&8u)!=0u)return false;
    if(edge.w>=header(5u)) {
        errorBit(2u);
        return false;
    }
    return (edgeCaches.Load(p.contactBias+edge.w*240u+72u)&1u)!=0u;
}
void unite(uint a,uint b,uint rigidCount,bool speculative) {
    for(uint attempt=0u;attempt<rigidCount;++attempt) {
        const uint rootA=rootOf(a,rigidCount,speculative);
        const uint rootB=rootOf(b,rigidCount,speculative);
        if(rootA==0xffffffffu || rootB==0xffffffffu) {
            errorBit(speculative?256u:8u);
            return;
        }
        if(rootA==rootB)return;
        const uint high=max(rootA,rootB),low=min(rootA,rootB);
        uint previous;
        graph.InterlockedCompareExchange(parentAddress(high,speculative),
                                         high,low,previous);
        if(previous==high)return;
    }
    errorBit(speculative?512u:16u);
}
void publish(uint ordinal,uint4 edge,uint node,uint rigidCount,
             bool speculative) {
    const uint root=graph.Load(parentAddress(node,speculative));
    if(root>=rigidCount ||
       ((edge.z&3u)==3u &&
        root!=graph.Load(parentAddress(edge.y,speculative)))) {
        errorBit(speculative?1024u:64u);
        return;
    }
    graph.Store(edgeRootAddress(edge.w,speculative),root);
    uint previous;
    graph.InterlockedExchange(componentHeadAddress(root,speculative),
                              ordinal,previous);
    graph.Store(componentNextAddress(ordinal,speculative),previous);
    graph.InterlockedAdd(componentCountAddress(root,speculative),1u,previous);
}
[numthreads(128,1,1)]
void main(uint3 dispatch:SV_DispatchThreadID) {
    const uint i=p.unused+dispatch.x;
    const uint edgeCount=header(0u);
    const uint rigidCount=header(1u);
    const uint pairBase=p.graphBias+header(2u);
    if(p.phase==0u) {
        if(i==0u)graph.Store(p.graphBias+header(4u),0u);
        if(i<rigidCount) {
            for(uint mode=0u;mode<2u;++mode) {
                const bool speculative=mode!=0u;
                graph.Store(parentAddress(i,speculative),i);
                graph.Store(componentCountAddress(i,speculative),0u);
                graph.Store(componentHeadAddress(i,speculative),0xffffffffu);
            }
        }
        if(i<header(10u)) {
            graph.Store(edgeRootAddress(i,false),0xffffffffu);
            graph.Store(edgeRootAddress(i,true),0xffffffffu);
        }
        return;
    }
    if(p.phase==1u) {
        if(i>=edgeCount)return;
        const uint4 edge=graph.Load4(pairBase+i*16u);
        if(edge.x>=rigidCount || edge.y>=rigidCount) {
            errorBit(1u);
            return;
        }
        if(edge.w>=header(10u)) {
            errorBit(128u);
            return;
        }
        const bool accurate=physicallyConnected(edge);
        if(edge.x==edge.y || (edge.z&3u)!=3u)return;
        // Source speculative connectivity follows contact admission;
        // accurate connectivity follows physical touch and live joints.
        unite(edge.x,edge.y,rigidCount,true);
        if(accurate)unite(edge.x,edge.y,rigidCount,false);
        return;
    }
    if(p.phase==2u && i<rigidCount) {
        for(uint mode=0u;mode<2u;++mode) {
            const bool speculative=mode!=0u;
            const uint root=rootOf(i,rigidCount,speculative);
            if(root==0xffffffffu) {
                errorBit(speculative?2048u:32u);
                continue;
            }
            graph.Store(parentAddress(i,speculative),root);
        }
        return;
    }
    if(p.phase==3u && i<edgeCount) {
        const uint4 edge=graph.Load4(pairBase+i*16u);
        if(edge.x>=rigidCount || edge.y>=rigidCount ||
           (edge.z&3u)==0u)return;
        if(edge.w>=header(10u)) {
            errorBit(128u);
            return;
        }
        const uint node=(edge.z&1u)!=0u?edge.x:edge.y;
        publish(i,edge,node,rigidCount,true);
        if(physicallyConnected(edge))publish(i,edge,node,rigidCount,false);
    }
}
