// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Accurate island admission consumes Sc's existing EdgeIndex touch state.
// Dynamic endpoints are dense eligible-node ordinals; a nonresponding
// static/kinematic endpoint is INVALID and never merges two islands.
// The output stays in original Sc edge order for Dy descriptor lowering.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer scEdges:register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer edgeHistory:register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer work:register(u2,space0);

struct Parameters {
    uint edgeBias,touchBias,workBias,rootBias;
    uint countBias,admissionBias,nodeCount,edgeCount;
    uint edgeCapacity,phase,first,unused;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p:register(b0,space0);
#endif

static const uint invalid=0xffffffffu;
static const uint contactEdge=0u;
static const uint jointEdge=1u;

void reject(uint bits) {
    uint previous;
    work.InterlockedOr(p.workBias,bits,previous);
}
bool validNode(uint node) {
    if(node==invalid)return true;
    if(node<p.nodeCount)return true;
    reject(2u);
    return false;
}
bool edgeConnected(uint edge,uint kind) {
    if(kind==jointEdge)return true;
    if(kind!=contactEdge || edge>=p.edgeCapacity) {
        reject(1u);
        return false;
    }
    return (edgeHistory.Load(p.touchBias+edge*240u+72u)&1u)!=0u;
}

// Every successful union links a current root to a strictly smaller
// root. Concurrent unions cannot form cycles. Halving preserves the
// monotone parent invariant while other workgroups attach roots.
uint findRoot(uint node) {
    for(uint step=0u;step<p.nodeCount;++step) {
        const uint parent=work.Load(p.rootBias+node*4u);
        if(parent==node)return node;
        if(parent>=node) {
            reject(4u);
            return invalid;
        }
        const uint grand=work.Load(p.rootBias+parent*4u);
        if(grand>parent) {
            reject(4u);
            return invalid;
        }
        if(grand!=parent) {
            uint previous;
            work.InterlockedMin(p.rootBias+node*4u,grand,previous);
        }
        node=grand;
    }
    reject(4u);
    return invalid;
}
void connectNodes(uint a,uint b) {
    if(a==b)return;
    for(uint attempt=0u;attempt<p.nodeCount;++attempt) {
        const uint rootA=findRoot(a);
        const uint rootB=findRoot(b);
        if(rootA==invalid || rootB==invalid || rootA==rootB)return;
        const uint high=max(rootA,rootB);
        const uint low=min(rootA,rootB);
        uint prior;
        work.InterlockedCompareExchange(p.rootBias+high*4u,
                                        high,low,prior);
        if(prior==high)return;
    }
    reject(8u);
}

[numthreads(128,1,1)]
void main(uint3 thread:SV_DispatchThreadID) {
    const uint index=p.first+thread.x;
    if(p.phase==0u) {
        if(index==0u)work.Store(p.workBias,0u);
        if(index<p.nodeCount) {
            work.Store(p.rootBias+index*4u,index);
            work.Store2(p.countBias+index*8u,uint2(0u,0u));
        }
        return;
    }
    if(p.phase==1u) {
        if(index>=p.edgeCount)return;
        const uint input=p.edgeBias+index*20u;
        const uint4 edge=scEdges.Load4(input);
        const uint kind=scEdges.Load(input+16u);
        if(!validNode(edge.y) || !validNode(edge.z))return;
        if(!edgeConnected(edge.x,kind))return;
        if(edge.y!=invalid && edge.z!=invalid)
            connectNodes(edge.y,edge.z);
        return;
    }
    if(p.phase==2u) {
        if(index<p.nodeCount) {
            const uint root=findRoot(index);
            if(root!=invalid) {
                work.Store(p.rootBias+index*4u,root);
                uint previous;
                work.InterlockedAdd(p.countBias+root*8u,1u,previous);
            }
        }
        if(index>=p.edgeCount)return;
        const uint input=p.edgeBias+index*20u;
        const uint4 edge=scEdges.Load4(input);
        const uint kind=scEdges.Load(input+16u);
        uint owner=invalid;
        if(validNode(edge.y) && validNode(edge.z) &&
           edgeConnected(edge.x,kind)) {
            const uint node=edge.y!=invalid?edge.y:edge.z;
            if(node!=invalid) {
                owner=findRoot(node);
                if(owner!=invalid) {
                    uint previous;
                    work.InterlockedAdd(p.countBias+owner*8u+4u,
                                        1u,previous);
                }
            }
        }
        // Source contact-manager/joint order is unchanged. Invalid
        // owner means no admitted response for this same-frame edge.
        work.Store4(p.admissionBias+index*24u,
                    uint4(edge.w,edge.x,edge.y,edge.z));
        work.Store2(p.admissionBias+index*24u+16u,
                    uint2(owner,kind));
    }
}
