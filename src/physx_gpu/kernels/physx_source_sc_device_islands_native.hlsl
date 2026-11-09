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
RWByteAddressBuffer nodeFlags:register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer work:register(u3,space0);

struct Parameters {
    uint edgeBias,touchBias,nodeBias,workBias;
    uint rootBias,countBias,admissionBias,taskBias;
    uint blockBias,nodeCount,edgeCount,edgeCapacity;
    uint phase,first,maxGroups,blockCount;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p:register(b0,space0);
#endif

static const uint invalid=0xffffffffu;
static const uint contactEdge=0u;
static const uint jointEdge=1u;
static const uint noResponse=2u;
static const uint hostTouch=4u;
static const uint hostFound=8u;
static const uint permittedKinds=jointEdge|noResponse|hostTouch|hostFound;

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
    if(edge==invalid) {
        reject(1u);
        return false;
    }
    if((kind&~permittedKinds)!=0u ||
       ((kind&hostFound)!=0u && (kind&hostTouch)==0u) ||
       ((kind&jointEdge)!=0u && (kind&(hostTouch|hostFound))!=0u)) {
        reject(1u);
        return false;
    }
    if((kind&jointEdge)!=0u)return true;
    // Only CPU Gu may author this bit; it must already have published
    // physical touch. Device PCM contacts never bypass EdgeIndex history.
    if((kind&hostTouch)!=0u)return true;
    if(edge>=p.edgeCapacity) {
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

// Root-ordered task admission has deterministic numerics even when
// independent source islands are processed by different workgroups.
// One group scans each 128-root tile; one group prefixes their totals.
// This is linear work, with no host count fetch or global spin barrier.
groupshared uint taskScan[128],responseScan[128];
groupshared uint taskCarry,responseCarry;

bool runnableRoot(uint index) {
    if(index>=p.nodeCount || work.Load(p.workBias)!=0u ||
       work.Load(p.rootBias+index*4u)!=index)return false;
    const uint count=p.countBias+index*24u;
    return work.Load(count)!=0u && work.Load(count+8u)!=0u;
}
void inclusiveTaskScan(uint lane) {
    [unroll]
    for(uint step=1u;step<128u;step<<=1u) {
        const uint previous=lane>=step?taskScan[lane-step]:0u;
        const uint edges=lane>=step?responseScan[lane-step]:0u;
        GroupMemoryBarrierWithGroupSync();
        taskScan[lane]+=previous;
        responseScan[lane]+=edges;
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(128,1,1)]
void main(uint3 thread:SV_DispatchThreadID,
          uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint index=p.first+thread.x;
    if(p.phase==0u) {
        if(index==0u) {
            work.Store4(p.workBias,uint4(0u,0u,0u,0u));
            work.Store4(p.workBias+16u,uint4(0u,1u,1u,0u));
        }
        if(index<p.nodeCount) {
            work.Store(p.rootBias+index*4u,index);
            const uint count=p.countBias+index*24u;
            work.Store4(count,uint4(0u,0u,0u,0u));
            work.Store2(count+16u,uint2(0u,invalid));
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
                const uint flags=nodeFlags.Load(p.nodeBias+index*4u);
                const uint position=flags&255u;
                const uint velocity=(flags>>8u)&255u;
                if(!position || !velocity ||
                   (flags&0xfffe0000u)!=0u)reject(16u);
                const uint count=p.countBias+root*24u;
                uint previous;
                work.InterlockedAdd(count,1u,previous);
                if((flags&0x10000u)!=0u)
                    work.InterlockedAdd(count+8u,1u,previous);
                work.InterlockedMax(count+12u,position,previous);
                work.InterlockedMax(count+16u,velocity,previous);
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
                    if((kind&noResponse)==0u)
                        work.InterlockedAdd(p.countBias+owner*24u+4u,
                                            1u,previous);
                    // A contact that became physically touching in
                    // this PCM epoch activates its accurate component.
                    // Joint edges and persistent contacts do not
                    // manufacture an additional wake reason.
                    if((kind&hostFound)!=0u) {
                        work.InterlockedAdd(p.countBias+owner*24u+8u,
                                            1u,previous);
                    }else if((kind&(jointEdge|hostTouch))==0u) {
                        const uint cache=p.touchBias+edge.x*240u;
                        const uint prior=edgeHistory.Load(cache+76u)&1u;
                        if(prior==0u)
                            work.InterlockedAdd(p.countBias+owner*24u+8u,
                                                1u,previous);
                    }
                }
            }
        }
        // Source contact-manager/joint order is unchanged. Invalid
        // owner means no admitted response for this same-frame edge.
        work.Store4(p.admissionBias+index*24u,
                    uint4(edge.w,edge.x,edge.y,edge.z));
        work.Store2(p.admissionBias+index*24u+16u,
                    uint2(owner,kind));
        return;
    }
    if(p.phase==3u) {
        const uint active=runnableRoot(index)?1u:0u;
        taskScan[lane.x]=active;
        responseScan[lane.x]=active!=0u
            ?work.Load(p.countBias+index*24u+4u):0u;
        GroupMemoryBarrierWithGroupSync();
        for(uint stride=64u;stride!=0u;stride>>=1u) {
            if(lane.x<stride) {
                taskScan[lane.x]+=taskScan[lane.x+stride];
                responseScan[lane.x]+=responseScan[lane.x+stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if(lane.x==0u && p.blockCount!=0u)
            work.Store2(p.blockBias+(index/128u)*8u,
                        uint2(taskScan[0],responseScan[0]));
        return;
    }
    if(p.phase==4u) {
        if(group.x!=0u)return;
        if(lane.x==0u) {
            taskCarry=0u;
            responseCarry=0u;
        }
        GroupMemoryBarrierWithGroupSync();
        for(uint start=0u;start<p.blockCount;start+=128u) {
            const uint block=start+lane.x;
            const uint2 value=block<p.blockCount
                ?work.Load2(p.blockBias+block*8u):uint2(0u,0u);
            taskScan[lane.x]=value.x;
            responseScan[lane.x]=value.y;
            GroupMemoryBarrierWithGroupSync();
            inclusiveTaskScan(lane.x);
            const uint2 prefix=uint2(
                taskCarry+taskScan[lane.x]-value.x,
                responseCarry+responseScan[lane.x]-value.y);
            if(block<p.blockCount)
                work.Store2(p.blockBias+block*8u,prefix);
            GroupMemoryBarrierWithGroupSync();
            if(lane.x==0u) {
                taskCarry+=taskScan[127u];
                responseCarry+=responseScan[127u];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if(lane.x==0u) {
            work.Store(p.workBias+4u,taskCarry);
            work.Store(p.workBias+12u,responseCarry);
            work.Store(p.workBias+16u,min(taskCarry,p.maxGroups));
        }
        return;
    }
    if(p.phase==5u) {
        const uint active=runnableRoot(index)?1u:0u;
        const uint count=p.countBias+index*24u;
        const uint edges=active!=0u?work.Load(count+4u):0u;
        taskScan[lane.x]=active;
        responseScan[lane.x]=edges;
        GroupMemoryBarrierWithGroupSync();
        inclusiveTaskScan(lane.x);
        if(active==0u)return;
        const uint2 base=work.Load2(p.blockBias+(index/128u)*8u);
        const uint task=base.x+taskScan[lane.x]-1u;
        const uint response=base.y+responseScan[lane.x]-edges;
        const uint nodes=work.Load(count);
        const uint position=work.Load(count+12u);
        const uint velocity=work.Load(count+16u);
        work.Store(count+20u,task);
        const uint dst=p.taskBias+task*24u;
        work.Store4(dst,uint4(index,nodes,edges,position));
        work.Store2(dst+16u,uint2(velocity,response));
        uint previous;
        work.InterlockedAdd(p.workBias+8u,nodes,previous);
    }
}
