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
    uint responseBiasA,responseBiasB,histBias,primaryCapacity;
    uint tokenBias,radixPasses,bodyOrderBias,combinedCount;
    uint responseOrderBias,jointRowCount,fullDescriptorCount,blockTilesBias;
    uint extentBias,maxGroupsY,contactAdmissionBias,contactPrefixBias;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p:register(b0,space0);
#endif

static const uint invalid=0xffffffffu;
static const uint contactEdge=0u;
static const uint numericalExtentValid=0x80000000u;
static const uint jointEdge=1u;
static const uint noResponse=2u;
static const uint deviceJoint=16u;
static const uint nonrespondingNode=0x20000u;
static const uint requireContactTask=0x40000u;
static const uint permittedKinds=jointEdge|noResponse|deviceJoint;

void reject(uint bits) {
    uint previous;
    work.InterlockedOr(p.workBias,bits,previous);
}
void publishDispatch(uint groups) {
    const uint y=groups==0u?1u:1u+(groups-1u)/p.maxGroups;
    if(y>p.maxGroupsY) {
        reject(128u);
        work.Store2(p.workBias+16u,uint2(0u,1u));
        return;
    }
    const uint x=groups==0u?0u:1u+(groups-1u)/y;
    work.Store2(p.workBias+16u,uint2(x,y));
}
bool validNode(uint node) {
    if(node==invalid)return true;
    if(node<p.nodeCount)return true;
    reject(2u);
    return false;
}
bool validSource(uint source,uint kind) {
    // Nonresponding edges still carry accurate connectivity but do not
    // address any numerical table. All admitted numerical response
    // tokens must resolve within their typed Dy input table.
    if((kind&noResponse)!=0u)return true;
    const uint limit=(kind&jointEdge)==0u?p.primaryCapacity:
        (kind&deviceJoint)!=0u?p.jointRowCount:p.fullDescriptorCount;
    if(source>=limit || source>=0x40000000u) {
        reject(64u);
        return false;
    }
    return true;
}
bool edgeConnected(uint edge,uint kind) {
    if(edge==invalid) {
        reject(1u);
        return false;
    }
    if((kind&~permittedKinds)!=0u ||
       ((kind&deviceJoint)!=0u && (kind&jointEdge)==0u)) {
        reject(1u);
        return false;
    }
    if((kind&jointEdge)!=0u)return true;
    // Both host Gu and device PCM publish physical touch to the same
    // EdgeIndex history before accurate admission. No source flag
    // substitutes for the committed/current resident touch state.
    if(edge>=p.edgeCapacity) {
        reject(1u);
        return false;
    }
    const uint touch=edgeHistory.Load(p.touchBias+edge*240u+72u);
    if((touch&0x80000000u)!=0u) {
        reject(256u);
        return false;
    }
    return (touch&1u)!=0u;
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
bool respondingNode(uint node) {
    return node!=invalid && node<p.nodeCount &&
        (nodeFlags.Load(p.nodeBias+node*4u)&nonrespondingNode)==0u;
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

// The same prefix owns runnable task, response and body ranges.
groupshared uint taskScan[128],responseScan[128],bodyScan[128];
groupshared uint taskCarry,responseCarry,bodyCarry;
groupshared uint radixBuckets[256],radixOffsets[256],radixTotals[256];
groupshared uint radixDigits[128],radixRanks[128];
groupshared uint compactContactScan[256];
groupshared uint contactGroupCarry;
groupshared uint compactBodyKeys[256],compactBodyStarts[256];
groupshared uint compactEdgeKeys[256],compactEdgeStarts[256];

uint responseRoot(uint edgeIndex) {
    if(work.Load(p.workBias)!=0u)return p.nodeCount;
    const uint address=p.admissionBias+edgeIndex*24u;
    const uint root=work.Load(address+16u);
    if(root==invalid || root>=p.nodeCount ||
       (work.Load(address+20u)&noResponse)!=0u)
        return p.nodeCount;
    return work.Load(p.countBias+root*24u+20u)!=invalid?root:p.nodeCount;
}
uint responseToken(uint edgeIndex) {
    const uint address=p.admissionBias+edgeIndex*24u;
    const uint source=work.Load(address);
    if(source>=0x40000000u) {
        reject(64u);
        work.Store(p.workBias+16u,0u);
        return invalid;
    }
    const uint kind=work.Load(address+20u);
    // Dy token kinds: 0=full 1D descriptor, 2=device joint row,
    // 3=contact manager. Every source index has a typed table owner.
    const uint tag=(kind&jointEdge)==0u?0xc0000000u:
        ((kind&deviceJoint)!=0u?0x80000000u:0u);
    return tag|source;
}
// Admitted source responses retain the device-published numerical extent.
uint responseExtent(uint edgeIndex) {
    const uint address=p.admissionBias+edgeIndex*24u;
    if((work.Load(address+20u)&jointEdge)!=0u)return 0u;
    const uint edge=work.Load(address+4u);
    if(edge>=p.edgeCapacity) {
        reject(1u);
        return 0u;
    }
    const uint touch=edgeHistory.Load(p.touchBias+edge*240u+72u);
    // Both producers publish exact response extents in one EdgeIndex
    // journal; CPU Gu supports the full 255-contact source range.
    const bool pcm=(touch&0x10000u)!=0u;
    const bool gu=(touch&0x20000u)!=0u;
    if(!pcm && !gu)return 0u;
    const uint emitted=pcm?(touch>>8u)&7u:(touch>>8u)&255u;
    if((pcm && gu) || (pcm && emitted>4u) ||
       ((touch&1u)!=0u)!=(emitted!=0u)) {
        reject(256u);
        work.Store(p.workBias+16u,0u);
        return 0u;
    }
    return numericalExtentValid|emitted;
}
uint itemRoot(uint item) {
    if(item>=p.nodeCount)return responseRoot(item-p.nodeCount);
    if(!respondingNode(item) || work.Load(p.workBias)!=0u)
        return p.nodeCount;
    const uint root=work.Load(p.rootBias+item*4u);
    if(root>=p.nodeCount)return p.nodeCount;
    return work.Load(p.countBias+root*24u+20u)!=invalid?root:p.nodeCount;
}
uint itemKey(uint item) {
    const uint root=itemRoot(item);
    if(root==p.nodeCount)return invalid;
    if(item<p.nodeCount)return root*3u;
    const uint edge=item-p.nodeCount;
    const uint kind=work.Load(p.admissionBias+edge*24u+20u);
    // setupDescTask admits 1D rows before contact managers. Preserve
    // original source order within each type; Sc supplies joint edges
    // in descending Constraint.index order.
    return root*3u+((kind&jointEdge)!=0u?1u:2u);
}
void scatterItem(uint position,uint item) {
    const uint root=itemRoot(item);
    if(root==p.nodeCount)return;
    const uint ordinal=work.Load(p.countBias+root*24u+20u);
    const uint task=p.taskBias+ordinal*36u;
    const uint responseBegin=work.Load(task);
    const uint responseCount=work.Load(task+4u)-responseBegin;
    const uint bodyBegin=work.Load(task+16u);
    const uint bodyCount=work.Load(task+20u)-bodyBegin;
    const uint start=bodyBegin+responseBegin;
    if(position<start) {
        reject(128u);work.Store(p.workBias+16u,0u);return;
    }
    const uint local=position-start;
    if(item<p.nodeCount) {
        if(local>=bodyCount) {
            reject(128u);work.Store(p.workBias+16u,0u);return;
        }
        work.Store(p.bodyOrderBias+(bodyBegin+local)*4u,item);
    } else {
        if(local<bodyCount || local-bodyCount>=responseCount) {
            reject(128u);work.Store(p.workBias+16u,0u);return;
        }
        const uint response=responseBegin+local-bodyCount;
        const uint edge=item-p.nodeCount;
        work.Store(p.responseOrderBias+response*4u,edge);
        work.Store(p.tokenBias+response*4u,responseToken(edge));
        work.Store(p.extentBias+response*4u,responseExtent(edge));
        if((work.Load(p.admissionBias+edge*24u+20u)&jointEdge)==0u) {
            uint unused;
            const uint source=work.Load(p.admissionBias+edge*24u);
            work.InterlockedOr(p.contactAdmissionBias+(source>>5u)*4u,
                1u<<(source&31u),unused);
        }
    }
}
bool runnableRoot(uint index) {
    if(index>=p.nodeCount || work.Load(p.workBias)!=0u ||
       work.Load(p.rootBias+index*4u)!=index)return false;
    const uint count=p.countBias+index*24u;
    return work.Load(count)!=0u && work.Load(count+8u)!=0u &&
        ((nodeFlags.Load(p.nodeBias+index*4u)&requireContactTask)==0u ||
         work.Load(count+4u)!=0u);
}
void inclusiveTaskScan(uint lane) {
    [unroll]
    for(uint step=1u;step<128u;step<<=1u) {
        const uint previous=lane>=step?taskScan[lane-step]:0u;
        const uint edges=lane>=step?responseScan[lane-step]:0u;
        const uint nodes=lane>=step?bodyScan[lane-step]:0u;
        GroupMemoryBarrierWithGroupSync();
        taskScan[lane]+=previous;
        responseScan[lane]+=edges;
        bodyScan[lane]+=nodes;
        GroupMemoryBarrierWithGroupSync();
    }
}

// Small graphs complete accurate admission in one ordered dispatch.
void admitCompact(uint lane) {
    if(lane==0u) {
        work.Store4(p.workBias,uint4(0u,0u,0u,0u));
        work.Store4(p.workBias+16u,uint4(0u,1u,1u,p.edgeCount));
        work.Store4(p.workBias+32u,uint4(0u,0u,0u,0u));
    }
    for(uint word=lane;word<(p.primaryCapacity/32u+
            uint((p.primaryCapacity&31u)!=0u));word+=128u)
        work.Store(p.contactAdmissionBias+word*4u,0u);
    for(uint node=lane;node<p.nodeCount;node+=128u) {
        work.Store(p.rootBias+node*4u,node);
        const uint count=p.countBias+node*24u;
        work.Store4(count,uint4(0u,0u,0u,0u));
        work.Store2(count+16u,uint2(0u,invalid));
    }
    AllMemoryBarrierWithGroupSync();

    for(uint item=lane;item<p.edgeCount;item+=128u) {
        const uint input=p.edgeBias+item*20u;
        const uint4 edge=scEdges.Load4(input);
        const uint kind=scEdges.Load(input+16u);
        if(validNode(edge.y) && validNode(edge.z) &&
           edgeConnected(edge.x,kind) && validSource(edge.w,kind) &&
           respondingNode(edge.y) && respondingNode(edge.z))
            connectNodes(edge.y,edge.z);
    }
    AllMemoryBarrierWithGroupSync();

    for(uint node=lane;node<p.nodeCount;node+=128u) {
        const uint root=findRoot(node);
        if(root!=invalid) {
            work.Store(p.rootBias+node*4u,root);
            const uint flags=nodeFlags.Load(p.nodeBias+node*4u);
            const uint position=flags&255u;
            const bool responds=(flags&nonrespondingNode)==0u;
            if((flags&0xfff80000u)!=0u ||
               (responds && position==0u))
                reject(16u);
            const uint count=p.countBias+root*24u;
            uint previous;
            if(responds) {
                work.InterlockedAdd(count,1u,previous);
                work.InterlockedMax(count+12u,position,previous);
                work.InterlockedMax(count+16u,(flags>>8u)&255u,previous);
            }
            if((flags&0x10000u)!=0u)
                work.InterlockedAdd(count+8u,1u,previous);
        }
    }
    AllMemoryBarrierWithGroupSync();

    for(uint item=lane;item<p.edgeCount;item+=128u) {
        const uint input=p.edgeBias+item*20u;
        const uint4 edge=scEdges.Load4(input);
        const uint kind=scEdges.Load(input+16u);
        uint owner=invalid;
        if(validNode(edge.y) && validNode(edge.z) &&
           edgeConnected(edge.x,kind) && validSource(edge.w,kind)) {
            const uint node=respondingNode(edge.y)?edge.y:
                respondingNode(edge.z)?edge.z:invalid;
            if(node!=invalid) {
                owner=findRoot(node);
                if(owner!=invalid) {
                    const uint count=p.countBias+owner*24u;
                    uint previous;
                    if((kind&noResponse)==0u)
                        work.InterlockedAdd(count+4u,1u,previous);
                    if((kind&jointEdge)==0u &&
                       (edgeHistory.Load(p.touchBias+edge.x*240u+76u)&1u)==0u)
                        work.InterlockedAdd(count+8u,1u,previous);
                }
            }
        }
        const uint address=p.admissionBias+item*24u;
        work.Store4(address,uint4(edge.w,edge.x,edge.y,edge.z));
        work.Store2(address+16u,uint2(owner,kind));
    }
    AllMemoryBarrierWithGroupSync();

    if(lane==0u && work.Load(p.workBias)==0u) {
        uint taskCount=0u,bodyCount=0u,responseCount=0u;
        uint nonSingleton=0u;
        for(uint root=0u;root<p.nodeCount;++root) {
            if(!runnableRoot(root))continue;
            const uint count=p.countBias+root*24u;
            const uint bodies=work.Load(count);
            const uint edges=work.Load(count+4u);
            nonSingleton|=(bodies!=1u || edges!=1u)?1u:0u;
            work.Store(count+20u,taskCount);
            const uint task=p.taskBias+taskCount*36u;
            work.Store4(task,uint4(responseCount,responseCount+edges,
                                    responseCount,responseCount+edges));
            work.Store4(task+16u,uint4(bodyCount,bodyCount+bodies,
                                       work.Load(count+12u),
                                       work.Load(count+16u)));
            work.Store(task+32u,0u);
            bodyCount+=bodies;
            responseCount+=edges;
            ++taskCount;
        }
        work.Store(p.workBias+4u,taskCount);
        work.Store(p.workBias+8u,bodyCount);
        work.Store(p.workBias+12u,responseCount);
        work.Store(p.workBias+32u,nonSingleton);
        const uint dispatch=nonSingleton!=0u?taskCount:
            ((taskCount+127u)/128u);
        publishDispatch(dispatch);
    }
    AllMemoryBarrierWithGroupSync();
    if(work.Load(p.workBias)!=0u)return;

    // Two source entries per lane stay in one workgroup and retain
    // the canonical root, kind and source order.
    const uint bodyShift=p.nodeCount<=128u?7u:8u;
    const uint edgeShift=p.edgeCount<=128u?8u:9u;
    const uint contactBit=1u<<(edgeShift-1u);
    for(uint item=lane;item<256u;item+=128u) {
        const uint bodyRoot=item<p.nodeCount && respondingNode(item)
            ?work.Load(p.rootBias+item*4u):invalid;
        compactBodyKeys[item]=bodyRoot<p.nodeCount
            ?(bodyRoot<<bodyShift)|item:invalid;
        const uint edgeRoot=item<p.edgeCount?responseRoot(item):invalid;
        const uint edgeKind=item<p.edgeCount
            ?work.Load(p.admissionBias+item*24u+20u):0u;
        compactEdgeKeys[item]=edgeRoot<p.nodeCount
            ?(edgeRoot<<edgeShift)|
                ((edgeKind&jointEdge)!=0u?0u:contactBit)|item
            :invalid;
        compactBodyStarts[item]=invalid;
        compactEdgeStarts[item]=invalid;
    }
    GroupMemoryBarrierWithGroupSync();

    uint sortWidth=1u;
    while(sortWidth<max(p.nodeCount,p.edgeCount))sortWidth<<=1u;
    [loop]
    for(uint width=2u;width<=sortWidth;width<<=1u) {
        [loop]
        for(uint stride=width>>1u;stride!=0u;stride>>=1u) {
            const uint other=lane+128u;
            const uint body0=compactBodyKeys[lane];
            const uint body1=compactBodyKeys[other];
            const uint edge0=compactEdgeKeys[lane];
            const uint edge1=compactEdgeKeys[other];
            const uint peerBody0=compactBodyKeys[lane^stride];
            const uint peerBody1=compactBodyKeys[other^stride];
            const uint peerEdge0=compactEdgeKeys[lane^stride];
            const uint peerEdge1=compactEdgeKeys[other^stride];
            const bool ascending0=((lane&width)==0u)==
                ((lane&stride)==0u);
            const bool ascending1=((other&width)==0u)==
                ((other&stride)==0u);
            const uint nextBody0=ascending0
                ?min(body0,peerBody0):max(body0,peerBody0);
            const uint nextBody1=ascending1
                ?min(body1,peerBody1):max(body1,peerBody1);
            const uint nextEdge0=ascending0
                ?min(edge0,peerEdge0):max(edge0,peerEdge0);
            const uint nextEdge1=ascending1
                ?min(edge1,peerEdge1):max(edge1,peerEdge1);
            GroupMemoryBarrierWithGroupSync();
            compactBodyKeys[lane]=nextBody0;
            compactBodyKeys[other]=nextBody1;
            compactEdgeKeys[lane]=nextEdge0;
            compactEdgeKeys[other]=nextEdge1;
            GroupMemoryBarrierWithGroupSync();
        }
    }
    for(uint item=lane;item<256u;item+=128u) {
        const uint sortedBody=compactBodyKeys[item];
        const uint sortedEdge=compactEdgeKeys[item];
        const uint bodyRootSorted=sortedBody>>bodyShift;
        const uint edgeRootSorted=sortedEdge>>edgeShift;
        if(bodyRootSorted<p.nodeCount) {
            const uint previous=item==0u
                ?invalid:compactBodyKeys[item-1u];
            if((previous>>bodyShift)!=bodyRootSorted)
                compactBodyStarts[bodyRootSorted]=item;
        }
        if(edgeRootSorted<p.nodeCount) {
            const uint previous=item==0u
                ?invalid:compactEdgeKeys[item-1u];
            if((previous>>edgeShift)!=edgeRootSorted)
                compactEdgeStarts[edgeRootSorted]=item;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for(uint item=lane;item<256u;item+=128u) {
        const uint sortedBody=compactBodyKeys[item];
        const uint sortedEdge=compactEdgeKeys[item];
        const uint bodyRootSorted=sortedBody>>bodyShift;
        const uint edgeRootSorted=sortedEdge>>edgeShift;
        if(bodyRootSorted<p.nodeCount) {
            const uint ordinal=work.Load(
                p.countBias+bodyRootSorted*24u+20u);
            if(ordinal!=invalid) {
                const uint task=p.taskBias+ordinal*36u;
                const uint begin=work.Load(task+16u);
                const uint end=work.Load(task+20u);
                const uint rank=item-compactBodyStarts[bodyRootSorted];
                if(begin+rank>=end)reject(128u);
                else work.Store(p.bodyOrderBias+(begin+rank)*4u,
                                sortedBody&((1u<<bodyShift)-1u));
            }
        }
        if(edgeRootSorted<p.nodeCount) {
            const uint edge=sortedEdge&((1u<<(edgeShift-1u))-1u);
            const uint ordinal=work.Load(
                p.countBias+edgeRootSorted*24u+20u);
            const uint task=p.taskBias+ordinal*36u;
            const uint begin=work.Load(task);
            const uint end=work.Load(task+4u);
            const uint rank=item-compactEdgeStarts[edgeRootSorted];
            if(begin+rank>=end)reject(128u);
            else {
                const uint position=begin+rank;
                work.Store(p.responseOrderBias+position*4u,edge);
                work.Store(p.tokenBias+position*4u,responseToken(edge));
                work.Store(p.extentBias+position*4u,responseExtent(edge));
                if((work.Load(p.admissionBias+edge*24u+20u)&jointEdge)==0u) {
                    uint unused;
                    const uint source=work.Load(p.admissionBias+edge*24u);
                    work.InterlockedOr(p.contactAdmissionBias+(source>>5u)*4u,
                        1u<<(source&31u),unused);
                }
            }
        }
    }
    // The resident response stream owns exact contact-row offsets.
    // Compact graphs compute their global prefix inside admission.
    AllMemoryBarrierWithGroupSync();
    if(work.Load(p.workBias)!=0u)return;
    const uint responses=work.Load(p.workBias+12u);
    if(responses<=1u) {
        if(lane==0u && responses!=0u)
            work.Store(p.contactPrefixBias,0u);
        return;
    }
    for(uint index=lane;index<256u;index+=128u) {
        const uint extent=index<responses
            ?work.Load(p.extentBias+index*4u):0u;
        compactContactScan[index]=(extent&numericalExtentValid)!=0u
            ?(extent&255u):0u;
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for(uint step=1u;step<256u;step<<=1u) {
        const uint a=lane>=step?compactContactScan[lane-step]:0u;
        const uint other=lane+128u;
        const uint b=other>=step?compactContactScan[other-step]:0u;
        GroupMemoryBarrierWithGroupSync();
        compactContactScan[lane]+=a;
        compactContactScan[other]+=b;
        GroupMemoryBarrierWithGroupSync();
    }
    for(uint index=lane;index<responses;index+=128u) {
        const uint extent=work.Load(p.extentBias+index*4u);
        const uint count=(extent&numericalExtentValid)!=0u
            ?(extent&255u):0u;
        work.Store(p.contactPrefixBias+index*4u,
                   compactContactScan[index]-count);
    }
}
// The large-graph scan uses the discarded radix histogram as its
// block/group prefix workspace after response ordering has completed.
void scanContactBlock(uint lane) {
    [unroll]
    for(uint step=1u;step<128u;step<<=1u) {
        const uint prior=lane>=step?compactContactScan[lane-step]:0u;
        GroupMemoryBarrierWithGroupSync();
        compactContactScan[lane]+=prior;
        GroupMemoryBarrierWithGroupSync();
    }
}
[numthreads(128,1,1)]
void main(uint3 thread:SV_DispatchThreadID,
          uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    const uint index=p.first+thread.x;
    if(p.phase==32u) {
        if(group.x==0u)admitCompact(lane.x);
        return;
    }
    if(p.phase==33u) {
        // Compact source ordinals map to the final Dy body and joint rows.
        const uint count=work.Load(p.workBias+8u);
        if(index<count) {
            const uint actor=work.Load(p.bodyOrderBias+index*4u);
            const uint row=actor<p.nodeCount
                ?nodeFlags.Load(p.nodeBias+actor*4u):invalid;
            if(row==invalid || row>=p.primaryCapacity) {
                reject(512u);
                work.Store(p.workBias+16u,0u);
            } else work.Store(p.bodyOrderBias+index*4u,row);
        }
        if(p.fullDescriptorCount==0u)return;
        const uint responses=work.Load(p.workBias+12u);
        if(index<responses) {
            const uint address=p.tokenBias+index*4u;
            const uint token=work.Load(address);
            if((token&0xc0000000u)==0x80000000u) {
                const uint ordinal=token&0x3fffffffu;
                const uint row=ordinal<p.fullDescriptorCount
                    ?nodeFlags.Load(p.nodeBias+
                        (p.nodeCount+ordinal)*4u):invalid;
                if(row==invalid || row>=p.jointRowCount) {
                    reject(512u);
                    work.Store(p.workBias+16u,0u);
                } else work.Store(address,0x80000000u|row);
            }
        }
        return;
    }
    if(p.phase==38u) {
        // Bounded response streams retain their exact row prefix in
        // one dispatch without intermediate block and group launches.
        if(group.x!=0u)return;
        if(lane.x==0u)contactGroupCarry=0u;
        GroupMemoryBarrierWithGroupSync();
        const uint responses=work.Load(p.workBias+12u);
        if(responses>p.edgeCount || p.edgeCount>512u) {
            if(lane.x==0u)reject(128u);
            return;
        }
        for(uint start=0u;start<p.edgeCount;start+=128u) {
            const uint item=start+lane.x;
            const uint extent=item<responses
                ?work.Load(p.extentBias+item*4u):0u;
            const uint amount=(extent&numericalExtentValid)!=0u
                ?(extent&255u):0u;
            compactContactScan[lane.x]=amount;
            GroupMemoryBarrierWithGroupSync();
            scanContactBlock(lane.x);
            if(item<responses)
                work.Store(p.contactPrefixBias+item*4u,
                           contactGroupCarry+
                           compactContactScan[lane.x]-amount);
            GroupMemoryBarrierWithGroupSync();
            if(lane.x==0u)
                contactGroupCarry+=compactContactScan[127u];
            GroupMemoryBarrierWithGroupSync();
        }
        return;
    }
    if(p.phase>=34u && p.phase<=37u) {
        const uint responses=work.Load(p.workBias+12u);
        const uint blocks=(p.edgeCount+127u)/128u;
        const uint groups=(blocks+127u)/128u;
        if(p.phase==34u) {
            const uint extent=index<responses
                ?work.Load(p.extentBias+index*4u):0u;
            const uint amount=(extent&numericalExtentValid)!=0u
                ?(extent&255u):0u;
            compactContactScan[lane.x]=amount;
            GroupMemoryBarrierWithGroupSync();
            scanContactBlock(lane.x);
            if(index<responses)
                work.Store(p.contactPrefixBias+index*4u,
                           compactContactScan[lane.x]-amount);
            if(lane.x==127u)
                work.Store(p.histBias+
                    ((p.first>>7u)+group.x)*4u,
                    compactContactScan[127u]);
            return;
        }
        if(p.phase==35u) {
            const uint amount=index<blocks
                ?work.Load(p.histBias+index*4u):0u;
            compactContactScan[lane.x]=amount;
            GroupMemoryBarrierWithGroupSync();
            scanContactBlock(lane.x);
            if(index<blocks)
                work.Store(p.histBias+index*4u,
                           compactContactScan[lane.x]-amount);
            if(lane.x==127u)
                work.Store(p.histBias+
                    (blocks+(p.first>>7u)+group.x)*4u,
                    blocks<=128u?0u:compactContactScan[127u]);
            return;
        }
        if(p.phase==36u) {
            if(group.x!=0u)return;
            if(lane.x==0u)contactGroupCarry=0u;
            GroupMemoryBarrierWithGroupSync();
            for(uint start=0u;start<groups;start+=128u) {
                const uint i=start+lane.x;
                const uint amount=i<groups
                    ?work.Load(p.histBias+(blocks+i)*4u):0u;
                compactContactScan[lane.x]=amount;
                GroupMemoryBarrierWithGroupSync();
                scanContactBlock(lane.x);
                if(i<groups)
                    work.Store(p.histBias+(blocks+i)*4u,
                               contactGroupCarry+
                               compactContactScan[lane.x]-amount);
                GroupMemoryBarrierWithGroupSync();
                if(lane.x==0u)
                    contactGroupCarry+=compactContactScan[127u];
                GroupMemoryBarrierWithGroupSync();
            }
            return;
        }
        if(index<responses) {
            const uint block=index>>7u;
            const uint groupIndex=block>>7u;
            const uint local=work.Load(p.contactPrefixBias+index*4u);
            const uint blockOffset=work.Load(p.histBias+block*4u);
            const uint groupOffset=work.Load(
                p.histBias+(blocks+groupIndex)*4u);
            work.Store(p.contactPrefixBias+index*4u,
                       local+blockOffset+groupOffset);
        }
        return;
    }
    if(p.phase==0u) {
        if(index==0u) {
            work.Store4(p.workBias,uint4(0u,0u,0u,0u));
            work.Store4(p.workBias+16u,uint4(0u,1u,1u,p.edgeCount));
            work.Store(p.workBias+32u,0u);
        }
        if(index<(p.primaryCapacity/32u+
            uint((p.primaryCapacity&31u)!=0u)))
            work.Store(p.contactAdmissionBias+index*4u,0u);
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
        if(!validNode(edge.y) || !validNode(edge.z) ||
           !edgeConnected(edge.x,kind))return;
        if(!validSource(edge.w,kind))return;
        // Static/kinematic endpoints do not connect two dynamic
        // components through a shared nonresponding body.
        if(respondingNode(edge.y) && respondingNode(edge.z))
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
                const bool responds=(flags&nonrespondingNode)==0u;
                if((flags&0xfff80000u)!=0u ||
                   (responds && !position))reject(16u);
                const uint count=p.countBias+root*24u;
                uint previous;
                if(responds) {
                    work.InterlockedAdd(count,1u,previous);
                    work.InterlockedMax(count+12u,position,previous);
                    work.InterlockedMax(count+16u,velocity,previous);
                }
                if((flags&0x10000u)!=0u)
                    work.InterlockedAdd(count+8u,1u,previous);
            }
        }
        if(index>=p.edgeCount)return;
        const uint input=p.edgeBias+index*20u;
        const uint4 edge=scEdges.Load4(input);
        const uint kind=scEdges.Load(input+16u);
        uint owner=invalid;
        if(validNode(edge.y) && validNode(edge.z) &&
           edgeConnected(edge.x,kind) && validSource(edge.w,kind)) {
            const uint node=respondingNode(edge.y)?edge.y:
                respondingNode(edge.z)?edge.z:invalid;
            if(node!=invalid) {
                owner=findRoot(node);
                if(owner!=invalid) {
                    uint previous;
                    if((kind&noResponse)==0u)
                        work.InterlockedAdd(p.countBias+owner*24u+4u,
                                            1u,previous);
                    // Physical new-touch activation is derived from
                    // the journaled prior state for both Gu producers.
                    if((kind&jointEdge)==0u) {
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
        bodyScan[lane.x]=active!=0u
            ?work.Load(p.countBias+index*24u):0u;
        if(active!=0u &&
           (responseScan[lane.x]!=1u || bodyScan[lane.x]!=1u)) {
            uint previous;
            work.InterlockedOr(p.workBias+32u,1u,previous);
        }
        GroupMemoryBarrierWithGroupSync();
        for(uint stride=64u;stride!=0u;stride>>=1u) {
            if(lane.x<stride) {
                taskScan[lane.x]+=taskScan[lane.x+stride];
                responseScan[lane.x]+=responseScan[lane.x+stride];
                bodyScan[lane.x]+=bodyScan[lane.x+stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if(lane.x==0u && p.blockCount!=0u)
            work.Store4(p.blockBias+(index/128u)*16u,
                uint4(taskScan[0],responseScan[0],bodyScan[0],0u));
        return;
    }
    if(p.phase==4u) {
        // Small root sets use one workgroup and no tile carry.
        if(p.blockCount<=128u) {
            if(group.x!=0u)return;
            if(lane.x==0u) {
                taskCarry=0u;responseCarry=0u;bodyCarry=0u;
            }
            GroupMemoryBarrierWithGroupSync();
            for(uint start=0u;start<p.blockCount;start+=128u) {
                const uint block=start+lane.x;
                const uint4 value=block<p.blockCount
                    ?work.Load4(p.blockBias+block*16u):uint4(0u,0u,0u,0u);
                taskScan[lane.x]=value.x;
                responseScan[lane.x]=value.y;
                bodyScan[lane.x]=value.z;
                GroupMemoryBarrierWithGroupSync();
                inclusiveTaskScan(lane.x);
                if(block<p.blockCount)
                    work.Store4(p.blockBias+block*16u,uint4(
                        taskCarry+taskScan[lane.x]-value.x,
                        responseCarry+responseScan[lane.x]-value.y,
                        bodyCarry+bodyScan[lane.x]-value.z,0u));
                GroupMemoryBarrierWithGroupSync();
                if(lane.x==0u) {
                    taskCarry+=taskScan[127u];
                    responseCarry+=responseScan[127u];
                    bodyCarry+=bodyScan[127u];
                }
                GroupMemoryBarrierWithGroupSync();
            }
            if(lane.x==0u) {
                work.Store(p.workBias+4u,taskCarry);
                work.Store(p.workBias+8u,bodyCarry);
                work.Store(p.workBias+12u,responseCarry);
                publishDispatch(work.Load(p.workBias+32u)==0u && taskCarry>1u
                        ?(taskCarry>>7u)+uint((taskCarry&127u)!=0u)
                        :taskCarry);
            }
            return;
        }

        const uint block=index;
        const uint4 value=block<p.blockCount
            ?work.Load4(p.blockBias+block*16u):uint4(0u,0u,0u,0u);
        taskScan[lane.x]=value.x;
        responseScan[lane.x]=value.y;
        bodyScan[lane.x]=value.z;
        GroupMemoryBarrierWithGroupSync();
        inclusiveTaskScan(lane.x);
        if(block<p.blockCount)
            work.Store4(p.blockBias+block*16u,uint4(
                taskScan[lane.x]-value.x,
                responseScan[lane.x]-value.y,
                bodyScan[lane.x]-value.z,0u));
        if(lane.x==127u)
            work.Store4(p.blockTilesBias+(block/128u)*16u,
                uint4(taskScan[127u],responseScan[127u],bodyScan[127u],0u));
        return;
    }
    if(p.phase==5u) {
        if(group.x!=0u)return;
        if(lane.x==0u) {
            taskCarry=0u;responseCarry=0u;bodyCarry=0u;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint tileCount=(p.blockCount+127u)/128u;
        for(uint start=0u;start<tileCount;start+=128u) {
            const uint tile=start+lane.x;
            const uint4 value=tile<tileCount
                ?work.Load4(p.blockTilesBias+tile*16u):
                 uint4(0u,0u,0u,0u);
            taskScan[lane.x]=value.x;
            responseScan[lane.x]=value.y;
            bodyScan[lane.x]=value.z;
            GroupMemoryBarrierWithGroupSync();
            inclusiveTaskScan(lane.x);
            if(tile<tileCount)
                work.Store4(p.blockTilesBias+tile*16u,uint4(
                    taskCarry+taskScan[lane.x]-value.x,
                    responseCarry+responseScan[lane.x]-value.y,
                    bodyCarry+bodyScan[lane.x]-value.z,0u));
            GroupMemoryBarrierWithGroupSync();
            if(lane.x==0u) {
                taskCarry+=taskScan[127u];
                responseCarry+=responseScan[127u];
                bodyCarry+=bodyScan[127u];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if(lane.x==0u) {
            work.Store(p.workBias+4u,taskCarry);
            work.Store(p.workBias+8u,bodyCarry);
            work.Store(p.workBias+12u,responseCarry);
            publishDispatch(work.Load(p.workBias+32u)==0u && taskCarry>1u
                        ?(taskCarry>>7u)+uint((taskCarry&127u)!=0u)
                        :taskCarry);
        }
        return;
    }
    if(p.phase==6u) {
        const uint active=runnableRoot(index)?1u:0u;
        const uint count=p.countBias+index*24u;
        const uint edges=active!=0u?work.Load(count+4u):0u;
        const uint nodes=active!=0u?work.Load(count):0u;
        taskScan[lane.x]=active;
        responseScan[lane.x]=edges;
        bodyScan[lane.x]=nodes;
        GroupMemoryBarrierWithGroupSync();
        inclusiveTaskScan(lane.x);
        if(active==0u)return;
        uint3 base=work.Load3(p.blockBias+(index/128u)*16u);
        if(p.blockCount>128u)
            base+=work.Load3(p.blockTilesBias+(index/16384u)*16u);
        const uint task=base.x+taskScan[lane.x]-1u;
        const uint response=base.y+responseScan[lane.x]-edges;
        const uint bodyBegin=base.z+bodyScan[lane.x]-nodes;
        const uint position=work.Load(count+12u);
        const uint velocity=work.Load(count+16u);
        work.Store(count+20u,task);
        const uint dst=p.taskBias+task*36u;
        work.Store4(dst,uint4(response,response+edges,
                              response,response+edges));
        work.Store4(dst+16u,uint4(bodyBegin,bodyBegin+nodes,
                                  position,velocity));
        work.Store(dst+32u,0u);
        return;
    }
    if(p.phase==7u) {
        if(p.combinedCount>256u) {
            if(index<p.combinedCount)
                work.Store(p.responseBiasA+index*4u,index);
            return;
        }
        // One workgroup sorts all body and response identities.
        // The lower nine bits preserve source order among equal keys.
        const uint first=lane.x;
        const uint second=first+128u;
        const uint firstKey=first<p.combinedCount?itemKey(first):invalid;
        const uint secondKey=second<p.combinedCount?itemKey(second):invalid;
        radixBuckets[first]=firstKey==invalid
            ?invalid:(firstKey<<9u)|first;
        radixBuckets[second]=secondKey==invalid
            ?invalid:(secondKey<<9u)|second;
        GroupMemoryBarrierWithGroupSync();
        [loop]
        for(uint width=2u;width<=256u;width<<=1u) {
            [loop]
            for(uint stride=width>>1u;stride!=0u;stride>>=1u) {
                const uint a=radixBuckets[first];
                const uint b=radixBuckets[second];
                const uint otherA=radixBuckets[first^stride];
                const uint otherB=radixBuckets[second^stride];
                const bool upA=((first&width)==0u)==
                    ((first&stride)==0u);
                const bool upB=((second&width)==0u)==
                    ((second&stride)==0u);
                const uint nextA=upA?min(a,otherA):max(a,otherA);
                const uint nextB=upB?min(b,otherB):max(b,otherB);
                GroupMemoryBarrierWithGroupSync();
                radixBuckets[first]=nextA;
                radixBuckets[second]=nextB;
                GroupMemoryBarrierWithGroupSync();
            }
        }
        const uint sortedA=radixBuckets[first];
        const uint sortedB=radixBuckets[second];
        if(sortedA!=invalid)scatterItem(first,sortedA&511u);
        if(sortedB!=invalid)scatterItem(second,sortedB&511u);
        return;
    }
    if(p.phase<8u)return;
    const uint pass=(p.phase-8u)/3u;
    const uint operation=(p.phase-8u)%3u;
    const uint shift=pass*8u;
    const uint source=(pass&1u)!=0u?p.responseBiasB:p.responseBiasA;
    const uint target=(pass&1u)!=0u?p.responseBiasA:p.responseBiasB;
    const uint blocks=(p.combinedCount+127u)/128u;
    const uint block=index/128u;
    if(operation==0u) {
        radixBuckets[lane.x]=0u;
        radixBuckets[lane.x+128u]=0u;
        GroupMemoryBarrierWithGroupSync();
        if(index<p.combinedCount) {
            const uint item=work.Load(source+index*4u);
            const uint digit=(itemKey(item)>>shift)&255u;
            uint previous;
            InterlockedAdd(radixBuckets[digit],1u,previous);
        }
        GroupMemoryBarrierWithGroupSync();
        const uint base=p.histBias+block*1024u;
        work.Store(base+lane.x*4u,radixBuckets[lane.x]);
        work.Store(base+(lane.x+128u)*4u,radixBuckets[lane.x+128u]);
        return;
    }
    if(operation==1u) {
        if(group.x!=0u)return;
        const uint bucket0=lane.x,bucket1=lane.x+128u;
        uint total0=0u,total1=0u;
        for(uint b=0u;b<blocks;++b) {
            total0+=work.Load(p.histBias+b*1024u+bucket0*4u);
            total1+=work.Load(p.histBias+b*1024u+bucket1*4u);
        }
        radixTotals[bucket0]=total0;
        radixTotals[bucket1]=total1;
        GroupMemoryBarrierWithGroupSync();
        if(lane.x==0u) {
            uint carry=0u;
            for(uint b=0u;b<256u;++b) {
                radixOffsets[b]=carry;
                carry+=radixTotals[b];
            }
            if(carry!=p.combinedCount) {
                reject(32u);
                work.Store(p.workBias+16u,0u);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        uint cursor0=radixOffsets[bucket0];
        uint cursor1=radixOffsets[bucket1];
        for(uint b=0u;b<blocks;++b) {
            const uint a=p.histBias+b*1024u;
            const uint c0=work.Load(a+bucket0*4u);
            const uint c1=work.Load(a+bucket1*4u);
            work.Store(a+bucket0*4u,cursor0);
            work.Store(a+bucket1*4u,cursor1);
            cursor0+=c0;
            cursor1+=c1;
        }
        return;
    }
    if(operation==2u) {
        radixDigits[lane.x]=invalid;
        if(index<p.combinedCount) {
            const uint item=work.Load(source+index*4u);
            radixDigits[lane.x]=(itemKey(item)>>shift)&255u;
        }
        GroupMemoryBarrierWithGroupSync();
        if(lane.x==0u) {
            for(uint i=0u;i<256u;++i)radixBuckets[i]=0u;
            for(uint i=0u;i<128u;++i) {
                const uint digit=radixDigits[i];
                if(digit==invalid)continue;
                radixRanks[i]=radixBuckets[digit]++;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if(index<p.combinedCount) {
            const uint digit=radixDigits[lane.x];
            const uint position=work.Load(
                p.histBias+block*1024u+digit*4u)+radixRanks[lane.x];
            const uint item=work.Load(source+index*4u);
            work.Store(target+position*4u,item);
            if(pass+1u==p.radixPasses)scatterItem(position,item);
        }
    }
}
