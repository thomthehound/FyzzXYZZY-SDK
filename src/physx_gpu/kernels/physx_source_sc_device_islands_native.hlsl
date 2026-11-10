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
    uint responseBiasA,responseBiasB,histBias,contactManagerCount;
    uint tokenBias,radixPasses,bodyOrderBias,combinedCount;
    uint responseOrderBias,jointRowCount,fullDescriptorCount,blockTilesBias;
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
static const uint deviceJoint=16u;
static const uint nonrespondingNode=0x20000u;
static const uint permittedKinds=jointEdge|noResponse|deviceJoint;

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
bool validSource(uint source,uint kind) {
    // Nonresponding edges still carry accurate connectivity but do not
    // address any numerical table. All admitted numerical response
    // tokens must resolve within their typed Dy input table.
    if((kind&noResponse)!=0u)return true;
    const uint limit=(kind&jointEdge)==0u?p.contactManagerCount:
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
    }
}
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
        const uint nodes=lane>=step?bodyScan[lane-step]:0u;
        GroupMemoryBarrierWithGroupSync();
        taskScan[lane]+=previous;
        responseScan[lane]+=edges;
        bodyScan[lane]+=nodes;
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
            work.Store4(p.workBias+16u,uint4(0u,1u,1u,p.edgeCount));
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
                if((flags&0xfffc0000u)!=0u ||
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
                work.Store(p.workBias+16u,min(taskCarry,p.maxGroups));
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
            work.Store(p.workBias+16u,min(taskCarry,p.maxGroups));
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
        if(index>=p.combinedCount)return;
        if(p.combinedCount>256u) {
            work.Store(p.responseBiasA+index*4u,index);
            return;
        }
        const uint key=itemKey(index);
        if(key==invalid)return;
        // Rank across bodies, joints and contacts in the same root.
        // Equal-key items retain original source encounter order.
        uint ordinal=0u;
        for(uint i=0u;i<p.combinedCount;++i) {
            const uint other=itemKey(i);
            if(other/3u==key/3u &&
               (other<key || (other==key && i<index)))
                ++ordinal;
        }
        const uint taskId=work.Load(p.countBias+(key/3u)*24u+20u);
        const uint task=p.taskBias+taskId*36u;
        const uint begin=work.Load(task)+work.Load(task+16u);
        scatterItem(begin+ordinal,index);
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
