// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Physical-touch history belongs to the persistent Sc EdgeIndex record.
// The candidate pass snapshots the prior published touch at +76, so GJK,
// EPA and convex face continuations can publish repeatedly in one frame.
uint sourceScTouchTransition(uint edge,uint emitted,uint status,
                             uint edgeBias,uint edgeCapacity) {
    if(edge>=edgeCapacity)return 0u;
    uint state=edgeBias+edge*240u;
    uint prior=edgeCaches.Load(state+76u)&1u;
    // The high bit marks a failed GPU contact producer for device
    // numerical admission. Successful publication clears that mark.
    if(status!=0u) {
        edgeCaches.Store(state+72u,0x80000000u);
        return 0u;
    }
    uint current=emitted!=0u?1u:0u;
    edgeCaches.Store(state+72u,current);
    return current!=0u?(prior!=0u?2u:1u):(prior!=0u?3u:0u);
}

// Publish Gu world contacts from the lane that finalizes the Sc EdgeIndex.
void publishScContacts(uint i,uint first,uint second,uint edge,uint result,
                       uint outputBias,uint headerBias,
                       uint transformBias,uint geometryBias,
                       uint edgeBias,uint shapeCapacity,uint edgeCapacity) {
    uint resultEdge=results.Load(result+12u);
    uint status=results.Load(result+8u);
    uint selected=results.Load(result+20u);
    uint regenerated=results.Load(result+28u);
    // Source Sc receives the full 96-byte host record after fetch.
    // Dy consumes this 84-byte device stream without expanding it.
    uint dst=outputBias+i*84u;
    uint selectedBits=min(selected,7u)<<6u;
    uint regenBit=(regenerated!=0u ? 1u : 0u)<<9u;
    if(edge>=edgeCapacity || second>=shapeCapacity) {
        if(edge<edgeCapacity)
            sourceScTouchTransition(edge,0u,1u,edgeBias,edgeCapacity);
        uint invalidCounts=7u | (1u<<3u) | selectedBits | regenBit;
        streamContacts.Store2(dst,uint2(resultEdge,invalidCounts));
        if(headerBias!=0xffffffffu) {
            uint header=headerBias+i*20u;
            streamContacts.Store2(header,uint2(resultEdge,invalidCounts));
            streamContacts.Store3(header+8u,uint3(0u,0u,0u));
        }
        return;
    }
    uint src=edgeBias+edge*240u;
    uint count=edgeCaches.Load(src+12u);
    if(status==0u && (resultEdge!=edge || count>4u ||
       selected>4u || selected>count ||
       (regenerated!=0u && selected!=count)))
        status=4u;
    uint emitted=0u;
    float3 normal=float3(0.f,0.f,0.f);
    float4 qb=float4(0.f,0.f,0.f,1.f);
    float3 pb=float3(0.f,0.f,0.f);
    if(status==0u) {
        uint tb=transformBias+second*32u;
        qb=asfloat(transforms.Load4(tb));
        pb=asfloat(transforms.Load3(tb+16u));
        if(!all(isfinite(qb)) || !all(isfinite(pb)))status=4u;
    }
    if(status==0u && count>0u && count<=4u) {
        float distance=0.f;
        if(first<shapeCapacity) {
            uint ga=geometryBias+first*64u;
            uint gb=geometryBias+second*64u;
            distance=asfloat(geometry.Load(ga+12u))+
                     asfloat(geometry.Load(gb+12u));
        } else {
            status=1u;
        }
        if(status==0u) {
            float3 localNormal=float3(0.f,0.f,0.f);
            [unroll]
            for(uint j=0u;j<4u;++j) {
                if(j>=count)break;
                float3 current=asfloat(edgeCaches.Load3(
                    src+80u+j*40u+24u));
                if(regenerated!=0u) {
                    if(j==0u)localNormal=current;
                }else
                    localNormal+=current;
            }
            float sq=dot(localNormal,localNormal);
            if(!(sq>0.f) || !isfinite(sq)) {
                status=4u;
            }else {
                normal=rotateSource(qb,localNormal*(1.f/sqrt(sq)));
                if(!all(isfinite(normal)) || !isfinite(distance))
                    status=4u;
                [unroll]
                for(uint j=0u;j<4u && status==0u;++j) {
                    if(j>=count)break;
                    uint contact=src+80u+j*40u;
                    float depth=asfloat(edgeCaches.Load(contact+36u));
                    if(regenerated==0u && depth>distance)continue;
                    float3 b=asfloat(edgeCaches.Load3(contact+12u));
                    float3 world=rotateSource(qb,b)+pb;
                    if(!all(isfinite(world)) || !isfinite(depth)) {
                        status=4u;
                        break;
                    }
                    streamContacts.Store4(dst+20u+emitted*16u,
                        asuint(float4(world,depth)));
                    ++emitted;
                }
            }
        }
    }
    if(status!=0u) {
        emitted=0u;
        normal=float3(0.f,0.f,0.f);
    }
    uint transition=sourceScTouchTransition(
        edge,emitted,status,edgeBias,edgeCapacity);
    uint packed=min(count,7u) | (min(status,7u)<<3u) |
        selectedBits | regenBit | (min(emitted,7u)<<10u) |
        (transition<<13u);
    streamContacts.Store2(dst,uint2(resultEdge,packed));
    streamContacts.Store3(dst+8u,asuint(normal));
    if(headerBias!=0xffffffffu) {
        uint header=headerBias+i*20u;
        streamContacts.Store2(header,uint2(resultEdge,packed));
        streamContacts.Store3(header+8u,asuint(normal));
    }
}
