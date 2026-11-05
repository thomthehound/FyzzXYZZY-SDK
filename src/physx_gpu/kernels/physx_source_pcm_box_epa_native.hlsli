// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// GuEPA box continuation. Per-pair facet and vertex storage belongs to
// the sparse workgroup, outside the ordinary box SAT/GJK kernel.

struct EpaFacet {
    uint i,j,k;
    float3 normal;
    float distance;
    uint valid,inHeap;
};
struct EpaEdge { uint from,to; };

// Each sparse continuation owns one workgroup. Bounded EPA topology is
// shared storage rather than repeated private-array function arguments.
groupshared GjkSupportPoint epa_vertices[64];
groupshared EpaFacet epa_facets[64];
groupshared EpaEdge epa_boundary[32];

EpaFacet epaFacet(uint i,uint j,uint k) {
    EpaFacet result=(EpaFacet)0;
    float3 a=epa_vertices[i].q;
    float3 e0=epa_vertices[j].q-a,e1=epa_vertices[k].q-a;
    float3 normal=cross(e0,e1);
    float norm2=gjkLength2(normal);
    if(!(norm2>1.1920928955078125e-7f))return result;
    normal*=1.f/sqrt(norm2);
    result.i=i;result.j=j;result.k=k;
    result.normal=normal;
    result.distance=gjkDot(normal,a);
    result.valid=1u;result.inHeap=1u;
    return result;
}
bool epaInitialize(
    GjkBoxPair pair,inout uint count,out uint facetCount) {
    facetCount=0u;
    if(count==0u || count>4u)return false;
    if(count==1u) {
        GjkSupportPoint next=gjkSupport(pair,float3(1.f,0.f,0.f));
        if(gjkLength2(next.q-epa_vertices[0].q)==0.f)return false;
        epa_vertices[count++]=next;
    }
    if(count==2u) {
        float3 edge=epa_vertices[1].q-epa_vertices[0].q;
        float3 ab=abs(edge);
        float3 axis=float3(1.f,0.f,0.f);
        if(ab.x>ab.y && ab.z>ab.y)
            axis=float3(0.f,1.f,0.f);
        else if(ab.x>ab.z)
            axis=float3(0.f,0.f,1.f);
        float3 perpendicular=cross(axis,edge);
        float sq=gjkLength2(perpendicular);
        if(!(sq>0.f))return false;
        perpendicular*=1.f/sqrt(sq);
        epa_vertices[count++]=gjkSupport(pair,perpendicular);
    }
    if(count==3u) {
        EpaFacet f0=epaFacet(0u,1u,2u);
        EpaFacet f1=epaFacet(1u,0u,2u);
        if(f0.valid==0u || f1.valid==0u)return false;
        epa_facets[facetCount++]=f0;
        epa_facets[facetCount++]=f1;
        return true;
    }
    float3 p0=epa_vertices[0].q;
    if(gjkDot(cross(epa_vertices[1].q-p0,epa_vertices[2].q-p0),
              epa_vertices[3].q-p0)>0.f) {
        GjkSupportPoint swap=epa_vertices[1];
        epa_vertices[1]=epa_vertices[2];
        epa_vertices[2]=swap;
    }
    uint3 faces[4]={
        uint3(0u,1u,2u),uint3(0u,3u,1u),
        uint3(0u,2u,3u),uint3(1u,3u,2u)};
    for(uint i=0u;i<4u;++i) {
        EpaFacet f=epaFacet(faces[i].x,faces[i].y,faces[i].z);
        if(f.valid==0u)return false;
        epa_facets[facetCount++]=f;
    }
    return true;
}

bool epaContact(GjkBoxPair pair,EpaFacet facet,
                out GjkWitness witness) {
    witness=(GjkWitness)0;
    GjkSupportPoint a=epa_vertices[facet.i];
    GjkSupportPoint b=epa_vertices[facet.j];
    GjkSupportPoint c=epa_vertices[facet.k];
    float3 e0=b.q-a.q,e1=c.q-a.q;
    float x=gjkDot(e0,e0),y=gjkDot(e0,e1),z=gjkDot(e1,e1);
    float denominator=x*z-y*y;
    if(!(denominator>1.1920928955078125e-7f*x*z))return false;
    float h=-gjkDot(a.q,e0),k=-gjkDot(a.q,e1);
    float u=(h*z-k*y)/denominator;
    float v=(k*x-h*y)/denominator;
    float w=1.f-u-v;
    float3 pa=(a.a*w+b.a*u)+c.a*v;
    float3 pb=(a.b*w+b.b*u)+c.b*v;
    if(!isfinite(pa.x) || !isfinite(pb.x) ||
       !isfinite(facet.distance))return false;
    float depth=abs(facet.distance);
    witness.normalB=-facet.normal;
    witness.localA=rotateSource(conjugate(pair.relative),
        pa-pair.aInB-witness.normalB*pair.radiusA);
    witness.localB=pb+witness.normalB*pair.radiusB;
    witness.separation=-depth-pair.radiusA-pair.radiusB;
    return sqrt(gjkLength2(pa-pb))<=depth+1.e-3f;
}
// One workgroup owns a penetration pair. Facet visibility and closest-facet
// selection run across lanes; topology updates retain deterministic edge order.
groupshared uint epa_vertex_count;
groupshared uint epa_facet_count;
groupshared uint epa_finished;
groupshared uint epa_succeeded;
groupshared uint epa_visible[64];
groupshared float epa_distances[64];
groupshared GjkSupportPoint epa_next_support;
groupshared GjkWitness epa_finished_witness;

bool epaBoxContinuation(GjkBoxPair pair,
                        GjkSupportPoint simplex[4],uint simplexCount,
                        uint lane,out GjkWitness witness) {
    witness=(GjkWitness)0;
    if(lane==0u) {
        uint vertices=simplexCount,facets=0u;
        for(uint i=0u;i<simplexCount;++i)epa_vertices[i]=simplex[i];
        const bool ready=epaInitialize(pair,vertices,facets);
        epa_vertex_count=vertices;
        epa_facet_count=facets;
        epa_finished=ready?0u:1u;
        epa_succeeded=0u;
    }
    GroupMemoryBarrierWithGroupSync();
    const float epsilon=0.005f*min(
        min(pair.halfA.x,min(pair.halfA.y,pair.halfA.z)),
        min(pair.halfB.x,min(pair.halfB.y,pair.halfB.z)));
    float upperBound=3.402823466e+38f;
    [loop]
    for(uint iteration=0u;iteration<60u && epa_finished==0u;++iteration) {
        const uint facetCount=epa_facet_count;
        float distance=3.402823466e+38f;
        if(lane<facetCount) {
            const EpaFacet facet=epa_facets[lane];
            if(facet.valid!=0u && facet.inHeap!=0u)
                distance=facet.distance;
        }
        epa_distances[lane]=distance;
        GroupMemoryBarrierWithGroupSync();
        if(lane==0u) {
            int closest=-1;
            float best=3.402823466e+38f;
            for(uint i=0u;i<facetCount;++i)
                if(epa_distances[i]<best) {
                    closest=int(i);
                    best=epa_distances[i];
                }
            if(closest<0) {
                epa_finished=1u;
            }else {
                const EpaFacet selected=epa_facets[closest];
                const GjkSupportPoint next=gjkSupport(pair,selected.normal);
                const float projection=gjkDot(next.q,selected.normal);
                if(abs(projection-best)<=epsilon) {
                    epa_succeeded=epaContact(pair,selected,epa_finished_witness)?1u:0u;
                    epa_finished=1u;
                }else if(epa_vertex_count==64u) {
                    epa_finished=1u;
                }else {
                    upperBound=min(upperBound,projection);
                    bool duplicate=false;
                    for(uint i=0u;i<epa_vertex_count;++i)
                        if(epa_vertices[i].ai==next.ai &&
                           epa_vertices[i].bi==next.bi) {
                            duplicate=true;
                            break;
                        }
                    if(duplicate) {
                        epa_succeeded=epaContact(pair,selected,epa_finished_witness)?1u:0u;
                        epa_finished=1u;
                    }else {
                        epa_next_support=next;
                        epa_vertices[epa_vertex_count++]=next;
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if(epa_finished!=0u)break;

        const GjkSupportPoint next=epa_next_support;
        uint visible=0u;
        if(lane<facetCount) {
            const EpaFacet facet=epa_facets[lane];
            visible=(facet.valid!=0u &&
                gjkDot(facet.normal,next.q-epa_vertices[facet.i].q)>=0.f)?1u:0u;
        }
        epa_visible[lane]=visible;
        GroupMemoryBarrierWithGroupSync();
        if(lane==0u) {
            uint boundaryCount=0u;
            for(uint i=0u;i<facetCount;++i) {
                if(epa_visible[i]==0u)continue;
                const EpaFacet f=epa_facets[i];
                epa_facets[i].valid=0u;
                const uint2 edges[3]={
                    uint2(f.i,f.j),uint2(f.j,f.k),uint2(f.k,f.i)};
                for(uint e=0u;e<3u;++e) {
                    int reverse=-1;
                    for(uint j=0u;j<boundaryCount;++j)
                        if(epa_boundary[j].from==edges[e].y &&
                           epa_boundary[j].to==edges[e].x) {
                            reverse=int(j);
                            break;
                        }
                    if(reverse>=0)
                        epa_boundary[reverse]=epa_boundary[--boundaryCount];
                    else if(boundaryCount==32u) {
                        epa_finished=1u;
                        break;
                    }else {
                        EpaEdge entry;
                        entry.from=edges[e].x;
                        entry.to=edges[e].y;
                        epa_boundary[boundaryCount++]=entry;
                    }
                }
                if(epa_finished!=0u)break;
            }
            if(epa_finished==0u) {
                if(boundaryCount==0u) {
                    epa_finished=1u;
                }else {
                    uint slots=64u-facetCount;
                    for(uint i=0u;i<facetCount;++i)
                        if(epa_facets[i].valid==0u)++slots;
                    if(boundaryCount>slots) {
                        epa_finished=1u;
                    }else {
                        const uint newIndex=epa_vertex_count-1u;
                        for(uint i=0u;i<boundaryCount;++i) {
                            const EpaEdge edge=epa_boundary[i];
                            EpaFacet f=epaFacet(edge.from,edge.to,newIndex);
                            if(f.valid==0u) {
                                epa_finished=1u;
                                break;
                            }
                            f.inHeap=f.distance<=upperBound?1u:0u;
                            int slot=-1;
                            for(uint j=0u;j<epa_facet_count;++j)
                                if(epa_facets[j].valid==0u) {
                                    slot=int(j);
                                    break;
                                }
                            if(slot<0)slot=int(epa_facet_count++);
                            epa_facets[slot]=f;
                        }
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    GroupMemoryBarrierWithGroupSync();
    if(lane==0u && epa_succeeded!=0u)
        witness=epa_finished_witness;
    return epa_succeeded!=0u;
}
