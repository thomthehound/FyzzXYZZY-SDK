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
bool epaBoxContinuation(GjkBoxPair pair,
                        GjkSupportPoint simplex[4],uint simplexCount,
                        out GjkWitness witness) {
    witness=(GjkWitness)0;
    uint vertexCount=simplexCount,facetCount=0u;
    for(uint i=0u;i<simplexCount;++i)epa_vertices[i]=simplex[i];
    if(!epaInitialize(pair,vertexCount,facetCount))
        return false;
    float epsilon=0.005f*min(
        min(pair.halfA.x,min(pair.halfA.y,pair.halfA.z)),
        min(pair.halfB.x,min(pair.halfB.y,pair.halfB.z)));
    float upperBound=3.402823466e+38f;
    for(uint iteration=0u;iteration<60u;++iteration) {
        int closest=-1;
        float distance=3.402823466e+38f;
        for(uint i=0u;i<facetCount;++i)
            if(epa_facets[i].valid!=0u && epa_facets[i].inHeap!=0u &&
               epa_facets[i].distance<distance) {
                closest=int(i);
                distance=epa_facets[i].distance;
            }
        if(closest<0)return false;
        EpaFacet selected=epa_facets[closest];
        GjkSupportPoint next=gjkSupport(pair,selected.normal);
        float projection=gjkDot(next.q,selected.normal);
        if(abs(projection-distance)<=epsilon)
            return epaContact(pair,selected,witness);
        if(vertexCount==64u)return false;
        upperBound=min(upperBound,projection);
        bool duplicate=false;
        for(uint i=0u;i<vertexCount;++i)
            if(epa_vertices[i].ai==next.ai && epa_vertices[i].bi==next.bi) {
                duplicate=true;
                break;
            }
        if(duplicate)return epaContact(pair,selected,witness);
        uint newIndex=vertexCount++;
        epa_vertices[newIndex]=next;
        uint boundaryCount=0u;
        for(uint i=0u;i<facetCount;++i) {
            EpaFacet f=epa_facets[i];
            if(f.valid==0u)continue;
            if(gjkDot(f.normal,next.q-epa_vertices[f.i].q)<0.f)continue;
            epa_facets[i].valid=0u;
            uint2 edges[3]={
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
                else {
                    if(boundaryCount==32u)return false;
                    EpaEdge entry;
                    entry.from=edges[e].x;
                    entry.to=edges[e].y;
                    epa_boundary[boundaryCount++]=entry;
                }
            }
        }
        if(boundaryCount==0u)return false;
        uint slots=64u-facetCount;
        for(uint i=0u;i<facetCount;++i)
            if(epa_facets[i].valid==0u)++slots;
        if(boundaryCount>slots)return false;
        for(uint i=0u;i<boundaryCount;++i) {
            EpaEdge edge=epa_boundary[i];
            EpaFacet f=epaFacet(edge.from,edge.to,newIndex);
            if(f.valid==0u)return false;
            f.inHeap=f.distance<=upperBound?1u:0u;
            int slot=-1;
            for(uint j=0u;j<facetCount;++j)
                if(epa_facets[j].valid==0u) {
                    slot=int(j);
                    break;
                }
            if(slot<0)slot=int(facetCount++);
            epa_facets[slot]=f;
        }
    }
    return false;
}
