// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// GuEPA box continuation. Large EPA facet/vertex storage is private to
// sparse indirect threads; the ordinary box SAT/GJK kernel has none.

struct EpaFacet {
    uint i,j,k;
    float3 normal;
    float distance;
    uint valid,inHeap;
};
struct EpaEdge { uint from,to; };

EpaFacet epaFacet(GjkSupportPoint vertices[64],uint i,uint j,uint k) {
    EpaFacet result=(EpaFacet)0;
    float3 a=vertices[i].q;
    float3 e0=vertices[j].q-a,e1=vertices[k].q-a;
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
    GjkBoxPair pair,inout GjkSupportPoint vertices[64],
    inout uint count,inout EpaFacet facets[64],out uint facetCount) {
    facetCount=0u;
    if(count==0u || count>4u)return false;
    if(count==1u) {
        GjkSupportPoint next=gjkSupport(pair,float3(1.f,0.f,0.f));
        if(gjkLength2(next.q-vertices[0].q)==0.f)return false;
        vertices[count++]=next;
    }
    if(count==2u) {
        float3 edge=vertices[1].q-vertices[0].q;
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
        vertices[count++]=gjkSupport(pair,perpendicular);
    }
    if(count==3u) {
        EpaFacet f0=epaFacet(vertices,0u,1u,2u);
        EpaFacet f1=epaFacet(vertices,1u,0u,2u);
        if(f0.valid==0u || f1.valid==0u)return false;
        facets[facetCount++]=f0;
        facets[facetCount++]=f1;
        return true;
    }
    float3 p0=vertices[0].q;
    if(gjkDot(cross(vertices[1].q-p0,vertices[2].q-p0),
              vertices[3].q-p0)>0.f) {
        GjkSupportPoint swap=vertices[1];
        vertices[1]=vertices[2];
        vertices[2]=swap;
    }
    uint3 faces[4]={
        uint3(0u,1u,2u),uint3(0u,3u,1u),
        uint3(0u,2u,3u),uint3(1u,3u,2u)};
    for(uint i=0u;i<4u;++i) {
        EpaFacet f=epaFacet(vertices,faces[i].x,faces[i].y,faces[i].z);
        if(f.valid==0u)return false;
        facets[facetCount++]=f;
    }
    return true;
}

bool epaContact(GjkBoxPair pair,GjkSupportPoint vertices[64],
                EpaFacet facet,out GjkWitness witness) {
    witness=(GjkWitness)0;
    GjkSupportPoint a=vertices[facet.i];
    GjkSupportPoint b=vertices[facet.j];
    GjkSupportPoint c=vertices[facet.k];
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
    witness.localA=rotateSource(conjugate(pair.relative),pa-pair.aInB);
    witness.localB=pb;
    witness.normalB=-facet.normal;
    witness.separation=-depth;
    return sqrt(gjkLength2(pa-pb))<=depth+1.e-3f;
}
bool epaBoxContinuation(GjkBoxPair pair,
                        GjkSupportPoint simplex[4],uint simplexCount,
                        out GjkWitness witness) {
    witness=(GjkWitness)0;
    GjkSupportPoint vertices[64];
    EpaFacet facets[64];
    EpaEdge boundary[32];
    uint vertexCount=simplexCount,facetCount=0u;
    for(uint i=0u;i<simplexCount;++i)vertices[i]=simplex[i];
    if(!epaInitialize(pair,vertices,vertexCount,facets,facetCount))
        return false;
    float epsilon=0.005f*min(
        min(pair.halfA.x,min(pair.halfA.y,pair.halfA.z)),
        min(pair.halfB.x,min(pair.halfB.y,pair.halfB.z)));
    float upperBound=3.402823466e+38f;
    for(uint iteration=0u;iteration<60u;++iteration) {
        int closest=-1;
        float distance=3.402823466e+38f;
        for(uint i=0u;i<facetCount;++i)
            if(facets[i].valid!=0u && facets[i].inHeap!=0u &&
               facets[i].distance<distance) {
                closest=int(i);
                distance=facets[i].distance;
            }
        if(closest<0)return false;
        EpaFacet selected=facets[closest];
        GjkSupportPoint next=gjkSupport(pair,selected.normal);
        float projection=gjkDot(next.q,selected.normal);
        if(abs(projection-distance)<=epsilon)
            return epaContact(pair,vertices,selected,witness);
        if(vertexCount==64u)return false;
        upperBound=min(upperBound,projection);
        bool duplicate=false;
        for(uint i=0u;i<vertexCount;++i)
            if(vertices[i].ai==next.ai && vertices[i].bi==next.bi) {
                duplicate=true;
                break;
            }
        if(duplicate)return epaContact(pair,vertices,selected,witness);
        uint newIndex=vertexCount++;
        vertices[newIndex]=next;
        uint boundaryCount=0u;
        for(uint i=0u;i<facetCount;++i) {
            EpaFacet f=facets[i];
            if(f.valid==0u)continue;
            if(gjkDot(f.normal,next.q-vertices[f.i].q)<0.f)continue;
            facets[i].valid=0u;
            uint2 edges[3]={
                uint2(f.i,f.j),uint2(f.j,f.k),uint2(f.k,f.i)};
            for(uint e=0u;e<3u;++e) {
                int reverse=-1;
                for(uint j=0u;j<boundaryCount;++j)
                    if(boundary[j].from==edges[e].y &&
                       boundary[j].to==edges[e].x) {
                        reverse=int(j);
                        break;
                    }
                if(reverse>=0)
                    boundary[reverse]=boundary[--boundaryCount];
                else {
                    if(boundaryCount==32u)return false;
                    EpaEdge entry;
                    entry.from=edges[e].x;
                    entry.to=edges[e].y;
                    boundary[boundaryCount++]=entry;
                }
            }
        }
        if(boundaryCount==0u)return false;
        uint slots=64u-facetCount;
        for(uint i=0u;i<facetCount;++i)
            if(facets[i].valid==0u)++slots;
        if(boundaryCount>slots)return false;
        for(uint i=0u;i<boundaryCount;++i) {
            EpaEdge edge=boundary[i];
            EpaFacet f=epaFacet(vertices,edge.from,edge.to,newIndex);
            if(f.valid==0u)return false;
            f.inHeap=f.distance<=upperBound?1u:0u;
            int slot=-1;
            for(uint j=0u;j<facetCount;++j)
                if(facets[j].valid==0u) {
                    slot=int(j);
                    break;
                }
            if(slot<0)slot=int(facetCount++);
            facets[slot]=f;
        }
    }
    return false;
}
