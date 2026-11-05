// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
//
// GuGJKPenetration box support and GJK terminal cases. EPA admission is
// returned explicitly; no substitute penetration witness is fabricated.

struct GjkBoxPair {
    float3 halfA,halfB,aInB;
    float4 relative;
    uint kindA,kindB,firstA,countA,firstB,countB;
    float radiusA,radiusB;
};
struct GjkSupportPoint {
    float3 a,b,q;
    uint ai,bi;
};
struct GjkClosest {
    float3 closestPoint,a,b;
    float weights[4];
    uint mask;
    float dist2;
};
struct GjkWitness {
    float3 localA,localB,normalB;
    float separation;
};
float gjkDot(float3 a,float3 b) {
    precise float x=a.x*b.x;
    precise float y=a.y*b.y;
    precise float z=a.z*b.z;
    precise float xy=x+y;
    return xy+z;
}
float gjkLength2(float3 v) { return gjkDot(v,v); }

#ifndef FYZZXYZZY_GJK_BOX_ONLY
float3 gjkShapeSupport(uint kind,float3 half,float3 direction,
                       uint first,uint count,out uint index) {
    index=0u;
    if(kind==1u)return float3(0.f,0.f,0.f);
    if(kind==2u) {
        float3 bestVertex=asfloat(hullVertices.Load3(
            p.hullBias+first*12u));
        float best=gjkDot(bestVertex,direction);
        for(uint i=1u;i<count;++i) {
            const float3 vertex=asfloat(hullVertices.Load3(
                p.hullBias+(first+i)*12u));
            const float candidate=gjkDot(vertex,direction);
            if(candidate>best) {
                best=candidate;
                bestVertex=vertex;
                index=i;
            }
        }
        return bestVertex;
    }
    bool3 positive=direction>0.f;
    index=(positive.x?1u:0u)|(positive.y?2u:0u)|(positive.z?4u:0u);
    return float3(positive.x?half.x:-half.x,
                  positive.y?half.y:-half.y,
                  positive.z?half.z:-half.z);
}
#endif

GjkSupportPoint gjkSupport(GjkBoxPair pair,float3 direction) {
    const float3 directionA=rotateSource(
        conjugate(pair.relative),direction);
    GjkSupportPoint s;
#ifdef FYZZXYZZY_GJK_BOX_ONLY
    const bool3 positiveA=directionA>0.f;
    const bool3 positiveB=-direction>0.f;
    s.ai=(positiveA.x?1u:0u)|(positiveA.y?2u:0u)|(positiveA.z?4u:0u);
    s.bi=(positiveB.x?1u:0u)|(positiveB.y?2u:0u)|(positiveB.z?4u:0u);
    const float3 localA=float3(
        positiveA.x?pair.halfA.x:-pair.halfA.x,
        positiveA.y?pair.halfA.y:-pair.halfA.y,
        positiveA.z?pair.halfA.z:-pair.halfA.z);
    s.b=float3(
        positiveB.x?pair.halfB.x:-pair.halfB.x,
        positiveB.y?pair.halfB.y:-pair.halfB.y,
        positiveB.z?pair.halfB.z:-pair.halfB.z);
#else
    const float3 localA=gjkShapeSupport(
        pair.kindA,pair.halfA,directionA,
        pair.firstA,pair.countA,s.ai);
    s.b=gjkShapeSupport(
        pair.kindB,pair.halfB,-direction,
        pair.firstB,pair.countB,s.bi);
#endif
    s.a=pair.aInB+rotateSource(pair.relative,localA);
    s.q=s.a-s.b;
    return s;
}
// Closest-origin features have at most three affine directions.
// Explicit barycentric solves avoid a dynamically indexed elimination
// matrix in the GPU continuation while preserving mask order.
GjkClosest gjkClosestSimplex(GjkSupportPoint simplex[4],uint count) {
    GjkClosest best=(GjkClosest)0;
    best.dist2=3.402823466e+38f;
    for(uint mask=1u;mask<(1u<<count);++mask) {
        uint ids[4]={0u,0u,0u,0u};
        uint n=0u;
        for(uint i=0u;i<count;++i)
            if((mask&(1u<<i))!=0u)ids[n++]=i;
        float weights[4]={0.f,0.f,0.f,0.f};
        weights[ids[0]]=1.f;
        if(n>1u) {
            const float3 origin=simplex[ids[0]].q;
            const float3 d1=simplex[ids[1]].q-origin;
            const float a=gjkDot(d1,d1);
            const float x=-gjkDot(d1,origin);
            const float tolerance=1.1920928955078125e-7f;
            float u=0.f,v=0.f,w=0.f;
            if(n==2u) {
                if(!(a>0.f) || !isfinite(a))continue;
                u=x/a;
            }else {
                const float3 d2=simplex[ids[2]].q-origin;
                const float b=gjkDot(d1,d2);
                const float c=gjkDot(d2,d2);
                const float y=-gjkDot(d2,origin);
                if(n==3u) {
                    const float scale=max(a,c);
                    const float determinant=a*c-b*b;
                    if(!(scale>0.f) ||
                       !(determinant>tolerance*scale*scale) ||
                       !isfinite(determinant))continue;
                    u=(x*c-y*b)/determinant;
                    v=(y*a-x*b)/determinant;
                }else {
                    const float3 d3=simplex[ids[3]].q-origin;
                    const float3 cross23=cross(d2,d3);
                    const float determinant=gjkDot(d1,cross23);
                    const float scale=max(a,max(c,gjkDot(d3,d3)));
                    if(!(scale>0.f) ||
                       !(determinant*determinant>
                         tolerance*scale*scale*scale) ||
                       !isfinite(determinant))continue;
                    const float3 minusOrigin=-origin;
                    u=gjkDot(minusOrigin,cross23)/determinant;
                    v=gjkDot(minusOrigin,cross(d3,d1))/determinant;
                    w=gjkDot(minusOrigin,cross(d1,d2))/determinant;
                }
            }
            weights[ids[0]]=1.f-u-v-w;
            weights[ids[1]]=u;
            if(n>=3u)weights[ids[2]]=v;
            if(n==4u)weights[ids[3]]=w;
        }
        bool feasible=true;
        for(uint j=0u;j<n;++j)
            if(weights[ids[j]]< -1.e-5f)feasible=false;
        if(!feasible)continue;
        precise float3 q=0.f,pa=0.f,pb=0.f;
        for(uint j=0u;j<count;++j) {
            precise float3 dq=simplex[j].q*weights[j];
            precise float3 da=simplex[j].a*weights[j];
            precise float3 db=simplex[j].b*weights[j];
            q=q+dq;
            pa=pa+da;
            pb=pb+db;
        }
        const float distance2=gjkLength2(q);
        if(distance2<best.dist2) {
            best.closestPoint=q;
            best.a=pa;
            best.b=pb;
            best.mask=mask;
            best.dist2=distance2;
            for(uint j=0u;j<4u;++j)best.weights[j]=weights[j];
        }
    }
    return best;
}
void gjkCompact(inout GjkSupportPoint simplex[4],
                inout uint count,GjkClosest closest) {
    uint write=0u;
    for(uint read=0u;read<count;++read)
        if((closest.mask&(1u<<read))!=0u &&
           closest.weights[read]>1.e-6f)
            simplex[write++]=simplex[read];
    if(write==0u && count>0u)write=1u;
    count=write;
}
bool gjkContact(GjkBoxPair pair,GjkClosest nearest,
                out GjkWitness witness) {
    witness=(GjkWitness)0;
    float distance=sqrt(nearest.dist2);
    if(!(distance>0.f) || !isfinite(distance))return false;
    witness.normalB=nearest.closestPoint*(1.f/distance);
    witness.localA=rotateSource(
        conjugate(pair.relative),
        nearest.a-pair.aInB-nearest.closestPoint*
            (pair.radiusA/distance));
    witness.localB=nearest.b+
        nearest.closestPoint*(pair.radiusB/distance);
    witness.separation=distance-pair.radiusA-pair.radiusB;
    return true;
}
// 0: no source GJK contact; 1: source GJK contact;
// 3: simplex requires the source EPA continuation.
uint gjkBoxContinuation(GjkBoxPair pair,float contactDistance,
                        out GjkWitness witness,
                        out GjkSupportPoint simplex[4],out uint count) {
    witness=(GjkWitness)0;
    const float epsilon=0.005f*min(
        min(pair.halfA.x,min(pair.halfA.y,pair.halfA.z)),
        min(pair.halfB.x,min(pair.halfB.y,pair.halfB.z)));
    const float progressFactor=1.f-0.000225f;
    const bool pointCore=pair.radiusA>0.f || pair.radiusB>0.f;
    count=0u;
    float3 closest=gjkLength2(pair.aInB)>0.f
        ?pair.aInB:float3(1.f,0.f,0.f);
    float previous=3.402823466e+38f;
    for(uint iteration=0u;iteration<32u;++iteration) {
        float dist=sqrt(gjkLength2(closest));
        if(count>0u && dist<=epsilon && !pointCore)return 3u;
        if(dist<=0.f)return pointCore && count>0u?3u:0u;
        float3 normal=closest*(1.f/dist);
        GjkSupportPoint support=gjkSupport(pair,-closest);
        float projection=gjkDot(normal,support.q);
        if(projection>contactDistance+pair.radiusA+pair.radiusB)
            return 0u;
        if(count>0u && projection>dist*progressFactor) {
            GjkClosest current=gjkClosestSimplex(simplex,count);
            return current.mask!=0u && gjkContact(pair,current,witness) &&
                   witness.separation<=contactDistance
                ?1u:0u;
        }
        for(uint j=0u;j<count;++j)
            if(support.ai==simplex[j].ai &&
               support.bi==simplex[j].bi) {
                if(projection>0.f) {
                    const GjkClosest current=gjkClosestSimplex(simplex,count);
                    return current.mask!=0u &&
                           gjkContact(pair,current,witness) &&
                           witness.separation<=contactDistance?1u:0u;
                }
                return pointCore?3u:0u;
            }
        if(count==4u)return 3u;
        simplex[count++]=support;
        GjkClosest next=gjkClosestSimplex(simplex,count);
        if(next.mask==0u)return 0u;
        closest=next.closestPoint;
        float nextDist=sqrt(next.dist2);
        gjkCompact(simplex,count,next);
        if(nextDist>=previous) {
            if(pointCore) {
                if(projection>0.f)
                    return gjkContact(pair,next,witness) &&
                           witness.separation<=contactDistance?1u:0u;
                return 3u;
            }
            return 0u;
        }
        previous=nextDist;
    }
    return pointCore && count>0u?3u:0u;
}

uint gjkBoxContinuation(GjkBoxPair pair,float contactDistance,
                        out GjkWitness witness) {
    GjkSupportPoint simplex[4];
    uint count;
    return gjkBoxContinuation(
        pair,contactDistance,witness,simplex,count);
}
