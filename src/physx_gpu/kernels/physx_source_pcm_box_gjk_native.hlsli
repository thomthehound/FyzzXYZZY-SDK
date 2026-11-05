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
// The simplex is carried in fixed fields rather than a dynamically
// indexed private array across the GJK loop.
struct GjkSimplex {
    GjkSupportPoint point0,point1,point2,point3;
    uint count;
};
void gjkAppend(inout GjkSimplex simplex,GjkSupportPoint support) {
    if(simplex.count==0u)simplex.point0=support;
    else if(simplex.count==1u)simplex.point1=support;
    else if(simplex.count==2u)simplex.point2=support;
    else simplex.point3=support;
    ++simplex.count;
}
struct GjkClosest {
    float3 closestPoint,a,b;
    float4 weights;
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
// Closest-origin Voronoi features. Each simplex dimension has a fixed
// feature calculation; no subset masks, dynamic matrix, or variable-size
// simplex search is needed in the GPU inner loop.
struct GjkFeature {
    float3 closest;
    float4 weights;
    uint mask;
    float dist2;
};
GjkFeature gjkFeature(float3 closest,float4 weights,uint mask) {
    GjkFeature result;
    result.closest=closest;
    result.weights=weights;
    result.mask=mask;
    result.dist2=gjkLength2(closest);
    return result;
}
GjkFeature gjkChoose(GjkFeature previous,GjkFeature candidate) {
    if(candidate.dist2<previous.dist2)return candidate;
    return previous;
}
GjkFeature gjkEdgeFeature(
    float3 a,float3 b,float4 wa,float4 wb,uint ma,uint mb) {
    const float3 ab=b-a;
    const float denom=gjkLength2(ab);
    if(!(denom>0.f) || !isfinite(denom))
        return gjkChoose(gjkFeature(a,wa,ma),gjkFeature(b,wb,mb));
    const float t=clamp(-gjkDot(a,ab)/denom,0.f,1.f);
    const float u=1.f-t;
    precise float3 closest=a*u+b*t;
    const float4 weights=wa*u+wb*t;
    const uint mask=t<=0.f?ma:(t>=1.f?mb:(ma|mb));
    return gjkFeature(closest,weights,mask);
}
GjkFeature gjkTriangleFeature(
    float3 a,float3 b,float3 c,
    float4 wa,float4 wb,float4 wc,uint ma,uint mb,uint mc) {
    GjkFeature closest=gjkEdgeFeature(a,b,wa,wb,ma,mb);
    closest=gjkChoose(closest,gjkEdgeFeature(a,c,wa,wc,ma,mc));
    closest=gjkChoose(closest,gjkEdgeFeature(b,c,wb,wc,mb,mc));
    const float3 ab=b-a,ac=c-a;
    const float aa=gjkLength2(ab);
    const float bb=gjkDot(ab,ac);
    const float cc=gjkLength2(ac);
    const float scale=max(aa,cc);
    const float determinant=aa*cc-bb*bb;
    if(scale>0.f &&
       determinant>1.1920928955078125e-7f*scale*scale &&
       isfinite(determinant)) {
        const float x=-gjkDot(ab,a);
        const float y=-gjkDot(ac,a);
        const float u=(x*cc-y*bb)/determinant;
        const float v=(y*aa-x*bb)/determinant;
        const float w=1.f-u-v;
        if(u>=-1.e-5f && v>=-1.e-5f && w>=-1.e-5f) {
            precise float3 projected=a*w+b*u+c*v;
            const float4 weights=wa*w+wb*u+wc*v;
            closest=gjkChoose(closest,
                gjkFeature(projected,weights,ma|mb|mc));
        }
    }
    return closest;
}
GjkClosest gjkClosestSimplex(GjkSimplex simplex) {
    GjkClosest result=(GjkClosest)0;
    result.dist2=3.402823466e+38f;
    const uint count=simplex.count;
    if(count==0u)return result;
    const GjkSupportPoint s0=simplex.point0;
    const GjkSupportPoint s1=simplex.point1;
    const GjkSupportPoint s2=simplex.point2;
    const GjkSupportPoint s3=simplex.point3;
    const float4 w0=float4(1.f,0.f,0.f,0.f);
    const float4 w1=float4(0.f,1.f,0.f,0.f);
    const float4 w2=float4(0.f,0.f,1.f,0.f);
    const float4 w3=float4(0.f,0.f,0.f,1.f);
    GjkFeature best=gjkFeature(s0.q,w0,1u);
    if(count==2u)
        best=gjkEdgeFeature(s0.q,s1.q,w0,w1,1u,2u);
    else if(count==3u)
        best=gjkTriangleFeature(
            s0.q,s1.q,s2.q,w0,w1,w2,1u,2u,4u);
    else if(count==4u) {
        best=gjkTriangleFeature(
            s0.q,s1.q,s2.q,w0,w1,w2,1u,2u,4u);
        best=gjkChoose(best,gjkTriangleFeature(
            s0.q,s1.q,s3.q,w0,w1,w3,1u,2u,8u));
        best=gjkChoose(best,gjkTriangleFeature(
            s0.q,s2.q,s3.q,w0,w2,w3,1u,4u,8u));
        best=gjkChoose(best,gjkTriangleFeature(
            s1.q,s2.q,s3.q,w1,w2,w3,2u,4u,8u));

        const float3 d1=s1.q-s0.q,d2=s2.q-s0.q,d3=s3.q-s0.q;
        const float3 cross23=cross(d2,d3);
        const float determinant=gjkDot(d1,cross23);
        const float scale=max(gjkLength2(d1),
            max(gjkLength2(d2),gjkLength2(d3)));
        if(scale>0.f &&
           determinant*determinant>
               1.1920928955078125e-7f*scale*scale*scale &&
           isfinite(determinant)) {
            const float3 target=-s0.q;
            const float u=gjkDot(target,cross23)/determinant;
            const float v=gjkDot(target,cross(d3,d1))/determinant;
            const float w=gjkDot(target,cross(d1,d2))/determinant;
            const float t=1.f-u-v-w;
            if(u>=-1.e-5f && v>=-1.e-5f &&
               w>=-1.e-5f && t>=-1.e-5f) {
                precise float3 closest=((s0.q*t+s1.q*u)+s2.q*v)+s3.q*w;
                best=gjkChoose(best,
                    gjkFeature(closest,float4(t,u,v,w),15u));
            }
        }
    }

    result.mask=best.mask;
    result.closestPoint=best.closest;
    result.dist2=best.dist2;
    result.weights=best.weights;
    precise float3 pa=s0.a*best.weights.x;
    precise float3 pb=s0.b*best.weights.x;
    if(count>1u) {
        pa=pa+s1.a*best.weights.y;
        pb=pb+s1.b*best.weights.y;
    }
    if(count>2u) {
        pa=pa+s2.a*best.weights.z;
        pb=pb+s2.b*best.weights.z;
    }
    if(count>3u) {
        pa=pa+s3.a*best.weights.w;
        pb=pb+s3.b*best.weights.w;
    }
    result.a=pa;
    result.b=pb;
    return result;
}
void gjkCompact(inout GjkSimplex simplex,GjkClosest closest) {
    GjkSimplex packed=(GjkSimplex)0;
    if(simplex.count>0u && (closest.mask&1u)!=0u &&
       closest.weights.x>1.e-6f)
        gjkAppend(packed,simplex.point0);
    if(simplex.count>1u && (closest.mask&2u)!=0u &&
       closest.weights.y>1.e-6f)
        gjkAppend(packed,simplex.point1);
    if(simplex.count>2u && (closest.mask&4u)!=0u &&
       closest.weights.z>1.e-6f)
        gjkAppend(packed,simplex.point2);
    if(simplex.count>3u && (closest.mask&8u)!=0u &&
       closest.weights.w>1.e-6f)
        gjkAppend(packed,simplex.point3);
    if(packed.count==0u && simplex.count>0u)
        gjkAppend(packed,simplex.point0);
    simplex=packed;
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
uint gjkBoxContinuationState(GjkBoxPair pair,float contactDistance,
                             out GjkWitness witness,
                             out GjkSimplex simplex) {
    witness=(GjkWitness)0;
    const float epsilon=0.005f*min(
        min(pair.halfA.x,min(pair.halfA.y,pair.halfA.z)),
        min(pair.halfB.x,min(pair.halfB.y,pair.halfB.z)));
    const float progressFactor=1.f-0.000225f;
    const bool pointCore=pair.radiusA>0.f || pair.radiusB>0.f;
    simplex=(GjkSimplex)0;
    float3 closest=gjkLength2(pair.aInB)>0.f
        ?pair.aInB:float3(1.f,0.f,0.f);
    float previous=3.402823466e+38f;
#ifdef FYZZXYZZY_GJK_PROBE_ITERATION
    const uint maxIterations=FYZZXYZZY_GJK_PROBE_ITERATION;
#else
    const uint maxIterations=32u;
#endif
    [loop]
    for(uint iteration=0u;iteration<maxIterations;++iteration) {
#ifdef FYZZXYZZY_GJK_PROBE_SECOND_ENTRY
        if(iteration==1u)return 0x24u;
#endif
        float dist=sqrt(gjkLength2(closest));
        if(simplex.count>0u && dist<=epsilon && !pointCore)return 3u;
        if(dist<=0.f)return pointCore && simplex.count>0u?3u:0u;
        float3 normal=closest*(1.f/dist);
#ifdef FYZZXYZZY_GJK_PROBE_SECOND_NORMAL
        if(iteration==1u)return 0x25u;
#endif
        GjkSupportPoint support=gjkSupport(pair,-closest);
#ifdef FYZZXYZZY_GJK_PROBE_SECOND_SUPPORT
        if(iteration==1u)return 0x21u;
#endif
        float projection=gjkDot(normal,support.q);
        if(projection>contactDistance+pair.radiusA+pair.radiusB)
            return 0u;
        if(simplex.count>0u && projection>dist*progressFactor) {
            GjkClosest current=gjkClosestSimplex(simplex);
            return current.mask!=0u && gjkContact(pair,current,witness) &&
                   witness.separation<=contactDistance
                ?1u:0u;
        }
        const bool duplicate=
            (simplex.count>0u && support.ai==simplex.point0.ai &&
             support.bi==simplex.point0.bi) ||
            (simplex.count>1u && support.ai==simplex.point1.ai &&
             support.bi==simplex.point1.bi) ||
            (simplex.count>2u && support.ai==simplex.point2.ai &&
             support.bi==simplex.point2.bi) ||
            (simplex.count>3u && support.ai==simplex.point3.ai &&
             support.bi==simplex.point3.bi);
        if(duplicate) {
            if(projection>0.f) {
                const GjkClosest current=gjkClosestSimplex(simplex);
                return current.mask!=0u &&
                       gjkContact(pair,current,witness) &&
                       witness.separation<=contactDistance?1u:0u;
            }
            return pointCore?3u:0u;
        }
        if(simplex.count==4u)return 3u;
        gjkAppend(simplex,support);
        GjkClosest next=gjkClosestSimplex(simplex);
#ifdef FYZZXYZZY_GJK_PROBE_SECOND_FEATURE
        if(iteration==1u)return 0x22u;
#endif
        if(next.mask==0u)return 0u;
        closest=next.closestPoint;
        float nextDist=sqrt(next.dist2);
        gjkCompact(simplex,next);
#ifdef FYZZXYZZY_GJK_PROBE_SECOND_COMPACT
        if(iteration==1u)return 0x23u;
#endif
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
    return pointCore && simplex.count>0u?3u:0u;
}

uint gjkBoxContinuation(GjkBoxPair pair,float contactDistance,
                        out GjkWitness witness,
                        out GjkSupportPoint simplex[4],out uint count) {
    GjkSimplex state;
    const uint outcome=gjkBoxContinuationState(
        pair,contactDistance,witness,state);
    count=state.count;
    simplex[0]=state.point0;
    simplex[1]=state.point1;
    simplex[2]=state.point2;
    simplex[3]=state.point3;
    return outcome;
}

uint gjkBoxContinuation(GjkBoxPair pair,float contactDistance,
                        out GjkWitness witness) {
    GjkSupportPoint simplex[4];
    uint count;
    return gjkBoxContinuation(
        pair,contactDistance,witness,simplex,count);
}
