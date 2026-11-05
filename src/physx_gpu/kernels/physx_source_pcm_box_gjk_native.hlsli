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
// Only the Minkowski point and feature IDs persist between iterations.
// Shape-space witnesses are reconstructed on convergence or EPA admission.
struct GjkCompactPoint {
    float3 q;
    uint ai,bi;
};
struct GjkSimplex {
    GjkCompactPoint point0,point1,point2,point3;
    uint count;
};
void gjkAppend(inout GjkSimplex simplex,GjkCompactPoint support) {
    if(simplex.count==0u)simplex.point0=support;
    else if(simplex.count==1u)simplex.point1=support;
    else if(simplex.count==2u)simplex.point2=support;
    else simplex.point3=support;
    ++simplex.count;
}
void gjkAppend(inout GjkSimplex simplex,GjkSupportPoint support) {
    GjkCompactPoint compact;
    compact.q=support.q;
    compact.ai=support.ai;
    compact.bi=support.bi;
    gjkAppend(simplex,compact);
}
struct GjkClosest {
    float3 closestPoint;
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
float3 gjkVertex(uint kind,float3 half,uint first,uint index) {
#ifndef FYZZXYZZY_GJK_BOX_ONLY
    if(kind==1u)return float3(0.f,0.f,0.f);
    if(kind==2u)
        return asfloat(hullVertices.Load3(p.hullBias+(first+index)*12u));
#endif
    return float3((index&1u)!=0u?half.x:-half.x,
                  (index&2u)!=0u?half.y:-half.y,
                  (index&4u)!=0u?half.z:-half.z);
}
GjkSupportPoint gjkRestoreSupport(GjkBoxPair pair,GjkCompactPoint compact) {
    GjkSupportPoint support;
    const float3 localA=gjkVertex(
        pair.kindA,pair.halfA,pair.firstA,compact.ai);
    support.b=gjkVertex(pair.kindB,pair.halfB,pair.firstB,compact.bi);
    support.a=pair.aInB+rotateSource(pair.relative,localA);
    support.q=compact.q;
    support.ai=compact.ai;
    support.bi=compact.bi;
    return support;
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
// Keep the triangle Voronoi classifier callable on DXIL. Its scalar
// interface prevents the vector-valued intermediate instructions that
// cannot appear in a non-inlined DXIL helper.
#ifndef __spirv__
[noinline]
#endif
uint gjkTriangleBarycentric(
    float ax,float ay,float az,float bx,float by,float bz,
    float cx,float cy,float cz,
    out float baryA,out float baryB,out float baryC) {
    baryA=1.f;
    baryB=0.f;
    baryC=0.f;
    const float abx=bx-ax,aby=by-ay,abz=bz-az;
    const float acx=cx-ax,acy=cy-ay,acz=cz-az;
    const float aa=(abx*abx+aby*aby)+abz*abz;
    const float bb=(abx*acx+aby*acy)+abz*acz;
    const float cc=(acx*acx+acy*acy)+acz*acz;
    const float scale=max(aa,cc);
    const float determinant=aa*cc-bb*bb;
    if(!(scale>0.f) ||
       !(determinant>1.1920928955078125e-7f*scale*scale) ||
       !isfinite(determinant)) {
        // Degenerate triangles reduce to their closest finite edge.
        float t=aa>0.f && isfinite(aa)
            ?clamp(-((ax*abx+ay*aby)+az*abz)/aa,0.f,1.f):0.f;
        float u=1.f-t;
        float qx=ax*u+bx*t,qy=ay*u+by*t,qz=az*u+bz*t;
        float best=(qx*qx+qy*qy)+qz*qz;
        baryA=u;baryB=t;
        uint mask=t<=0.f?1u:(t>=1.f?2u:3u);

        t=cc>0.f && isfinite(cc)
            ?clamp(-((ax*acx+ay*acy)+az*acz)/cc,0.f,1.f):0.f;
        u=1.f-t;
        qx=ax*u+cx*t;qy=ay*u+cy*t;qz=az*u+cz*t;
        float dist=(qx*qx+qy*qy)+qz*qz;
        if(dist<best) {
            best=dist;
            baryA=u;baryB=0.f;baryC=t;
            mask=t<=0.f?1u:(t>=1.f?4u:5u);
        }

        const float bcx=cx-bx,bcy=cy-by,bcz=cz-bz;
        const float bc=(bcx*bcx+bcy*bcy)+bcz*bcz;
        t=bc>0.f && isfinite(bc)
            ?clamp(-((bx*bcx+by*bcy)+bz*bcz)/bc,0.f,1.f):0.f;
        u=1.f-t;
        qx=bx*u+cx*t;qy=by*u+cy*t;qz=bz*u+cz*t;
        dist=(qx*qx+qy*qy)+qz*qz;
        if(dist<best) {
            baryA=0.f;baryB=u;baryC=t;
            mask=t<=0.f?2u:(t>=1.f?4u:6u);
        }
        return mask;
    }

    const float d1=-((abx*ax+aby*ay)+abz*az);
    const float d2=-((acx*ax+acy*ay)+acz*az);
    if(d1<=0.f && d2<=0.f)return 1u;
    const float d3=-((abx*bx+aby*by)+abz*bz);
    const float d4=-((acx*bx+acy*by)+acz*bz);
    if(d3>=0.f && d4<=d3) {
        baryA=0.f;baryB=1.f;
        return 2u;
    }
    const float vc=d1*d4-d3*d2;
    if(vc<=0.f && d1>=0.f && d3<=0.f) {
        const float t=clamp(d1/aa,0.f,1.f);
        baryA=1.f-t;baryB=t;
        return t<=0.f?1u:(t>=1.f?2u:3u);
    }
    const float d5=-((abx*cx+aby*cy)+abz*cz);
    const float d6=-((acx*cx+acy*cy)+acz*cz);
    if(d6>=0.f && d5<=d6) {
        baryA=0.f;baryC=1.f;
        return 4u;
    }
    const float vb=d5*d2-d1*d6;
    if(vb<=0.f && d2>=0.f && d6<=0.f) {
        const float t=clamp(d2/cc,0.f,1.f);
        baryA=1.f-t;baryC=t;
        return t<=0.f?1u:(t>=1.f?4u:5u);
    }
    const float va=d3*d6-d5*d4;
    if(va<=0.f && d4>=d3 && d5>=d6) {
        const float bcx=cx-bx,bcy=cy-by,bcz=cz-bz;
        const float bc=(bcx*bcx+bcy*bcy)+bcz*bcz;
        const float t=clamp(-((bx*bcx+by*bcy)+bz*bcz)/bc,0.f,1.f);
        baryA=0.f;baryB=1.f-t;baryC=t;
        return t<=0.f?2u:(t>=1.f?4u:6u);
    }
    baryB=(d1*cc-d2*bb)/determinant;
    baryC=(d2*aa-d1*bb)/determinant;
    baryA=1.f-baryB-baryC;
    return 7u;
}
GjkFeature gjkTriangleFeature(
    float3 a,float3 b,float3 c,
    float4 wa,float4 wb,float4 wc,uint ma,uint mb,uint mc) {
    float baryA,baryB,baryC;
    const uint localMask=gjkTriangleBarycentric(
        a.x,a.y,a.z,b.x,b.y,b.z,c.x,c.y,c.z,
        baryA,baryB,baryC);
    precise float3 closest=(a*baryA+b*baryB)+c*baryC;
    const float4 weights=(wa*baryA+wb*baryB)+wc*baryC;
    const uint mask=((localMask&1u)!=0u?ma:0u) |
                    ((localMask&2u)!=0u?mb:0u) |
                    ((localMask&4u)!=0u?mc:0u);
    return gjkFeature(closest,weights,mask);
}

GjkClosest gjkClosestSimplex(GjkSimplex simplex) {
    GjkClosest result=(GjkClosest)0;
    result.dist2=3.402823466e+38f;
    const uint count=simplex.count;
    if(count==0u)return result;
    const GjkCompactPoint s0=simplex.point0;
    const GjkCompactPoint s1=simplex.point1;
    const GjkCompactPoint s2=simplex.point2;
    const GjkCompactPoint s3=simplex.point3;
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
                GjkSimplex simplex,out GjkWitness witness) {
    witness=(GjkWitness)0;
    float distance=sqrt(nearest.dist2);
    if(!(distance>0.f) || !isfinite(distance) ||
       simplex.count==0u)return false;
    const GjkSupportPoint s0=gjkRestoreSupport(pair,simplex.point0);
    precise float3 pa=s0.a*nearest.weights.x;
    precise float3 pb=s0.b*nearest.weights.x;
    if(simplex.count>1u) {
        const GjkSupportPoint s1=gjkRestoreSupport(pair,simplex.point1);
        pa=pa+s1.a*nearest.weights.y;
        pb=pb+s1.b*nearest.weights.y;
    }
    if(simplex.count>2u) {
        const GjkSupportPoint s2=gjkRestoreSupport(pair,simplex.point2);
        pa=pa+s2.a*nearest.weights.z;
        pb=pb+s2.b*nearest.weights.z;
    }
    if(simplex.count>3u) {
        const GjkSupportPoint s3=gjkRestoreSupport(pair,simplex.point3);
        pa=pa+s3.a*nearest.weights.w;
        pb=pb+s3.b*nearest.weights.w;
    }
    witness.normalB=nearest.closestPoint*(1.f/distance);
    witness.localA=rotateSource(
        conjugate(pair.relative),
        pa-pair.aInB-nearest.closestPoint*
            (pair.radiusA/distance));
    witness.localB=pb+
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
    uint outcome=0u;
    uint iteration=0u;
#ifdef FYZZXYZZY_GJK_PROBE_FORCE_UNROLL
    [unroll]
#else
    [loop]
#endif
    for(;iteration<maxIterations;++iteration) {
#ifdef FYZZXYZZY_GJK_PROBE_SECOND_ENTRY
        if(iteration==1u)return 0x24u;
#endif
        float dist=sqrt(gjkLength2(closest));
        if(simplex.count>0u && dist<=epsilon && !pointCore) {
            outcome=3u;
            break;
        }
        if(dist<=0.f) {
            outcome=pointCore && simplex.count>0u?3u:0u;
            break;
        }
        float3 normal=closest*(1.f/dist);
#ifdef FYZZXYZZY_GJK_PROBE_SECOND_NORMAL
        if(iteration==1u)return 0x25u;
#endif
        GjkSupportPoint support=gjkSupport(pair,-closest);
#ifdef FYZZXYZZY_GJK_PROBE_SECOND_SUPPORT
        if(iteration==1u)return 0x21u;
#endif
        float projection=gjkDot(normal,support.q);
        if(projection>contactDistance+pair.radiusA+pair.radiusB) {
            outcome=0u;
            break;
        }
        if(simplex.count>0u && projection>dist*progressFactor) {
            GjkClosest current=gjkClosestSimplex(simplex);
            outcome=current.mask!=0u && gjkContact(pair,current,simplex,witness) &&
                    witness.separation<=contactDistance?1u:0u;
            break;
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
                outcome=current.mask!=0u &&
                        gjkContact(pair,current,simplex,witness) &&
                        witness.separation<=contactDistance?1u:0u;
            } else outcome=pointCore?3u:0u;
            break;
        }
        if(simplex.count==4u) {
            outcome=3u;
            break;
        }
        gjkAppend(simplex,support);
        GjkClosest next=gjkClosestSimplex(simplex);
#ifdef FYZZXYZZY_GJK_PROBE_SECOND_FEATURE
        if(iteration==1u)return 0x22u;
#endif
        if(next.mask==0u) {
            outcome=0u;
            break;
        }
        closest=next.closestPoint;
        float nextDist=sqrt(next.dist2);
        bool noProgressContact=false;
        if(nextDist>=previous && pointCore && projection>0.f)
            noProgressContact=gjkContact(pair,next,simplex,witness) &&
                witness.separation<=contactDistance;
        gjkCompact(simplex,next);
#ifdef FYZZXYZZY_GJK_PROBE_SECOND_COMPACT
        if(iteration==1u)return 0x23u;
#endif
        if(nextDist>=previous) {
            if(pointCore)
                outcome=projection>0.f?(noProgressContact?1u:0u):3u;
            else outcome=0u;
            break;
        }
        previous=nextDist;
    }
    return iteration==maxIterations
        ?(pointCore && simplex.count>0u?3u:0u):outcome;
}

uint gjkBoxContinuation(GjkBoxPair pair,float contactDistance,
                        out GjkWitness witness,
                        out GjkSupportPoint simplex[4],out uint count) {
    GjkSimplex state;
    const uint outcome=gjkBoxContinuationState(
        pair,contactDistance,witness,state);
    count=state.count;
    simplex[0]=(GjkSupportPoint)0;
    simplex[1]=(GjkSupportPoint)0;
    simplex[2]=(GjkSupportPoint)0;
    simplex[3]=(GjkSupportPoint)0;
    if(outcome==3u) {
        if(count>0u)simplex[0]=gjkRestoreSupport(pair,state.point0);
        if(count>1u)simplex[1]=gjkRestoreSupport(pair,state.point1);
        if(count>2u)simplex[2]=gjkRestoreSupport(pair,state.point2);
        if(count>3u)simplex[3]=gjkRestoreSupport(pair,state.point3);
    }
    return outcome;
}

uint gjkBoxContinuation(GjkBoxPair pair,float contactDistance,
                        out GjkWitness witness) {
    GjkSupportPoint simplex[4];
    uint count;
    return gjkBoxContinuation(
        pair,contactDistance,witness,simplex,count);
}
