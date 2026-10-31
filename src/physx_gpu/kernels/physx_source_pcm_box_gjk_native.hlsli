// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
//
// GuGJKPenetration box support and GJK terminal cases. EPA admission is
// returned explicitly; no substitute penetration witness is fabricated.

struct GjkBoxPair {
    float3 halfA,halfB,aInB;
    float4 relative;
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
float gjkLength2(float3 v) { return dot(v,v); }

GjkSupportPoint gjkSupport(GjkBoxPair pair,float3 direction) {
    float3 directionA=rotateSource(conjugate(pair.relative),direction);
    bool3 positiveA=directionA>0.f,positiveB=-direction>0.f;
    GjkSupportPoint s;
    s.ai=(positiveA.x?1u:0u)|(positiveA.y?2u:0u)|(positiveA.z?4u:0u);
    s.bi=(positiveB.x?1u:0u)|(positiveB.y?2u:0u)|(positiveB.z?4u:0u);
    float3 localA=float3(
        positiveA.x?pair.halfA.x:-pair.halfA.x,
        positiveA.y?pair.halfA.y:-pair.halfA.y,
        positiveA.z?pair.halfA.z:-pair.halfA.z);
    s.b=float3(
        positiveB.x?pair.halfB.x:-pair.halfB.x,
        positiveB.y?pair.halfB.y:-pair.halfB.y,
        positiveB.z?pair.halfB.z:-pair.halfB.z);
    s.a=pair.aInB+rotateSource(pair.relative,localA);
    s.q=s.a-s.b;
    return s;
}
bool gjkSolve(uint n,inout float a[3][4],out float result[3]) {
    float scale=0.f;
    for(uint i=0u;i<n;++i)scale=max(scale,abs(a[i][i]));
    if(!(scale>0.f) || !isfinite(scale))return false;
    const float tolerance=1.1920928955078125e-7f*scale;
    for(uint col=0u;col<n;++col) {
        uint pivot=col;
        for(uint row=col+1u;row<n;++row)
            if(abs(a[row][col])>abs(a[pivot][col]))pivot=row;
        if(!(abs(a[pivot][col])>tolerance))return false;
        for(uint j=col;j<=n;++j) {
            float swap=a[col][j];
            a[col][j]=a[pivot][j];
            a[pivot][j]=swap;
        }
        float divisor=a[col][col];
        for(uint j=col;j<=n;++j)a[col][j]/=divisor;
        for(uint row=0u;row<n;++row)if(row!=col) {
            float factor=a[row][col];
            for(uint j=col;j<=n;++j)a[row][j]-=factor*a[col][j];
        }
    }
    for(uint i=0u;i<3u;++i)
        result[i]=i<n?a[i][n]:0.f;
    return true;
}
GjkClosest gjkClosestSimplex(GjkSupportPoint simplex[4],uint count) {
    GjkClosest best;
    best.closestPoint=0.f;best.a=0.f;best.b=0.f;
    best.mask=0u;
    best.dist2=3.402823466e+38f;
    for(uint i=0u;i<4u;++i)best.weights[i]=0.f;
    for(uint mask=1u;mask<(1u<<count);++mask) {
        uint ids[4]={0u,0u,0u,0u};
        uint n=0u;
        for(uint i=0u;i<count;++i)
            if((mask&(1u<<i))!=0u)ids[n++]=i;
        float weights[4]={0.f,0.f,0.f,0.f};
        weights[ids[0]]=1.f;
        if(n>1u) {
            float3 origin=simplex[ids[0]].q;
            float3 directions[3];
            float a[3][4]={{0.f,0.f,0.f,0.f},
                           {0.f,0.f,0.f,0.f},
                           {0.f,0.f,0.f,0.f}};
            for(uint j=1u;j<n;++j)
                directions[j-1u]=simplex[ids[j]].q-origin;
            for(uint j=1u;j<n;++j) {
                for(uint k=1u;k<n;++k)
                    a[j-1u][k-1u]=dot(directions[j-1u],directions[k-1u]);
                a[j-1u][n-1u]=-dot(directions[j-1u],origin);
            }
            float coeff[3]={0.f,0.f,0.f};
            if(!gjkSolve(n-1u,a,coeff))continue;
            for(uint j=1u;j<n;++j) {
                weights[ids[j]]=coeff[j-1u];
                weights[ids[0]]-=coeff[j-1u];
            }
        }
        bool feasible=true;
        for(uint j=0u;j<n;++j)
            if(weights[ids[j]]< -1.e-5f)feasible=false;
        if(!feasible)continue;
        float3 q=0.f,pa=0.f,pb=0.f;
        for(uint j=0u;j<count;++j) {
            q+=simplex[j].q*weights[j];
            pa+=simplex[j].a*weights[j];
            pb+=simplex[j].b*weights[j];
        }
        float d=gjkLength2(q);
        if(d<best.dist2) {
            best.closestPoint=q;best.a=pa;best.b=pb;
            best.mask=mask;best.dist2=d;
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
    witness.normalB=nearest.closestPoint/distance;
    witness.localA=rotateSource(
        conjugate(pair.relative),nearest.a-pair.aInB);
    witness.localB=nearest.b;
    witness.separation=distance;
    return true;
}
// 0: no source GJK contact; 1: source GJK contact;
// 3: simplex requires the source EPA continuation.
uint gjkBoxContinuation(GjkBoxPair pair,float contactDistance,
                        out GjkWitness witness) {
    witness=(GjkWitness)0;
    const float epsilon=0.005f*min(
        min(pair.halfA.x,min(pair.halfA.y,pair.halfA.z)),
        min(pair.halfB.x,min(pair.halfB.y,pair.halfB.z)));
    const float progressFactor=1.f-0.000225f;
    GjkSupportPoint simplex[4];
    uint count=0u;
    float3 closest=gjkLength2(pair.aInB)>0.f
        ?pair.aInB:float3(1.f,0.f,0.f);
    float previous=3.402823466e+38f;
    for(uint iteration=0u;iteration<32u;++iteration) {
        float dist=sqrt(gjkLength2(closest));
        if(count>0u && dist<=epsilon)return 3u;
        if(dist<=0.f)return 0u;
        float3 normal=closest/dist;
        GjkSupportPoint support=gjkSupport(pair,-closest);
        float projection=dot(normal,support.q);
        if(projection>contactDistance)return 0u;
        if(count>0u && projection>dist*progressFactor) {
            GjkClosest current=gjkClosestSimplex(simplex,count);
            return current.mask!=0u && gjkContact(pair,current,witness)
                ?1u:0u;
        }
        for(uint j=0u;j<count;++j)
            if(support.ai==simplex[j].ai &&
               support.bi==simplex[j].bi)
                return 0u;
        if(count==4u)return 3u;
        simplex[count++]=support;
        GjkClosest next=gjkClosestSimplex(simplex,count);
        if(next.mask==0u)return 0u;
        closest=next.closestPoint;
        float nextDist=sqrt(next.dist2);
        gjkCompact(simplex,count,next);
        if(nextDist>=previous)return 0u;
        previous=nextDist;
    }
    return 0u;
}
