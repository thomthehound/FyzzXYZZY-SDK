// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// GuPCMContactBoxBox: source box SAT, reference-face selection and
// incident-polygon clipping. Manifold persistence belongs to Sc's manager.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer transforms : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer geometry : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer pairs : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer results : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer candidates : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer edgeCaches : register(u5,space0);

struct Parameters {
    uint pairCount;
    uint transformBias;
    uint geometryBias;
    uint pairBias;
    uint resultBias;
    uint candidateBias;
    uint edgeCacheBias;
    uint shapeCapacity;
    uint candidateCapacity;
    uint edgeCapacity;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

struct Frame {
    float3 x,y,z,position;
};
float3 rotateSource(float4 q,float3 v) {
    float3 u=q.xyz;
    precise float3 t=cross(u,v)*2.f;
    precise float3 rotated=v+(t*q.w+cross(u,t));
    return rotated;
}
float3x3 quaternionBasis(float4 q) {
    // GuPCMContactBoxBox builds axes by rotating unit vectors. Retain
    // that operation order where incident-face alignment is nearly tied.
    return float3x3(
        rotateSource(q,float3(1.f,0.f,0.f)),
        rotateSource(q,float3(0.f,1.f,0.f)),
        rotateSource(q,float3(0.f,0.f,1.f)));
}
float3 apply(float3x3 basis,float3 v) {
    return basis[0]*v.x+basis[1]*v.y+basis[2]*v.z;
}
float3 local(float3x3 basis,float3 position,float3 world) {
    float3 v=world-position;
    return float3(dot(v,basis[0]),dot(v,basis[1]),dot(v,basis[2]));
}
float3 project(Frame f,float3 world) {
    float3 v=world-f.position;
    return float3(dot(v,f.x),dot(v,f.y),dot(v,f.z));
}
float3 unproject(Frame f,float3 v) {
    return f.position+((f.x*v.x+f.y*v.y)+f.z*v.z);
}
float component(float3 v,uint i) {
    return i==0u?v.x:(i==1u?v.y:v.z);
}
// Preserve fused float32 source arithmetic without relying on
// backend-specific multiply-add contraction.
float fusedProductSum(float a,float b,float c) {
    precise float product=a*b;
    precise float sa=4097.f*a;
    precise float sb=4097.f*b;
    precise float highA=sa-(sa-a);
    precise float highB=sb-(sb-b);
    precise float lowA=a-highA,lowB=b-highB;
    precise float e0=highA*highB-product;
    precise float e1=e0+highA*lowB;
    precise float e2=e1+lowA*highB;
    precise float productError=e2+lowA*lowB;
    precise float sum=product+c;
    precise float bv=sum-product;
    precise float sumError=(product-(sum-bv))+(c-bv);
    precise float fused=sum+(productError+sumError);
    return fused;
}
#ifdef __spirv__
// Preserve the source float32 incident-axis comparison on near-tied faces.
float fusedDot(float3 a,float3 b) {
    return fusedProductSum(a.x,b.x,
                           fusedProductSum(a.y,b.y,a.z*b.z));
}
#endif


float4 quatProduct(float4 a,float4 b) {
    return float4(
        a.w*b.x+b.w*a.x+a.y*b.z-a.z*b.y,
        a.w*b.y+b.w*a.y+a.z*b.x-a.x*b.z,
        a.w*b.z+b.w*a.z+a.x*b.y-a.y*b.x,
        a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z);
}
float4 conjugate(float4 q) {return float4(-q.xyz,q.w);}
#include "physx_source_pcm_box_gjk_native.hlsli"

uint cachePointAddress(uint edgeBase,uint contactIndex) {
    return edgeBase+80u+contactIndex*40u;
}
void copyCachedContact(uint edgeBase,uint contactIndex,uint destination) {
    uint sourceOffset=cachePointAddress(edgeBase,contactIndex);
    [unroll]
    for(uint k=0u;k<10u;++k)
        candidates.Store(destination+k*4u,edgeCaches.Load(sourceOffset+k*4u));
}
void storeSelectedContact(uint edgeBase,uint slot,uint source) {
    uint destinationOffset=cachePointAddress(edgeBase,slot);
    [unroll]
    for(uint k=0u;k<10u;++k)
        edgeCaches.Store(destinationOffset+k*4u,candidates.Load(source+k*4u));
}
// GuPersistentContactManifold::refreshContactPoints and
// invalidate_BoxConvex, operating directly on the Sc EdgeIndex record.
// The relative transform belongs to B; no independently owned cache.
bool refreshEdge(uint edgeBase,float3 pa,float3 pb,float4 qa,float4 qb,
                 float3 ea,float3 eb,float contactDistance) {
    const float4 inverseB=conjugate(qb);
    const float3 relative=rotateSource(inverseB,pa-pb);
    const float4 rotation=quatProduct(inverseB,qa);
    const uint initialized=edgeCaches.Load(edgeBase+64u);
    uint count=edgeCaches.Load(edgeBase+12u);
    if(initialized==0u)count=0u;
    const uint initial=count;
    float minA=min(ea.x,min(ea.y,ea.z));
    float minB=min(eb.x,min(eb.y,eb.z));
    const float margin=min(min(0.15f*minA,0.15f),min(0.15f*minB,0.15f));
    const float breakingSq=(margin*0.8f)*(margin*0.8f);
    for(uint i=count;i>0u;--i) {
        uint offset=cachePointAddress(edgeBase,i-1u);
        float3 localA=asfloat(edgeCaches.Load3(offset));
        float3 localB=asfloat(edgeCaches.Load3(offset+12u));
        float3 normal=asfloat(edgeCaches.Load3(offset+24u));
        float3 v=relative+rotateSource(rotation,localA)-localB;
        float distance=dot(v,normal);
        float3 tangent=v-normal*distance;
        if(dot(tangent,tangent)>breakingSq) {
            --count;
            if(i-1u!=count) {
                uint from=cachePointAddress(edgeBase,count);
                [unroll]
                for(uint k=0u;k<10u;++k)
                    edgeCaches.Store(offset+k*4u,
                                    edgeCaches.Load(from+k*4u));
            }
        }else
            edgeCaches.Store(offset+36u,asuint(distance));
    }
    edgeCaches.Store(edgeBase+12u,count);
    const float3 previous=asfloat(edgeCaches.Load3(edgeBase));
    const float4 previousA=asfloat(edgeCaches.Load4(edgeBase+32u));
    const float4 previousB=asfloat(edgeCaches.Load4(edgeBase+48u));
    const float translation=max(abs(relative.x-previous.x),
        max(abs(relative.y-previous.y),abs(relative.z-previous.z)));
    const float thresholds[5]={0.5f,0.125f,0.25f,0.375f,0.375f};
    const float quaternionThresholds[5]={
        0.9998f,0.9999f,0.9999f,0.9999f,0.9999f};
    float qaDot=dot(qa,previousA),qbDot=dot(qb,previousB);
    float limit=margin*thresholds[count];
    bool regenerate=initialized==0u || count!=initial ||
        translation>limit || qaDot<quaternionThresholds[count] ||
        qbDot<quaternionThresholds[count];
    if(!regenerate) {
        float arcA=qaDot<1.f?acos(max(-1.f,qaDot)):0.f;
        float arcB=qbDot<1.f?acos(max(-1.f,qbDot)):0.f;
        regenerate=arcA*sqrt(dot(ea,ea))>limit ||
                   arcB*sqrt(dot(eb,eb))>limit;
    }
    if(regenerate) {
        edgeCaches.Store3(edgeBase,asuint(relative));
        edgeCaches.Store4(edgeBase+16u,asuint(rotation));
        edgeCaches.Store4(edgeBase+32u,asuint(qa));
        edgeCaches.Store4(edgeBase+48u,asuint(qb));
        edgeCaches.Store(edgeBase+64u,1u);
    }
    return regenerate;
}
void publishPair(uint offset,uint count,uint overlap,uint status,uint edge,
                 uint first,uint selectedCount,uint packed,uint regenerated) {
    results.Store4(offset,uint4(count,overlap,status,edge));
    results.Store4(offset+16u,uint4(first,selectedCount,packed,regenerated));
}

void emit(Frame f,float3 reference,float3 incident,float3 normal,
          float separation,bool flipped,
          float3x3 A,float3 pa,float3x3 B,float3 pb,
          uint first,uint limit,inout uint count,inout uint status) {
    if(count>=limit || count>=64u) {
        status=2u;
        return;
    }
    float3 w0=unproject(f,flipped?incident:reference);
    float3 w1=unproject(f,flipped?reference:incident);
    uint dst=p.candidateBias+(first+count)*40u;
    candidates.Store3(dst,asuint(local(A,pa,w0)));
    candidates.Store3(dst+12u,asuint(local(B,pb,w1)));
    candidates.Store3(dst+24u,asuint(local(B,float3(0.f,0.f,0.f),normal)));
    candidates.Store(dst+36u,asuint(separation));
    ++count;
}
bool contains(float3 poly[4],float3 v,float3 low,float3 high) {
    if(v.x<low.x || v.y<low.y || v.x>high.x || v.y>high.y)return false;
    uint hits=0u;
    float eps=1.1920928955078125e-7f;
    [unroll]
    for(uint i=0u,j=3u;i<4u;j=i++) {
        float3 a=poly[j],b=poly[i];
        if((v.x==a.x && v.y==a.y) ||
           (v.x==b.x && v.y==b.y))return true;
        if((a.y>v.y)!=(b.y>v.y)) {
            float dx=b.x-a.x,dy=b.y-a.y;
            float part=(v.y-a.y)*dx+(a.x+eps)*dy;
            float target=v.x*dy;
            bool intersection=dy>0.f?part>=target:target>=part;
            if(intersection && ++hits==2u)return false;
        }
    }
    return hits>0u;
}
bool intersectSegment(float3 start,float3 delta,float3 high,float3 low,
                      out float enter,out float leave) {
    enter=0.f; leave=1.f;
    [unroll]
    for(uint axis=0u;axis<3u;++axis) {
        float x=component(start,axis),d=component(delta,axis);
        float mn=component(low,axis),mx=component(high,axis);
        if(abs(d)<1.e-6f) {
            if(x<mn || x>mx)return false;
            continue;
        }
        float one=(mn-x)/d,two=(mx-x)/d;
        enter=max(enter,min(one,two));
        leave=min(leave,max(one,two));
    }
    return enter<=leave && enter<=1.f;
}
void clip(Frame f,float3 poly[4],float3 incidentNormal,
          float faceX,float faceY,float3 normal,float distance,bool flipped,
          float3x3 A,float3 pa,float3x3 B,float3 pb,
          uint first,uint limit,inout uint count,inout uint status) {
    faceX*=1.0001f;faceY*=1.0001f;
    bool penetrating[4],inside[4];
    float3 minimum=float3(3.402823466e38f,3.402823466e38f,3.402823466e38f);
    float3 maximum=-minimum;
    float3 high=float3(faceX,faceY,3.402823466e38f);
    [unroll]
    for(uint i=0u;i<4u;++i) {
        float3 v=poly[i];
        minimum.xy=min(minimum.xy,v.xy);
        maximum.xy=max(maximum.xy,v.xy);
        penetrating[i]=-v.z<distance;
        inside[i]=penetrating[i] &&
            abs(v.x)<=faceX && abs(v.y)<=faceY;
        if(inside[i])
            emit(f,float3(v.x,v.y,0.f),v,normal,-v.z,flipped,
                 A,pa,B,pb,first,limit,count,status);
    }
    if(count==4u)return;
    if(incidentNormal.z!=0.f) {
        float3 corners[4]={
            float3(faceX,faceY,0.f),float3(faceX,-faceY,0.f),
            float3(-faceX,faceY,0.f),float3(-faceX,-faceY,0.f)};
        [unroll]
        for(uint i=0u;i<4u;++i) {
            float3 corner=corners[i];
            if(!contains(poly,corner,minimum,maximum))continue;
            float nom=dot(incidentNormal,poly[0]-corner);
            float depth=nom/incidentNormal.z;
            float separation=-depth;
            if(distance>separation)
                emit(f,corner,float3(corner.x,corner.y,depth),
                     normal,separation,flipped,
                     A,pa,B,pb,first,limit,count,status);
        }
    }
    float3 low=float3(-faceX,-faceY,
                       -distance-1.1920928955078125e-7f);
    [unroll]
    for(uint i=0u,j=3u;i<4u;j=i++) {
        if(!penetrating[i]&&!penetrating[j])continue;
        bool c0=penetrating[i]&&inside[i];
        bool c1=penetrating[j]&&inside[j];
        if(c0&&c1)continue;
        float3 edge=poly[j]-poly[i];
        float t0,t1;
        if(!intersectSegment(poly[i],edge,high,low,t0,t1))continue;
        if(!c0) {
            float3 v=poly[i]+edge*t0;
            emit(f,float3(v.x,v.y,0.f),v,normal,-v.z,flipped,
                 A,pa,B,pb,first,limit,count,status);
        }
        if(!c1) {
            float3 v=poly[i]+edge*t1;
            emit(f,float3(v.x,v.y,0.f),v,normal,-v.z,flipped,
                 A,pa,B,pb,first,limit,count,status);
        }
    }
}

// GuPersistentContactManifold::reduceBatchContacts retains its swap-last
// candidate order. Only indices are stored; source candidates stay resident.
float3 candidateLocalB(uint base,uint index) {
    return asfloat(candidates.Load3(p.candidateBias+(base+index)*40u+12u));
}
float3 candidateNormalB(uint base,uint index) {
    return asfloat(candidates.Load3(p.candidateBias+(base+index)*40u+24u));
}
float candidateDepth(uint base,uint index) {
    return asfloat(candidates.Load(p.candidateBias+(base+index)*40u+36u));
}
float candidateDistanceSquared(float3 a,float3 b) {
    float3 d=a-b;
    return dot(d,d);
}
uint4 selectBatch(uint base,uint count) {
    uint available[64];
    for(uint i=0u;i<count;++i)available[i]=i;
    uint remaining=count;
    uint4 selected=uint4(0u,0u,0u,0u);
    float deepestDepth=candidateDepth(base,0u);
    float shallowestDepth=deepestDepth;
    uint deepest=0u;
    for(uint i=1u;i<count;++i) {
        float depth=candidateDepth(base,i);
        shallowestDepth=max(shallowestDepth,depth);
        if(deepestDepth>depth) {
            deepestDepth=depth;
            deepest=i;
        }
    }
    selected.x=deepest;
    --remaining;
    available[deepest]=available[remaining];

    uint furthestPosition=0u;
    float3 anchor=candidateLocalB(base,selected.x);
    float farthestSquared=candidateDistanceSquared(
        candidateLocalB(base,available[0]),anchor);
    for(uint i=1u;i<remaining;++i) {
        float value=candidateDistanceSquared(
            candidateLocalB(base,available[i]),anchor);
        if(value>farthestSquared) {
            farthestSquared=value;
            furthestPosition=i;
        }
    }
    selected.y=available[furthestPosition];
    --remaining;
    available[furthestPosition]=available[remaining];

    float3 baseline=candidateLocalB(base,selected.y)-anchor;
    float3 transverse=cross(baseline,candidateNormalB(base,selected.x));
    float transverseSquared=dot(transverse,transverse);
    if(transverseSquared>0.f)
        transverse*=1.f/sqrt(transverseSquared);
    else
        transverse=candidateNormalB(base,selected.x);

    float greatest=-3.402823466e38f;
    float least=3.402823466e38f;
    uint greatestPosition=0u,leastPosition=0u;
    for(uint i=0u;i<remaining;++i) {
        float side=dot(candidateLocalB(base,available[i])-anchor,transverse);
        if(side>greatest) {
            greatest=side;
            greatestPosition=i;
        }
        if(side<least) {
            least=side;
            leastPosition=i;
        }
    }
    selected.z=available[greatestPosition];
    --remaining;
    available[greatestPosition]=available[remaining];
    if(leastPosition==remaining)leastPosition=greatestPosition;
    if(least*greatest>0.f) {
        greatest=-3.402823466e38f;
        for(uint i=0u;i<remaining;++i) {
            float side=dot(candidateLocalB(base,available[i])-anchor,transverse);
            if(side>greatest) {
                greatest=side;
                leastPosition=i;
            }
        }
    }
    selected.w=available[leastPosition];
    --remaining;
    available[leastPosition]=available[remaining];

    const float threshold=0.02f; // PxTolerancesScale.length = 1
    if(deepestDepth<threshold && shallowestDepth>threshold) {
        for(uint i=0u;i<4u;++i) {
            float depth=candidateDepth(base,selected[i]);
            if(depth>threshold) {
                uint replacement=remaining;
                for(uint j=0u;j<remaining;++j) {
                    float candidate=candidateDepth(base,available[j]);
                    if(depth>candidate && threshold>candidate) {
                        depth=candidate;
                        replacement=j;
                    }
                }
                if(replacement<remaining) {
                    uint prior=selected[i];
                    selected[i]=available[replacement];
                    available[replacement]=prior;
                }
            }
        }
    }
    return selected;
}

[numthreads(128,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint i=thread.x;
    if(i>=p.pairCount)return;
    uint source=p.pairBias+i*20u;
    uint a=pairs.Load(source),b=pairs.Load(source+4u);
    uint edge=pairs.Load(source+8u);
    uint first=pairs.Load(source+12u),limit=pairs.Load(source+16u);
    uint result=p.resultBias+i*32u;
    uint count=0u,overlap=0u,status=0u,regenerated=0u;
    if(a>=p.shapeCapacity || b>=p.shapeCapacity || a==b ||
       edge>=p.edgeCapacity ||
       first>p.candidateCapacity ||
       limit>p.candidateCapacity-first) {
        status=1u;
    }else {
        uint ta=p.transformBias+a*32u,tb=p.transformBias+b*32u;
        float3 pa=asfloat(transforms.Load3(ta+16u));
        float3 pb=asfloat(transforms.Load3(tb+16u));
        uint ga=p.geometryBias+a*16u,gb=p.geometryBias+b*16u;
        float3 ea=asfloat(geometry.Load3(ga));
        float3 eb=asfloat(geometry.Load3(gb));
        float distance=asfloat(geometry.Load(ga+12u))+
                       asfloat(geometry.Load(gb+12u));
        uint edgeBase=p.edgeCacheBias+edge*240u;
        float4 qa=asfloat(transforms.Load4(ta));
        float4 qb=asfloat(transforms.Load4(tb));
        if(edgeCaches.Load(edgeBase+12u)>4u ||
           edgeCaches.Load(edgeBase+64u)>1u) {
            publishPair(result,0u,0u,1u,edge,first,0u,0u,0u);
            return;
        }
        bool regenerate=refreshEdge(edgeBase,pa,pb,qa,qb,ea,eb,distance);
        if(!regenerate) {
            uint cached=edgeCaches.Load(edgeBase+12u);
            uint eligible=0u;
            for(uint j=0u;j<cached;++j) {
                uint source=cachePointAddress(edgeBase,j);
                float depth=asfloat(edgeCaches.Load(source+36u));
                if(depth<=distance) {
                    if(eligible<limit)
                        copyCachedContact(edgeBase,j,
                            p.candidateBias+(first+eligible)*40u);
                    ++eligible;
                }
            }
            count=min(eligible,limit);
            uint needsGjk=edgeCaches.Load(edgeBase+68u);
            overlap=(eligible>0u || needsGjk!=0u)?1u:0u;
            if(eligible>limit)status=2u;
            else if(needsGjk!=0u)status=3u;
            uint selection=count==0u?0u:
                0u|(1u<<8u)|(2u<<16u)|(3u<<24u);
            publishPair(result,count,overlap,status,edge,first,
                        min(count,4u),selection,0u);
            return;
        }
        regenerated=1u;
        float3x3 A=quaternionBasis(qa);
        float3x3 B=quaternionBasis(qb);
        float3 delta=pb-pa;
        float3 s[6]={A[0],A[1],A[2],B[0],B[1],B[2]};
        float3x3 incident;
        float overlapFaces[6],sign[6];
        bool separated=false;
        [unroll]
        for(uint axis=0u;axis<6u;++axis) {
            bool isA=axis<3u;
            float3 n=s[axis];
            uint k=axis%3u;
            float3x3 other=isA?B:A;
            float3 h=isA?eb:ea;
            float radius=(abs(dot(n,other[0]))+1.e-6f)*h.x+
                         (abs(dot(n,other[1]))+1.e-6f)*h.y+
                         (abs(dot(n,other[2]))+1.e-6f)*h.z;
            sign[axis]=dot(delta,n);
            overlapFaces[axis]=component(isA?ea:eb,k)+radius-
                               abs(sign[axis])+distance;
            if(overlapFaces[axis]<0.f)separated=true;
        }
        [unroll]
        for(uint j=0u;j<3u;++j) {
            [unroll]
            for(uint k=0u;k<3u;++k) {
                float3 axis=cross(A[j],B[k]);
                float ra=0.f,rb=0.f;
                [unroll]
                for(uint m=0u;m<3u;++m) {
                    if(m!=j)ra+=(abs(dot(axis,A[m]))+1.e-6f)*ea[m];
                    if(m!=k)rb+=(abs(dot(axis,B[m]))+1.e-6f)*eb[m];
                }
                if(abs(dot(delta,axis))>ra+rb+distance)separated=true;
            }
        }
        if(!separated) {
            overlap=1u;
            uint feature=0u;
            float best=overlapFaces[0];
            [unroll]
            for(uint axis=1u;axis<6u;++axis) {
                if(best>overlapFaces[axis]) {
                    best=overlapFaces[axis];
                    feature=axis;
                }
            }
            bool flipped=feature>=3u;
            bool nonpositive=sign[feature]<=0.f;
            float3x3 face=flipped?B:A;
            float3 referenceHalf=flipped?eb:ea;
            float3 facePosition=flipped?pb:pa;
            incident=flipped?A:B;
            float3 incidentPosition=flipped?pa:pb;
            float3 incidentHalf=flipped?ea:eb;
            Frame f;
            float3 normal;
            float faceX,faceY;
            if(feature%3u==0u) {
                normal=nonpositive?face[0]:-face[0];
                f.x=flipped
                    ? (nonpositive?face[2]:-face[2])
                    : (nonpositive?-face[2]:face[2]);
                f.y=face[1];f.z=flipped?-normal:normal;
                f.position=facePosition+face[0]*
                    (flipped?(nonpositive?referenceHalf.x:-referenceHalf.x):
                             (nonpositive?-referenceHalf.x:referenceHalf.x));
                faceX=referenceHalf.z;faceY=referenceHalf.y;
            }else if(feature%3u==1u) {
                normal=nonpositive?face[1]:-face[1];
                f.x=face[0];
                f.y=flipped
                    ? (nonpositive?face[2]:-face[2])
                    : (nonpositive?-face[2]:face[2]);
                f.z=flipped?-normal:normal;
                f.position=facePosition+face[1]*
                    (flipped?(nonpositive?referenceHalf.y:-referenceHalf.y):
                             (nonpositive?-referenceHalf.y:referenceHalf.y));
                faceX=referenceHalf.x;faceY=referenceHalf.z;
            }else {
                normal=nonpositive?face[2]:-face[2];
                f.x=face[0];
                f.y=flipped
                    ? (nonpositive?-face[1]:face[1])
                    : (nonpositive?face[1]:-face[1]);
                f.z=flipped?-normal:normal;
                f.position=facePosition+face[2]*
                    (flipped?(nonpositive?referenceHalf.z:-referenceHalf.z):
                             (nonpositive?-referenceHalf.z:referenceHalf.z));
                faceX=referenceHalf.x;faceY=referenceHalf.y;
            }
            float3 search=-f.z;
            float dx=abs(dot(search,incident[0]));
            float dy=abs(dot(search,incident[1]));
            float dz=abs(dot(search,incident[2]));
#ifdef __spirv__
            // Preserve the source rounding on near-tied faces in SPIR-V.
            // DXIL's native dot already matches the Windows source.
            uint bx=asuint(dx),by=asuint(dy),bz=asuint(dz);
            uint bestScoreBits=max(bx,max(by,bz));
            uint tied=(bestScoreBits-bx<=1u?1u:0u)+
                      (bestScoreBits-by<=1u?1u:0u)+
                      (bestScoreBits-bz<=1u?1u:0u);
            if(tied>=2u) {
                dx=abs(fusedDot(search,incident[0]));
                dy=abs(fusedDot(search,incident[1]));
                dz=abs(fusedDot(search,incident[2]));
            }
#endif
            uint axis=dx>=dy&&dx>=dz?0u:(dy>=dz?1u:2u);
            float facing=dot(search,incident[axis]);
            float signedExtent=facing>0.f
                ?-component(incidentHalf,axis)
                : component(incidentHalf,axis);
            float3 center=incidentPosition+
                incident[axis]*signedExtent;
            uint uAxis=axis==0u?1u:0u;
            uint vAxis=axis==2u?1u:2u;
            float3 u=incident[uAxis]*component(incidentHalf,uAxis);
            float3 v=incident[vAxis]*component(incidentHalf,vAxis);
            float3 poly[4]={
                project(f,center+(u+v)),
                project(f,center+(u-v)),
                project(f,center-(u+v)),
                project(f,center-(u-v))};
            float3 incidentNormal=project(f,
                f.position+(facing>0.f?-incident[axis]:incident[axis]));
            clip(f,poly,incidentNormal,faceX,faceY,normal,distance,flipped,
                 A,pa,B,pb,first,limit,count,status);
        }
        // GuPCMContactBoxBox's GJK continuation only runs when face
        // clipping found no witness. EPA remains explicitly pending.
        if(overlap!=0u && count==0u && status==0u) {
            uint cache=p.edgeCacheBias+edge*240u;
            if(edgeCaches.Load(cache+12u)==0u) {
                GjkBoxPair boxPair;
                boxPair.halfA=ea;
                boxPair.halfB=eb;
                boxPair.aInB=rotateSource(conjugate(qb),pa-pb);
                boxPair.relative=quatProduct(conjugate(qb),qa);
                GjkWitness witness;
                uint outcome=gjkBoxContinuation(boxPair,distance,witness);
                if(outcome==1u) {
                    if(limit==0u) {
                        status=2u;
                    }else {
                        uint offset=p.candidateBias+first*40u;
                        candidates.Store3(offset,asuint(witness.localA));
                        candidates.Store3(offset+12u,asuint(witness.localB));
                        candidates.Store3(offset+24u,asuint(witness.normalB));
                        candidates.Store(offset+36u,asuint(witness.separation));
                        count=1u;
                    }
                }else if(outcome==0u)
                    overlap=0u;
                else status=3u;
            }else status=3u;
        }
    }
    uint selectedCount=status==0u?min(count,4u):0u;
    uint4 selection=uint4(0u,1u,2u,3u);
    if(status==0u && count>4u)
        selection=selectBatch(first,count);
    uint packed=selectedCount==0u?0u:
        selection.x|(selection.y<<8u)|(selection.z<<16u)|(selection.w<<24u);
    if(regenerated!=0u && status==2u) {
        // A partial clip is not a valid PCM manifold. Require full
        // regeneration even when the next frame has the same pose.
        uint edgeBase=p.edgeCacheBias+edge*240u;
        edgeCaches.Store(edgeBase+12u,0u);
        edgeCaches.Store(edgeBase+64u,0u);
        edgeCaches.Store(edgeBase+68u,0u);
    }
    if(regenerated!=0u && (status==0u || status==3u)) {
        uint edgeBase=p.edgeCacheBias+edge*240u;
        for(uint j=0u;j<selectedCount;++j)
            storeSelectedContact(edgeBase,j,
                p.candidateBias+(first+selection[j])*40u);
        edgeCaches.Store(edgeBase+12u,selectedCount);
        // GJK/EPA owns cases where overlap has no clipped witnesses.
        uint needsGjk=status==3u?1u:0u;
        edgeCaches.Store(edgeBase+68u,needsGjk);
        if(needsGjk!=0u)status=3u;
    }
    publishPair(result,count,overlap,status,edge,first,selectedCount,
                packed,regenerated);
}
