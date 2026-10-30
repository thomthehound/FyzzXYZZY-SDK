// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// GuContactBoxBox::generateContacts. The prior SAT stage supplies the six-axis
// selection; contact witnesses remain in Gu's original insertion order.
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
RWByteAddressBuffer axes : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer results : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer contacts : register(u5,space0);

struct Parameters {
    uint pairCount;
    uint transformBias;
    uint geometryBias;
    uint pairBias;
    uint axisBias;
    uint resultBias;
    uint contactBias;
    uint shapeCapacity;
    uint contactCapacity;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif

float3x3 quaternionBasis(float4 q) {
    float x2=q.x+q.x, y2=q.y+q.y, z2=q.z+q.z;
    float xx=x2*q.x, yy=y2*q.y, zz=z2*q.z;
    float xy=x2*q.y, xz=x2*q.z, xw=x2*q.w;
    float yz=y2*q.z, yw=y2*q.w, zw=z2*q.w;
    return float3x3(
        float3(1.f-yy-zz,xy+zw,xz-yw),
        float3(xy-zw,1.f-xx-zz,yz+xw),
        float3(xz+yw,yz-xw,1.f-xx-yy));
}

float3 apply(float3x3 basis,float3 local) {
    return basis[0]*local.x+basis[1]*local.y+basis[2]*local.z;
}
float3 inverseApply(float3x3 basis,float3 world) {
    return float3(dot(basis[0],world),dot(basis[1],world),dot(basis[2],world));
}
void emit(uint outputStart,uint outputLimit,inout uint count,inout uint overflow,
          float3 point,float3 normal,float separation,
          float3x3 frame,float3 origin) {
    if(count<outputLimit && count<255u) {
        uint dst=p.contactBias+(outputStart+count)*32u;
        float3 world=apply(frame,point)+origin;
        contacts.Store3(dst,asuint(world));
        contacts.Store3(dst+12u,asuint(normal));
        contacts.Store(dst+24u,asuint(separation));
        contacts.Store(dst+28u,0u);
    }else {
        overflow=1u;
    }
    if(count<255u)++count;
}

// GuContactBoxBox::IsInYZ: preserve the source quad winding and edge tests.
float isInYZ(float y,float z,in float3 vertices[8],
             uint f0,uint f1,uint f2,uint f3) {
    uint f[4]={f0,f1,f2,f3};
    float previousY=vertices[f3].y, previousZ=vertices[f3].z;
    [unroll]
    for(uint i=0u;i<4u;++i) {
        float currentY=vertices[f[i]].y;
        float currentZ=vertices[f[i]].z;
        if((currentY-previousY)*(z-previousZ)-
           (currentZ-previousZ)*(y-previousY)>=0.f)return -1.f;
        previousY=currentY;
        previousZ=currentZ;
    }
    float x=vertices[f0].x;
    float ay=y-vertices[f0].y,az=z-vertices[f0].z;
    float3 b=vertices[f1]-vertices[f0];
    x+=b.x*(ay*b.y+az*b.z)/dot(b,b);
    b=vertices[f3]-vertices[f0];
    x+=b.x*(ay*b.y+az*b.z)/dot(b,b);
    return x;
}

uint generate(float3x3 frame,float3 framePosition,
              float3x3 incident,float3 incidentPosition,float3 half,
              float y1,float z1,float3 normal,float distance,
              uint outputStart,uint outputLimit,out uint overflow) {
    uint count=0u;
    overflow=0u;
    y1+=distance; z1+=distance;

    float3 position=inverseApply(frame,incidentPosition-framePosition);
    float3 ex=inverseApply(frame,incident[0])*half.x;
    float3 ey=inverseApply(frame,incident[1])*half.y;
    float3 ez=inverseApply(frame,incident[2])*half.z;
    float3 vtx[8];
    vtx[0]=vtx[2]=vtx[4]=vtx[6]=position-ex;
    vtx[1]=vtx[3]=vtx[5]=vtx[7]=position+ex;
    float3 e=ey+ez;
    vtx[0]-=e;vtx[1]-=e;vtx[6]+=e;vtx[7]+=e;
    e=ey-ez;
    vtx[2]+=e;vtx[3]+=e;vtx[4]-=e;vtx[5]-=e;
    bool penetrating[8],inside[8];

    [unroll]
    for(uint i=0u;i<8u;++i) {
        if(vtx[i].x<-distance) {
            penetrating[i]=false;
            inside[i]=false;
            continue;
        }
        penetrating[i]=true;
        inside[i]=abs(vtx[i].y)<=y1 && abs(vtx[i].z)<=z1;
        if(inside[i])
            emit(outputStart,outputLimit,count,overflow,vtx[i],
                 normal,-vtx[i].x,frame,framePosition);
    }

    static const uint edge[24]={
        0u,1u,1u,3u,3u,2u,2u,0u,
        4u,5u,5u,7u,7u,6u,6u,4u,
        0u,4u,1u,5u,2u,6u,3u,7u
    };
    [unroll]
    for(uint index=0u;index<12u;++index) {
        uint first=edge[index*2u],second=edge[index*2u+1u];
        if(!(penetrating[first]||penetrating[second]))continue;
        if(!inside[first]||!inside[second]) {
            if(vtx[first].y>vtx[second].y) {
                uint temp=first;first=second;second=temp;
            }
            if(vtx[first].y<y1 && vtx[second].y>=y1) {
                float a=(y1-vtx[first].y)/(vtx[second].y-vtx[first].y);
                float z=vtx[first].z+(vtx[second].z-vtx[first].z)*a;
                if(abs(z)<=z1) {
                    float x=vtx[first].x+(vtx[second].x-vtx[first].x)*a;
                    if(x+distance>=0.f)
                        emit(outputStart,outputLimit,count,overflow,
                             float3(x,y1,z),normal,-x,frame,framePosition);
                }
            }
            if(vtx[first].y<-y1 && vtx[second].y>=-y1) {
                float a=(-y1-vtx[first].y)/(vtx[second].y-vtx[first].y);
                float z=vtx[first].z+(vtx[second].z-vtx[first].z)*a;
                if(abs(z)<=z1) {
                    float x=vtx[first].x+(vtx[second].x-vtx[first].x)*a;
                    if(x+distance>=0.f)
                        emit(outputStart,outputLimit,count,overflow,
                             float3(x,-y1,z),normal,-x,frame,framePosition);
                }
            }
            if(vtx[first].z>vtx[second].z) {
                uint temp=first;first=second;second=temp;
            }
            if(vtx[first].z<z1 && vtx[second].z>=z1) {
                float a=(z1-vtx[first].z)/(vtx[second].z-vtx[first].z);
                float y=vtx[first].y+(vtx[second].y-vtx[first].y)*a;
                if(abs(y)<=y1) {
                    float x=vtx[first].x+(vtx[second].x-vtx[first].x)*a;
                    if(x+distance>=0.f)
                        emit(outputStart,outputLimit,count,overflow,
                             float3(x,y,z1),normal,-x,frame,framePosition);
                }
            }
            if(vtx[first].z<-z1 && vtx[second].z>=-z1) {
                float a=(-z1-vtx[first].z)/(vtx[second].z-vtx[first].z);
                float y=vtx[first].y+(vtx[second].y-vtx[first].y)*a;
                if(abs(y)<=y1) {
                    float x=vtx[first].x+(vtx[second].x-vtx[first].x)*a;
                    if(x+distance>=0.f)
                        emit(outputStart,outputLimit,count,overflow,
                             float3(x,y,-z1),normal,-x,frame,framePosition);
                }
            }
        }
        if((!penetrating[first]&&!inside[second])||
           (!penetrating[second]&&!inside[first])) {
            float a=-vtx[first].x/(vtx[second].x-vtx[first].x);
            float y=vtx[first].y+(vtx[second].y-vtx[first].y)*a;
            if(abs(y)<=y1) {
                float z=vtx[first].z+(vtx[second].z-vtx[first].z)*a;
                if(abs(z)<=z1)
                    emit(outputStart,outputLimit,count,overflow,
                         float3(0.f,y,z),normal,0.f,frame,framePosition);
            }
        }
    }

    static const uint face[24]={
        0u,1u,3u,2u, 1u,5u,7u,3u,
        5u,4u,6u,7u, 4u,0u,2u,6u,
        2u,3u,7u,6u, 0u,4u,5u,1u
    };
    uint added=0u;
    [loop]
    for(uint i=0u;i<6u && added!=15u;++i) {
        uint a=face[i*4u],b=face[i*4u+1u];
        uint c=face[i*4u+2u],d=face[i*4u+3u];
        if(!(penetrating[a]&&penetrating[b]&&
             penetrating[c]&&penetrating[d]))continue;
        if(inside[a]&&inside[b]&&inside[c]&&inside[d])continue;
        if(!(added&1u)) {
            float x=isInYZ(-y1,-z1,vtx,a,b,c,d);
            if(x>=0.f) {
                added|=1u;
                emit(outputStart,outputLimit,count,overflow,
                     float3(x,-y1,-z1),normal,-x,frame,framePosition);
            }
        }
        if(!(added&2u)) {
            float x=isInYZ(y1,-z1,vtx,a,b,c,d);
            if(x>=0.f) {
                added|=2u;
                emit(outputStart,outputLimit,count,overflow,
                     float3(x,y1,-z1),normal,-x,frame,framePosition);
            }
        }
        if(!(added&4u)) {
            float x=isInYZ(-y1,z1,vtx,a,b,c,d);
            if(x>=0.f) {
                added|=4u;
                emit(outputStart,outputLimit,count,overflow,
                     float3(x,-y1,z1),normal,-x,frame,framePosition);
            }
        }
        if(!(added&8u)) {
            float x=isInYZ(y1,z1,vtx,a,b,c,d);
            if(x>=0.f) {
                added|=8u;
                emit(outputStart,outputLimit,count,overflow,
                     float3(x,y1,z1),normal,-x,frame,framePosition);
            }
        }
    }
    return count;
}

[numthreads(128,1,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint index=thread.x;
    if(index>=p.pairCount)return;
    uint pairBase=p.pairBias+index*24u;
    uint a=pairs.Load(pairBase),b=pairs.Load(pairBase+4u);
    uint previousAxis=pairs.Load(pairBase+8u);
    uint outputStart=pairs.Load(pairBase+12u);
    uint outputLimit=pairs.Load(pairBase+16u);
    uint resultBase=p.resultBias+index*16u;
    uint axis=axes.Load(p.axisBias+index*4u);
    uint status=0u,count=0u;
    if(a>=p.shapeCapacity || b>=p.shapeCapacity || a==b ||
       previousAxis>6u || axis>6u ||
       outputStart>p.contactCapacity ||
       outputLimit>p.contactCapacity-outputStart) {
        status=1u;
        axis=0u;
    }else if(axis!=0u) {
        uint ta=p.transformBias+a*32u,tb=p.transformBias+b*32u;
        float3 pa=asfloat(transforms.Load3(ta+16u));
        float3 pb=asfloat(transforms.Load3(tb+16u));
        float3x3 A=quaternionBasis(asfloat(transforms.Load4(ta)));
        float3x3 B=quaternionBasis(asfloat(transforms.Load4(tb)));
        uint ga=p.geometryBias+a*16u,gb=p.geometryBias+b*16u;
        float3 ea=asfloat(geometry.Load3(ga));
        float3 eb=asfloat(geometry.Load3(gb));
        float distance=asfloat(geometry.Load(ga+12u))+
                       asfloat(geometry.Load(gb+12u));
        float3 delta=pb-pa;
        bool sign=(axis<=3u ? dot(A[axis-1u],delta)
                              : dot(B[axis-4u],delta))<0.f;
        float3x3 frame,incident;
        float3 origin,incidentOrigin,incidentHalf;
        float width,height;
        float3 normal;
        if(axis<=3u) {
            incident=B;incidentOrigin=pb;incidentHalf=eb;
            if(axis==1u) {
                width=ea.y;height=ea.z;
                normal=sign?A[0]:-A[0];
                frame=sign ? A : float3x3(-A[0],-A[1],A[2]);
                origin=sign?pa-ea.x*A[0]:pa+ea.x*A[0];
            }else if(axis==2u) {
                width=ea.z;height=ea.x;
                normal=sign?A[1]:-A[1];
                frame=sign ? float3x3(A[1],A[2],A[0])
                           : float3x3(-A[1],-A[2],A[0]);
                origin=sign?pa-ea.y*A[1]:pa+ea.y*A[1];
            }else {
                width=ea.x;height=ea.y;
                normal=sign?A[2]:-A[2];
                frame=sign ? float3x3(A[2],A[0],A[1])
                           : float3x3(-A[2],-A[0],A[1]);
                origin=sign?pa-ea.z*A[2]:pa+ea.z*A[2];
            }
        }else {
            incident=A;incidentOrigin=pa;incidentHalf=ea;
            if(axis==4u) {
                width=eb.y;height=eb.z;
                normal=sign?B[0]:-B[0];
                frame=sign ? float3x3(-B[0],-B[1],B[2]) : B;
                origin=sign?pb+eb.x*B[0]:pb-eb.x*B[0];
            }else if(axis==5u) {
                width=eb.z;height=eb.x;
                normal=sign?B[1]:-B[1];
                frame=sign ? float3x3(-B[1],-B[2],B[0])
                           : float3x3(B[1],B[2],B[0]);
                origin=sign?pb+eb.y*B[1]:pb-eb.y*B[1];
            }else {
                width=eb.x;height=eb.y;
                normal=sign?B[2]:-B[2];
                frame=sign ? float3x3(-B[2],-B[0],B[1])
                           : float3x3(B[2],B[0],B[1]);
                origin=sign?pb+eb.z*B[2]:pb-eb.z*B[2];
            }
        }
        uint overflow;
        count=generate(frame,origin,incident,incidentOrigin,incidentHalf,
                       width,height,normal,distance,
                       outputStart,outputLimit,overflow);
        if(overflow)status=2u;
        if(count==0u)axis=0u;
    }
    results.Store4(resultBase,uint4(count,axis,status,0u));
}
