// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Scalar translation of GuPCMContactBoxBox.cpp's box SAT, reference-face
// selection, incident polygon, and projected witness generation. The
// shape-local persistent manifold and source GJK/EPA continuation are
// owned by Scene, not this stateless generator.

#include "native/scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fyzz::native {
namespace {

Vec3 add(Vec3 a,Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
Vec3 sub(Vec3 a,Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
Vec3 mul(Vec3 a,float f) { return {a.x*f,a.y*f,a.z*f}; }
Vec3 negate(Vec3 a) { return {-a.x,-a.y,-a.z}; }
float dot(Vec3 a,Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
Vec3 cross(Vec3 a,Vec3 b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
Vec3 inverse_rotate(Quat q,Vec3 v) {
    const Vec3 u{-q.x,-q.y,-q.z};
    const Vec3 t=mul(cross(u,v),2.0f);
    return add(v,add(mul(t,q.w),cross(u,t)));
}
Vec3 rotate(Quat q,Vec3 v) {
    const Vec3 u{q.x,q.y,q.z};
    const Vec3 t=mul(cross(u,v),2.0f);
    return add(v,add(mul(t,q.w),cross(u,t)));
}
struct Box {
    Vec3 half{}, position{};
    Quat rotation{};
    std::array<Vec3,3> axis{};
    Box(Vec3 h,Vec3 p,Quat q):half(h),position(p),rotation(q),
        axis{rotate(q,{1,0,0}),rotate(q,{0,1,0}),rotate(q,{0,0,1})} {}
};
float extent(Vec3 v,unsigned index) {
    return index==0?v.x:(index==1?v.y:v.z);
}
Vec3 local(Box const& b,Vec3 world) {
    return inverse_rotate(b.rotation,sub(world,b.position));
}
Vec3 world(Box const& b,Vec3 point) {
    return add(b.position,rotate(b.rotation,point));
}
struct Frame {
    Vec3 x{}, y{}, z{}, position{};
    Vec3 project(Vec3 w) const {
        const Vec3 d=sub(w,position);
        return {dot(d,x),dot(d,y),dot(d,z)};
    }
    Vec3 unproject(Vec3 v) const {
        return add(position,add(add(mul(x,v.x),mul(y,v.y)),mul(z,v.z)));
    }
};

// GuPCMContactGenUtil.cpp::contains, including the epsilon-adjusted
// ray intersection and the original endpoint equality test.
bool contains(const std::array<Vec3,4>& poly,Vec3 p,Vec3 low,Vec3 high) {
    if(p.x<low.x || p.y<low.y || p.x>high.x || p.y>high.y)
        return false;
    unsigned hits=0;
    const float eps=std::numeric_limits<float>::epsilon();
    for(unsigned i=0,j=3;i<4;j=i++) {
        const Vec3 a=poly[j],b=poly[i];
        if((p.x==a.x && p.y==a.y) ||
           (p.x==b.x && p.y==b.y)) return true;
        if((a.y>p.y)!=(b.y>p.y)) {
            const float dx=b.x-a.x,dy=b.y-a.y;
            const float part=(p.y-a.y)*dx+(a.x+eps)*dy;
            const float target=p.x*dy;
            const bool intersection=dy>0.0f?part>=target:target>=part;
            if(intersection && ++hits==2)return false;
        }
    }
    return hits>0;
}

// GuPCMContactBoxBox.cpp::intersectSegmentAABB. The generator clips
// incident polygon edges against the reference face and contact depth.
bool intersect_segment(Vec3 p,Vec3 delta,Vec3 high,Vec3 low,
                       float& enter,float& exit) {
    enter=0.0f;
    exit=1.0f;
    for(unsigned axis=0;axis<3;++axis) {
        const float x=extent(p,axis),d=extent(delta,axis);
        const float mn=extent(low,axis),mx=extent(high,axis);
        if(std::abs(d)<1.0e-6f) {
            if(x<mn || x>mx)return false;
            continue;
        }
        const float one=(mn-x)/d,two=(mx-x)/d;
        enter=std::max(enter,std::min(one,two));
        exit=std::min(exit,std::max(one,two));
    }
    return enter<=exit && enter<=1.0f;
}

struct Generation {
    const Box& a;
    const Box& b;
    std::span<PcmContactCandidate> output;
    std::size_t count=0;
    void emit(const Frame& frame,Vec3 reference,Vec3 incident,
              Vec3 normal,float separation,bool flipped) {
        if(count>=output.size())
            throw std::overflow_error("PCM direct box-box candidate arena exhausted");
        Vec3 w0=frame.unproject(flipped?incident:reference);
        Vec3 w1=frame.unproject(flipped?reference:incident);
        output[count++]={local(a,w0),local(b,w1),
            inverse_rotate(b.rotation,normal),separation};
    }
    void calculate(const Frame& frame,std::array<Vec3,4> const& poly,
                   Vec3 incident_normal,float face_x,float face_y,
                   Vec3 normal,float contact_distance,bool flipped) {
        // GuPCMContactBoxBox::calculateContacts. The 1.0001 reference
        // expansion belongs to this contact clipping stage, not SAT.
        face_x*=1.0001f;face_y*=1.0001f;
        std::array<bool,4> penetrating{},inside{};
        Vec3 minimum{std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max()};
        Vec3 maximum=negate(minimum);
        const Vec3 high{face_x,face_y,std::numeric_limits<float>::max()};
        for(unsigned i=0;i<4;++i) {
            auto p=poly[i];
            minimum.x=std::min(minimum.x,p.x);
            minimum.y=std::min(minimum.y,p.y);
            maximum.x=std::max(maximum.x,p.x);
            maximum.y=std::max(maximum.y,p.y);
            if(-p.z<contact_distance) {
                penetrating[i]=true;
                inside[i]=std::abs(p.x)<=face_x && std::abs(p.y)<=face_y;
                if(inside[i])
                    emit(frame,{p.x,p.y,0},p,normal,-p.z,flipped);
            }
        }
        if(count==4)return;
        // Project reference corners back to the incident polygon.
        if(incident_normal.z!=0.0f) {
            for(Vec3 corner : std::array<Vec3,4>{{
                    {face_x,face_y,0},{face_x,-face_y,0},
                    {-face_x,face_y,0},{-face_x,-face_y,0}}}) {
                if(!contains(poly,corner,minimum,maximum))continue;
                const float nom=dot(incident_normal,sub(poly[0],corner));
                const float t=nom/incident_normal.z;
                const float separation=-t;
                if(contact_distance>separation)
                    emit(frame,corner,{corner.x,corner.y,t},
                         normal,separation,flipped);
            }
        }
        const Vec3 low{-face_x,-face_y,
            -contact_distance-std::numeric_limits<float>::epsilon()};
        for(unsigned start=0,end=3;start<4;end=start++) {
            if(!penetrating[start] && !penetrating[end])continue;
            const bool c0=penetrating[start] && inside[start];
            const bool c1=penetrating[end] && inside[end];
            if(c0 && c1)continue;
            const Vec3 edge=sub(poly[end],poly[start]);
            float t0,t1;
            if(!intersect_segment(poly[start],edge,high,low,t0,t1))continue;
            if(!c0) {
                const Vec3 p=add(poly[start],mul(edge,t0));
                emit(frame,{p.x,p.y,0},p,normal,-p.z,flipped);
            }
            if(!c1) {
                const Vec3 p=add(poly[start],mul(edge,t1));
                emit(frame,{p.x,p.y,0},p,normal,-p.z,flipped);
            }
        }
    }
};

} // namespace

bool source_pcm_box_box_candidates(
    Vec3 half0,Vec3 position0,Quat rotation0,
    Vec3 half1,Vec3 position1,Quat rotation1,
    float contact_distance,std::span<PcmContactCandidate> candidates,
    std::size_t& count) {
    const Box a{half0,position0,rotation0},b{half1,position1,rotation1};
    Generation gen{a,b,candidates,0};
    const Vec3 delta=sub(b.position,a.position);
    std::array<float,6> sign{},overlap{};
    constexpr float sat_epsilon=1.0e-6f;
    const auto radius=[&](const Box& box,Vec3 axis) {
        return (std::abs(dot(axis,box.axis[0]))+sat_epsilon)*box.half.x+
               (std::abs(dot(axis,box.axis[1]))+sat_epsilon)*box.half.y+
               (std::abs(dot(axis,box.axis[2]))+sat_epsilon)*box.half.z;
    };
    for(unsigned i=0;i<6;++i) {
        const bool first=i<3;
        const Box& reference=first?a:b, &incident=first?b:a;
        const unsigned axis=i%3;
        const Vec3 n=reference.axis[axis];
        sign[i]=first?dot(delta,n):dot(delta,n);
        overlap[i]=extent(reference.half,axis)+radius(incident,n)-
                   std::abs(sign[i])+contact_distance;
        if(overlap[i]<0.0f) {
            count=0;
            return false;
        }
    }
    // The original cross-axis test uses an unnormalized axis and
    // adds contactDistance without dividing by its length.
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j) {
        const Vec3 cross_axis=cross(a.axis[i],b.axis[j]);
        // The source's nine cross-axis projections omit their own
        // parallel axes: the 1e-6 stabilization belongs only to the
        // other two half-extents on each body.
        float ra=0.0f,rb=0.0f;
        for(unsigned k=0;k<3;++k) {
            if(k!=i)
                ra+=(std::abs(dot(cross_axis,a.axis[k]))+
                     sat_epsilon)*extent(a.half,k);
            if(k!=j)
                rb+=(std::abs(dot(cross_axis,b.axis[k]))+
                     sat_epsilon)*extent(b.half,k);
        }
        if(std::abs(dot(delta,cross_axis))>
           ra+rb+contact_distance) {
            count=0;
            return false;
        }
    }
    unsigned feature=0;
    float min_overlap=overlap[0];
    for(unsigned i=1;i<6;++i)
        if(min_overlap>overlap[i]) {
            min_overlap=overlap[i];
            feature=i;
        }
    const bool flipped=feature>=3;
    const Box& incident=flipped?a:b;
    const bool nonpositive=sign[feature]<=0.0f;
    Frame frame{};
    Vec3 normal{};
    float face_x=0.0f,face_y=0.0f;
    // Source switch(feature): preserve face bases, origin and sign.
    switch(feature) {
    case 0:
        normal=nonpositive?a.axis[0]:negate(a.axis[0]);
        frame={nonpositive?negate(a.axis[2]):a.axis[2],a.axis[1],normal,
               add(a.position,mul(a.axis[0],nonpositive?-a.half.x:a.half.x))};
        face_x=a.half.z;face_y=a.half.y;break;
    case 1:
        normal=nonpositive?a.axis[1]:negate(a.axis[1]);
        frame={a.axis[0],nonpositive?negate(a.axis[2]):a.axis[2],normal,
               add(a.position,mul(a.axis[1],nonpositive?-a.half.y:a.half.y))};
        face_x=a.half.x;face_y=a.half.z;break;
    case 2:
        normal=nonpositive?a.axis[2]:negate(a.axis[2]);
        frame={a.axis[0],nonpositive?a.axis[1]:negate(a.axis[1]),normal,
               add(a.position,mul(a.axis[2],nonpositive?-a.half.z:a.half.z))};
        face_x=a.half.x;face_y=a.half.y;break;
    case 3:
        normal=nonpositive?b.axis[0]:negate(b.axis[0]);
        frame={nonpositive?b.axis[2]:negate(b.axis[2]),b.axis[1],
               negate(normal),add(b.position,
                  mul(b.axis[0],nonpositive?b.half.x:-b.half.x))};
        face_x=b.half.z;face_y=b.half.y;break;
    case 4:
        normal=nonpositive?b.axis[1]:negate(b.axis[1]);
        frame={b.axis[0],nonpositive?b.axis[2]:negate(b.axis[2]),
               negate(normal),add(b.position,
                  mul(b.axis[1],nonpositive?b.half.y:-b.half.y))};
        face_x=b.half.x;face_y=b.half.z;break;
    case 5:
        normal=nonpositive?b.axis[2]:negate(b.axis[2]);
        frame={b.axis[0],nonpositive?negate(b.axis[1]):b.axis[1],
               negate(normal),add(b.position,
                  mul(b.axis[2],nonpositive?b.half.z:-b.half.z))};
        face_x=b.half.x;face_y=b.half.y;break;
    default: throw std::logic_error("Invalid PCM reference face");
    }
    // GuPCMContactBoxBox::getIncidentPolygon: choose the incident face
    // most aligned with the search axis and emit its four source corners.
    // Both source branches search along reference-frame -Z.
    // Tie preference: X >= Y,Z; otherwise Y >= Z.
    const Vec3 search=negate(frame.z);
    const float dx=std::abs(dot(search,incident.axis[0]));
    const float dy=std::abs(dot(search,incident.axis[1]));
    const float dz=std::abs(dot(search,incident.axis[2]));
    const unsigned incident_axis=dx>=dy && dx>=dz?0:(dy>=dz?1:2);
    const float facing=dot(search,incident.axis[incident_axis]);
    const float signed_extent=facing>0.0f?
        -extent(incident.half,incident_axis):
        extent(incident.half,incident_axis);
    const Vec3 face_center=add(incident.position,
                       mul(incident.axis[incident_axis],signed_extent));
    const unsigned u_axis=incident_axis==0?1:0;
    const unsigned v_axis=incident_axis==2?1:2;
    const Vec3 u=mul(incident.axis[u_axis],extent(incident.half,u_axis));
    const Vec3 v=mul(incident.axis[v_axis],extent(incident.half,v_axis));
    const std::array<Vec3,4> polygon_world{{
        add(face_center,add(u,v)),
        add(face_center,sub(u,v)),
        sub(face_center,add(u,v)),
        sub(face_center,sub(u,v))}};
    std::array<Vec3,4> polygon{};
    for(unsigned i=0;i<4;++i)
        polygon[i]=frame.project(polygon_world[i]);
    const Vec3 face_normal=frame.project(
        add(frame.position,
            facing>0.0f?negate(incident.axis[incident_axis]):
                         incident.axis[incident_axis]));
    gen.calculate(frame,polygon,face_normal,face_x,face_y,
                  normal,contact_distance,flipped);
    count=gen.count;
    return true;
}
} // namespace fyzz::native
