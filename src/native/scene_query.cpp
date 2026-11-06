// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "native/scene.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace fyzz::native {
namespace {

float dot(Vec3 a,Vec3 b) noexcept {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
Vec3 subtract(Vec3 a,Vec3 b) noexcept {
    return {a.x-b.x,a.y-b.y,a.z-b.z};
}
Vec3 add_scaled(Vec3 a,Vec3 b,float distance) noexcept {
    return {a.x+b.x*distance,a.y+b.y*distance,a.z+b.z*distance};
}
Vec3 negate(Vec3 a) noexcept {
    return {-a.x,-a.y,-a.z};
}
Vec3 cross(Vec3 a,Vec3 b) noexcept {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
Vec3 rotate(Quat q,Vec3 v) noexcept {
    const Vec3 u{q.x,q.y,q.z};
    const Vec3 c=cross(u,v);
    const Vec3 t{2.f*c.x,2.f*c.y,2.f*c.z};
    const Vec3 d=cross(u,t);
    return {v.x+q.w*t.x+d.x,
            v.y+q.w*t.y+d.y,
            v.z+q.w*t.z+d.z};
}
Vec3 rotate_inverse(Quat q,Vec3 v) noexcept {
    return rotate({-q.x,-q.y,-q.z,q.w},v);
}
struct LocalHit {
    float distance=0.f;
    Vec3 normal{};
};

// Intersect an inward half-space using the source geometry's exact
// supporting planes; no enlarged SAP bounds or contact offsets enter queries.
bool clip_plane(Vec3 normal,float distance,
                Vec3 origin,Vec3 direction,
                float& entry,float& exit,Vec3& entry_normal) noexcept {
    const float signed_distance=distance-dot(normal,origin);
    const float slope=dot(normal,direction);
    if(slope==0.f)return signed_distance>=0.f;
    const float t=signed_distance/slope;
    if(slope<0.f) {
        if(t>entry) {
            entry=t;
            entry_normal=normal;
        }
    } else {
        exit=std::min(exit,t);
    }
    return entry<=exit;
}

std::optional<LocalHit> ray_box(
    Vec3 origin,Vec3 direction,Vec3 half,float limit) noexcept {
    float entry=0.f,exit=limit;
    Vec3 normal=negate(direction);
    if(!clip_plane({1.f,0.f,0.f},half.x,origin,direction,entry,exit,normal) ||
       !clip_plane({-1.f,0.f,0.f},half.x,origin,direction,entry,exit,normal) ||
       !clip_plane({0.f,1.f,0.f},half.y,origin,direction,entry,exit,normal) ||
       !clip_plane({0.f,-1.f,0.f},half.y,origin,direction,entry,exit,normal) ||
       !clip_plane({0.f,0.f,1.f},half.z,origin,direction,entry,exit,normal) ||
       !clip_plane({0.f,0.f,-1.f},half.z,origin,direction,entry,exit,normal))
        return std::nullopt;
    return LocalHit{entry,normal};
}

std::optional<LocalHit> ray_convex(
    Vec3 origin,Vec3 direction,const ConvexMesh& mesh,float limit) noexcept {
    float entry=0.f,exit=limit;
    Vec3 normal=negate(direction);
    for(const ConvexFace& face:mesh.faces())
        if(!clip_plane(face.normal,face.distance,
            origin,direction,entry,exit,normal))
            return std::nullopt;
    return LocalHit{entry,normal};
}

std::optional<LocalHit> ray_sphere(
    Vec3 origin,Vec3 direction,float radius,float limit) noexcept {
    const float c=dot(origin,origin)-radius*radius;
    if(c<=0.f)return LocalHit{0.f,negate(direction)};
    const float b=dot(origin,direction);
    if(b>=0.f)return std::nullopt;
    // Project first: b*b-c subtracts nearly equal large values for
    // a distant small sphere and can erase its radius in float arithmetic.
    const Vec3 closest=add_scaled(origin,direction,-b);
    const float radial_square=radius*radius-dot(closest,closest);
    if(radial_square<0.f)return std::nullopt;
    const float distance=-b-std::sqrt(radial_square);
    if(!(distance>=0.f) || distance>limit)return std::nullopt;
    const Vec3 point=add_scaled(origin,direction,distance);
    const float length=std::sqrt(dot(point,point));
    if(!(length>0.f))return std::nullopt;
    return LocalHit{distance,{point.x/length,point.y/length,point.z/length}};
}
} // namespace

void Scene::set_attached_shape_query_enabled(ShapeId id,bool enabled) {
    require_shape(id).query_shape_enabled=enabled;
}

std::optional<RaycastHit> Scene::raycast(
    Vec3 origin,Vec3 direction,float max_distance) const {
    const float direction_scale=std::max({
        std::abs(direction.x),std::abs(direction.y),std::abs(direction.z)});
    if(!std::isfinite(origin.x) || !std::isfinite(origin.y) ||
       !std::isfinite(origin.z) || !std::isfinite(direction.x) ||
       !std::isfinite(direction.y) || !std::isfinite(direction.z) ||
       !(direction_scale>0.f) || !std::isfinite(max_distance) ||
       !(max_distance>0.f))
        throw std::invalid_argument("Raycast requires finite origin, nonzero direction and positive range");

    // Normalize before squaring: every finite nonzero float direction is
    // valid, including magnitudes outside the range of a float dot product.
    direction={direction.x/direction_scale,direction.y/direction_scale,
               direction.z/direction_scale};
    const float length=std::sqrt(dot(direction,direction));
    direction={direction.x/length,direction.y/length,direction.z/length};
    std::optional<RaycastHit> closest;

    for(const ShapeId id:live_shape_ids_) {
        const Shape& shape=shapes_[id];
        if(!shape.query_shape_enabled)continue;
        const float limit=closest?closest->distance:max_distance;
        std::optional<LocalHit> local;

        if(shape.geometry==Geometry::Plane) {
            if(direction.y==0.f)continue;
            const float t=(shape.position.y-origin.y)/direction.y;
            if(t<0.f || t>limit)continue;
            const Vec3 normal=direction.y<0.f
                ?Vec3{0.f,1.f,0.f}:Vec3{0.f,-1.f,0.f};
            local=LocalHit{t,normal};
        } else {
            const Vec3 local_origin=rotate_inverse(
                shape.orientation,subtract(origin,shape.position));
            const Vec3 local_direction=rotate_inverse(shape.orientation,direction);
            switch(shape.geometry) {
            case Geometry::Box:
                local=ray_box(local_origin,local_direction,shape.half_extents,limit);
                break;
            case Geometry::Sphere:
                local=ray_sphere(local_origin,local_direction,shape.radius,limit);
                break;
            case Geometry::Convex:
                local=ray_convex(local_origin,local_direction,*shape.convex,limit);
                break;
            case Geometry::Plane:
                break;
            }
            if(local)
                local->normal=rotate(shape.orientation,local->normal);
        }
        if(!local || local->distance<0.f || local->distance>limit)
            continue;
        if(closest && (local->distance>closest->distance ||
           (local->distance==closest->distance && id>closest->shape)))
            continue;
        closest=RaycastHit{
            shape.body,id,add_scaled(origin,direction,local->distance),
            local->normal,local->distance};
    }
    return closest;
}
} // namespace fyzz::native
