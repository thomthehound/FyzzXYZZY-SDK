// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "native/scene.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fyzz::native {
namespace {
Vec3 subtract(Vec3 a,Vec3 b) {
    return {a.x-b.x,a.y-b.y,a.z-b.z};
}
Vec3 add(Vec3 a,Vec3 b) {
    return {a.x+b.x,a.y+b.y,a.z+b.z};
}
Vec3 scale(Vec3 a,float v) {
    return {a.x*v,a.y*v,a.z*v};
}
float dot(Vec3 a,Vec3 b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
Vec3 cross(Vec3 a,Vec3 b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
Vec3 rotate(Quat q,Vec3 v) {
    const Vec3 u{q.x,q.y,q.z};
    const Vec3 t=scale(cross(u,v),2.f);
    return add(v,add(scale(t,q.w),cross(u,t)));
}
Vec3 inverse_rotate(Quat q,Vec3 v) {
    return rotate({-q.x,-q.y,-q.z,q.w},v);
}
std::uint8_t publish(ContactPoint point,std::span<ContactPoint> output) {
    if(output.empty())
        throw std::logic_error("Gu sphere contact output capacity is zero");
    if(!std::isfinite(point.position.x) || !std::isfinite(point.position.y) ||
       !std::isfinite(point.position.z) || !std::isfinite(point.normal.x) ||
       !std::isfinite(point.normal.y) || !std::isfinite(point.normal.z) ||
       !std::isfinite(point.separation))
        throw std::overflow_error("Gu sphere contact is not finite");
    output[0]=point;
    return 1u;
}
}

std::uint8_t source_pcm_sphere_sphere(
    Vec3 center0,float radius0,Vec3 center1,float radius1,
    float contact_distance,std::span<ContactPoint> output) {
    // Gu::pcmContactSphereSphere: same strict inflated-distance test,
    // singularity normal and shape1 surface witness.
    const Vec3 delta=subtract(center0,center1);
    const float distance_squared=dot(delta,delta);
    const float radius_sum=radius0+radius1;
    const float inflated=radius_sum+contact_distance;
    if(!(inflated*inflated>distance_squared))return 0u;
    const float distance=std::sqrt(distance_squared);
    const Vec3 normal=distance<=0.00001f
        ? Vec3{1.f,0.f,0.f}:scale(delta,1.f/distance);
    return publish({
        add(center1,scale(normal,radius1)),
        normal,distance-radius_sum},output);
}

std::uint8_t source_pcm_sphere_plane(
    Vec3 center,float radius,float plane_height,
    float contact_distance,std::span<ContactPoint> output) {
    // Gu::pcmContactSpherePlane: the native horizontal plane corresponds
    // to plane-local +X, with a world +Y normal.
    const float separation=center.y-plane_height-radius;
    if(!(contact_distance>=separation))return 0u;
    return publish({
        {center.x,center.y-radius,center.z},
        {0.f,1.f,0.f},separation},output);
}

std::uint8_t source_pcm_sphere_box(
    Vec3 center,float radius,Vec3 box_center,Quat box_orientation,
    Vec3 box_half_extents,float contact_distance,
    std::span<ContactPoint> output) {
    // Gu::pcmContactSphereBox: the closest point on the oriented box
    // determines the witness and outward normal, with the embedded
    // center case selecting the nearest exit face.
    const Vec3 local=inverse_rotate(
        box_orientation,subtract(center,box_center));
    const Vec3 closest{
        std::clamp(local.x,-box_half_extents.x,box_half_extents.x),
        std::clamp(local.y,-box_half_extents.y,box_half_extents.y),
        std::clamp(local.z,-box_half_extents.z,box_half_extents.z)};
    const Vec3 gap=subtract(local,closest);
    const float length_squared=dot(gap,gap);
    const float inflated=radius+contact_distance;
    if(!(inflated*inflated>length_squared))return 0u;

    const bool inside=
        std::abs(local.x)<=box_half_extents.x &&
        std::abs(local.y)<=box_half_extents.y &&
        std::abs(local.z)<=box_half_extents.z;
    if(inside) {
        const Vec3 distance{
            box_half_extents.x-std::abs(closest.x),
            box_half_extents.y-std::abs(closest.y),
            box_half_extents.z-std::abs(closest.z)};
        Vec3 normal_local{};
        float depth;
        if(distance.z<=distance.x && distance.z<=distance.y) {
            normal_local={0.f,0.f,closest.z>=0.f?1.f:-1.f};
            depth=distance.z;
        }else if(distance.x<=distance.y && distance.x<=distance.z) {
            normal_local={closest.x>=0.f?1.f:-1.f,0.f,0.f};
            depth=distance.x;
        }else {
            normal_local={0.f,closest.y>=0.f?1.f:-1.f,0.f};
            depth=distance.y;
        }
        const Vec3 normal=rotate(box_orientation,normal_local);
        return publish({
            add(center,scale(normal,depth)),
            normal,-depth-radius},output);
    }

    const float length=std::sqrt(length_squared);
    const Vec3 normal=rotate(box_orientation,scale(gap,1.f/length));
    return publish({
        add(box_center,rotate(box_orientation,closest)),
        normal,length-radius},output);
}
} // namespace fyzz::native
