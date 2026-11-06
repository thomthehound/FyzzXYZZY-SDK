// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/convex.h"
#include "fyzzxyzzy/world.h"

#include <array>
#include <cmath>
#include <stdexcept>

using namespace fyzz;
namespace {
void require(bool value,const char* message) {
    if(!value)throw std::runtime_error(message);
}
void near(float actual,float expected,float tolerance,const char* message) {
    require(std::isfinite(actual) && std::abs(actual-expected)<=tolerance,message);
}
void verify_boxes_and_attachment_poses() {
    World world({.execution=ExecutionMode::NativeCpu});
    const auto body=world.add_static_box({1.f,1.f,1.f},{0.f,0.f,0.f});
    auto hit=world.raycast({-3.f,0.f,0.f},{2.f,0.f,0.f},10.f);
    require(hit && hit->body==body,"Native scene box ray lost its body");
    near(hit->distance,2.f,1.e-5f,"Native box ray entry distance");
    near(hit->normal.x,-1.f,1.e-5f,"Native box ray normal");
    near(hit->position.x,-1.f,1.e-5f,"Native box ray position");
    require(!world.raycast({-3.f,0.f,0.f},{1.f,0.f,0.f},1.99f),
        "Ray hit beyond its maximum distance");
    require(!world.raycast({-3.f,0.f,0.f},{-1.f,0.f,0.f},10.f),
        "Ray pointing away from a box hit it");
    hit=world.raycast({0.f,0.f,0.f},{1.f,0.f,0.f},10.f);
    require(hit.has_value(),"Ray starting inside box missed");
    near(hit->distance,0.f,1.e-6f,"Inside-box initial hit distance");
    near(hit->normal.x,-1.f,1.e-5f,"Inside-box initial hit normal");

    constexpr float s=0.7071067811865475f;
    const auto compound=world.create_static_body({{5.f,1.f,0.f},{0.f,0.f,s,s}});
    const auto attached=world.attach_box_shape(
        compound,{1.f,0.5f,0.5f},{{1.f,0.f,0.f},{}});
    hit=world.raycast({5.f,-2.f,0.f},{0.f,1.f,0.f},8.f);
    require(hit && hit->shape==attached && hit->body==compound,
        "Rotated compound attachment lost shape identity");
    near(hit->distance,3.f,1.e-4f,"Rotated compound box ray distance");
    near(hit->normal.y,-1.f,1.e-5f,"Rotated compound box ray normal");
    world.set_shape_local_pose(attached,{{2.f,0.f,0.f},{}});
    hit=world.raycast({5.f,-2.f,0.f},{0.f,1.f,0.f},8.f);
    require(hit && hit->shape==attached,"Moving compound shape lost query");
    near(hit->distance,4.f,1.e-4f,"Updated attachment pose not queried");
    world.set_body_pose(compound,{6.f,1.f,0.f},{0.f,0.f,s,s});
    hit=world.raycast({6.f,-2.f,0.f},{0.f,1.f,0.f},8.f);
    require(hit && hit->shape==attached,"Body pose edit lost query");
    near(hit->distance,4.f,1.e-4f,"Updated actor pose not queried");
}
void verify_sphere_flags_and_lifetime() {
    World world({.execution=ExecutionMode::NativeCpu});
    const auto body=world.create_static_body({});
    const auto shape=world.attach_sphere_shape(
        body,1.f,{{4.f,0.f,0.f},{}});
    auto hit=world.raycast({0.f,0.f,0.f},{1.f,0.f,0.f},10.f);
    require(hit && hit->shape==shape,"Attached sphere did not enter scene query");
    near(hit->distance,3.f,1.e-5f,"Sphere ray distance");
    near(hit->normal.x,-1.f,1.e-5f,"Sphere surface normal");

    world.set_shape_simulation_enabled(shape,false);
    require(world.raycast({0.f,0.f,0.f},{1.f,0.f,0.f},10.f).has_value(),
        "Simulation flag incorrectly disabled scene query");
    world.set_shape_query_enabled(shape,false);
    require(!world.raycast({0.f,0.f,0.f},{1.f,0.f,0.f},10.f),
        "Query-disabled attachment remained visible");
    world.set_shape_query_enabled(shape,true);
    require(world.raycast({0.f,0.f,0.f},{1.f,0.f,0.f},10.f).has_value(),
        "Query-enabled attachment was not readmitted");
    world.detach_shape(shape);
    require(!world.raycast({0.f,0.f,0.f},{1.f,0.f,0.f},10.f),
        "Retired shape was returned by scene query");
}
void verify_planes() {
    World world({.execution=ExecutionMode::NativeCpu});
    const auto floor=world.add_static_floor(1.f);
    auto hit=world.raycast({0.f,3.f,0.f},{0.f,-1.f,0.f},10.f);
    require(hit && hit->body==floor,"Floor ray lost plane actor");
    near(hit->distance,2.f,1.e-5f,"Floor ray distance");
    near(hit->normal.y,1.f,1.e-5f,"Floor ray normal from above");
    hit=world.raycast({0.f,-1.f,0.f},{0.f,1.f,0.f},10.f);
    require(hit.has_value(),"Floor ray from below missed");
    near(hit->distance,2.f,1.e-5f,"Floor ray from below distance");
    near(hit->normal.y,-1.f,1.e-5f,"Floor ray normal from below");
    require(!world.raycast({0.f,3.f,0.f},{1.f,0.f,0.f},10.f),
        "Parallel ray hit an infinite plane");
}
void verify_cooked_convex() {
    constexpr std::array<Vec3,8> vertices{{
        {99.f,-1.f,-1.f},{101.f,-1.f,-1.f},
        {101.f,1.f,-1.f},{99.f,1.f,-1.f},
        {99.f,-1.f,1.f},{101.f,-1.f,1.f},
        {101.f,1.f,1.f},{99.f,1.f,1.f}}};
    constexpr std::array<ConvexTriangle,12> triangles{{
        {0,2,1},{0,3,2},{4,5,6},{4,6,7},
        {0,1,5},{0,5,4},{3,7,6},{3,6,2},
        {0,4,7},{0,7,3},{1,2,6},{1,6,5}}};
    const auto mesh=cook_convex_mesh(vertices,triangles);
    World world({.execution=ExecutionMode::NativeCpu});
    const auto body=world.create_static_body({{-100.f,0.f,0.f},{}});
    const auto shape=world.attach_convex_shape(body,mesh);
    auto hit=world.raycast({-3.f,0.f,0.f},{1.f,0.f,0.f},10.f);
    require(hit && hit->body==body && hit->shape==shape,
        "Cooked convex ray lost geometry or attachment");
    near(hit->distance,2.f,1.e-4f,"Cooked convex face ray distance");
    near(hit->normal.x,-1.f,1.e-5f,"Cooked convex ray normal");
    require(!world.raycast({-3.f,3.f,0.f},{1.f,0.f,0.f},10.f),
        "Cooked convex exterior ray produced a hit");
    hit=world.raycast({0.f,0.f,0.f},{0.f,0.f,1.f},10.f);
    require(hit.has_value(),"Inside convex ray missed");
    near(hit->distance,0.f,1.e-6f,"Inside convex hit distance");
    near(hit->normal.z,-1.f,1.e-5f,"Inside convex hit normal");
}
void verify_input_validation() {
    World world({.execution=ExecutionMode::NativeCpu});
    bool rejected=false;
    try {world.raycast({},{},10.f);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"Zero ray direction not rejected");
    rejected=false;
    try {world.raycast({},{1.f,0.f,0.f},0.f);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"Zero ray distance not rejected");
}
}
int main() {
    verify_boxes_and_attachment_poses();
    verify_sphere_flags_and_lifetime();
    verify_planes();
    verify_cooked_convex();
    verify_input_validation();
}
