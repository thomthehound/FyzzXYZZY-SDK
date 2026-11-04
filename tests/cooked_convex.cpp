// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/convex.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace fyzz;
namespace {
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
void near(double a,double b,double tolerance,const char* message) {
    require(std::isfinite(a)&&std::abs(a-b)<tolerance,message);
}
constexpr std::array<ConvexTriangle,4> tetra_faces{{
    {0,2,1},{0,1,3},{0,3,2},{1,2,3}}};
constexpr std::array<Vec3,4> tetra_vertices{{
    {0,0,0},{1,0,0},{0,1,0},{0,0,1}}};

void verify_tetra() {
    const auto mesh=cook_convex_mesh(tetra_vertices,tetra_faces);
    const auto& mass=mesh->mass_properties();
    near(mass.volume,1.0/6.0,1.e-8,"Tetrahedron volume");
    near(mass.center.x,0.25,1.e-6,"Tetrahedron COM x");
    near(mass.center.y,0.25,1.e-6,"Tetrahedron COM y");
    near(mass.center.z,0.25,1.e-6,"Tetrahedron COM z");
    near(mass.inertia_columns[0].x,1.0/80.0,1.e-7,
         "Tetrahedron density-one Ixx");
    near(mass.inertia_columns[1].x,1.0/480.0,1.e-7,
         "Tetrahedron density-one Ixy");
    near(mass.inertia_columns[2].x,1.0/480.0,1.e-7,
         "Tetrahedron density-one Ixz");
    near(mass.inertia_columns[1].y,1.0/80.0,1.e-7,
         "Tetrahedron density-one Iyy");
    near(mass.inertia_columns[2].z,1.0/80.0,1.e-7,
         "Tetrahedron density-one Izz");
    std::array<Vec3,4> translated=tetra_vertices;
    for(auto& v:translated) {
        v.x+=10000.f;v.y-=20000.f;v.z+=30000.f;
    }
    const auto offset=cook_convex_mesh(translated,tetra_faces);
    const auto& moved=offset->mass_properties();
    near(moved.volume,mass.volume,1.e-8,"Translated tetrahedron volume");
    near(moved.center.x,10000.25,1.e-3,"Translated COM x");
    near(moved.center.y,-19999.75,1.e-3,"Translated COM y");
    near(moved.center.z,30000.25,1.e-3,"Translated COM z");
    near(moved.inertia_columns[0].x,mass.inertia_columns[0].x,1.e-7,
         "Translated tetrahedron inertia");
}

void verify_cube() {
    constexpr std::array<Vec3,8> vertices{{
        {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
        {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}}};
    constexpr std::array<ConvexTriangle,12> faces{{
        {0,2,1},{0,3,2},{4,5,6},{4,6,7},
        {0,1,5},{0,5,4},{3,7,6},{3,6,2},
        {0,4,7},{0,7,3},{1,2,6},{1,6,5}}};
    const auto mesh=cook_convex_mesh(vertices,faces);
    const auto& mass=mesh->mass_properties();
    near(mass.volume,8.,1.e-7,"Cube volume");
    near(mass.center.x,0.,1.e-7,"Cube COM");
    near(mass.inertia_columns[0].x,16./3.,1.e-6,"Cube inertia x");
    near(mass.inertia_columns[1].y,16./3.,1.e-6,"Cube inertia y");
    near(mass.inertia_columns[2].z,16./3.,1.e-6,"Cube inertia z");
    near(mass.inertia_columns[1].x,0.,1.e-7,"Cube product of inertia");
}

template<class F>void rejects(F&& f,const char* message) {
    try {f();}catch(const std::invalid_argument&){return;}
    throw std::runtime_error(message);
}
void verify_wedge() {
    // A triangular prism has an asymmetric mass frame and nonzero
    // products of inertia; its upper and side faces are triangulated.
    constexpr std::array<Vec3,6> vertices{{
        {0,0,0},{1,0,0},{0,1,0},
        {0,0,2},{1,0,2},{0,1,2}}};
    constexpr std::array<ConvexTriangle,8> faces{{
        {0,2,1},{3,4,5},
        {0,1,4},{0,4,3},
        {0,3,5},{0,5,2},
        {1,2,5},{1,5,4}}};
    const auto hull=cook_convex_mesh(vertices,faces);
    const auto& mass=hull->mass_properties();
    near(mass.volume,1.0,1.e-8,"Wedge volume");
    near(mass.center.x,1.0/3.0,1.e-6,"Wedge COM x");
    near(mass.center.y,1.0/3.0,1.e-6,"Wedge COM y");
    near(mass.center.z,1.0,1.e-6,"Wedge COM z");
    near(mass.inertia_columns[0].x,7.0/18.0,1.e-6,"Wedge Ixx");
    near(mass.inertia_columns[1].y,7.0/18.0,1.e-6,"Wedge Iyy");
    near(mass.inertia_columns[2].z,1.0/9.0,1.e-6,"Wedge Izz");
    near(mass.inertia_columns[1].x,1.0/36.0,1.e-6,"Wedge Ixy");
}

void verify_native_rigid_contact() {
    World world({.execution=ExecutionMode::NativeCpu,
                 .contact_notifications=true});
    world.add_static_floor(0.f);
    const auto hull=cook_convex_mesh(tetra_vertices,tetra_faces);
    const auto body=world.create_dynamic_body({{0.f,-0.025f,0.f},{}});
    world.attach_convex_shape(body,hull);
    world.update_mass_properties(body,6.f);
    const auto mass=world.body_mass_properties(body);
    near(mass.inverse_mass,1.0,1.e-5,
         "Native convex body did not receive cooked volume mass");
    near(mass.mass_local_pose.position.x,0.25,1.e-5,
         "Native convex body COM differs from cooked hull");
    near(mass.mass_local_pose.position.y,0.25,1.e-5,
         "Native convex body COM y differs from cooked hull");
    world.set_body_velocity(body,{0.f,-1.f,0.f});
    for(unsigned frame=0u;frame<5u;++frame)
        world.step(1.f/120.f);
    const auto state=world.body_state(body);
    require(std::isfinite(state.position.y) &&
            std::isfinite(state.velocity.y) &&
            state.position.y>-0.15f,
            "Cooked convex rigid contact did not constrain body motion");
}

void verify_invalid() {
    auto open=tetra_faces;
    rejects([&]{cook_convex_mesh(tetra_vertices,
        std::span<const ConvexTriangle>(open.data(),3u));},
        "Open tetrahedron admitted");
    auto inward=tetra_faces;
    inward[3]={1,3,2};
    rejects([&]{cook_convex_mesh(tetra_vertices,inward);},
        "Inward face admitted");
    auto bad=tetra_vertices;
    bad[3].z=std::numeric_limits<float>::infinity();
    rejects([&]{cook_convex_mesh(bad,tetra_faces);},
        "Nonfinite vertex admitted");
    auto repeat=tetra_faces;
    repeat[0]={0,2,2};
    rejects([&]{cook_convex_mesh(tetra_vertices,repeat);},
        "Degenerate triangle admitted");
    constexpr std::array<Vec3,5> pyramid{{
        {-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0},{0,0,1}}};
    constexpr std::array<ConvexTriangle,6> concave{{
        {0,3,2},{0,2,1},{0,1,4},{1,2,4},{2,3,4},{3,0,4}}};
    auto pinched=pyramid;
    pinched[4].z=-0.25f;
    rejects([&]{cook_convex_mesh(pinched,concave);},
        "Concave or inverted convex mesh admitted");
}
}
int main() {
    try {
        verify_tetra();
        verify_cube();
        verify_wedge();
        verify_invalid();
        verify_native_rigid_contact();
        std::cout<<"Cooked convex volume, inertia and topology OK\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
