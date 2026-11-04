// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/convex.h"

#include <algorithm>
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
constexpr std::array<Vec3,8> cube_vertices{{
    {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
    {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}}};
constexpr std::array<ConvexTriangle,12> cube_faces{{
    {0,2,1},{0,3,2},{4,5,6},{4,6,7},
    {0,1,5},{0,5,4},{3,7,6},{3,6,2},
    {0,4,7},{0,7,3},{1,2,6},{1,6,5}}};

void verify_faces(const ConvexMesh& mesh,std::size_t expected) {
    const auto faces=mesh.faces();
    const auto indices=mesh.face_indices();
    const auto vertices=mesh.vertices();
    require(faces.size()==expected,"Cooked planar polygon count");
    for(const auto& face:faces) {
        require(face.index_count>=3u && face.first_index<=indices.size() &&
                face.index_count<=indices.size()-face.first_index,
                "Cooked polygon index window invalid");
        const auto n=face.normal;
        near(double(n.x)*n.x+double(n.y)*n.y+double(n.z)*n.z,
             1.0,1.e-5,"Cooked face normal not unit length");
        double twice_area=0.0;
        const auto a=vertices[indices[face.first_index]];
        for(std::uint32_t i=0u;i<face.index_count;++i) {
            const auto index=indices[face.first_index+i];
            require(index<vertices.size(),"Cooked face vertex outside hull");
            for(std::uint32_t j=0u;j<i;++j)
                require(index!=indices[face.first_index+j],
                        "Cooked polygon repeats a boundary vertex");
            const auto v=vertices[index];
            near(double(n.x)*v.x+double(n.y)*v.y+
                 double(n.z)*v.z,face.distance,1.e-4,
                 "Cooked face vertex not in its plane");
            if(i>=2u) {
                const auto b=vertices[indices[face.first_index+i-1u]];
                const auto c=v;
                const Vec3 e{b.x-a.x,b.y-a.y,b.z-a.z};
                const Vec3 f{c.x-a.x,c.y-a.y,c.z-a.z};
                twice_area+=double(n.x)*(double(e.y)*f.z-double(e.z)*f.y)+
                    double(n.y)*(double(e.z)*f.x-double(e.x)*f.z)+
                    double(n.z)*(double(e.x)*f.y-double(e.y)*f.x);
            }
        }
        require(twice_area>1.e-6,
                "Cooked polygon lost its outward boundary winding");
    }
}
void verify_tetra() {
    const auto mesh=cook_convex_mesh(tetra_vertices,tetra_faces);
    verify_faces(*mesh,4u);
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
    const auto mesh=cook_convex_mesh(cube_vertices,cube_faces);
    verify_faces(*mesh,6u);
    for(const auto& face:mesh->faces())
        require(face.index_count==4u,
                "Cube face split across triangulation diagonals");
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
    verify_faces(*hull,5u);
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

void verify_pentagonal_face() {
    constexpr std::array<Vec3,6> vertices{{
        {-1,0,-1},{-1,0,1},{0,0,1.5f},
        {1,0,1},{1,0,-1},{0,1,0}}};
    constexpr std::array<ConvexTriangle,8> faces{{
        {0,2,1},{0,3,2},{0,4,3},
        {0,1,5},{1,2,5},{2,3,5},{3,4,5},{4,0,5}}};
    const auto hull=cook_convex_mesh(vertices,faces);
    verify_faces(*hull,6u);
    require(std::any_of(hull->faces().begin(),hull->faces().end(),
        [](const ConvexFace& face){return face.index_count==5u;}),
        "Cooked pentagon failed to merge coplanar triangles");
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

void verify_sphere_face_contact() {
    // The local vertex origin may be far outside a cooked hull without
    // changing its physical size, contact geometry, or margin thresholds.
    for(float origin:std::array{0.f,100.f}) {
        auto vertices=cube_vertices;
        for(auto& vertex:vertices) {
            vertex.x+=origin;
            vertex.y+=origin;
            vertex.z+=origin;
        }
        World world({.execution=ExecutionMode::NativeCpu,
                     .contact_notifications=true});
        const auto cube=cook_convex_mesh(vertices,cube_faces);
        const auto fixed=world.create_static_body({
            {-origin,-origin,-origin},{}});
        world.attach_convex_shape(fixed,cube);
        const auto sphere=world.add_dynamic_sphere(
            0.2f,{0.273157f,-0.794949f,-1.0039f},1.f);
        RigidBodyDynamics settings{};
        settings.gravity_enabled=false;
        settings.position_iterations=24u;
        world.set_body_dynamics(sphere,settings);
        world.step(1.f/120.f);
        const auto events=world.contact_events();
        require(events.size()==1u &&
                events[0].kind==ContactEventKind::Found &&
                events[0].contact_count>0u,
                "Near-face convex/sphere penetration lost its contact manager");
        const auto points=world.contact_event_points();
        require(events[0].point_offset<points.size(),
                "Near-face contact lost its published witness");
        const auto& point=points[events[0].point_offset];
        near(point.separation,-0.1961,0.015,
             "Near-face sphere penetration differs from geometric depth");
        require(std::abs(point.normal.z)>0.95f,
                "Near-face sphere contact normal is not face-aligned");
    }
}

void verify_off_origin_hull_pair() {
    auto vertices=tetra_vertices;
    for(auto& vertex:vertices) {
        vertex.x+=100.f;
        vertex.y+=100.f;
        vertex.z+=100.f;
    }
    World world({.execution=ExecutionMode::NativeCpu,
                 .contact_notifications=true});
    const auto mesh=cook_convex_mesh(vertices,tetra_faces);
    const auto fixed=world.create_static_body({{-72.f,-100.f,-100.f},{}});
    world.attach_convex_shape(fixed,mesh);
    const auto moving=world.create_dynamic_body({
        {-71.85f,-99.88f,-99.92f},{}});
    world.attach_convex_shape(moving,mesh);
    world.update_mass_properties(moving,1.f);
    RigidBodyDynamics settings{};
    settings.gravity_enabled=false;
    settings.position_iterations=24u;
    world.set_body_dynamics(moving,settings);
    world.step(1.f/120.f);
    const auto contacts=world.contact_events();
    require(contacts.size()==1u &&
            contacts[0].kind==ContactEventKind::Found &&
            contacts[0].contact_count>0u,
            "Off-origin hull pair lost finite convex contact");
    const auto points=world.contact_event_points();
    const auto& contact=points[contacts[0].point_offset];
    require(contact.position.x>27.f && contact.position.x<30.f &&
            contact.separation<0.f &&
            std::isfinite(contact.normal.x),
            "Off-origin convex pair lost physical contact geometry");
}

void verify_convex_face_manifold() {
    const auto hull=cook_convex_mesh(cube_vertices,cube_faces);
    World world({.execution=ExecutionMode::NativeCpu,
                 .contact_notifications=true});
    const auto fixed=world.create_static_body({});
    world.attach_convex_shape(fixed,hull);
    const auto moving=world.create_dynamic_body({{0.2f,1.85f,0.1f},{}});
    world.attach_convex_shape(moving,hull);
    world.update_mass_properties(moving,1.f);
    RigidBodyDynamics dynamics{};
    dynamics.gravity_enabled=false;
    dynamics.position_iterations=24u;
    world.set_body_dynamics(moving,dynamics);
    world.step(1.f/120.f);
    const auto events=world.contact_events();
    require(events.size()==1u &&
            events[0].kind==ContactEventKind::Found &&
            events[0].contact_count==4u,
            "Overlapping cooked cube faces did not emit four contact points");
    const auto contacts=world.contact_event_points();
    float min_x=std::numeric_limits<float>::max();
    float max_x=-min_x,min_z=min_x,max_z=-min_x;
    for(std::size_t i=0;i<events[0].contact_count;++i) {
        const auto& point=contacts[events[0].point_offset+i];
        require(std::abs(point.normal.y)>0.95f &&
                point.separation<0.f,
                "Clipped convex face generated invalid contact geometry");
        min_x=std::min(min_x,point.position.x);
        max_x=std::max(max_x,point.position.x);
        min_z=std::min(min_z,point.position.z);
        max_z=std::max(max_z,point.position.z);
    }
    require(max_x-min_x>1.5f && max_z-min_z>1.5f,
            "Clipped convex face collapsed its contact footprint");
    for(unsigned frame=0u;frame<4u;++frame) {
        world.step(1.f/120.f);
        const auto current=world.contact_events();
        require(current.size()==1u &&
                current[0].kind==ContactEventKind::Persists &&
                current[0].contact_count>=3u,
                "Clipped convex manifold did not persist through rigid response");
    }
}

void verify_finite_pair_matrix() {
    World world({.execution=ExecutionMode::NativeCpu,
                 .contact_notifications=true});
    const auto hull=cook_convex_mesh(tetra_vertices,tetra_faces);
    std::array<BodyId,5> moving{};
    world.add_static_box({1.f,0.5f,1.f},{0.f,-0.5f,0.f});
    moving[0]=world.create_dynamic_body({{0.f,-0.12f,0.f},{}});
    world.attach_convex_shape(moving[0],hull);
    world.update_mass_properties(moving[0],1.f);

    world.add_static_sphere(0.38f,{4.45f,0.25f,0.25f});
    moving[1]=world.create_dynamic_body({{4.f,0.f,0.f},{}});
    world.attach_convex_shape(moving[1],hull);
    world.update_mass_properties(moving[1],1.f);

    const auto fixed_hull=world.create_static_body({{8.f,0.f,0.f},{}});
    world.attach_convex_shape(fixed_hull,hull);
    moving[2]=world.create_dynamic_body({{8.15f,0.12f,0.08f},{}});
    world.attach_convex_shape(moving[2],hull);
    world.update_mass_properties(moving[2],1.f);

    const auto fixed_sphere_hull=world.create_static_body({{12.f,0.f,0.f},{}});
    world.attach_convex_shape(fixed_sphere_hull,hull);
    moving[3]=world.add_dynamic_sphere(
        0.35f,{12.35f,0.2f,0.25f},1.f);

    const auto fixed_box_hull=world.create_static_body({{16.f,0.f,0.f},{}});
    world.attach_convex_shape(fixed_box_hull,hull);
    moving[4]=world.add_dynamic_box(
        {0.4f,0.4f,0.4f},{16.35f,0.15f,0.25f},1.f);

    RigidBodyDynamics settings{};
    settings.gravity_enabled=false;
    settings.position_iterations=24u;
    for(const auto body:moving)
        world.set_body_dynamics(body,settings);
    world.step(1.f/120.f);
    unsigned found=0;
    for(const auto& event:world.contact_events()) {
        require(event.kind==ContactEventKind::Found,
                "Finite convex initial contact was not a found event");
        require(event.contact_count>0u,
                "Finite convex manifold did not publish a Gu witness");
        ++found;
    }
    require(found==moving.size(),
            "Cooked convex failed hull-box, hull-sphere or hull-hull admission");
    // Moving a hull out of SAP must retire its independent manager
    // without disturbing the other four live contacts.
    world.set_body_position(moving[2],{25.f,0.f,0.f});
    world.step(1.f/120.f);
    unsigned lost=0,persistent=0;
    for(const auto& event:world.contact_events()) {
        lost+=event.kind==ContactEventKind::Lost;
        persistent+=event.kind==ContactEventKind::Persists;
    }
    require(lost==1u && persistent==4u,
            "Finite convex manager lifetime lost separation or persistence");
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
    std::array<Vec3,8> doubled_vertices{};
    std::array<ConvexTriangle,8> doubled_faces{};
    for(std::size_t i=0u;i<4u;++i) {
        doubled_vertices[i]=tetra_vertices[i];
        doubled_vertices[i+4u]=tetra_vertices[i];
        doubled_faces[i]=tetra_faces[i];
        for(unsigned axis=0u;axis<3u;++axis)
            doubled_faces[i+4u][axis]=tetra_faces[i][axis]+4u;
    }
    rejects([&]{cook_convex_mesh(doubled_vertices,doubled_faces);},
        "Disconnected coincident shells doubled the cooked mass");
}
}
int main() {
    try {
        verify_tetra();
        verify_cube();
        verify_wedge();
        verify_pentagonal_face();
        verify_invalid();
        verify_native_rigid_contact();
        verify_sphere_face_contact();
        verify_off_origin_hull_pair();
        verify_convex_face_manifold();
        verify_finite_pair_matrix();
        std::cout<<"Cooked convex volume, inertia and topology OK\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
