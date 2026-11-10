// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fyzzxyzzy/world.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fyzz::test {
inline void compound_require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
inline float compound_error(Vec3 a,Vec3 b) {
    return std::max({std::abs(a.x-b.x),std::abs(a.y-b.y),std::abs(a.z-b.z)});
}
inline Vec3 compound_rotate(Quat q,Vec3 v) {
    const Vec3 t{2*(q.y*v.z-q.z*v.y),2*(q.z*v.x-q.x*v.z),2*(q.x*v.y-q.y*v.x)};
    return {v.x+q.w*t.x+q.y*t.z-q.z*t.y,
            v.y+q.w*t.y+q.z*t.x-q.x*t.z,
            v.z+q.w*t.z+q.x*t.y-q.y*t.x};
}
inline void compound_compare_mass(const BodyMassProperties& a,const BodyMassProperties& b) {
    compound_require(std::abs(a.inverse_mass-b.inverse_mass)<2.0e-6f,
                     "Compound inverse mass differs from reference");
    compound_require(compound_error(a.mass_local_pose.position,b.mass_local_pose.position)<2.0e-6f,
                     "Compound center of mass differs from reference");
    // Principal-axis signs and permutations are equivalent. Compare the
    // reconstructed actor-local inverse inertia tensor rather than eigenvectors.
    const auto tensor=[](const BodyMassProperties& m) {
        const auto x=compound_rotate(m.mass_local_pose.orientation,{1,0,0});
        const auto y=compound_rotate(m.mass_local_pose.orientation,{0,1,0});
        const auto z=compound_rotate(m.mass_local_pose.orientation,{0,0,1});
        return std::array<float,6>{
            x.x*x.x*m.inverse_inertia.x+y.x*y.x*m.inverse_inertia.y+z.x*z.x*m.inverse_inertia.z,
            x.y*x.y*m.inverse_inertia.x+y.y*y.y*m.inverse_inertia.y+z.y*z.y*m.inverse_inertia.z,
            x.z*x.z*m.inverse_inertia.x+y.z*y.z*m.inverse_inertia.y+z.z*z.z*m.inverse_inertia.z,
            x.x*x.y*m.inverse_inertia.x+y.x*y.y*m.inverse_inertia.y+z.x*z.y*m.inverse_inertia.z,
            x.x*x.z*m.inverse_inertia.x+y.x*y.z*m.inverse_inertia.y+z.x*z.z*m.inverse_inertia.z,
            x.y*x.z*m.inverse_inertia.x+y.y*y.z*m.inverse_inertia.y+z.y*z.z*m.inverse_inertia.z};
    };
    const auto ta=tensor(a),tb=tensor(b);
    for(std::size_t i=0;i<ta.size();++i)
        compound_require(std::abs(ta[i]-tb[i])<2.0e-5f*std::max(1.0f,std::abs(tb[i])),
                         "Compound inverse inertia tensor differs from reference");
}
inline void compound_compare_state(BodyState a,BodyState b,float tolerance,const char* stage) {
    const float direct=std::max({std::abs(a.orientation.x-b.orientation.x),std::abs(a.orientation.y-b.orientation.y),
        std::abs(a.orientation.z-b.orientation.z),std::abs(a.orientation.w-b.orientation.w)});
    const float negated=std::max({std::abs(a.orientation.x+b.orientation.x),std::abs(a.orientation.y+b.orientation.y),
        std::abs(a.orientation.z+b.orientation.z),std::abs(a.orientation.w+b.orientation.w)});
    const float error=std::max({compound_error(a.position,b.position),compound_error(a.velocity,b.velocity),
        compound_error(a.angular_velocity,b.angular_velocity),std::min(direct,negated)});
    if(error>tolerance)throw std::runtime_error(std::string(stage)+": body state error "+std::to_string(error)+
        "; linear "+std::to_string(compound_error(a.velocity,b.velocity))+"; angular "+std::to_string(compound_error(a.angular_velocity,b.angular_velocity)));
}

// Public API conformance: no private solver rows or alternate dynamics.
inline void compound_mass_and_motion(World& world,World* reference) {
    const auto create=[](World& w) {
        const auto body=w.create_dynamic_body({{0,8,0},{0,0.19866933f,0,0.98006658f}});
        const auto a=w.attach_box_shape(body,{0.6f,0.3f,0.25f},{{-0.8f,0.1f,0.2f},{0,0,0.24740396f,0.96891242f}});
        const auto b=w.attach_box_shape(body,{0.3f,0.5f,0.2f},{{0.9f,0.4f,-0.3f},{0.14943813f,0,0,0.98877108f}});
        w.attach_sphere_shape(body,0.35f,{{0.1f,-0.2f,0.6f},{}});
        w.update_mass_properties(body,7.0f);
        return std::array<std::uint32_t,3>{body,a,b};
    };
    const auto ids=create(world);
    const auto ref=reference?create(*reference):ids;
    auto props=world.body_mass_properties(ids[0]);
    compound_require(std::abs(props.mass_local_pose.position.x)>0.02f,
                     "Compound center of mass stayed at the actor origin");
    if(reference)compound_compare_mass(props,reference->body_mass_properties(ref[0]));
    const auto set=[&](World& w,BodyId id) {
        auto d=RigidBodyDynamics{};d.angular_damping=0.0f;
        w.set_body_dynamics(id,d);
        w.set_body_velocity(id,{0.7f,-0.2f,0.1f});
        w.set_body_angular_velocity(id,{0.3f,-0.4f,0.7f});
        w.apply_body_torque_impulse(id,{0.2f,-0.1f,0.3f});
    };
    set(world,ids[0]);if(reference)set(*reference,ref[0]);
    for(unsigned i=0;i<24;++i) {
        world.step(1.0f/120.0f);
        if(reference) {
            reference->step(1.0f/120.0f);
            compound_compare_state(world.body_state(ids[0]),reference->body_state(ref[0]),4.0e-5f,"Compound free motion");
        }
        compound_require(world.contact_events().empty(),"Sibling shapes generated self contacts");
    }
    world.set_body_kinematic(ids[0],true);
    if(reference)reference->set_body_kinematic(ref[0],true);
    const Transform target{{0.2f,8.5f,-0.1f},{0,0,0.14943813f,0.98877108f}};
    world.set_body_kinematic_target(ids[0],target.position,target.orientation);
    if(reference)reference->set_body_kinematic_target(ref[0],target.position,target.orientation);
    world.step(1.0f/120.0f);
    if(reference) {
        reference->step(1.0f/120.0f);
        compound_compare_state(world.body_state(ids[0]),reference->body_state(ref[0]),8.0e-4f,"Compound kinematic target");
    }
    world.set_body_kinematic(ids[0],false);if(reference)reference->set_body_kinematic(ref[0],false);
    world.detach_shape(ids[2]);
    if(reference)reference->detach_shape(ref[2]);
    // Authored attachments do not silently change existing mass properties.
    const auto unchanged=world.body_mass_properties(ids[0]);
    compound_require(unchanged.inverse_mass==props.inverse_mass,"Detach implicitly changed mass");
    world.update_mass_properties(ids[0],7.0f);
    if(reference) {
        reference->update_mass_properties(ref[0],7.0f);
        compound_compare_mass(world.body_mass_properties(ids[0]),reference->body_mass_properties(ref[0]));
    }
    bool rejected=false;
    try {world.set_shape_local_pose(ids[2],{});}catch(const std::out_of_range&){rejected=true;}
    compound_require(rejected,"Detached shape ID remained writable");
    world.remove_body(ids[0]);if(reference)reference->remove_body(ref[0]);
}

inline void compound_contact_lifetimes(World& world,World* reference) {
    struct Ids {BodyId ground,body;ShapeId left,right,top;};
    const auto create=[](World& w) {
        const auto ground=w.create_static_body({});
        w.attach_box_shape(ground,{0.45f,0.3f,0.6f},{{-0.7f,0,0},{}});
        w.attach_box_shape(ground,{0.45f,0.3f,0.6f},{{0.7f,0,0},{}});
        const auto body=w.create_dynamic_body({{0,0.59f,0},{}});
        const auto left=w.attach_box_shape(body,{0.3f,0.3f,0.4f},{{-0.7f,0,0},{}});
        const auto right=w.attach_box_shape(body,{0.3f,0.3f,0.4f},{{0.7f,0,0},{}});
        const auto top=w.attach_sphere_shape(body,0.2f,{{0.25f,0.6f,0.1f},{}});
        w.update_mass_properties(body,5.0f);
        // Converge simultaneous support constraints across manager orders.
        RigidBodyDynamics dynamics;dynamics.position_iterations=32;dynamics.velocity_iterations=8;
        w.set_body_dynamics(body,dynamics);
        Material frictionless;frictionless.static_friction=0;frictionless.dynamic_friction=0;frictionless.restitution=0;
        w.set_body_material(ground,frictionless);w.set_body_material(body,frictionless);
        return Ids{ground,body,left,right,top};
    };
    const auto ids=create(world);const auto ref=reference?create(*reference):ids;
    world.step(1.0f/240.0f);if(reference)reference->step(1.0f/240.0f);
    unsigned contacts=0;
    for(const auto& event:world.contact_events()) {
        compound_require(event.body0==ids.ground && event.body1==ids.body,
                         "Compound shape contact used a shape ID as a body ID");
        compound_require(event.shape1==ids.left || event.shape1==ids.right,
                         "Compound contact lost its attached shape identity");
        compound_require(event.kind==ContactEventKind::Found,"Compound first contacts did not report found");
        ++contacts;
    }
    compound_require(contacts==2,"Compound body did not retain two independent contact managers");
    if(reference) {
        try {
            compound_compare_state(world.body_state(ids.body),
                reference->body_state(ref.body),2.0e-4f,"Compound contacts");
        }catch(const std::runtime_error& failure) {
            // Report both contact streams when PGS diverges despite
            // matching manager lifetimes; distinguish Gu from Dy errors.
            std::string detail=failure.what();
            const auto report=[&](const World& w,BodyId body,const char* tag) {
                const auto state=w.body_state(body);
                detail+=" "+std::string(tag)+" pos=("+
                    std::to_string(state.position.x)+","+
                    std::to_string(state.position.y)+","+
                    std::to_string(state.position.z)+") vel=("+
                    std::to_string(state.velocity.x)+","+
                    std::to_string(state.velocity.y)+","+
                    std::to_string(state.velocity.z)+")";
                const auto points=w.contact_event_points();
                for(const auto& event:w.contact_events()) {
                    detail+=" shape="+std::to_string(event.shape1)+
                        " count="+std::to_string(event.contact_count)+
                        " impulse="+std::to_string(event.normal_impulse);
                    for(std::size_t i=0;i<event.contact_count;++i) {
                        const auto& point=points[event.point_offset+i];
                        detail+=" [xyz="+std::to_string(point.position.x)+","+
                            std::to_string(point.position.y)+","+
                            std::to_string(point.position.z)+
                            " n="+std::to_string(point.normal.x)+","+
                            std::to_string(point.normal.y)+","+
                            std::to_string(point.normal.z)+
                            " sep="+std::to_string(point.separation)+
                            " j="+std::to_string(point.normal_impulse)+"]";
                    }
                }
            };
            report(world,ids.body,"actual");
            report(*reference,ref.body,"reference");
            throw std::runtime_error(detail);
        }
    }
    world.detach_shape(ids.right);if(reference)reference->detach_shape(ref.right);
    const auto replacement=world.attach_sphere_shape(ids.body,0.1f,{{0,4,0},{}});
    if(reference)reference->attach_sphere_shape(ref.body,0.1f,{{0,4,0},{}});
    compound_require(replacement!=ids.right,"Replacement attachment reused a retired shape identity");
    world.step(1.0f/240.0f);if(reference)reference->step(1.0f/240.0f);
    bool lost=false,persists=false;
    for(const auto& event:world.contact_events()) {
        lost|=event.kind==ContactEventKind::Lost && event.shape1==ids.right;
        persists|=event.kind==ContactEventKind::Persists && event.shape1==ids.left;
    }
    if(!lost || !persists) {
        std::string detail="Compound detach events:";
        for(const auto& e:world.contact_events())detail+=" ["+std::to_string(e.shape0)+","+std::to_string(e.shape1)+","+std::to_string(static_cast<unsigned>(e.kind))+"]";
        throw std::runtime_error(detail);
    }
    if(reference)compound_compare_state(world.body_state(ids.body),reference->body_state(ref.body),3.0e-4f,"Compound detach response");
    const auto before=world.body_state(ids.body);
    const auto reports=world.contact_events();
    world.add_body_force(ids.body,{0.2f,0.1f,-0.1f},ForceMode::Impulse);
    if(reference)reference->add_body_force(ref.body,{0.2f,0.1f,-0.1f},ForceMode::Impulse);
    bool failed=false;
    try {world.step(std::numeric_limits<float>::max());}catch(const std::overflow_error&){failed=true;}
    compound_require(failed,"Compound invalid frame was accepted");
    compound_compare_state(world.body_state(ids.body),before,0.0f,"Compound failed-frame rollback");
    compound_require(world.contact_events().data()==reports.data() && world.contact_events().size()==reports.size(),
                     "Compound failed frame replaced published contacts");
    world.set_shape_local_pose(ids.top,{{0,2,0},{}});
    if(reference)reference->set_shape_local_pose(ref.top,{{0,2,0},{}});
    world.set_shape_simulation_enabled(ids.left,false);
    if(reference)reference->set_shape_simulation_enabled(ref.left,false);
    world.step(1.0f/240.0f);if(reference)reference->step(1.0f/240.0f);
    compound_require(std::any_of(world.contact_events().begin(),world.contact_events().end(),[&](const ContactEvent& e) {
        return e.kind==ContactEventKind::Lost && e.shape1==ids.left;
    }),"Disabled compound shape retained physical contact");
    world.set_shape_simulation_enabled(ids.left,true);if(reference)reference->set_shape_simulation_enabled(ref.left,true);
    for(unsigned i=0;i<8;++i) {
        world.step(1.0f/240.0f);
        if(reference) {
            reference->step(1.0f/240.0f);
            try {
                compound_compare_state(world.body_state(ids.body),
                    reference->body_state(ref.body),4.0e-4f,"Compound retained response");
            }catch(const std::runtime_error& error) {
                const auto gpu=world.body_state(ids.body);
                const auto cpu=reference->body_state(ref.body);
                const auto velocity=[](Vec3 v) {
                    return "("+std::to_string(v.x)+","+
                        std::to_string(v.y)+","+std::to_string(v.z)+")";
                };
                throw std::runtime_error(std::string(error.what())+
                    " frame="+std::to_string(i)+
                    " gpu_pos="+velocity(gpu.position)+
                    " cpu_pos="+velocity(cpu.position)+
                    " gpu_vel="+velocity(gpu.velocity)+
                    " cpu_vel="+velocity(cpu.velocity)+
                    " gpu_events="+std::to_string(world.contact_events().size())+
                    " cpu_events="+std::to_string(reference->contact_events().size()));
            }
        }
    }
    world.remove_body(ids.body);world.remove_body(ids.ground);
    if(reference) {reference->remove_body(ref.body);reference->remove_body(ref.ground);}
    world.step(1.0f/240.0f);if(reference)reference->step(1.0f/240.0f);
}
inline void compound_default_contact_response(World& world,World* reference) {
    const auto create=[](World& w) {
        const BodyId floor=w.add_static_floor(0);
        w.attach_box_shape(floor,{0.2f,0.2f,0.2f},{{5,3,0},{}});
        const BodyId body=w.create_dynamic_body({{0,0.49f,0},{}});
        const auto foot=w.attach_box_shape(body,{0.5f,0.5f,0.4f},{{0.7f,0,0},{0,0.09983342f,0,0.99500417f}});
        w.attach_sphere_shape(body,0.3f,{{-0.8f,0.9f,0.2f},{}});
        w.update_mass_properties(body,4.0f);
        Material surface;surface.restitution=0;surface.static_friction=0.3f;surface.dynamic_friction=0.2f;
        w.set_shape_material(foot,surface);
        return std::array<BodyId,2>{floor,body};
    };
    const auto ids=create(world),ref=reference?create(*reference):ids;
    bool rotated=false;
    for(unsigned i=0;i<16;++i) {
        world.step(1.0f/240.0f);
        rotated|=std::abs(world.body_state(ids[1]).angular_velocity.z)>0.01f;
        if(reference) {
            reference->step(1.0f/240.0f);
            compound_compare_state(world.body_state(ids[1]),reference->body_state(ref[1]),3.0e-4f,"Compound default contact response");
        }
    }
    compound_require(rotated,"Offset compound contact did not rotate the shared body");
    world.remove_body(ids[1]);world.remove_body(ids[0]);
    if(reference) {reference->remove_body(ref[1]);reference->remove_body(ref[0]);}
    world.step(1.0f/240.0f);if(reference)reference->step(1.0f/240.0f);
}
inline void compound_joint_response(World& world,World* reference) {
    const auto create=[](World& w) {
        const auto fixed=w.create_static_body({{0,5,0},{}});
        const auto body=w.create_dynamic_body({{4.2f,5,0},{}});
        w.attach_box_shape(body,{0.4f,0.3f,0.2f},{{0.3f,0.6f,0},{0,0,0.19866933f,0.98006658f}});
        w.attach_sphere_shape(body,0.3f,{{-0.2f,0.8f,0.2f},{}});
        w.update_mass_properties(body,3.0f);
        w.add_anchored_distance_constraint(fixed,body,{0,0.2f,0},{0,-0.1f,0},4.0f);
        return std::array<BodyId,2>{fixed,body};
    };
    const auto ids=create(world),ref=reference?create(*reference):ids;
    for(unsigned i=0;i<16;++i) {
        world.step(1.0f/120.0f);
        if(reference) {
            reference->step(1.0f/120.0f);
            compound_compare_state(world.body_state(ids[1]),reference->body_state(ref[1]),2.0e-4f,"Compound joint response");
        }
    }
    world.remove_body(ids[1]);world.remove_body(ids[0]);
    if(reference) {reference->remove_body(ref[1]);reference->remove_body(ref[0]);}
    world.step(1.0f/120.0f);if(reference)reference->step(1.0f/120.0f);
}
inline void fixed_joint_response(World& world,World* reference) {
    constexpr Quat frame_rotation{0.0f,0.17364818f,0.0f,0.98480775f};
    constexpr Transform local_dynamic{{0.85f,0.25f,-0.4f},frame_rotation};
    constexpr Transform local_static{{-1.15f,0.25f,-0.4f},frame_rotation};
    constexpr Vec3 target{0.0f,5.0f,0.0f};
    struct Assembly {
        BodyId anchor,body;
        ConstraintId joint;
    };
    const auto create=[&](World& w) {
        const auto anchor=w.create_static_body({{2.0f,5.0f,0.0f},{}});
        const auto body=w.create_dynamic_body(
            {{0.04f,5.06f,-0.05f},{0.04997917f,0.0f,0.0f,0.99875026f}});
        w.attach_box_shape(body,{0.4f,0.3f,0.2f},
            {{0.3f,0.6f,0.0f},{0.0f,0.0f,0.19866933f,0.98006658f}});
        w.attach_sphere_shape(body,0.3f,{{-0.2f,0.8f,0.2f},{}});
        w.update_mass_properties(body,3.0f);
        const auto joint=w.add_fixed_constraint(body,anchor,local_dynamic,local_static);
        w.set_body_velocity(body,{0.9f,-0.6f,0.2f});
        w.set_body_angular_velocity(body,{0.5f,1.2f,0.7f});
        return Assembly{anchor,body,joint};
    };
    const auto actual=create(world),expected=reference?create(*reference):actual;
    constexpr float dt=1.0f/120.0f;
    for(unsigned frame=0;frame<12u;++frame) {
        world.add_body_torque(actual.body,{0.13f,0.39f,-0.24f});
        world.step(dt);
        if(reference) {
            reference->add_body_torque(expected.body,{0.13f,0.39f,-0.24f});
            reference->step(dt);
            compound_compare_state(world.body_state(actual.body),
                reference->body_state(expected.body),0.002f,"Fixed joint response");
        }
    }
    const auto held=world.body_state(actual.body);
    compound_require(compound_error(held.position,target)<0.002f &&
                     std::abs(held.orientation.w)>0.99999f &&
                     compound_error(held.angular_velocity,{})<0.002f,
                     "Fixed joint failed to hold the authored frame");
    const auto retire=[&](World& w,Assembly a) {
        w.remove_constraint(a.joint);
        const auto replacement=w.add_fixed_constraint(
            a.body,a.anchor,local_dynamic,local_static);
        bool rejected=false;
        try {w.remove_constraint(a.joint);} catch(const std::out_of_range&) {rejected=true;}
        compound_require(rejected,"Retired joint token reached a recycled joint");
        w.remove_constraint(replacement);
        w.set_body_angular_velocity(a.body,{0.0f,1.0f,0.0f});
        w.step(4.0f*dt);
        const auto released=w.body_state(a.body);
        compound_require(released.position.y<held.position.y-0.002f &&
                         std::abs(released.orientation.y)>0.01f,
                         "Retiring a fixed joint did not restore free motion");
        w.remove_body(a.body);
        w.remove_body(a.anchor);
        w.step(dt);
    };
    retire(world,actual);
    if(reference)retire(*reference,expected);
}
inline void spherical_joint_response(World& world,World* reference) {
    constexpr Transform local_dynamic{{0.5f,0.6f,-0.2f},{}};
    constexpr Transform local_static{{-1.5f,0.6f,-0.2f},{}};
    constexpr Vec3 pivot{0.5f,5.6f,-0.2f};
    struct Assembly {BodyId anchor,body;ConstraintId joint;};
    const auto create=[&](World& w) {
        const auto anchor=w.create_static_body({{2.0f,5.0f,0.0f},{}});
        const auto body=w.create_dynamic_body({{0.0f,5.0f,0.0f},{}});
        w.attach_box_shape(body,{0.4f,0.25f,0.2f},{{0.2f,0.3f,0.1f},{}});
        w.attach_sphere_shape(body,0.22f,{{-0.3f,0.15f,-0.2f},{}});
        w.update_mass_properties(body,3.0f);
        const auto joint=w.add_spherical_constraint(
            body,anchor,local_dynamic,local_static);
        w.set_body_angular_velocity(body,{0.4f,1.3f,0.7f});
        return Assembly{anchor,body,joint};
    };
    const auto actual=create(world),expected=reference?create(*reference):actual;
    constexpr float dt=1.0f/120.0f;
    for(unsigned frame=0;frame<18u;++frame) {
        world.add_body_torque(actual.body,{0.1f,0.3f,0.2f});
        world.step(dt);
        if(reference) {
            reference->add_body_torque(expected.body,{0.1f,0.3f,0.2f});
            reference->step(dt);
            compound_compare_state(world.body_state(actual.body),
                reference->body_state(expected.body),0.005f,
                "Spherical joint response");
        }
        const auto state=world.body_state(actual.body);
        const auto anchor=compound_rotate(state.orientation,local_dynamic.position);
        const Vec3 position{state.position.x+anchor.x,
                            state.position.y+anchor.y,
                            state.position.z+anchor.z};
        compound_require(compound_error(position,pivot)<0.01f,
                         "Spherical joint lost its anchored pivot");
    }
    const auto held=world.body_state(actual.body);
    compound_require(std::abs(held.orientation.y)>0.025f,
                     "Spherical joint incorrectly locked free rotation");
    const auto retire=[&](World& w,Assembly assembly) {
        w.remove_constraint(assembly.joint);
        const auto replacement=w.add_spherical_constraint(
            assembly.body,assembly.anchor,local_dynamic,local_static);
        bool rejected=false;
        try {w.remove_constraint(assembly.joint);}
        catch(const std::out_of_range&) {rejected=true;}
        compound_require(rejected,"Retired spherical token reached a recycled joint");
        w.remove_constraint(replacement);
        w.set_body_velocity(assembly.body,{0.7f,0.0f,0.0f});
        w.step(0.2f);
        const auto free=w.body_state(assembly.body);
        compound_require(free.position.x>held.position.x+0.05f,
                         "Retiring a spherical joint did not restore free motion");
        w.remove_body(assembly.body);
        w.remove_body(assembly.anchor);
        w.step(dt);
    };
    retire(world,actual);
    if(reference)retire(*reference,expected);
}
inline void revolute_joint_response(World& world,World* reference) {
    constexpr Transform local_dynamic{{0.5f,0.6f,-0.2f},{}};
    constexpr Transform local_static{{-1.5f,0.6f,-0.2f},{}};
    constexpr Vec3 pivot{0.5f,5.6f,-0.2f};
    struct Assembly {BodyId anchor,body;ConstraintId joint;};
    const auto create=[&](World& w) {
        const auto anchor=w.create_static_body({{2.0f,5.0f,0.0f},{}});
        const auto body=w.create_dynamic_body({{0.0f,5.0f,0.0f},{}});
        w.attach_box_shape(body,{0.4f,0.25f,0.2f},{{0.2f,0.3f,0.1f},{}});
        w.attach_sphere_shape(body,0.22f,{{-0.3f,0.15f,-0.2f},{}});
        w.update_mass_properties(body,3.0f);
        const auto joint=w.add_revolute_constraint(
            body,anchor,local_dynamic,local_static);
        w.set_body_angular_velocity(body,{2.0f,0.2f,-0.2f});
        return Assembly{anchor,body,joint};
    };
    const auto actual=create(world),expected=reference?create(*reference):actual;
    constexpr float dt=1.0f/120.0f;
    for(unsigned frame=0;frame<24u;++frame) {
        world.add_body_torque(actual.body,{0.45f,0.0f,0.0f});
        world.step(dt);
        if(reference) {
            reference->add_body_torque(expected.body,{0.45f,0.0f,0.0f});
            reference->step(dt);
            compound_compare_state(world.body_state(actual.body),
                reference->body_state(expected.body),0.008f,
                "Revolute joint response");
        }
        const auto state=world.body_state(actual.body);
        const auto offset=compound_rotate(state.orientation,local_dynamic.position);
        const Vec3 position{state.position.x+offset.x,
                            state.position.y+offset.y,
                            state.position.z+offset.z};
        compound_require(compound_error(position,pivot)<0.015f,
                         "Revolute joint lost its anchored pivot");
        compound_require(std::abs(state.orientation.y)<0.02f &&
                         std::abs(state.orientation.z)<0.02f,
                         "Revolute joint failed to lock off-axis rotation");
    }
    const auto held=world.body_state(actual.body);
    compound_require(std::abs(held.orientation.x)>0.05f,
                     "Revolute joint incorrectly locked hinge rotation");
    const auto retire=[&](World& w,Assembly assembly) {
        w.remove_constraint(assembly.joint);
        const auto replacement=w.add_revolute_constraint(
            assembly.body,assembly.anchor,local_dynamic,local_static);
        bool rejected=false;
        try {w.remove_constraint(assembly.joint);}
        catch(const std::out_of_range&) {rejected=true;}
        compound_require(rejected,"Retired revolute token reached recycled joint");
        w.remove_constraint(replacement);
        w.set_body_velocity(assembly.body,{0.7f,0.0f,0.0f});
        w.step(0.2f);
        const auto free=w.body_state(assembly.body);
        compound_require(free.position.x>held.position.x+0.05f,
                         "Retiring a revolute joint did not restore free motion");
        w.remove_body(assembly.body);
        w.remove_body(assembly.anchor);
        w.step(dt);
    };
    retire(world,actual);
    if(reference)retire(*reference,expected);
}
inline void prismatic_joint_response(World& world,World* reference) {
    constexpr Quat frame_rotation{0.0f,0.258819045f,0.0f,0.965925826f};
    constexpr Transform local_dynamic{{0.4f,0.2f,0.0f},frame_rotation};
    constexpr Transform local_static{{-1.6f,0.2f,0.0f},frame_rotation};
    constexpr Vec3 pivot{0.4f,5.2f,0.0f};
    const Vec3 axis=compound_rotate(frame_rotation,{1.0f,0.0f,0.0f});
    struct Assembly {BodyId anchor,body;ConstraintId joint;};
    const auto create=[&](World& w) {
        const auto anchor=w.create_static_body({{2.0f,5.0f,0.0f},{}});
        const auto body=w.create_dynamic_body({{0.0f,5.0f,0.0f},{}});
        w.attach_box_shape(body,{0.36f,0.2f,0.26f},{{0.25f,0.3f,0.1f},{}});
        w.attach_sphere_shape(body,0.2f,{{-0.3f,0.2f,-0.1f},{}});
        w.update_mass_properties(body,3.0f);
        const auto joint=w.add_prismatic_constraint(
            body,anchor,local_dynamic,local_static);
        w.set_body_velocity(body,{0.7f,0.2f,-0.3f});
        w.set_body_angular_velocity(body,{0.3f,-0.4f,0.2f});
        return Assembly{anchor,body,joint};
    };
    const auto actual=create(world),expected=reference?create(*reference):actual;
    constexpr float dt=1.0f/120.0f;
    float last_slide=0.0f;
    for(unsigned frame=0;frame<24u;++frame) {
        world.add_body_torque(actual.body,{0.1f,0.12f,-0.08f});
        try { world.step(dt); }
        catch(...) {
            std::cerr<<"FYZZXYZZY_METRIC suite=prismatic"
                     <<" rejected_frame="<<frame<<'\n';
            throw;
        }
        if(reference) {
            reference->add_body_torque(expected.body,{0.1f,0.12f,-0.08f});
            reference->step(dt);
            compound_compare_state(world.body_state(actual.body),
                reference->body_state(expected.body),0.008f,
                "Prismatic joint response");
        }
        const auto state=world.body_state(actual.body);
        const auto rotated=compound_rotate(state.orientation,local_dynamic.position);
        const Vec3 origin{state.position.x+rotated.x,
                          state.position.y+rotated.y,
                          state.position.z+rotated.z};
        const Vec3 offset{origin.x-pivot.x,origin.y-pivot.y,origin.z-pivot.z};
        const float slide=offset.x*axis.x+offset.y*axis.y+offset.z*axis.z;
        const Vec3 transverse{offset.x-slide*axis.x,
                              offset.y-slide*axis.y,
                              offset.z-slide*axis.z};
        compound_require(compound_error(transverse,{})<0.015f,
                         "Prismatic joint lost its transverse lock");
        compound_require(std::abs(state.orientation.x)<0.02f &&
                         std::abs(state.orientation.y)<0.02f &&
                         std::abs(state.orientation.z)<0.02f,
                         "Prismatic joint failed to lock rotation");
        last_slide=slide;
    }
    compound_require(last_slide>0.05f,
                     "Prismatic joint incorrectly locked axial translation");
    const auto held=world.body_state(actual.body);
    const auto retire=[&](World& w,Assembly assembly) {
        w.remove_constraint(assembly.joint);
        const auto replacement=w.add_prismatic_constraint(
            assembly.body,assembly.anchor,local_dynamic,local_static);
        bool rejected=false;
        try {w.remove_constraint(assembly.joint);}
        catch(const std::out_of_range&) {rejected=true;}
        compound_require(rejected,"Retired prismatic token reached a recycled joint");
        w.remove_constraint(replacement);
        w.set_body_velocity(assembly.body,{0.0f,2.0f,0.0f});
        w.step(0.12f);
        const auto free=w.body_state(assembly.body);
        compound_require(free.position.y>held.position.y+0.05f,
                         "Retiring a prismatic joint did not restore transverse motion");
        w.remove_body(assembly.body);
        w.remove_body(assembly.anchor);
        w.step(dt);
    };
    retire(world,actual);
    if(reference)retire(*reference,expected);

    // At nonzero extension, rotating the prismatic axis changes the
    // permitted transverse velocity even if both local anchors coincide.
    const auto create_rotating_slider=[](World& w) {
        const auto guide=w.create_dynamic_body({{0.0f,12.0f,0.0f},{}});
        w.attach_box_shape(guide,{0.2f,0.2f,0.2f},{});
        w.update_mass_properties(guide,1.0f);
        w.set_body_kinematic(guide,true);
        const auto follower=w.create_dynamic_body({{8.0f,12.0f,0.0f},{}});
        w.attach_box_shape(follower,{0.25f,0.2f,0.2f},{});
        w.update_mass_properties(follower,2.0f);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        dynamics.linear_damping=0.0f;
        dynamics.angular_damping=0.0f;
        dynamics.position_iterations=8;
        dynamics.velocity_iterations=2;
        w.set_body_dynamics(follower,dynamics);
        const auto joint=w.add_prismatic_constraint(guide,follower,{},{});
        return Assembly{guide,follower,joint};
    };
    const auto rotating=create_rotating_slider(world);
    const auto rotating_ref=reference?create_rotating_slider(*reference):rotating;
    for(unsigned frame=1;frame<=12u;++frame) {
        const float half_angle=0.005f*static_cast<float>(frame);
        const Quat orientation{0.0f,0.0f,std::sin(half_angle),std::cos(half_angle)};
        world.set_body_kinematic_target(
            rotating.anchor,{0.0f,12.0f,0.0f},orientation);
        if(reference)reference->set_body_kinematic_target(
            rotating_ref.anchor,{0.0f,12.0f,0.0f},orientation);
        world.step(0.01f);
        if(reference) {
            reference->step(0.01f);
            compound_compare_state(world.body_state(rotating.body),
                reference->body_state(rotating_ref.body),0.06f,
                "Rotating extended prismatic joint");
        }
        const auto guide=world.body_state(rotating.anchor);
        const auto follower=world.body_state(rotating.body);
        const auto axis=compound_rotate(guide.orientation,{1.0f,0.0f,0.0f});
        const Vec3 span{follower.position.x-guide.position.x,
                        follower.position.y-guide.position.y,
                        follower.position.z-guide.position.z};
        const float length=span.x*axis.x+span.y*axis.y+span.z*axis.z;
        const Vec3 transverse{span.x-length*axis.x,
                              span.y-length*axis.y,
                              span.z-length*axis.z};
        compound_require(compound_error(transverse,{})<0.03f && length>7.0f,
                         "Extended prismatic joint lost rotating slider axis");
    }
    const auto remove_rotating=[](World& w,Assembly a) {
        w.remove_constraint(a.joint);
        w.remove_body(a.body);
        w.remove_body(a.anchor);
        w.step(0.01f);
    };
    remove_rotating(world,rotating);
    if(reference)remove_rotating(*reference,rotating_ref);
}

inline void compound_plane_box_device_pcm(World& world,World* comparison) {
    // Exercise both shape orders. The plane has no finite device volume,
    // while each live Sc contact manager retains its own GPU EdgeIndex.
    struct PairIds {BodyId plane,box;};
    const auto create=[](World& w,bool plane_first) {
        PairIds ids{};
        const auto add_box=[&] {
            ids.box=w.add_dynamic_box({0.5f,0.5f,0.35f},
                {0.0f,0.46f,0.0f},2.0f);
            RigidBodyDynamics d{};
            d.gravity_enabled=false;
            d.angular_damping=0.0f;
            d.position_iterations=16u;
            d.velocity_iterations=4u;
            w.set_body_dynamics(ids.box,d);
        };
        if(plane_first) {ids.plane=w.add_static_floor(0.0f);add_box();}
        else {add_box();ids.plane=w.add_static_floor(0.0f);}
        Material m{};
        m.static_friction=0.0f;m.dynamic_friction=0.0f;m.restitution=0.0f;
        w.set_body_material(ids.plane,m);
        w.set_body_material(ids.box,m);
        return ids;
    };
    const auto contacts=[](const World& w,PairIds ids) {
        std::vector<ContactEventPoint> points;
        for(const auto& event:w.contact_events()) {
            if(!((event.body0==ids.plane && event.body1==ids.box) ||
                 (event.body1==ids.plane && event.body0==ids.box)))
                continue;
            for(std::size_t j=0;j<event.contact_count;++j)
                points.push_back(w.contact_event_points()[event.point_offset+j]);
        }
        return points;
    };
    for(const bool plane_first:{true,false}) {
        const auto ids=create(world,plane_first);
        const auto ref=comparison?create(*comparison,plane_first):ids;
        for(unsigned frame=0;frame<12u;++frame) {
            const float height=frame>=7u?0.03f:0.0f;
            const float x=(frame%3u==0u?0.003f:0.0f);
            const float angle=0.09f+0.003f*static_cast<float>(frame);
            const Quat q{0.0f,0.0f,std::sin(angle*0.5f),std::cos(angle*0.5f)};
            const auto author=[&](World& w,PairIds actor) {
                w.set_body_position(actor.plane,{0.0f,height,0.0f});
                w.set_body_pose(actor.box,{x,0.46f,0.0f},q);
                w.set_body_velocity(actor.box,{0.0f,0.0f,0.0f});
                w.set_body_angular_velocity(actor.box,{0.0f,0.0f,0.0f});
            };
            author(world,ids);
            if(comparison)author(*comparison,ref);
            world.step(1.0f/120.0f);
            if(comparison)comparison->step(1.0f/120.0f);
            const auto points=contacts(world,ids);
            compound_require(!points.empty() && points.size()<=4u,
                "Plane-box source PCM failed to publish bounded contacts");
            for(const auto& p:points) {
                if(std::abs(p.position.y-height)>=0.0008f ||
                   std::abs(std::abs(p.normal.y)-1.0f)>=0.0008f ||
                   std::abs(p.normal.x)>=0.0008f ||
                   std::abs(p.normal.z)>=0.0008f ||
                   p.separation>0.0408f)
                    throw std::runtime_error(
                        "Plane-box PCM witness invalid: frame="+
                        std::to_string(frame)+" plane="+std::to_string(height)+
                        " y="+std::to_string(p.position.y)+
                        " normal="+std::to_string(p.normal.y)+
                        " separation="+std::to_string(p.separation));
            }
            if(comparison) {
                const auto expected=contacts(*comparison,ref);
                compound_require(points.size()==expected.size(),
                    "GPU plane-box PCM changed persistent contact count");
                for(const auto& actual:points) {
                    bool matched=false;
                    for(const auto& reference:expected)
                        if(compound_error(actual.position,reference.position)<0.001f &&
                           compound_error(actual.normal,reference.normal)<0.001f &&
                           std::abs(actual.separation-reference.separation)<0.001f)
                            matched=true;
                    compound_require(matched,
                        "GPU plane-box PCM changed the source witness");
                }
                compound_compare_state(world.body_state(ids.box),
                    comparison->body_state(ref.box),0.004f,
                    "Plane-box PCM dynamics");
            }
        }
        world.remove_body(ids.box);
        world.remove_body(ids.plane);
        if(comparison) {
            comparison->remove_body(ref.box);
            comparison->remove_body(ref.plane);
        }
    }
}


inline void compound_plane_sphere_device_pcm(World& world,World* reference) {
    struct Bodies {BodyId plane,sphere;};
    const auto create=[](World& w,bool plane_first) {
        Bodies b{};
        const auto floor=[&] {b.plane=w.add_static_floor(0.0f);};
        const auto ball=[&] {
            b.sphere=w.add_dynamic_sphere(0.5f,{0.0f,0.45f,0.0f},2.0f);
            RigidBodyDynamics d{};
            d.gravity_enabled=false;
            d.angular_damping=0.0f;
            w.set_body_dynamics(b.sphere,d);
        };
        if(plane_first) {floor();ball();}
        else {ball();floor();}
        Material frictionless{};
        frictionless.static_friction=0.0f;
        frictionless.dynamic_friction=0.0f;
        frictionless.restitution=0.0f;
        w.set_body_material(b.plane,frictionless);
        w.set_body_material(b.sphere,frictionless);
        return b;
    };
    const auto find_contact=[](World& w,Bodies b) {
        std::vector<ContactEventPoint> contacts;
        for(const auto& event:w.contact_events()) {
            if(event.body0!=std::min(b.plane,b.sphere) ||
               event.body1!=std::max(b.plane,b.sphere))
                continue;
            for(std::size_t j=0;j<event.contact_count;++j)
                contacts.push_back(
                    w.contact_event_points()[event.point_offset+j]);
        }
        return contacts;
    };
    for(bool plane_first:{true,false}) {
        const auto bodies=create(world,plane_first);
        const auto oracle=reference?create(*reference,plane_first):bodies;
        for(unsigned frame=0;frame<9u;++frame) {
            const float plane_height=frame>=5u?0.02f:0.0f;
            const float sphere_height=frame==4u?0.74f:0.46f;
            const Vec3 center{0.07f*static_cast<float>(frame),
                              sphere_height,0.02f};
            const auto set=[&](World& w,Bodies b) {
                w.set_body_position(b.plane,{0.0f,plane_height,0.0f});
                w.set_body_position(b.sphere,center);
                w.set_body_velocity(b.sphere,{});
                w.set_body_angular_velocity(b.sphere,{});
            };
            set(world,bodies);
            if(reference)set(*reference,oracle);
            world.step(1.0f/120.0f);
            if(reference)reference->step(1.0f/120.0f);
            const auto points=find_contact(world,bodies);
            const bool touching=frame!=4u;
            compound_require(points.size()==(touching?1u:0u),
                "Sphere-plane PCM lost its single surface witness");
            if(touching) {
                const auto& point=points[0];
                compound_require(
                    compound_error(point.position,
                        {center.x,center.y-0.5f,center.z})<0.0008f &&
                    std::abs(std::abs(point.normal.y)-1.0f)<0.0008f &&
                    std::abs(point.normal.x)<0.0008f &&
                    std::abs(point.normal.z)<0.0008f &&
                    std::abs(point.separation-
                        (center.y-plane_height-0.5f))<0.0008f,
                    "Sphere-plane PCM contact differs from sphere surface");
            }
            if(reference) {
                const auto expected=find_contact(*reference,oracle);
                compound_require(points.size()==expected.size(),
                    "Device sphere-plane contact count diverged from native PCM");
                for(std::size_t k=0;k<points.size();++k) {
                    const auto& a=points[k],&b=expected[k];
                    compound_require(
                        compound_error(a.position,b.position)<0.0008f &&
                        compound_error(a.normal,b.normal)<0.0008f &&
                        std::abs(a.separation-b.separation)<0.0008f,
                        "Device sphere-plane witness diverged from native PCM");
                }
                compound_compare_state(world.body_state(bodies.sphere),
                    reference->body_state(oracle.sphere),0.002f,
                    "Sphere-plane PCM motion");
            }
        }
        world.remove_body(bodies.sphere);
        world.remove_body(bodies.plane);
        if(reference) {
            reference->remove_body(oracle.sphere);
            reference->remove_body(oracle.plane);
        }
    }
}

} // namespace fyzz::test
