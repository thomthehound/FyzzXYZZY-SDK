// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fyzzxyzzy/world.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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
    if(reference)compound_compare_state(world.body_state(ids.body),reference->body_state(ref.body),2.0e-4f,"Compound contacts");
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
            compound_compare_state(world.body_state(ids.body),reference->body_state(ref.body),4.0e-4f,"Compound retained response");
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
} // namespace fyzz::test
