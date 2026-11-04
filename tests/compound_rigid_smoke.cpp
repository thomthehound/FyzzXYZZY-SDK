// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "compound_rigid_cases.h"
#include <iostream>
int main() {
    try {
        fyzz::World native({.execution=fyzz::ExecutionMode::NativeCpu,.contact_notifications=true});
#if FYZZXYZZY_ENABLE_PHYSX_SOURCE
        fyzz::World reference({.execution=fyzz::ExecutionMode::ReferencePhysXCpu,.contact_notifications=true});
        auto* comparison=&reference;
#else
        fyzz::World* comparison=nullptr;
#endif
        fyzz::test::compound_mass_and_motion(native,comparison);
        fyzz::test::compound_contact_lifetimes(native,comparison);
        fyzz::test::compound_default_contact_response(native,comparison);
        fyzz::test::compound_joint_response(native,comparison);
        fyzz::test::fixed_joint_response(native,comparison);
        fyzz::test::spherical_joint_response(native,comparison);
        fyzz::test::revolute_joint_response(native,comparison);
        fyzz::test::prismatic_joint_response(native,comparison);
        fyzz::test::compound_plane_box_device_pcm(native,nullptr);
        fyzz::test::compound_plane_sphere_device_pcm(native,nullptr);
        std::cout<<"compound rigid smoke: passed\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"compound rigid smoke: "<<error.what()<<'\n';
        return 1;
    }
}
