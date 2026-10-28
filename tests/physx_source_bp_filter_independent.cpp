// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/lowlevelaabb/include/BpFiltering.h"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool correct, const char* message) {
    if (!correct) throw std::runtime_error(message);
}

void test_original_source_pair_filter() {
    using namespace physx;
    const auto stat=Bp::getFilterGroup_Statics();
    const auto dynamicA=Bp::getFilterGroup_Dynamics(3u,false);
    const auto dynamicB=Bp::getFilterGroup_Dynamics(7u,false);
    const auto kinematicA=Bp::getFilterGroup_Dynamics(3u,true);
    const auto kinematicB=Bp::getFilterGroup_Dynamics(7u,true);

    for (bool suppressKinematicKinematic : {false,true}) {
        for (bool suppressStaticKinematic : {false,true}) {
            const Bp::BpFilter filter(
                suppressKinematicKinematic,suppressStaticKinematic);
            const auto accepts=[&](Bp::FilterGroup::Enum a,
                                   Bp::FilterGroup::Enum b) {
                return Bp::groupFiltering(a,b,filter.getLUT());
            };
            require(!accepts(stat,stat),
                    "Original PhysX GPU SAP accepted static/static group");
            require(accepts(stat,dynamicA) && accepts(dynamicA,stat),
                    "Original PhysX GPU SAP rejected static/dynamic group");
            require(accepts(dynamicA,dynamicB) &&
                    accepts(dynamicB,dynamicA),
                    "Original PhysX GPU SAP rejected distinct dynamic actors");
            require(!accepts(dynamicA,dynamicA),
                    "Original PhysX GPU SAP admitted identical actor group");
            require(accepts(dynamicA,kinematicA) &&
                    accepts(kinematicA,dynamicA),
                    "Original PhysX GPU SAP lost dynamic/kinematic interaction");
            require(accepts(kinematicA,kinematicB)==
                    !suppressKinematicKinematic,
                    "Original PhysX GPU SAP kinematic policy was not applied");
            require(accepts(stat,kinematicA)==
                    !suppressStaticKinematic &&
                    accepts(kinematicA,stat)==!suppressStaticKinematic,
                    "Original PhysX GPU SAP static/kinematic policy was not applied");
            // Source GPU filters derive type from the low three bits and
            // retain full 32-bit group identity for self-pair rejection.
            require((static_cast<PxU32>(dynamicA)&7u)==
                    Bp::FilterType::DYNAMIC &&
                    (static_cast<PxU32>(kinematicA)&7u)==
                    Bp::FilterType::KINEMATIC &&
                    (static_cast<PxU32>(dynamicA)&~7u)==
                    (static_cast<PxU32>(kinematicA)&~7u),
                    "Original PhysX GPU pair filter group layout changed");
        }
    }
}
}

int main() {
    try {
        test_original_source_pair_filter();
        std::cout << "Original PhysX source broadphase pair filter PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
