// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"
#include "fyzzxyzzy/compute.h"

#include <stdexcept>
#include <string>

// This target deliberately builds without any PhysX dependency. Until
// the native scene/solver ships, neither execution mode may masquerade as
// working physics. The compute types must remain independent of PhysX.
int main() {
    using fyzz::ExecutionMode;
    using fyzz::World;
    using fyzz::WorldOptions;

    bool reference_unavailable = false;
    try {
        World world(WorldOptions{ExecutionMode::ReferencePhysXCpu});
    } catch (const std::logic_error& error) {
        reference_unavailable =
            std::string(error.what()).find("disabled at SDK build time") !=
            std::string::npos;
    }
    if (!reference_unavailable)
        throw std::runtime_error("Reference-only build boundary bypassed");

    bool native_unavailable = false;
    try {
        World world(WorldOptions{ExecutionMode::NativePortableGpu});
    } catch (const std::logic_error& error) {
        native_unavailable =
            std::string(error.what()).find("not yet implemented") !=
            std::string::npos;
    }
    if (!native_unavailable)
        throw std::runtime_error("Native GPU request silently admitted");

    bool invalid_rejected = false;
    try {
        World world(WorldOptions{static_cast<ExecutionMode>(255)});
    } catch (const std::invalid_argument&) {
        invalid_rejected = true;
    }
    if (!invalid_rejected)
        throw std::runtime_error("Unknown execution mode accepted");

    static_assert(sizeof(fyzz::compute::BufferId) == sizeof(std::uint64_t));
    return 0;
}
