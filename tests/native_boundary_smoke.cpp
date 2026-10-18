// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"
#include "fyzzxyzzy/compute.h"

#include <stdexcept>
#include <string>

// Built without any PhysX dependency: verify a real native CPU timestep
// and explicit native GPU rejection, never a reference backend fallback.
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

    World native_cpu(WorldOptions{ExecutionMode::NativeCpu});
    const auto native_id = native_cpu.add_dynamic_box(
        {0.25f,0.25f,0.25f}, {0,2,0}, 1);
    if (native_cpu.body_state(native_id).position.y != 2)
        throw std::runtime_error("Native-only build did not own actor state");
    native_cpu.set_body_position(native_id, {0,3,0});
    if (native_cpu.body_state(native_id).position.y != 3)
        throw std::runtime_error("Native-only actor relocation failed");
    native_cpu.set_body_velocity(native_id, {2,0,0});
    native_cpu.step(1.0f / 60.0f);
    if (native_cpu.body_state(native_id).position.x <= 0.0f ||
        native_cpu.body_state(native_id).position.y >= 3.0f)
        throw std::runtime_error("Native-only CPU did not integrate gravity");
    native_cpu.remove_body(native_id);
    bool retired = false;
    try { (void)native_cpu.body_state(native_id); }
    catch (const std::out_of_range&) { retired = true; }
    if (!retired)
        throw std::runtime_error("Native-only actor retirement failed");

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
