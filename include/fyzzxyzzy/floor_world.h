// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fyzzxyzzy/floor_contact.h"
#include "fyzzxyzzy/world.h"

namespace fyzz::compute {

// Experimental, non-PhysX GPU physics prototype (explicit build opt-in).
// Creates a World whose only simulation backend is Vulkan floor contact.
// Supported: one horizontal infinite floor, axis-aligned dynamic boxes,
// gravity and floor response. Unsupported: static box geometry, body-to-body
// collisions, rotations, joints and other PhysX scene features.
// Requires SPIR-V compiled from src/kernels/rigid_floor.comp.
World create_floor_world(ShaderBinary shader, std::uint32_t device_ordinal = 0);

} // namespace fyzz::compute
