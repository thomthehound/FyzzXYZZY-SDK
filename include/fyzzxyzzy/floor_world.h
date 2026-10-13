// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fyzzxyzzy/floor_contact.h"
#include "fyzzxyzzy/world.h"

namespace fyzz::compute {

// Constructs one authoritative Vulkan World using both integration/floor and
// vertical body-contact kernels. Both ShaderBinary objects must contain
// SPIR-V for the corresponding SDK shader sources. This experimental
// backend supports an infinite horizontal floor and nonrotating boxes with
// vertical-only motion and AABB contact; not general 3D rigid-body physics.
World create_floor_world(ShaderBinary floor_shader,
                         ShaderBinary vertical_contact_shader,
                         std::uint32_t device_ordinal = 0);

} // namespace fyzz::compute
