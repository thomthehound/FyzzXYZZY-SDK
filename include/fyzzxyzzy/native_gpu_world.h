// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/compute.h"
#include "fyzzxyzzy/world.h"

#include <array>

namespace fyzz {

// Compiled source SAP stages in the original update/fetch dispatch order.
// The caller owns shader compilation and supplies either SPIR-V or DXIL.
using NativeGpuSapShaders = std::array<compute::ShaderBinary,34>;

// Construct a World with one native Context, original PhysX GPU buddy heap,
// AABB/SAP owners, stream and source Sc/PCM/Dy scene. No reference PhysX
// scene or CPU broadphase is selected. The World owns all these resources.
World create_native_gpu_world(
    compute::Backend backend,NativeGpuSapShaders shaders,
    WorldOptions options={ExecutionMode::NativePortableGpu});

} // namespace fyzz
