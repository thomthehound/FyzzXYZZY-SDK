// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_pcm_box_candidates_native.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {

// Source-oriented Gu world contacts are the input to DyContactPrep.
// The original Sc EdgeIndex retains the persistent B-local witnesses.
struct SourcePcmScWorldContactNative {
    float position[3];
    float separation;
};
static_assert(sizeof(SourcePcmScWorldContactNative)==16u);
struct SourcePcmScContactStreamNative {
    std::uint32_t edge_handle;
    std::uint32_t cached_count;
    std::uint32_t status;
    // Lower 31 bits: selected Gu count; high bit: regenerated flag.
    std::uint32_t selected_and_regenerated;
    float normal[3];
    std::uint32_t emitted_count;
    SourcePcmScWorldContactNative contacts[4];
};
static_assert(sizeof(SourcePcmScContactStreamNative)==96u);

} // namespace fyzz::physx_gpu
