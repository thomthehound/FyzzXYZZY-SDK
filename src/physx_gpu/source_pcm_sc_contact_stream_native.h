// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_gpu/source_pcm_box_candidates_native.h"

#include <algorithm>
#include <cstddef>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {

// One Sc-owned Gu layout for GPU output, host publication and Dy input.
// The original Sc EdgeIndex retains the persistent B-local witnesses.
struct SourcePcmScWorldContactNative {
    float position[3];
    float separation;
};
static_assert(sizeof(SourcePcmScWorldContactNative)==16u);

// The packed stream keeps all Gu contact floats and bounded counters.
// CPU primitive records use the identical layout; no host repack is needed.
// Bits: cache count [0:2], status [3:5], selected [6:8], regenerated
// [9], emitted count [10:12], Sc touch transition [13:14].
// The value 7 signals an invalid counter.
struct SourcePcmGpuContactStreamNative {
    std::uint32_t edge_handle;
    std::uint32_t packed_counts;
    float normal[3];
    SourcePcmScWorldContactNative contacts[4];
};
static_assert(sizeof(SourcePcmGpuContactStreamNative)==84u);
static_assert(offsetof(SourcePcmGpuContactStreamNative,contacts)==20u);
static_assert(sizeof(SourcePcmGpuContactStreamNative)==
              sizeof(native::GpuPcmBoxResult));
static_assert(offsetof(SourcePcmGpuContactStreamNative,edge_handle)==
              offsetof(native::GpuPcmBoxResult,edge));
static_assert(offsetof(SourcePcmGpuContactStreamNative,packed_counts)==
              offsetof(native::GpuPcmBoxResult,packed_counts));
static_assert(offsetof(SourcePcmGpuContactStreamNative,normal)==
              offsetof(native::GpuPcmBoxResult,normal));
static_assert(offsetof(SourcePcmGpuContactStreamNative,contacts)==
              offsetof(native::GpuPcmBoxResult,points));

struct SourcePcmGpuContactHeaderNative {
    std::uint32_t edge_handle;
    std::uint32_t cached_count;
    std::uint32_t status;
    std::uint32_t selected_and_regenerated;
    std::uint32_t emitted_count;
    std::uint32_t touch_transition;
};
inline SourcePcmGpuContactHeaderNative unpack_source_pcm_gpu_header_native(
    const SourcePcmGpuContactStreamNative& source) noexcept {
    const std::uint32_t packed=source.packed_counts;
    return {source.edge_handle,packed&7u,(packed>>3u)&7u,
            ((packed>>6u)&7u)|(((packed>>9u)&1u)<<31u),
            (packed>>10u)&7u,(packed>>13u)&3u};
}

} // namespace fyzz::physx_gpu
