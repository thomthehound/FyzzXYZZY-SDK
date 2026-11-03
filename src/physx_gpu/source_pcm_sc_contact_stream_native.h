// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_pcm_box_candidates_native.h"

#include <algorithm>
#include <cstddef>
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

// Device-side Sc PCM stream keeps all Gu contact floats, but folds the
// bounded manager counters into one word. The host Sc record above is
// reconstructed exactly after fetch; Dy reads this stream in place.
// Bits: cache count [0:2], status [3:5], selected [6:8], regenerated
// [9], emitted count [10:12]. The value 7 signals an invalid counter.
struct SourcePcmGpuContactStreamNative {
    std::uint32_t edge_handle;
    std::uint32_t packed_counts;
    float normal[3];
    SourcePcmScWorldContactNative contacts[4];
};
static_assert(sizeof(SourcePcmGpuContactStreamNative)==84u);
static_assert(offsetof(SourcePcmGpuContactStreamNative,contacts)==20u);

struct SourcePcmGpuContactHeaderNative {
    std::uint32_t edge_handle;
    std::uint32_t cached_count;
    std::uint32_t status;
    std::uint32_t selected_and_regenerated;
    std::uint32_t emitted_count;
};
inline SourcePcmGpuContactHeaderNative unpack_source_pcm_gpu_header_native(
    const SourcePcmGpuContactStreamNative& source) noexcept {
    const std::uint32_t packed=source.packed_counts;
    return {source.edge_handle,packed&7u,(packed>>3u)&7u,
            ((packed>>6u)&7u)|(((packed>>9u)&1u)<<31u),
            (packed>>10u)&7u};
}
inline SourcePcmScContactStreamNative unpack_source_pcm_sc_contact_native(
    const SourcePcmGpuContactStreamNative& source) {
    const auto header=unpack_source_pcm_gpu_header_native(source);
    SourcePcmScContactStreamNative record{};
    record.edge_handle=header.edge_handle;
    record.cached_count=header.cached_count;
    record.status=header.status;
    record.selected_and_regenerated=header.selected_and_regenerated;
    record.emitted_count=header.emitted_count;
    if(record.status==0u && record.emitted_count<=4u) {
        std::memcpy(record.normal,source.normal,sizeof(record.normal));
        std::memcpy(record.contacts,source.contacts,
                    std::size_t(record.emitted_count)*sizeof(record.contacts[0]));
    }
    return record;
}

} // namespace fyzz::physx_gpu
