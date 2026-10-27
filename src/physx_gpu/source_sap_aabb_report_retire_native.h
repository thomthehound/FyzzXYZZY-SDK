// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_fill_d32_native.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// Borrowed original PxgAABBManager bitmap/map identities. The owning
// PhysX buffers and PinnableBitMaps retain their allocation and lifetime.
struct SourceSapAabbReportBitmapNative {
    CudaDeviceAddressMap::Address device=0u;
    std::uint64_t device_bytes=0u;
    std::span<std::uint32_t> host_words;
};

// PxgAABBManager::processFoundPairs/processLostPairs post-fetch bitmap
// retirement: clear both complete GPU allocations using the original
// mStream, then clear the original host map words. This follows successful
// fetch and source pair admission, not kernel submission.
//
// Stage BOTH GPU fills before touching the source stream or host maps. A
// malformed second resource cannot consume the first map. Native GPU work
// is enqueued but never waited here: the next source update submits it
// ahead of any following HtoD transfers, preserving CUDA mStream order.
inline void retire_source_sap_aabb_report_bitmaps_native(
    SourceSapDmaNative& stream,
    const SourceFillD32NativeStage& fill,
    const CudaDeviceAddressMap& addresses,
    SourceSapAabbReportBitmapNative added,
    SourceSapAabbReportBitmapNative removed) {
    if(!stream.gpu_complete())
        throw std::logic_error(
            "Source AABB pair publication must complete before bitmap retirement");
    const auto checked_words=[](const SourceSapAabbReportBitmapNative& map){
        if(map.device_bytes%sizeof(std::uint32_t) ||
           map.device_bytes/sizeof(std::uint32_t)>
               std::numeric_limits<std::uint32_t>::max() ||
           map.host_words.size()>
               map.device_bytes/sizeof(std::uint32_t))
            throw std::invalid_argument(
                "Source AABB report bitmap CPU/GPU capacities disagree");
        if(map.device_bytes && !map.device)
            throw std::invalid_argument(
                "Source AABB report bitmap GPU allocation is missing");
        return static_cast<std::uint32_t>(
            map.device_bytes/sizeof(std::uint32_t));
    };
    const auto added_words=checked_words(added);
    const auto removed_words=checked_words(removed);
    std::vector<compute::BatchCommand> prepared;
    fill.append(prepared,addresses,added.device,0u,added_words);
    fill.append(prepared,addresses,removed.device,0u,removed_words);

    auto& commands=stream.commands();
    commands.reserve(commands.size()+prepared.size());
    commands.insert(commands.end(),
                    std::make_move_iterator(prepared.begin()),
                    std::make_move_iterator(prepared.end()));
    std::fill(added.host_words.begin(),added.host_words.end(),0u);
    std::fill(removed.host_words.begin(),removed.host_words.end(),0u);
}

} // namespace fyzz::physx_gpu
