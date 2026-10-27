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

// Preflight both original GPU buffers and construct their D32 fills
// BEFORE Sc publishes either fetched actor-pair report. The prepared
// commands do not enter mStream until report publication succeeds.
struct SourceSapPreparedAabbReportRetirementNative {
    std::vector<compute::BatchCommand> commands;
    std::span<std::uint32_t> added_words;
    std::span<std::uint32_t> removed_words;
};

inline SourceSapPreparedAabbReportRetirementNative
prepare_source_sap_aabb_report_retirement_native(
    const SourceSapDmaNative& stream,
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
    SourceSapPreparedAabbReportRetirementNative prepared{
        {},added.host_words,removed.host_words};
    fill.append(prepared.commands,addresses,added.device,0u,added_words);
    fill.append(prepared.commands,addresses,removed.device,0u,removed_words);
    return prepared;
}

// PxgAABBManager::processFoundPairs/processLostPairs retire both host
// maps and zero the COMPLETE source GPU allocations after successful
// publication. The caller has reserved enough mStream capacity before
// entering Sc publication; inserting the prepared commands cannot
// allocate and therefore cannot fail after a frame has been published.
inline void commit_source_sap_aabb_report_retirement_native(
    SourceSapDmaNative& stream,
    SourceSapPreparedAabbReportRetirementNative&& prepared) {
    auto& commands=stream.commands();
    if(commands.capacity()-commands.size()<prepared.commands.size())
        throw std::logic_error(
            "Source report bitmap retirement was not reserved before publication");
    commands.insert(commands.end(),
                    std::make_move_iterator(prepared.commands.begin()),
                    std::make_move_iterator(prepared.commands.end()));
    std::fill(prepared.added_words.begin(),prepared.added_words.end(),0u);
    std::fill(prepared.removed_words.begin(),prepared.removed_words.end(),0u);
}

// Standalone fetch callers retain an atomic one-call operation.
inline void retire_source_sap_aabb_report_bitmaps_native(
    SourceSapDmaNative& stream,
    const SourceFillD32NativeStage& fill,
    const CudaDeviceAddressMap& addresses,
    SourceSapAabbReportBitmapNative added,
    SourceSapAabbReportBitmapNative removed) {
    auto prepared=prepare_source_sap_aabb_report_retirement_native(
        stream,fill,addresses,added,removed);
    auto& commands=stream.commands();
    commands.reserve(commands.size()+prepared.commands.size());
    commit_source_sap_aabb_report_retirement_native(
        stream,std::move(prepared));
}

} // namespace fyzz::physx_gpu
