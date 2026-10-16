// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/source_region_create_native.h"
#include "physx_gpu/source_region_histograms_native.h"
#include "physx_gpu/source_active_region_lists_native.h"
#include "physx_gpu/source_newbounds_sequence_native.h"

#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// Capacities belong to PxgCudaBroadPhaseSap::gpuDMAUp, not this lowering.
// The region stages accept padded projection allocations, while the pair
// stages use exactly 2 * (current boxes + retained removed handles).
struct SourceRegionNewBoundsNativeLayout {
    SourceRegionCreateNativeLayout create;
    SourceRegionHistogramsNativeLayout histograms;
    SourceActiveRegionListsNativeLayout active;
    SourceNewBoundsSequenceNativeLayout new_bounds;
};

// Contiguous part of the ORIGINAL PxgCudaBroadPhaseSap::update() mStream:
// computeRegionHistogramKernel (7 commands), computeStartAndActiveHistogramKernel
// (3 commands), generateNewPairsKernel (4 commands). Caller owns all PhysX
// allocations, original descriptor, stream, submission and task continuations.
// In particular, this is not a second scene driver or a submission boundary.
class SourceRegionNewBoundsNativeStage final {
public:
    SourceRegionNewBoundsNativeStage(
        const SourceRegionCreateNativeStage& create,
        const SourceRegionHistogramsNativeStage& histograms,
        const SourceActiveRegionListsNativeStage& active,
        const SourceNewBoundsSequenceNativeStage& new_bounds) noexcept
        : create_(create), histograms_(histograms), active_(active),
          new_bounds_(new_bounds) {}

    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceRegionNewBoundsNativeLayout layout) const {
        // The three original calls share the exact created-handle gate.
        if (!source.numCreatedHandles) return;

        const std::uint64_t retained =
            std::uint64_t(source.numPreviousHandles) + source.numCreatedHandles;
        if (retained < source.numRemovedHandles ||
            retained > std::numeric_limits<std::uint32_t>::max() / 2u)
            throw std::invalid_argument("Invalid PhysX retained projection population");
        const auto current = retained - source.numRemovedHandles;
        const auto projections = retained * 2u;
        if (!current || current != layout.new_bounds.current_boxes ||
            !layout.new_bounds.descriptor ||
            layout.histograms.descriptor != layout.new_bounds.descriptor ||
            layout.active.descriptor != layout.new_bounds.descriptor ||
            layout.new_bounds.projection_slots != projections ||
            layout.create.projection_slots < projections ||
            layout.histograms.projection_slots < projections ||
            layout.active.projection_slots < projections ||
            layout.active.box_capacity != layout.create.box_capacity ||
            layout.new_bounds.box_capacity != layout.create.box_capacity)
            throw std::invalid_argument("Inconsistent original PhysX source chain layout");

        // Do not commit partial work to the original stream if ANY later
        // source view fails validation, including the final found-pair report.
        std::vector<compute::BatchCommand> pending;
        pending.reserve(14u);
        create_.append(pending, source, addresses, layout.create);
        const auto created_commands = pending.size();
        // The two source D32 clears can be segmented over a native device's
        // maximum view range. Their physical command count need not be two.
        if (created_commands < 3u)
            throw std::logic_error("Missing original source region producer/clears");
        histograms_.append(pending, source, addresses, layout.histograms);
        active_.append(pending, source, addresses, layout.active);
        new_bounds_.append(pending, source, addresses, layout.new_bounds);
        // The remaining eleven original launches each lower to one native
        // dispatch, regardless of the number of D32 clear chunks.
        if (pending.size() != created_commands + 11u)
            throw std::logic_error("Original PhysX region/pair launch chain changed");
        commands.reserve(commands.size() + pending.size());
        commands.insert(commands.end(),
            std::make_move_iterator(pending.begin()),
            std::make_move_iterator(pending.end()));
    }

private:
    const SourceRegionCreateNativeStage& create_;
    const SourceRegionHistogramsNativeStage& histograms_;
    const SourceActiveRegionListsNativeStage& active_;
    const SourceNewBoundsSequenceNativeStage& new_bounds_;
};

} // namespace fyzz::physx_gpu
