// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapBroadphaseDescriptorNative.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace fyzz::physx_gpu {
namespace {
template<class T>
T* device_ptr(const physx::PxgCudaBuffer& b) {
    return reinterpret_cast<T*>(b.getDevicePtr());
}
template<class T>
T* device_alias(CUdeviceptr p) {
    return reinterpret_cast<T*>(p);
}
static_assert(std::is_trivially_copyable_v<physx::PxgBroadPhaseDesc>,
              "PhysX SAP descriptor must remain an original GPU-uploadable record");
}

void prepare_sap_broadphase_descriptor_native(
    physx::PxgBroadPhaseDesc& desc,
    SourceSapBroadphaseDescriptorBuffers b,
    const SourceSapBroadphaseDescriptorInputs& in) {

    // The source computes unsigned previousBoxes from post-update live
    // boxes. Reject impossible input before touching the caller's record.
    const auto sum = std::uint64_t(in.num_boxes) + in.removed_count;
    if (sum < in.created_count ||
        sum - in.created_count > std::numeric_limits<physx::PxU32>::max())
        throw std::overflow_error("PhysX source SAP previous box count overflows");
    const auto previous = static_cast<physx::PxU32>(sum - in.created_count);

    // PhysX source uses PxMemZero over the entire record. In particular,
    // totalIncrementalComparisons and unavailable manager pointer fields
    // start at zero rather than retaining data from the previous frame.
    std::memset(&desc, 0, sizeof(desc));
    desc.updateData_createdHandles = device_ptr<physx::PxU32>(b.created);
    desc.numCreatedHandles = in.created_count;
    desc.updateData_removedHandles = device_ptr<physx::PxU32>(b.removed);
    desc.numRemovedHandles = in.removed_count;

    if (in.aabb_manager) {
        // These GPU aliases must be provided from the owning PhysX
        // PxgAABBManager (NOT from BroadPhaseUpdateData or a mock bitmap).
        const auto& m = *in.aabb_manager;
        desc.aabbMngr_changedHandleMap = device_alias<physx::PxU32>(m.changed_handles);
        desc.aabbMngr_changedHandleBitMapWordCounts = m.changed_bitmap_word_count;
        desc.aabbMngr_addedHandleMap = device_alias<physx::PxU32>(m.added_handles);
        desc.aabbMngr_removedHandleMap = device_alias<physx::PxU32>(m.removed_handles);
        desc.aabbMngr_aggregatedBoundHandles = device_alias<physx::PxU32>(m.aggregated_bounds);
        desc.aabbMngr_volumeData = device_alias<physx::Bp::VolumeData>(m.volume_data);
    }
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
    else {
        // Pinned source descriptor header enables the standalone path.
        // AABB manager mode intentionally DOES NOT set updated array.
        desc.updateData_updatedHandles = device_ptr<physx::PxU32>(b.updated);
        desc.numUpdatedHandles = in.updated_count;
    }
#endif

    desc.oldIntegerBounds = device_ptr<physx::PxgIntegerAABB>(b.old_integer_bounds);
    desc.newIntegerBounds = device_ptr<physx::PxgIntegerAABB>(b.new_integer_bounds);
    desc.updateData_fpBounds = device_ptr<physx::PxBounds3>(b.fp_bounds);
    desc.updateData_contactDistances = device_ptr<physx::PxReal>(b.contact_distances);
    desc.updateData_groups = device_ptr<physx::PxU32>(b.groups);
    desc.updateData_envIDs = device_ptr<physx::PxU32>(b.env_ids);
    desc.numPreviousHandles = previous;

    desc.foundPairReport = device_ptr<physx::PxgBroadPhasePair>(b.found_pairs);
    desc.lostPairReport = device_ptr<physx::PxgBroadPhasePair>(b.lost_pairs);
    desc.foundAggPairReport = device_ptr<physx::PxgBroadPhasePair>(b.found_aggregate);
    desc.lostAggPairReport = device_ptr<physx::PxgBroadPhasePair>(b.lost_aggregate);
    desc.foundActorPairReport = device_ptr<physx::PxgBroadPhasePair>(b.found_actor);
    desc.lostActorPairReport = device_ptr<physx::PxgBroadPhasePair>(b.lost_actor);
    desc.foundPairReportMap = device_alias<physx::PxgBroadPhasePair>(in.mapped_found_actor_pairs);
    desc.lostPairReportMap = device_alias<physx::PxgBroadPhasePair>(in.mapped_lost_actor_pairs);

    for(physx::PxU32 i=0; i<3u; ++i) {
        desc.boxSapBox1D[i] = device_ptr<physx::PxgSapBox1D>(b.sap_boxes[i]);
        desc.boxNewSapBox1D[i] = device_ptr<physx::PxgSapBox1D>(b.new_sap_boxes[i]);
        desc.boxProjectionRanks[i] = device_ptr<physx::PxU32>(b.ranks[i]);
        desc.boxProjections[i] = device_ptr<physx::PxU32>(b.projections[i]);
        desc.boxHandles[0][i] = device_ptr<physx::PxU32>(b.projection_handles[i]);
        desc.boxHandles[1][i] = device_ptr<physx::PxU32>(b.projection_handles[i+3u]);
        desc.totalEndPtHistogram[i] = device_ptr<physx::PxU32>(b.total_end_histograms[i]);
        desc.blockTotalEndPtHistogram[i] = device_ptr<physx::PxU32>(b.block_total_end_histograms[i]);

        for(physx::PxU32 j=0; j<2u; ++j) {
            const auto index = j*3u+i;
            desc.endPtHistogram[j][i] = device_ptr<physx::PxU32>(b.end_histograms[index]);
            desc.blockEndPtHistogram[j][i] = device_ptr<physx::PxU32>(b.block_end_histograms[index]);
            desc.startPtHistogram[j][i] = device_ptr<physx::PxU32>(b.start_histograms[index]);
            desc.blockStartPtHistogram[j][i] = device_ptr<physx::PxU32>(b.block_start_histograms[index]);
            desc.endPointHandles[j][i] = device_ptr<physx::PxU32>(b.end_handles[index]);
            desc.startPointHandles[j][i] = device_ptr<physx::PxU32>(b.start_handles[index]);
        }
        desc.incrementalComparisons[i] = device_ptr<physx::PxU32>(b.incremental_comparisons[i]);
        desc.incrementalBlockComparisons[i] = device_ptr<physx::PxU32>(b.block_incremental_comparisons[i]);
    }
    for(physx::PxU32 i=0; i<2u; ++i) {
        desc.aggReportBlock[i] = device_ptr<physx::PxU32>(b.aggregate_report_blocks[i]);
        desc.actorReportBlock[i] = device_ptr<physx::PxU32>(b.actor_report_blocks[i]);
    }

    desc.activeRegionsHistogram = device_ptr<physx::PxU32>(b.active_regions);
    desc.startRegionsHistogram = device_ptr<physx::PxU32>(b.start_regions);
    desc.orderedActiveRegionHandles = device_ptr<physx::PxU32>(b.ordered_active_handles);
    desc.orderedStartRegionHandles = device_ptr<physx::PxU32>(b.ordered_start_handles);
    desc.blockOverlapChecksRegion = device_ptr<physx::regionOverlapType>(b.block_overlap_checks);
    desc.overlapChecksRegion = device_ptr<physx::regionOverlapType>(b.overlap_checks);
    desc.overlapChecksHandleRegiones = device_ptr<physx::PxgHandleRegion>(b.overlap_handle_regions);
    desc.regionRange = device_ptr<physx::PxgIntegerRegion>(b.region_range);
    desc.startRegionAccum = device_ptr<physx::PxU32>(b.start_region_accum);
    desc.blockStartRegionAccum = device_ptr<physx::PxU32>(b.block_start_region_accum);
    desc.regionAccum = device_ptr<physx::PxU32>(b.region_accum);
    desc.blockRegionAccum = device_ptr<physx::PxU32>(b.block_region_accum);

    desc.sharedFoundPairIndex = 0u;
    desc.sharedLostPairIndex = 0u;
    desc.sharedFoundAggPairIndex = 0u;
    desc.sharedLostAggPairIndex = 0u;
    desc.startRegionAccumTotal = 0u;
    desc.regionAccumTotal = in.region_accum_total;
    desc.overlapChecksTotalRegion = in.overlap_checks_total;
    desc.max_found_lost_pairs = in.max_found_lost_pairs;
    desc.found_lost_pairs_overflow_flags = false;
    desc.max_found_lost_agg_pairs = in.max_found_lost_aggregate_pairs;
}

} // namespace fyzz::physx_gpu
