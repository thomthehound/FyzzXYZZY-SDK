// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/native_gpu_world.h"
#include "physx_gpu/source_sap_scene_frame_native.h"
#include "physx_port/gpubroadphase/PxgAabbSourceMembersNative.h"

#include <utility>

namespace fyzz::physx_gpu {

// One device/stream owner constructs every original aggregate GPU stage.
// Views borrow pipeline objects; Sc and SAP never own a second stream,
// shader compiler, aggregate index allocator or report path.
class SourceSapAggregateOwnedStagesNative final {
    static compute::ShaderBinary take(
        NativeGpuAggregateShaders& shaders,NativeGpuAggregateStage stage) {
        return std::move(shaders[static_cast<std::size_t>(stage)]);
    }
public:
    SourceSapAggregateOwnedStagesNative(
        compute::Context& context,NativeGpuAggregateShaders shaders)
        : bounds_(context,
            take(shaders,NativeGpuAggregateStage::Bounds),
            take(shaders,NativeGpuAggregateStage::MarkBounds)),
          history_(context,
            take(shaders,NativeGpuAggregateStage::MemberHistory)),
          projection_(context,
            take(shaders,NativeGpuAggregateStage::Projection)),
          self_collision_(context,
            take(shaders,NativeGpuAggregateStage::SelfCollision)),
          report_copy_(context,
            take(shaders,NativeGpuAggregateStage::CopyReports)),
          pair_admission_(context,
            take(shaders,NativeGpuAggregateStage::PairAdmit),
            take(shaders,NativeGpuAggregateStage::PairPublish)),
          pair_collisions_(context,
            take(shaders,NativeGpuAggregateStage::PersistentCollisions)),
          pair_retirement_(context,
            take(shaders,NativeGpuAggregateStage::PairRetireHistogram),
            take(shaders,NativeGpuAggregateStage::PairRetireMark),
            take(shaders,NativeGpuAggregateStage::PairRetirePublish)),
          borrowed_{bounds_,history_,projection_,self_collision_,
                    report_copy_,pair_admission_,pair_collisions_,pair_retirement_} {}

    const SourceSapAggregateStagesNative& borrowed() const noexcept {
        return borrowed_;
    }

private:
    SourceAggregateGpuNativeStage bounds_;
    SourceAggregateMemberHistoryNativeStage history_;
    SourceAggregateProjectionNativeStage projection_;
    SourceAggregateSelfCollisionNativeStage self_collision_;
    SourceAggregateReportCopyNativeStage report_copy_;
    SourceAggregatePairAdmissionNativeStage pair_admission_;
    SourceAggregatePersistentCollisionNativeStage pair_collisions_;
    SourceAggregatePairRetirementNativeStage pair_retirement_;
    SourceSapAggregateStagesNative borrowed_;
};

} // namespace fyzz::physx_gpu
