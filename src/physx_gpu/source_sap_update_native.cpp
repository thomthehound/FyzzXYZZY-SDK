// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_update_native.h"
#include "BpBroadPhaseUpdate.h"

namespace fyzz::physx_gpu {
bool append_source_sap_update_native(
    SourceSapDmaNative& stream,
    const SourceSapUpdateNativeStages& stages,
    const physx::Bp::BroadPhaseUpdateData& update,
    bool& force_update,
    const physx::PxgBroadPhaseDesc& descriptor,
    std::span<const physx::PxgRadixSortDesc,6> wr_radix,
    std::span<const physx::PxgRadixSortDesc,6> wor_radix,
    const CudaDeviceAddressMap& addresses,
    const SourceSapUpdateNativeFrame& frame,
    physx::PxgCudaBuffer& new_integer_bounds,
    physx::PxgCudaBuffer& old_integer_bounds) {

    // The original update early-return is AFTER gpuDMAUp. Preserve the
    // member's previous value if a late stage rejects the source frame.
    bool next_force_update=force_update;
    const bool execute=should_execute_sap_update_after_dma_up_native(
        update,next_force_update);
    if(!execute)return false;
    if(descriptor.numCreatedHandles!=update.getNumCreatedHandles() ||
       descriptor.numRemovedHandles!=update.getNumRemovedHandles() ||
       !frame.device_descriptor ||
       frame.reports.descriptor!=frame.device_descriptor ||
       descriptor.newIntegerBounds!=reinterpret_cast<
           decltype(descriptor.newIntegerBounds)>(
               static_cast<std::uintptr_t>(new_integer_bounds.getDevicePtr())) ||
       descriptor.oldIntegerBounds!=reinterpret_cast<
           decltype(descriptor.oldIntegerBounds)>(
               static_cast<std::uintptr_t>(old_integer_bounds.getDevicePtr())))
        throw std::invalid_argument(
            "Source SAP GPU descriptor or integer-bounds owner disagrees with gpuDMAUp");

    std::vector<compute::BatchCommand> pending;
    SourceSapUpdateHeadNative head{
        frame.translation,frame.removed,frame.updated,
        descriptor.numPreviousHandles,update.getGpuStateChanged(),true};
    append_source_sap_update_head_native(
        pending,stages.translate,stages.removed,stages.updated,stages.radix,
        descriptor,wr_radix,addresses,head);

    append_source_sap_incremental_phase_native(
        pending,stages.histograms,stages.comparisons,
        stages.incremental_pairs,descriptor,frame.device_descriptor,
        addresses,frame.histograms,frame.comparisons,
        frame.incremental_pairs,update.getGpuStateChanged(),true);

    const SourceSapUpdateMiddleNative middle{
        frame.indices,frame.removed,frame.created,true};
    append_source_sap_update_middle_native(
        pending,stages.indices,stages.removed,stages.created,stages.radix,
        descriptor,wor_radix,addresses,middle);
    append_source_sap_final_histograms_native(
        pending,stages.histograms,descriptor,addresses,frame.histograms,true);

    append_source_sap_update_tail_commands_native(
        pending,stages.indices,stages.new_pairs,stages.clear,
        stages.reports,descriptor,addresses,frame.indices,
        frame.new_pairs,frame.clear,frame.reports,true);

    auto& commands=stream.commands();
    commands.reserve(commands.size()+pending.size());
    commands.insert(commands.end(),
                    std::make_move_iterator(pending.begin()),
                    std::make_move_iterator(pending.end()));
    physx::PxgCudaBuffer::swapBuffer(new_integer_bounds,old_integer_bounds);
    force_update=next_force_update;
    return true;
}
} // namespace fyzz::physx_gpu
