// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_region_create_native.h"
#include "physx_gpu/source_native_view.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kGroups = 32u, kThreads = 256u;

template<class Ptr> CudaDeviceAddressMap::Address address(Ptr* source) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(source));
}
} // namespace

SourceRegionCreateNativeStage::SourceRegionCreateNativeStage(
    compute::Context& context, compute::ShaderBinary shader,
    const SourceFillD32NativeStage& fill)
    : context_(context), fill_(fill) {
    const auto& gpu = context_.device_info();
    if (gpu.max_workgroup_invocations < kThreads ||
        gpu.max_workgroup_size[0] < kThreads ||
        gpu.max_workgroup_count[0] < kGroups)
        throw std::runtime_error("PhysX BP_CREATE_REGIONS requires original 32x256 grid");
    compute::ComputePipelineDesc pipeline;
    pipeline.shader = std::move(shader);
    pipeline.storage_buffer_binding_count = 10u;
    pipeline.push_constant_bytes = 12u * sizeof(std::uint32_t);
    pipeline.storage_buffer_views = {
        compute::StorageBufferViewKind::RawRead,
        compute::StorageBufferViewKind::RawRead,
        compute::StorageBufferViewKind::RawRead,
        compute::StorageBufferViewKind::RawRead,
        compute::StorageBufferViewKind::RawRead,
        compute::StorageBufferViewKind::RawRead,
        compute::StorageBufferViewKind::RawRead,
        compute::StorageBufferViewKind::RawReadWrite,
        compute::StorageBufferViewKind::RawReadWrite,
        compute::StorageBufferViewKind::RawReadWrite};
    pipeline_ = context_.create_compute_pipeline(pipeline);
}

SourceRegionCreateNativeStage::~SourceRegionCreateNativeStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void SourceRegionCreateNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceRegionCreateNativeLayout layout) const {
    // Original PxgCudaBroadPhaseSap::computeRegionHistogramKernel gate.
    if (!source.numCreatedHandles) return;
    const std::uint64_t before_remove =
        std::uint64_t(source.numPreviousHandles) + source.numCreatedHandles;
    if (source.numRemovedHandles > before_remove)
        throw std::invalid_argument("PhysX BP_CREATE_REGIONS removed too many boxes");
    const auto boxes = before_remove - source.numRemovedHandles;
    // Original host's (2 * boxes + 255)/256 and region-clear
    // (2 * (previous+created) * 64 + 3)&~3 must not wrap.
    if (!boxes || boxes > layout.box_capacity ||
        boxes > (std::numeric_limits<std::uint32_t>::max() - 255u) / 2u ||
        before_remove > std::numeric_limits<std::uint32_t>::max() / 128u ||
        !layout.projection_slots ||
        before_remove * 2u > layout.projection_slots)
        throw std::invalid_argument("Invalid original PhysX region creation sizes");

    const auto count = static_cast<std::uint32_t>(before_remove * 128u);
    const auto& gpu = context_.device_info();
    const auto alignment = static_cast<std::size_t>(
        gpu.storage_buffer_offset_alignment);
    if (!alignment || alignment % 4u || !gpu.max_storage_buffer_range)
        throw std::invalid_argument("Invalid native PhysX region storage capabilities");

    // Build the original two async clears and then BP_CREATE_REGIONS
    // off-list, so a missing last source allocation never commits half a
    // source call sequence to the caller's pre-existing command batch.
    std::vector<compute::BatchCommand> pending;
    fill_.append(pending, addresses, address(source.activeRegionsHistogram), 0u, count);
    fill_.append(pending, addresses, address(source.startRegionsHistogram), 0u, count);

    const std::array<CudaDeviceAddressMap::Address,10> pointers{{
        address(source.boxHandles[0][0]),
        address(source.boxSapBox1D[0]), address(source.boxSapBox1D[1]),
        address(source.boxSapBox1D[2]),
        address(source.totalEndPtHistogram[0]), address(source.totalEndPtHistogram[1]),
        address(source.totalEndPtHistogram[2]),
        address(source.regionRange), address(source.startRegionAccum),
        address(source.regionAccum)}};
    const std::array<std::size_t,10> sizes{{
        source_allocation_bytes(layout.projection_slots,4u),
        source_allocation_bytes(layout.box_capacity,8u),
        source_allocation_bytes(layout.box_capacity,8u),
        source_allocation_bytes(layout.box_capacity,8u),
        source_allocation_bytes(layout.projection_slots,4u),
        source_allocation_bytes(layout.projection_slots,4u),
        source_allocation_bytes(layout.projection_slots,4u),
        source_allocation_bytes(layout.box_capacity,32u),
        source_allocation_bytes(layout.projection_slots,4u),
        source_allocation_bytes(layout.projection_slots,4u)}};

    std::array<std::uint32_t,12> constants{};
    constants[0] = static_cast<std::uint32_t>(boxes);
    compute::DispatchCommand op;
    op.pipeline = pipeline_;
    op.groups = {kGroups,1u,1u};
    op.buffers.reserve(pointers.size());
    for (std::uint32_t i=0u; i<pointers.size(); ++i) {
        const auto access = i < 7u ? compute::BufferAccess::Read :
                                      compute::BufferAccess::Write;
        op.buffers.push_back(source_native_view(
            addresses,pointers[i],sizes[i],alignment,i,access,
            constants[1u+i],gpu.max_storage_buffer_range));
    }
    op.push_constants.resize(sizeof(constants));
    std::memcpy(op.push_constants.data(),constants.data(),sizeof(constants));
    pending.emplace_back(std::move(op));
    commands.reserve(commands.size() + pending.size());
    for(auto& command:pending)
        commands.emplace_back(std::move(command));
}
} // namespace fyzz::physx_gpu
