// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_newbounds_pairs_native.h"
#include "physx_gpu/source_native_view.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kViews = 16u, kWords = 24u;
CudaDeviceAddressMap::Address address_of(const void* p) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(p));
}
CudaDeviceAddressMap::Address source_scalar(
    CudaDeviceAddressMap::Address descriptor, std::size_t offset) {
    if (descriptor > std::numeric_limits<CudaDeviceAddressMap::Address>::max() - offset)
        throw std::out_of_range("PhysX original descriptor scalar pointer overflow");
    return descriptor + offset;
}
} // namespace

SourceNewBoundsPairsNativeStage::SourceNewBoundsPairsNativeStage(
    compute::Context& context, compute::ShaderBinary shader) : context_(context) {
    const auto& gpu = context_.device_info();
    if (!gpu.supports_int64_shader || !gpu.supports_wave_arithmetic ||
        gpu.subgroup_size < 4u || gpu.subgroup_size > 256u ||
        256u % gpu.subgroup_size != 0u ||
        gpu.max_workgroup_invocations < 256u ||
        gpu.max_workgroup_size[0] < 256u ||
        gpu.max_workgroup_count[0] < 64u)
        throw std::runtime_error("PhysX found-pairs requires original 64x256 launch and native uint64/wave operations");
    compute::ComputePipelineDesc pipeline;
    pipeline.shader = std::move(shader);
    pipeline.storage_buffer_binding_count = kViews;
    pipeline.push_constant_bytes = sizeof(std::uint32_t) * kWords;
    // The CUDA allocator may place read-only and writable fields in one
    // physical D3D12 buffer. RAW UAV descriptors support that mixed use.
    pipeline.storage_buffer_views.assign(kViews,
        compute::StorageBufferViewKind::RawReadWrite);
    pipeline_ = context_.create_compute_pipeline(pipeline);
}

SourceNewBoundsPairsNativeStage::~SourceNewBoundsPairsNativeStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void SourceNewBoundsPairsNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceNewBoundsPairsNativeLayout layout) const {
    if (!source.numCreatedHandles) return;
    const auto handles = std::uint64_t(source.numPreviousHandles) +
                         source.numCreatedHandles;
    const auto projections = std::uint64_t(layout.current_boxes) +
                             source.numRemovedHandles;
    if (!layout.descriptor || !layout.current_boxes ||
        layout.current_boxes > 0x7fffffffu ||
        !layout.box_capacity ||
        layout.current_boxes > layout.box_capacity ||
        handles < source.numRemovedHandles ||
        handles - source.numRemovedHandles != layout.current_boxes ||
        projections * 2u != layout.projection_slots ||
        layout.overlap_prefix_capacity <
            std::uint64_t(layout.current_boxes) * 64u ||
        layout.handle_region_capacity < layout.overlap_prefix_capacity ||
        !source.max_found_lost_pairs)
        throw std::invalid_argument("Invalid original source found-pairs allocation extents");

    const auto& gpu = context_.device_info();
    const auto align = std::max<std::size_t>(
        4u, static_cast<std::size_t>(gpu.storage_buffer_offset_alignment));
    if (align % 4u || !gpu.max_storage_buffer_range)
        throw std::runtime_error("Unsupported source found-pairs storage view limits");

    const bool has_env = source.updateData_envIDs != nullptr;
    const std::array<CudaDeviceAddressMap::Address, kViews> pointers{{
        address_of(source.newIntegerBounds),
        address_of(source.updateData_groups),
        // If no env-ID buffer was allocated, use a never-accessed valid
        // RAW view; no dummy GPU allocation or staging copy is introduced.
        has_env ? address_of(source.updateData_envIDs) : address_of(source.updateData_groups),
        address_of(source.boxHandles[0][0]),
        address_of(source.boxSapBox1D[0]),
        address_of(source.regionRange),
        address_of(source.overlapChecksRegion),
        source_scalar(layout.descriptor, offsetof(physx::PxgBroadPhaseDesc, regionAccumTotal)),
        source_scalar(layout.descriptor, offsetof(physx::PxgBroadPhaseDesc, overlapChecksTotalRegion)),
        address_of(source.overlapChecksHandleRegiones),
        address_of(source.activeRegionsHistogram),
        address_of(source.startRegionsHistogram),
        address_of(source.orderedActiveRegionHandles),
        address_of(source.orderedStartRegionHandles),
        address_of(source.foundPairReport),
        source_scalar(layout.descriptor, offsetof(physx::PxgBroadPhaseDesc, sharedFoundPairIndex))
    }};
    const std::uint64_t hist_elements =
        std::uint64_t(layout.projection_slots) * 64u;
    const std::array<std::size_t, kViews> lengths{{
        source_allocation_bytes(layout.box_capacity, 24u),
        source_allocation_bytes(layout.box_capacity, 4u),
        source_allocation_bytes(layout.box_capacity, 4u),
        source_allocation_bytes(layout.projection_slots, 4u),
        source_allocation_bytes(layout.box_capacity, 8u),
        source_allocation_bytes(layout.box_capacity, 32u),
        source_allocation_bytes(layout.overlap_prefix_capacity, 8u),
        4u,
        8u,
        source_allocation_bytes(layout.handle_region_capacity, 8u),
        source_allocation_bytes(hist_elements, 4u),
        source_allocation_bytes(hist_elements, 4u),
        source_allocation_bytes(hist_elements, 4u),
        source_allocation_bytes(hist_elements, 4u),
        source_allocation_bytes(source.max_found_lost_pairs, 8u),
        4u
    }};
    std::array<std::uint32_t, kWords> params{};
    params[16] = layout.current_boxes;
    params[17] = source.max_found_lost_pairs;
    params[18] = layout.box_capacity;
    params[19] = has_env ? 1u : 0u;
    compute::DispatchCommand dispatch;
    dispatch.pipeline = pipeline_;
    dispatch.groups = {64u, 1u, 1u};
    dispatch.buffers.reserve(kViews);
    for (std::uint32_t i = 0; i < kViews; ++i) {
        const auto access = i >= 14u
            ? compute::BufferAccess::ReadWrite : compute::BufferAccess::Read;
        dispatch.buffers.push_back(source_native_view(
            addresses, pointers[i], lengths[i], align, i, access,
            params[i], gpu.max_storage_buffer_range));
    }
    dispatch.buffers[14].access = compute::BufferAccess::Write;
    dispatch.push_constants.resize(sizeof(params));
    std::memcpy(dispatch.push_constants.data(), params.data(), sizeof(params));
    commands.emplace_back(std::move(dispatch));
}
} // namespace fyzz::physx_gpu
