// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_overlap_work_native.h"
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
constexpr std::uint32_t kBindings = 9u;
constexpr std::uint32_t kParameters = 12u;

CudaDeviceAddressMap::Address address_of(const void* pointer) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(pointer));
}

} // namespace

SourceOverlapWorkNativeStage::SourceOverlapWorkNativeStage(
    compute::Context& context, compute::ShaderBinary shader)
    : context_(context) {
    const auto& gpu = context_.device_info();
    if (gpu.max_workgroup_invocations < 256u ||
        gpu.max_workgroup_size[0] < 256u ||
        gpu.max_workgroup_count[0] < 32u)
        throw std::runtime_error("PhysX overlap-work requires the source 32x256 grid");
    compute::ComputePipelineDesc desc;
    desc.shader = std::move(shader);
    desc.storage_buffer_binding_count = kBindings;
    desc.push_constant_bytes = kParameters * sizeof(std::uint32_t);
    // D3D12 buffers are single stateful subresources. RAW UAVs for reads
    // allow input/output slices to coexist within a source-owned heap root;
    // BufferAccess describes which slots actually write.
    desc.storage_buffer_views.assign(kBindings,
        compute::StorageBufferViewKind::RawReadWrite);
    pipeline_ = context_.create_compute_pipeline(desc);
}

SourceOverlapWorkNativeStage::~SourceOverlapWorkNativeStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void SourceOverlapWorkNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceOverlapWorkLayout layout) const {
    if (!source.numCreatedHandles) return;
    if (!layout.descriptor || !layout.current_boxes ||
        layout.current_boxes > (std::numeric_limits<std::uint32_t>::max() - 255u) / 2u ||
        layout.current_boxes > layout.box_capacity ||
        std::uint64_t(source.numPreviousHandles) + source.numCreatedHandles <
            source.numRemovedHandles ||
        std::uint64_t(source.numPreviousHandles) + source.numCreatedHandles -
            source.numRemovedHandles != layout.current_boxes ||
        std::uint64_t(layout.current_boxes + std::uint64_t(source.numRemovedHandles)) * 2u !=
            layout.projection_slots ||
        layout.region_work_capacity < std::uint64_t(layout.current_boxes) * 64u ||
        layout.handle_region_capacity < layout.region_work_capacity)
        throw std::invalid_argument("Invalid source PhysX overlap-work frame or capacities");

    const auto& gpu = context_.device_info();
    const auto alignment = std::max<std::size_t>(
        4u, static_cast<std::size_t>(gpu.storage_buffer_offset_alignment));
    if (alignment % 4u || !gpu.max_storage_buffer_range)
        throw std::runtime_error("Unsupported native RAW storage view alignment");

    const std::uint64_t hist_entries =
        std::uint64_t(layout.projection_slots) * 64u;
    // Original descriptor stays resident: regionAccumTotal is written by the
    // preceding PhysX region producer, not read from the host-side 'source'.
    if (layout.descriptor >
        std::numeric_limits<CudaDeviceAddressMap::Address>::max() -
            offsetof(physx::PxgBroadPhaseDesc, regionAccumTotal))
        throw std::out_of_range("Source descriptor pointer overflow");
    const auto total_address = layout.descriptor +
        offsetof(physx::PxgBroadPhaseDesc, regionAccumTotal);

    const std::array<CudaDeviceAddressMap::Address, kBindings> pointers{{
        address_of(source.boxHandles[0][0]),
        address_of(source.boxSapBox1D[0]),
        address_of(source.regionRange),
        address_of(source.regionAccum),
        total_address,
        address_of(source.activeRegionsHistogram),
        address_of(source.startRegionsHistogram),
        address_of(source.overlapChecksRegion),
        address_of(source.overlapChecksHandleRegiones)
    }};
    const std::array<std::size_t, kBindings> sizes{{
        source_allocation_bytes(layout.projection_slots, 4u),
        source_allocation_bytes(layout.box_capacity, 8u),
        source_allocation_bytes(layout.box_capacity, 32u),
        source_allocation_bytes(layout.projection_slots, 4u),
        sizeof(std::uint32_t),
        source_allocation_bytes(hist_entries, 4u),
        source_allocation_bytes(hist_entries, 4u),
        source_allocation_bytes(layout.region_work_capacity, 8u),
        source_allocation_bytes(layout.handle_region_capacity, 8u)
    }};

    std::array<std::uint32_t, kParameters> params{};
    params[0] = layout.current_boxes;
    compute::DispatchCommand dispatch;
    dispatch.pipeline = pipeline_;
    dispatch.groups = {32u, 1u, 1u};
    dispatch.buffers.reserve(kBindings);
    for (std::uint32_t i = 0; i < kBindings; ++i) {
        const auto access = i >= 7u
            ? compute::BufferAccess::Write : compute::BufferAccess::Read;
        dispatch.buffers.push_back(source_native_view(
            addresses, pointers[i], sizes[i], alignment, i, access,
            params[i + 1u], gpu.max_storage_buffer_range));
    }
    dispatch.push_constants.resize(sizeof(params));
    std::memcpy(dispatch.push_constants.data(), params.data(), sizeof(params));
    // Commit only after all original PhysX fields validate. A failed source
    // pointer cannot leave half of a new-bounds launch in the caller's batch.
    commands.emplace_back(std::move(dispatch));
}
} // namespace fyzz::physx_gpu
