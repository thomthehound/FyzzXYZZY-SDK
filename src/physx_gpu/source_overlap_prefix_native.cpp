// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_overlap_prefix_native.h"
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
constexpr std::uint32_t kSlots = 4u;
constexpr std::uint32_t kBlockCount = 32u;
constexpr std::uint32_t kPushWords = 8u;

CudaDeviceAddressMap::Address pointer_address(const void* p) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(p));
}
CudaDeviceAddressMap::Address scalar_address(
    CudaDeviceAddressMap::Address desc, std::size_t member) {
    if (desc > std::numeric_limits<CudaDeviceAddressMap::Address>::max() - member)
        throw std::out_of_range("PhysX source descriptor scalar address overflow");
    return desc + member;
}
} // namespace

SourceOverlapPrefixNativeStage::SourceOverlapPrefixNativeStage(
    compute::Context& context, compute::ShaderBinary compute_wave,
    compute::ShaderBinary output_wave) : context_(context) {
    const auto& device = context_.device_info();
    if (!device.supports_int64_wave || !device.supports_wave_arithmetic ||
        device.subgroup_size < 4u || device.subgroup_size > 128u ||
        256u % device.subgroup_size != 0u ||
        device.max_workgroup_invocations < 256u ||
        device.max_workgroup_size[0] < 256u ||
        device.max_workgroup_count[0] < kBlockCount)
        throw std::runtime_error("Source overlap prefix requires native 64-bit wave scans and 32x256 dispatch");
    compute::ComputePipelineDesc pipe;
    pipe.storage_buffer_binding_count = kSlots;
    pipe.push_constant_bytes = kPushWords * sizeof(std::uint32_t);
    // All-UAV views allow read and write allocations in one D3D12 root.
    // Each dispatch's BufferAccess still records its actual writes.
    pipe.storage_buffer_views.assign(kSlots,
        compute::StorageBufferViewKind::RawReadWrite);
    try {
        pipe.shader = std::move(compute_wave);
        pipelines_[0] = context_.create_compute_pipeline(pipe);
        pipe.shader = std::move(output_wave);
        pipelines_[1] = context_.create_compute_pipeline(pipe);
    } catch (...) {
        for (const auto p : pipelines_) if (p) context_.destroy_compute_pipeline(p);
        throw;
    }
}

SourceOverlapPrefixNativeStage::~SourceOverlapPrefixNativeStage() {
    for (const auto p : pipelines_) if (p) context_.destroy_compute_pipeline(p);
}

void SourceOverlapPrefixNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceOverlapPrefixLayout layout) const {
    if (!source.numCreatedHandles) return;
    const auto population = std::uint64_t(source.numPreviousHandles) +
                            source.numCreatedHandles;
    if (!layout.descriptor || !layout.current_boxes ||
        population < source.numRemovedHandles ||
        population - source.numRemovedHandles != layout.current_boxes ||
        layout.overlap_capacity < std::uint64_t(layout.current_boxes) * 64u)
        throw std::invalid_argument("Invalid PhysX source overlap scan frame");
    const auto& device = context_.device_info();
    const auto alignment = std::max<std::size_t>(
        4u, static_cast<std::size_t>(device.storage_buffer_offset_alignment));
    if (alignment % 4u || !device.max_storage_buffer_range)
        throw std::runtime_error("Unsupported native source scan storage alignment");

    const std::array<CudaDeviceAddressMap::Address, kSlots> addresses_to_bind{{
        pointer_address(source.overlapChecksRegion),
        pointer_address(source.blockOverlapChecksRegion),
        scalar_address(layout.descriptor,
            offsetof(physx::PxgBroadPhaseDesc, regionAccumTotal)),
        scalar_address(layout.descriptor,
            offsetof(physx::PxgBroadPhaseDesc, overlapChecksTotalRegion))
    }};
    const std::array<std::size_t, kSlots> byte_counts{{
        source_allocation_bytes(layout.overlap_capacity, sizeof(physx::regionOverlapType)),
        kBlockCount * sizeof(physx::regionOverlapType),
        sizeof(std::uint32_t),
        sizeof(physx::regionOverlapType)
    }};

    std::array<std::uint32_t, kPushWords> constants{};
    std::array<compute::BufferBinding, kSlots> views{};
    for (std::uint32_t i = 0; i < kSlots; ++i) {
        views[i] = source_native_view(
            addresses, addresses_to_bind[i], byte_counts[i], alignment, i,
            compute::BufferAccess::Unspecified,
            constants[i], device.max_storage_buffer_range);
    }
    std::vector<compute::BatchCommand> pending;
    pending.reserve(2u);
    for (std::uint32_t i = 0u; i < 2u; ++i) {
        compute::DispatchCommand command;
        command.pipeline = pipelines_[i];
        command.groups = {kBlockCount, 1u, 1u};
        command.push_constants.resize(sizeof(constants));
        std::memcpy(command.push_constants.data(), constants.data(),
                    sizeof(constants));
        command.buffers.assign(views.begin(), views.end());
        command.buffers[0].access = compute::BufferAccess::ReadWrite;
        command.buffers[1].access = i == 0u
            ? compute::BufferAccess::Write : compute::BufferAccess::Read;
        command.buffers[2].access = compute::BufferAccess::Read;
        command.buffers[3].access = i == 0u
            ? compute::BufferAccess::Read : compute::BufferAccess::Write;
        pending.emplace_back(std::move(command));
    }
    commands.reserve(commands.size() + pending.size());
    for (auto& op : pending) commands.emplace_back(std::move(op));
}
} // namespace fyzz::physx_gpu
