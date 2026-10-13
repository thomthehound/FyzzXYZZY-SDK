// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fy_::compute {

// Backend-independent host API. No Vulkan resources escape through this header.
enum class Backend { Vulkan };

struct DeviceInfo {
    std::uint32_t ordinal = 0;
    std::string name;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t api_version = 0;
    std::uint64_t max_storage_buffer_range = 0;
    std::uint32_t max_workgroup_invocations = 0;
    std::uint32_t max_workgroup_count[3] = {};
    std::uint32_t max_workgroup_size[3] = {};
};

// The opaque identifier belongs to one Context. Zero is never a valid buffer.
using BufferId = std::uint64_t;

// Resource identifiers are owned by a Context and are never interchangeable.
using PipelineId = std::uint64_t;

enum class ShaderFormat { Spirv, Dxil };

struct ShaderBinary {
    ShaderFormat format = ShaderFormat::Spirv;
    std::vector<std::uint8_t> bytes;
    std::string entry_point = "main";
};

// Descriptor set 0 uses consecutive storage-buffer slots [0, count).
// The pipeline binary must declare compatible storage-buffer bindings.
struct ComputePipelineDesc {
    ShaderBinary shader;
    std::uint32_t storage_buffer_binding_count = 0;
    std::uint32_t push_constant_bytes = 0;
};

struct BufferBinding {
    std::uint32_t slot = 0;
    BufferId buffer = 0;
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

struct DispatchSize {
    std::uint32_t x = 1;
    std::uint32_t y = 1;
    std::uint32_t z = 1;
};

std::vector<DeviceInfo> enumerate_devices(Backend backend);

class Context final {
public:
    explicit Context(Backend backend, std::uint32_t device_ordinal = 0);
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;

    const DeviceInfo& device_info() const;

    // Initial host-coherent storage allocation; subsequent versions will
    // add explicit device-local and shared-memory placement.
    BufferId create_buffer(std::size_t bytes);
    void destroy_buffer(BufferId id);
    void upload(BufferId id, std::size_t offset, const void* data, std::size_t bytes);
    void download(BufferId id, std::size_t offset, void* data, std::size_t bytes) const;

    // Submits a GPU transfer and completes it before returning.
    // Operations on a Context are synchronous and not thread-safe yet.
    void copy_buffer(BufferId source, std::size_t source_offset,
                     BufferId destination, std::size_t destination_offset,
                     std::size_t bytes);

    PipelineId create_compute_pipeline(const ComputePipelineDesc& description);
    void destroy_compute_pipeline(PipelineId id);
    void dispatch(PipelineId pipeline, DispatchSize groups,
                  const std::vector<BufferBinding>& buffers,
                  const void* push_constants = nullptr,
                  std::size_t push_constant_bytes = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fy_::compute
