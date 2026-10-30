// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <variant>

namespace fyzz::compute {

// Backend-independent host API; the initial D3D12 backend is built as a
// separate implementation of Context until shared runtime selection lands.
enum class Backend { Vulkan, D3D12 };

// Auto preserves each backend's established default (Vulkan currently selects
// HostVisible; D3D12 selects DeviceLocal). HostVisible is an explicit, strict
// request for CPU-mappable, GPU-writeable storage: unsupported adapters
// reject it, never silently allocate a DEFAULT heap and label it mapped.
// DeviceLocal retains normal GPU-preferred storage with staging transfers.
enum class MemoryPlacement { Auto, HostVisible, DeviceLocal };

// Only this explicit exception permits a higher-level source allocator to
// fall back from direct mapped storage to staging. Allocation exhaustion,
// lost devices and other failures must propagate, not silently downgrade.
class UnsupportedMemoryPlacement final : public std::runtime_error {
public:
    explicit UnsupportedMemoryPlacement(const char* reason)
        : std::runtime_error(reason) {}
};
// Map intent avoids claiming every read-only PhysX report was CPU-written.
enum class HostMapAccess { Read, Write, ReadWrite };

struct DeviceInfo {
    std::uint32_t ordinal = 0;
    std::string name;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t api_version = 0;
    std::uint64_t max_storage_buffer_range = 0;
    // Device-creation-time descriptor binding alignment. Vulkan obtains its
    // physical-device limit; D3D12 structured 32-bit buffer views use 4.
    // Used to lower CUDA-style interior pointers without staging copies.
    std::uint64_t storage_buffer_offset_alignment = 4;
    std::uint32_t max_workgroup_invocations = 0;
    std::uint32_t max_workgroup_count[3] = {};
    std::uint32_t max_workgroup_size[3] = {};
    // Native subgroup/wave integer scan capability (compute stage).
    // Vulkan queries VK_SUBGROUP_FEATURE_ARITHMETIC_BIT; D3D12 queries
    // WaveOps and shader model via CheckFeatureSupport. Not a vendor ID.
    bool supports_wave_arithmetic = false;
    // Physical GPU-writable buffer memory that can also be CPU-mapped.
    // D3D12: device architecture + custom-heap support, not mere upload/
    // readback heaps. Vulkan: available host-visible/coherent memory type.
    bool supports_gpu_writable_host_visible = false;
    // A Vulkan DEVICE_LOCAL|HOST_VISIBLE memory type is possible even on
    // non-UMA systems (e.g. CPU-accessible discrete GPU memory). Do not
    // infer unified CPU/GPU architecture from this memory-type property.
    bool has_device_local_host_visible_memory = false;
    // These describe architecture-level UMA/cache guarantees where actually
    // exposed (D3D12 architecture query), not Vulkan HOST_COHERENT.
    bool unified_memory = false;
    bool cache_coherent_unified_memory = false;
    // Resolved during adapter admission / logical-device creation, never dispatch.
    bool supports_int64_shader = false;
    bool supports_int64_wave = false; // 64-bit subgroup add/prefix is enabled.
    std::uint32_t subgroup_size = 0;
    std::uint32_t min_subgroup_size = 0;
    std::uint32_t max_subgroup_size = 0;
    // Vulkan: enabled subgroupSizeControl and compute-stage required-size support.
    // D3D12: SM6.6 WaveOps with an advertised selectable wave range.
    bool supports_required_subgroup_size = false;
    std::uint32_t shader_model = 0; // D3D_SHADER_MODEL enum value, 0 on Vulkan.
};

// The opaque identifier belongs to one Context. Zero is never a valid buffer.
using BufferId = std::uint64_t;

// Resource identifiers are owned by a Context and are never interchangeable.
using PipelineId = std::uint64_t;

// Zero is never a valid submission. A completed submission remains queryable
// until wait_submission() releases its tracking resources.
using SubmissionId = std::uint64_t;

enum class ShaderFormat { Spirv, Dxil };

struct ShaderBinary {
    ShaderFormat format = ShaderFormat::Spirv;
    std::vector<std::uint8_t> bytes;
    std::string entry_point = "main";
};

// The shader's declared view ABI is independent of whether several
// bindings refer to allocations in the same native resource. Raw means
// ByteAddressBuffer/RWByteAddressBuffer; Structured32 means 32-bit
// structured elements. Read bindings use t-registers on D3D12, writable
// bindings use u-registers. Indices within each register class are assigned
// in declaration order; Vulkan uses ordinary storage-buffer bindings.
enum class StorageBufferViewKind {
    RawRead,
    RawReadWrite,
    Structured32Read,
    Structured32ReadWrite
};

// Logical bindings use slots [0, count). Explicit storage views define
// the shader ABI and native descriptor types without inferring it from the
// number of buffers. Empty views retain the existing one-buffer arena and
// four-buffer AABB layouts while their callers migrate.
struct ComputePipelineDesc {
    ShaderBinary shader;
    std::uint32_t storage_buffer_binding_count = 0;
    std::uint32_t push_constant_bytes = 0;
    std::vector<StorageBufferViewKind> storage_buffer_views;
    // Zero leaves the wave width implementation-selected. A nonzero value
    // requires that physical width, not a logical-warp emulation.
    // D3D12 shaders must also encode the same [WaveSize] in their DXIL.
    std::uint32_t required_subgroup_size = 0;
};

// Source-derived *access intent*, not a request to change shader bindings.
// Unspecified preserves legacy conservative read/write hazard tracking.
// Accurate intent enables a backend to avoid ordering independent readers.
// A lying Read declaration can create a GPU data race; eventually cross-check
// these declarations against shader reflection/source manifests.
enum class BufferAccess { Unspecified, Read, Write, ReadWrite };

struct BufferBinding {
    std::uint32_t slot = 0;
    BufferId buffer = 0;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    BufferAccess access = BufferAccess::Unspecified;
};

struct DispatchSize {
    std::uint32_t x = 1;
    std::uint32_t y = 1;
    std::uint32_t z = 1;
};

// Commands own their parameter bytes through submission. Both implementations
// record ordered copies/dispatches with explicit GPU resource dependencies.
// A source-backed host-to-device transfer. Source memory is read and copied
// into Context-owned staging before submit_upload_batch returns; it may be
// immediately modified or freed. Multiple spans execute in given order.
struct UploadSpan {
    BufferId destination = 0;
    std::size_t destination_offset = 0;
    const void* data = nullptr;
    std::size_t bytes = 0;
};

// Multiple GPU-to-host outputs share one native readback resource and
// completion fence. Each destination receives only its requested bytes.
struct DownloadSpan {
    BufferId source = 0;
    std::size_t source_offset = 0;
    void* data = nullptr;
    std::size_t bytes = 0;
};

struct CopyCommand {
    BufferId source = 0;
    std::size_t source_offset = 0;
    BufferId destination = 0;
    std::size_t destination_offset = 0;
    std::size_t bytes = 0;
};

struct DispatchCommand {
    PipelineId pipeline = 0;
    DispatchSize groups;
    std::vector<BufferBinding> buffers;
    std::vector<std::uint8_t> push_constants;
};

using BatchCommand = std::variant<CopyCommand, DispatchCommand>;

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

    // Auto retains the pre-existing backend default. An explicit
    // HostVisible request must produce CPU-mappable GPU-writable memory,
    // or fail at creation rather than silently allocating device-only memory.
    BufferId create_buffer(std::size_t bytes,
                           MemoryPlacement placement = MemoryPlacement::Auto);
    void destroy_buffer(BufferId id);

    // Invalidates the handle immediately without waiting for the GPU.
    // Its physical allocation remains alive until all GPU commands already
    // submitted on this Context's ordered queue have completed. No future
    // command may use id; this is a precursor to PhysX's deferred free on
    // async grow/copy. Pending submissions must still be waited or retired
    // through their normal API. The Context owns the retired allocation.
    void retire_buffer(BufferId id);

    // Direct CPU access to a GPU-writable HostVisible buffer; never
    // implicitly synchronize. Callers must wait the existing source GPU
    // fence BEFORE reading GPU writes; finish CPU accesses before GPU uses
    // the resource. map_buffer returns a pointer valid until unmap_buffer.
    // Unmap makes CPU writes visible per backend memory-coherency rules.
    // Neither API stages a transfer or takes ownership of the buffer.
    // Mapping twice or freeing/retiring a mapped buffer is invalid.
    bool buffer_is_host_visible(BufferId id) const;
    void* map_buffer(BufferId id,
                     HostMapAccess access = HostMapAccess::ReadWrite);
    void unmap_buffer(BufferId id);

    void upload(BufferId id, std::size_t offset, const void* data, std::size_t bytes);

    // Copies source bytes to Context-owned staging before returning, then
    // queues transfer to the device without waiting. Caller can immediately
    // reuse/free data; staging is retained until its submission completes.
    // The returned submission token participates in ordinary wait/retire.
    // Initial implementation requires nonempty, 4-byte-aligned transfers.
    SubmissionId submit_upload_buffer(BufferId destination, std::size_t offset,
                                      const void* data, std::size_t bytes);

    // Pack several copies into one Context-owned staging allocation and
    // submit them as one ordered GPU batch (one completion token). Entries
    // must be nonempty and 4-byte aligned; all sources are snapshotted.
    // Staging and its command resources survive until the token completes.
    SubmissionId submit_upload_batch(const std::vector<UploadSpan>& uploads);

    void download(BufferId id, std::size_t offset, void* data, std::size_t bytes) const;
    // One submission and wait for all nonempty device-local readbacks.
    // Source buffer ownership and original CPU output addresses are retained.
    void download_batch(const std::vector<DownloadSpan>& downloads) const;

    // Submits a GPU transfer and completes it before returning.
    // Operations on a Context are synchronous and not thread-safe yet.
    void copy_buffer(BufferId source, std::size_t source_offset,
                     BufferId destination, std::size_t destination_offset,
                     std::size_t bytes);

    // Record and submit work without blocking the calling thread.
    // Work is ordered on this Context's compute queue.
    SubmissionId submit_copy_buffer(BufferId source, std::size_t source_offset,
                                    BufferId destination, std::size_t destination_offset,
                                    std::size_t bytes);

    PipelineId create_compute_pipeline(const ComputePipelineDesc& description);
    void destroy_compute_pipeline(PipelineId id);
    void dispatch(PipelineId pipeline, DispatchSize groups,
                  const std::vector<BufferBinding>& buffers,
                  const void* push_constants = nullptr,
                  std::size_t push_constant_bytes = 0);

    SubmissionId submit_dispatch(PipelineId pipeline, DispatchSize groups,
                                 const std::vector<BufferBinding>& buffers,
                                 const void* push_constants = nullptr,
                                 std::size_t push_constant_bytes = 0);

    // Submit multiple operations in order using a single command buffer and
    // fence. Empty batches and invalid commands are rejected.
    SubmissionId submit_batch(const std::vector<BatchCommand>& commands);
    void execute_batch(const std::vector<BatchCommand>& commands);

    // Optional GPU timestamp span around a *single ordered batch*.
    // Returns nanoseconds on the device timeline, excluding CPU submission
    // and waiting. Includes in-batch copies and barriers, not isolated ALU
    // time. std::nullopt means this queue cannot timestamp compute work.
    // Blocks until the batch finishes; intended for profiling, not the
    // PhysX simulation critical path.
    std::optional<double> profile_batch_gpu_nanoseconds(
        const std::vector<BatchCommand>& commands);

    bool submission_complete(SubmissionId id) const;
    void wait_submission(SubmissionId id);
    // Ordered submissions on this Context queue retire after one fence
    // wait on the newest token. Each supplied token is released once.
    void wait_submissions(std::span<const SubmissionId> ordered);
    void wait_all();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fyzz::compute
