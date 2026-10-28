// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// D3D12 implementation of the common native compute contract.
// Pipeline-declared storage view types determine SRV/UAV registers and RAW
// versus structured descriptors. The PhysX host selects resources at launch
// setup; this layer never interprets CUDA virtual addresses.

#include "fyzzxyzzy/compute.h"

#ifndef _WIN32
#error D3D12 compute is Windows-only
#endif

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace fyzz::compute {
namespace {

using Microsoft::WRL::ComPtr;

void check(HRESULT hr, const char* operation) {
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(operation) + " failed (HRESULT " +
                                 std::to_string(static_cast<unsigned long>(hr)) + ")");
    }
}

std::string adapter_name(const DXGI_ADAPTER_DESC1& desc) {
    int length = WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                                     nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<std::size_t>(std::max(1, length)), '\0');
    if (length > 0) {
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                            result.data(), length, nullptr, nullptr);
        result.pop_back();
    }
    return result;
}

struct Candidate {
    ComPtr<IDXGIAdapter1> adapter;
    DeviceInfo info;
};

std::vector<Candidate> discover_adapters() {
    ComPtr<IDXGIFactory4> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");

    char* force_warp = nullptr;
    std::size_t env_bytes = 0;
    check(_dupenv_s(&force_warp, &env_bytes, "FYZZXYZZY_D3D12_FORCE_WARP") == 0
              ? S_OK : E_FAIL, "_dupenv_s");
    const bool warp_only = force_warp && std::string(force_warp) == "1";
    std::free(force_warp);
    std::vector<Candidate> candidates;
    auto add = [&](ComPtr<IDXGIAdapter1> adapter) {
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                     __uuidof(ID3D12Device), nullptr)))
            return;
        DXGI_ADAPTER_DESC1 description{};
        check(adapter->GetDesc1(&description), "GetDesc1");
        DeviceInfo info;
        info.ordinal = static_cast<std::uint32_t>(candidates.size());
        info.name = adapter_name(description);
        info.vendor_id = description.VendorId;
        info.device_id = description.DeviceId;
        info.api_version = D3D_FEATURE_LEVEL_12_0;
        info.max_storage_buffer_range = static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max()) * 4ull;
        info.max_workgroup_invocations = 1024;
        info.max_workgroup_count[0] = 65535;
        info.max_workgroup_count[1] = 65535;
        info.max_workgroup_count[2] = 65535;
        info.max_workgroup_size[0] = 1024;
        info.max_workgroup_size[1] = 1024;
        info.max_workgroup_size[2] = 64;
        // Query the real adapter: a DXIL shader compiling under DXC
        // says nothing about WaveOps support on the selected device.
        ComPtr<ID3D12Device> probe;
        check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                IID_PPV_ARGS(&probe)), "D3D12 feature probe");
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts{};
        const bool waveOps =
            SUCCEEDED(probe->CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1, &opts, sizeof(opts))) &&
            opts.WaveOps;
        D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{};
        shaderModel.HighestShaderModel = D3D_SHADER_MODEL_6_6;
        if (FAILED(probe->CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))))
            shaderModel.HighestShaderModel = D3D_SHADER_MODEL_5_1;
        info.shader_model = static_cast<std::uint32_t>(
            shaderModel.HighestShaderModel);
        info.supports_wave_arithmetic =
            // The fast DXIL binary is compiled for cs_6_6, so SM6.0
            // WaveOps alone are insufficient to select this pipeline.
            waveOps && shaderModel.HighestShaderModel >= D3D_SHADER_MODEL_6_6;
        // Device admission is a setup-time decision. The native PhysX
        // overlap scan requires both Int64ShaderOps and WaveOps.
        info.supports_int64_shader = opts.Int64ShaderOps != FALSE;
        info.supports_int64_wave = info.supports_int64_shader &&
                                   info.supports_wave_arithmetic;
        info.subgroup_size = waveOps ? opts.WaveLaneCountMin : 0u;
        D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
        architecture.NodeIndex = 0;
        if (SUCCEEDED(probe->CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture)))) {
            info.unified_memory = architecture.UMA == TRUE;
            info.cache_coherent_unified_memory = architecture.CacheCoherentUMA == TRUE;
            // UMA alone does not guarantee that a GPU-writable, CPU-map
            // capable CUSTOM heap can actually be created. Probe the exact
            // UAV resource + memory combination, not just an adapter name.
            if (info.unified_memory) {
                const auto heap = probe->GetCustomHeapProperties(0, D3D12_HEAP_TYPE_READBACK);
                if (heap.Type == D3D12_HEAP_TYPE_CUSTOM &&
                    heap.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK &&
                    heap.MemoryPoolPreference == D3D12_MEMORY_POOL_L0) {
                    D3D12_RESOURCE_DESC desc{};
                    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                    desc.Width = 256;
                    desc.Height = 1;
                    desc.DepthOrArraySize = 1;
                    desc.MipLevels = 1;
                    desc.SampleDesc.Count = 1;
                    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                    ComPtr<ID3D12Resource> trial;
                    if (SUCCEEDED(probe->CreateCommittedResource(
                            &heap, D3D12_HEAP_FLAG_NONE, &desc,
                            D3D12_RESOURCE_STATE_COMMON, nullptr,
                            IID_PPV_ARGS(&trial)))) {
                        void* pointer = nullptr;
                        D3D12_RANGE read{0, 0};
                        if (SUCCEEDED(trial->Map(0, &read, &pointer)) && pointer) {
                            D3D12_RANGE written{0, 0};
                            trial->Unmap(0, &written);
                            info.supports_gpu_writable_host_visible = true;
                            info.has_device_local_host_visible_memory = true;
                        }
                    }
                }
            }
        }
        candidates.push_back({std::move(adapter), std::move(info)});
    };

    if (!warp_only) {
        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 description{};
            check(adapter->GetDesc1(&description), "GetDesc1");
            if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0)
                add(std::move(adapter));
        }
    }
    ComPtr<IDXGIAdapter1> warp;
    check(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "EnumWarpAdapter");
    add(std::move(warp));
    return candidates;
}

ComPtr<ID3D12Resource> allocate_resource(ID3D12Device* device, UINT64 bytes,
                                         D3D12_HEAP_TYPE heap_type,
                                         D3D12_RESOURCE_FLAGS flags,
                                         D3D12_RESOURCE_STATES initial_state) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = heap_type;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = std::max<UINT64>(1, bytes);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                          initial_state, nullptr,
                                          IID_PPV_ARGS(&resource)), "CreateCommittedResource");
    return resource;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

void validate_range(std::size_t capacity, std::size_t offset, std::size_t bytes) {
    if (offset > capacity || bytes > capacity - offset)
        throw std::out_of_range("D3D12 compute buffer range exceeds allocation");
}

} // namespace

std::vector<DeviceInfo> enumerate_devices(Backend backend) {
    if (backend != Backend::D3D12) throw std::invalid_argument("Expected D3D12 backend");
    auto devices = discover_adapters();
    std::vector<DeviceInfo> result;
    for (const auto& device : devices) result.push_back(device.info);
    return result;
}

struct Context::Impl {
    struct Buffer {
        ComPtr<ID3D12Resource> resource;
        std::size_t bytes = 0;
        bool host_visible = false;
        bool cpu_mapped = false;
        HostMapAccess map_access = HostMapAccess::ReadWrite;
    };
    struct Pipeline {
        ComPtr<ID3D12RootSignature> signature;
        ComPtr<ID3D12PipelineState> state;
        std::uint32_t bindings = 0;
        std::uint32_t parameters = 0;
        std::uint32_t srv_count = 0;
        std::uint32_t uav_count = 0;
        std::vector<StorageBufferViewKind> view_kinds;
        std::vector<std::uint32_t> descriptor_slots;
    };
    struct DescriptorView {
        BufferId buffer = 0;
        std::size_t offset = 0;
        std::size_t bytes = 0;
        ID3D12Resource* resource = nullptr;
    };
    struct Submission {
        // A descriptor heap belongs to its completed command submission.
        // These identities describe the live entries in that exact heap.
        std::vector<PipelineId> cached_pipelines;
        std::vector<DescriptorView> cached_views;
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> commands;
        ComPtr<ID3D12DescriptorHeap> heap;
        ComPtr<ID3D12QueryHeap> query_heap;
        ComPtr<ID3D12Resource> query_readback;
        // One-off HtoD batch staging belongs to the submission itself.
        // It is dropped on fence completion, not retained in command cache.
        ComPtr<ID3D12Resource> owned_upload;
        // Legacy D3D12 cannot set one subresource to both COPY_SOURCE
        // and COPY_DEST. Non-overlapping same-resource DtoD therefore
        // uses an all-device-local scratch resource, fence-owned by this
        // submission. Enhanced-barrier native self-copy remains an
        // optional future performance optimization.
        std::vector<ComPtr<ID3D12Resource>> owned_copy_scratch;
        std::uint64_t fence_value = 0;
    };

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12InfoQueue> debug_messages;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    // Context is not thread-safe; one persistent completion event can serve
    // every serial fence wait without creating Win32 objects per submission.
    HANDLE completion_event = nullptr;
    DeviceInfo info;
    std::unordered_map<BufferId, Buffer> buffers;
    std::unordered_map<PipelineId, Pipeline> pipelines;
    std::unordered_map<SubmissionId, Submission> submissions;
    struct RetiredBuffer {
        std::uint64_t release_after_fence = 0;
        Buffer buffer;
    };
    std::vector<RetiredBuffer> retired_buffers;
    // Completed command allocator/list and descriptor heap reuse. Never
    // recycle an in-flight allocation: D3D12 Reset requires GPU completion.
    std::vector<Submission> recycled;
    static constexpr std::size_t kMaxRecycled = 8;
    BufferId next_buffer = 1;
    PipelineId next_pipeline = 1;
    SubmissionId next_submission = 1;
    std::uint64_t next_fence = 1;

    explicit Impl(Backend backend, std::uint32_t ordinal) {
        if (backend != Backend::D3D12) throw std::invalid_argument("Expected D3D12 backend");
        auto adapters = discover_adapters();
        if (ordinal >= adapters.size())
            throw std::out_of_range("D3D12 adapter ordinal does not exist");
        info = adapters[ordinal].info;
        const char* debug_env = std::getenv("FYZZXYZZY_D3D12_DEBUG");
        if (debug_env && std::string(debug_env) == "1") {
            ComPtr<ID3D12Debug> debug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
                debug->EnableDebugLayer();
                ComPtr<ID3D12Debug1> gpu_debug;
                if (SUCCEEDED(debug.As(&gpu_debug)))
                    gpu_debug->SetEnableGPUBasedValidation(TRUE);
                std::cerr << "Enabled D3D12 GPU-based validation for native WARP test\n";
            } else {
                std::cerr << "D3D12 debug layer not available on this runner\n";
            }
        }
        check(D3D12CreateDevice(adapters[ordinal].adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                IID_PPV_ARGS(&device)), "D3D12CreateDevice");
        if (debug_env && std::string(debug_env) == "1")
            device.As(&debug_messages);
        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                  IID_PPV_ARGS(&fence)), "CreateFence");
        completion_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!completion_event)
            throw std::runtime_error("CreateEvent failed for D3D12 fence");
    }

    ~Impl() {
        if (queue && fence) {
            try { wait_all(); } catch (...) {}
        }
        if (completion_event) CloseHandle(completion_event);
    }

    Buffer& get_buffer(BufferId id) {
        auto it = buffers.find(id);
        if (it == buffers.end()) throw std::invalid_argument("Unknown D3D12 buffer");
        return it->second;
    }
    Pipeline& get_pipeline(PipelineId id) {
        auto it = pipelines.find(id);
        if (it == pipelines.end()) throw std::invalid_argument("Unknown D3D12 pipeline");
        return it->second;
    }

    Submission new_commands(
            const std::vector<PipelineId>* pipeline_sequence = nullptr,
            std::size_t descriptor_count = 0) {
        std::size_t match = recycled.size();
        if (pipeline_sequence && !pipeline_sequence->empty()) {
            for (std::size_t i = recycled.size(); i > 0; --i) {
                const auto& candidate = recycled[i - 1u];
                if (candidate.heap &&
                    candidate.heap->GetDesc().NumDescriptors >= descriptor_count &&
                    candidate.cached_pipelines == *pipeline_sequence) {
                    match = i - 1u;
                    break;
                }
            }
        }
        // Do not evict a completed source stage's still-valid descriptors
        // merely because another stage reached the fetch boundary later.
        if (match == recycled.size() &&
            (!pipeline_sequence || pipeline_sequence->empty() ||
             recycled.size() == kMaxRecycled))
            match = recycled.empty() ? recycled.size() : recycled.size() - 1u;
        if (match < recycled.size()) {
            Submission submission = std::move(recycled[match]);
            if (match + 1u != recycled.size())
                recycled[match] = std::move(recycled.back());
            recycled.pop_back();
            check(submission.allocator->Reset(), "ResetCommandAllocator");
            check(submission.commands->Reset(submission.allocator.Get(), nullptr),
                  "ResetCommandList");
            submission.fence_value = 0;
            return submission;
        }
        Submission submission;
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             IID_PPV_ARGS(&submission.allocator)),
              "CreateCommandAllocator");
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        submission.allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&submission.commands)),
              "CreateCommandList");
        return submission;
    }

    void recycle_completed(Submission&& submission) {
        submission.owned_upload.Reset(); // GPU completed; staging no longer referenced.
        submission.owned_copy_scratch.clear(); // GPU-local copies completed.
        // The caller must first wait for the submission fence. A cached heap
        // keeps its descriptors only until the next recording overwrites them.
        if (recycled.size() < kMaxRecycled)
            recycled.push_back(std::move(submission));
    }

    SubmissionId enqueue(Submission&& submission) {
        if (next_submission == std::numeric_limits<SubmissionId>::max() ||
            next_fence == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("D3D12 submission or fence ID range exhausted");
        check(submission.commands->Close(), "Close command list");
        const SubmissionId id = next_submission;
        const std::uint64_t signal = next_fence;
        submission.fence_value = signal;
        // Own the command allocator, list, descriptor heap and any source
        // upload/copy staging before the GPU can begin using them.
        // unordered_map insertion can allocate and throw; submitting first
        // would allow its failure to release resources still in flight.
        const auto [it, inserted] = submissions.emplace(id, std::move(submission));
        if (!inserted)
            throw std::logic_error("D3D12 submission ID already registered");
        ++next_submission;
        ++next_fence;
        ID3D12CommandList* lists[] = {it->second.commands.Get()};
        queue->ExecuteCommandLists(1, lists);
        // A queue-signal failure must not destroy the submitted resources:
        // the owning entry stays recorded until device teardown.
        check(queue->Signal(fence.Get(), signal), "Signal D3D12 fence");
        return id;
    }

    void print_debug_messages() noexcept {
        if (!debug_messages) return;
        try {
            const auto count = debug_messages->GetNumStoredMessages();
            const auto first = count > 80u ? count - 80u : 0u;
            for (std::uint64_t i = first; i < count; ++i) {
                SIZE_T size = 0;
                if (FAILED(debug_messages->GetMessage(i, nullptr, &size)) || !size)
                    continue;
                std::vector<std::uint8_t> storage(size);
                auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                if (SUCCEEDED(debug_messages->GetMessage(i, message, &size)) &&
                    (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR ||
                     message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ||
                     message->Severity == D3D12_MESSAGE_SEVERITY_WARNING))
                    std::cerr << "D3D12 validation ["
                              << static_cast<int>(message->ID) << "]: "
                              << message->pDescription << '\n';
            }
        } catch (...) {}
    }

    void wait_value(std::uint64_t value) {
        // ID3D12Fence returns UINT64_MAX when the device has been removed.
        // Treating that as a completed fence conceals the failing dispatch;
        // the next resource allocation then reports a misleading failure.
        const auto completed = fence->GetCompletedValue();
        if (completed == std::numeric_limits<std::uint64_t>::max()) {
            const auto reason = device->GetDeviceRemovedReason();
            print_debug_messages();
            throw std::runtime_error(
                "D3D12 device removed during GPU submission (HRESULT " +
                std::to_string(static_cast<unsigned long>(reason)) + ")");
        }
        if (completed >= value) return;
        check(fence->SetEventOnCompletion(value, completion_event),
              "SetEventOnCompletion");
        DWORD result = WaitForSingleObject(completion_event, INFINITE);
        if (result != WAIT_OBJECT_0)
            throw std::runtime_error("D3D12 fence wait failed");
        if (fence->GetCompletedValue() == std::numeric_limits<std::uint64_t>::max()) {
            const auto reason = device->GetDeviceRemovedReason();
            print_debug_messages();
            throw std::runtime_error(
                "D3D12 device removed during GPU submission (HRESULT " +
                std::to_string(static_cast<unsigned long>(reason)) + ")");
        }
    }

    void release_retired_through(std::uint64_t completed_fence) noexcept {
        auto it = retired_buffers.begin();
        while (it != retired_buffers.end()) {
            if (it->release_after_fence <= completed_fence) {
                it = retired_buffers.erase(it); // ComPtr releases GPU allocation
            } else {
                ++it;
            }
        }
    }

    void wait_submission(SubmissionId id) {
        auto it = submissions.find(id);
        if (it == submissions.end()) throw std::invalid_argument("Unknown D3D12 submission");
        const auto completed = it->second.fence_value;
        wait_value(completed);
        recycle_completed(std::move(it->second));
        submissions.erase(it);
        release_retired_through(completed);
    }

    void wait_all() {
        if (submissions.empty()) return;
        const auto completed = next_fence - 1;
        wait_value(completed);
        for (auto& entry : submissions)
            recycle_completed(std::move(entry.second));
        submissions.clear();
        release_retired_through(completed);
    }

    BufferId create_buffer(std::size_t bytes, MemoryPlacement placement) {
        if (!bytes) throw std::invalid_argument("Compute buffer size must be positive");
        Buffer buffer;
        if (placement == MemoryPlacement::HostVisible) {
            // The PhysX actor-pair report output is GPU-writable AND
            // CPU-readable; UPLOAD (GENERIC_READ) and READBACK (COPY_DEST)
            // are not legal substitutes for a UAV destination.
            if (!info.supports_gpu_writable_host_visible)
                throw UnsupportedMemoryPlacement(
                    "D3D12 adapter has no GPU-writable CPU-mapped UMA buffer; use staged reports");

            // PhysX mapped actor reports are GPU-written then CPU-READ.
            // The READBACK-derived CUSTOM heap selects WRITE_BACK/L0 even
            // on non-cache-coherent UMA, avoiding very slow CPU reads from
            // upload-oriented WRITE_COMBINE pages. The CUSTOM heap permits
            // GPU UAV and normal state transitions; a regular READBACK heap
            // does not. The admission probe exercises precisely this pairing.
            const auto heap = device->GetCustomHeapProperties(0, D3D12_HEAP_TYPE_READBACK);
            if (heap.Type != D3D12_HEAP_TYPE_CUSTOM ||
                heap.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_WRITE_BACK ||
                heap.MemoryPoolPreference != D3D12_MEMORY_POOL_L0)
                throw UnsupportedMemoryPlacement("D3D12 UMA did not expose a CPU-mappable L0 custom heap");
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = static_cast<UINT64>(bytes);
            desc.Height = 1;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.SampleDesc.Count = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            check(device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                nullptr, IID_PPV_ARGS(&buffer.resource)),
                "Create GPU-writable CPU-visible D3D12 UMA buffer");
            buffer.host_visible = true;
        } else {
            // Auto preserves the established D3D12 default, while explicit
            // DeviceLocal remains a GPU-preferred DEFAULT allocation.
            buffer.resource = allocate_resource(device.Get(), bytes, D3D12_HEAP_TYPE_DEFAULT,
                                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                                 D3D12_RESOURCE_STATE_COMMON);
        }
        buffer.bytes = bytes;
        const BufferId id = next_buffer;
        buffers.emplace(id, std::move(buffer));
        ++next_buffer;
        return id;
    }

    bool buffer_is_host_visible(BufferId id) { return get_buffer(id).host_visible; }

    void* map_buffer(BufferId id, HostMapAccess access) {
        auto& buffer = get_buffer(id);
        if (!buffer.host_visible)
            throw std::invalid_argument("D3D12 buffer is not CPU-mappable GPU storage");
        if (buffer.cpu_mapped)
            throw std::logic_error("D3D12 buffer already CPU mapped");
        void* pointer = nullptr;
        D3D12_RANGE readable{0, access == HostMapAccess::Write ? 0u : buffer.bytes};
        // Map does NOT synchronize. The source fence must have completed
        // before CPU reads any GPU-produced PhysX report bytes.
        check(buffer.resource->Map(0, &readable, &pointer), "Map D3D12 UMA buffer");
        buffer.cpu_mapped = true;
        buffer.map_access = access;
        return pointer;
    }

    void unmap_buffer(BufferId id) {
        auto& buffer = get_buffer(id);
        if (!buffer.cpu_mapped)
            throw std::logic_error("D3D12 buffer is not CPU mapped");
        // General read/write map. The GPU must not read data until the
        // caller has released the CPU map and submitted ordered work.
        D3D12_RANGE written{0, buffer.map_access == HostMapAccess::Read
                               ? 0u : buffer.bytes};
        buffer.resource->Unmap(0, &written);
        buffer.cpu_mapped = false;
    }

    void destroy_buffer(BufferId id) {
        if (get_buffer(id).cpu_mapped)
            throw std::logic_error("Cannot destroy mapped D3D12 GPU storage");
        wait_all(); // original blocking API remains unchanged
        buffers.erase(id);
    }

    void retire_buffer(BufferId id) {
        auto it = buffers.find(id);
        if (it == buffers.end())
            throw std::invalid_argument("Unknown D3D12 buffer");
        if (it->second.cpu_mapped)
            throw std::logic_error("Cannot retire mapped D3D12 GPU storage");
        if (!submissions.empty()) {
            std::uint64_t last_fence = 0;
            for (const auto& pending : submissions)
                last_fence = std::max(last_fence, pending.second.fence_value);
            // Copy the owning ComPtr into the retirement record *before*
            // erasing the ID. On host allocation failure the original ID
            // still owns the live resource.
            retired_buffers.push_back({last_fence, it->second});
        }
        // The handle becomes unusable immediately. Its backing resource
        // lives in retired_buffers until the last dependent queue fence.
        buffers.erase(it);
    }

    void upload(BufferId id, std::size_t offset, const void* data, std::size_t bytes) {
        Buffer& buffer = get_buffer(id);
        validate_range(buffer.bytes, offset, bytes);
        if (!bytes) return;
        if (!data || offset % 4 || bytes % 4)
            throw std::invalid_argument("D3D12 upload requires aligned range and data");
        auto staging = allocate_resource(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD,
                                         D3D12_RESOURCE_FLAG_NONE,
                                         D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        D3D12_RANGE read{0, 0};
        check(staging->Map(0, &read, &mapped), "Map upload");
        std::memcpy(mapped, data, bytes);
        D3D12_RANGE written{0, bytes};
        staging->Unmap(0, &written);
        Submission cmd = new_commands();
        transition(cmd.commands.Get(), buffer.resource.Get(),
                   D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd.commands->CopyBufferRegion(buffer.resource.Get(), offset,
                                       staging.Get(), 0, bytes);
        transition(cmd.commands.Get(), buffer.resource.Get(),
                   D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        wait_submission(enqueue(std::move(cmd))); // staging remains alive to completion
    }

    SubmissionId submit_upload(BufferId id, std::size_t offset,
                               const void* data, std::size_t bytes) {
        Buffer& destination = get_buffer(id);
        validate_range(destination.bytes, offset, bytes);
        if (!bytes || !data || (offset % 4) || (bytes % 4))
            throw std::invalid_argument("Async D3D12 upload requires nonempty aligned data");
        // The staging allocation must be owned before submitting the queue
        // work; reserve its retirement entry before enqueue can succeed.
        retired_buffers.reserve(retired_buffers.size() + 1);
        auto staging = allocate_resource(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD,
                                         D3D12_RESOURCE_FLAG_NONE,
                                         D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        const D3D12_RANGE read{0, 0};
        check(staging->Map(0, &read, &mapped), "Map async upload");
        std::memcpy(mapped, data, bytes);
        const D3D12_RANGE written{0, bytes};
        staging->Unmap(0, &written);

        Submission cmd = new_commands();
        transition(cmd.commands.Get(), destination.resource.Get(),
                   D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd.commands->CopyBufferRegion(destination.resource.Get(), offset,
                                       staging.Get(), 0, bytes);
        transition(cmd.commands.Get(), destination.resource.Get(),
                   D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        const SubmissionId submission = enqueue(std::move(cmd));
        const auto fence_value = submissions.at(submission).fence_value;
        // No allocation after enqueue; ComPtr move retains upload heap
        // until wait_submission or a later completed queue-fence releases it.
        retired_buffers.push_back({fence_value, {std::move(staging), bytes}});
        return submission;
    }

    SubmissionId submit_uploads(const std::vector<UploadSpan>& uploads) {
        if (uploads.empty())
            throw std::invalid_argument("Async D3D12 upload batch cannot be empty");
        std::size_t total = 0;
        std::unordered_set<BufferId> destinations;
        destinations.reserve(uploads.size());
        for (const auto& upload : uploads) {
            const auto& target = get_buffer(upload.destination);
            validate_range(target.bytes, upload.destination_offset, upload.bytes);
            if (!upload.data || !upload.bytes ||
                (upload.destination_offset % 4) || (upload.bytes % 4) ||
                upload.bytes > std::numeric_limits<std::size_t>::max() - total)
                throw std::invalid_argument("Invalid D3D12 upload span or staging capacity");
            total += upload.bytes;
            destinations.insert(upload.destination);
        }

        auto staging = allocate_resource(device.Get(), total, D3D12_HEAP_TYPE_UPLOAD,
                                         D3D12_RESOURCE_FLAG_NONE,
                                         D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        const D3D12_RANGE read{0, 0};
        check(staging->Map(0, &read, &mapped), "Map batched upload");
        std::size_t position = 0;
        for (const auto& upload : uploads) {
            std::memcpy(static_cast<unsigned char*>(mapped) + position,
                        upload.data, upload.bytes);
            position += upload.bytes;
        }
        const D3D12_RANGE written{0, total};
        staging->Unmap(0, &written);

        Submission submission = new_commands();
        // A target enters COPY_DEST once for the whole batch; repeated
        // CopyBufferRegion writes are ordered under the legacy D3D12
        // copy contract. Restore COMMON once at the end.
        for (const auto id : destinations) {
            transition(submission.commands.Get(), get_buffer(id).resource.Get(),
                       D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        }
        position = 0;
        for (const auto& upload : uploads) {
            submission.commands->CopyBufferRegion(
                get_buffer(upload.destination).resource.Get(),
                upload.destination_offset, staging.Get(), position, upload.bytes);
            position += upload.bytes;
        }
        for (const auto id : destinations) {
            transition(submission.commands.Get(), get_buffer(id).resource.Get(),
                       D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        }
        // Ownership is attached BEFORE queue submission. A later allocation
        // failure cannot free the memory while the GPU still reads it.
        submission.owned_upload = std::move(staging);
        return enqueue(std::move(submission));
    }

    void download(BufferId id, std::size_t offset, void* data, std::size_t bytes) {
        Buffer& buffer = get_buffer(id);
        validate_range(buffer.bytes, offset, bytes);
        if (!bytes) return;
        if (!data || offset % 4 || bytes % 4)
            throw std::invalid_argument("D3D12 download requires aligned range and data");
        auto staging = allocate_resource(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK,
                                         D3D12_RESOURCE_FLAG_NONE,
                                         D3D12_RESOURCE_STATE_COPY_DEST);
        Submission cmd = new_commands();
        transition(cmd.commands.Get(), buffer.resource.Get(),
                   D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd.commands->CopyBufferRegion(staging.Get(), 0,
                                       buffer.resource.Get(), offset, bytes);
        transition(cmd.commands.Get(), buffer.resource.Get(),
                   D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        wait_submission(enqueue(std::move(cmd)));
        void* mapped = nullptr;
        D3D12_RANGE read{0, bytes};
        check(staging->Map(0, &read, &mapped), "Map readback");
        std::memcpy(data, mapped, bytes);
        D3D12_RANGE written{0, 0};
        staging->Unmap(0, &written);
    }

    PipelineId create_pipeline(const ComputePipelineDesc& desc) {
        if (desc.shader.format != ShaderFormat::Dxil || desc.shader.bytes.empty())
            throw std::invalid_argument("Native D3D12 requires compiled DXIL");
        if (desc.shader.entry_point != "main" ||
            !desc.storage_buffer_binding_count || !desc.push_constant_bytes ||
            desc.push_constant_bytes % 4u || desc.push_constant_bytes / 4u > 62u)
            throw std::invalid_argument("Invalid D3D12 compute pipeline declaration");

        std::vector<StorageBufferViewKind> kinds = desc.storage_buffer_views;
        if (kinds.empty()) {
            // Existing stage ABIs during migration to explicit declarations.
            if (desc.storage_buffer_binding_count == 1u)
                kinds = {StorageBufferViewKind::RawReadWrite};
            else if (desc.storage_buffer_binding_count == 4u)
                kinds = {StorageBufferViewKind::Structured32Read,
                         StorageBufferViewKind::Structured32Read,
                         StorageBufferViewKind::Structured32Read,
                         StorageBufferViewKind::Structured32ReadWrite};
            else
                throw std::invalid_argument(
                    "D3D12 multi-resource pipeline requires explicit storage views");
        }
        if (kinds.size() != desc.storage_buffer_binding_count)
            throw std::invalid_argument("D3D12 storage view declaration count mismatch");

        Pipeline pipeline;
        pipeline.view_kinds = kinds;
        pipeline.bindings = desc.storage_buffer_binding_count;
        pipeline.parameters = desc.push_constant_bytes;
        pipeline.descriptor_slots.resize(kinds.size());
        for (const auto kind : kinds) {
            switch (kind) {
                case StorageBufferViewKind::RawRead:
                case StorageBufferViewKind::Structured32Read:
                    ++pipeline.srv_count;
                    break;
                case StorageBufferViewKind::RawReadWrite:
                case StorageBufferViewKind::Structured32ReadWrite:
                    ++pipeline.uav_count;
                    break;
                default:
                    throw std::invalid_argument("Unknown D3D12 storage buffer view kind");
            }
        }

        std::uint32_t read_index = 0u, write_index = 0u;
        for (std::size_t slot = 0; slot < kinds.size(); ++slot) {
            const bool read_only = kinds[slot] == StorageBufferViewKind::RawRead ||
                                   kinds[slot] == StorageBufferViewKind::Structured32Read;
            pipeline.descriptor_slots[slot] = read_only
                ? read_index++ : pipeline.srv_count + write_index++;
        }

        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        D3D12_ROOT_PARAMETER roots[3]{};
        UINT root_count = 0u;
        if (pipeline.srv_count) {
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = pipeline.srv_count;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].RegisterSpace = 0;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;
            roots[root_count].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            roots[root_count].DescriptorTable.NumDescriptorRanges = 1;
            roots[root_count].DescriptorTable.pDescriptorRanges = &ranges[0];
            roots[root_count].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            ++root_count;
        }
        if (pipeline.uav_count) {
            ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[1].NumDescriptors = pipeline.uav_count;
            ranges[1].BaseShaderRegister = 0;
            ranges[1].RegisterSpace = 0;
            ranges[1].OffsetInDescriptorsFromTableStart = 0;
            roots[root_count].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            roots[root_count].DescriptorTable.NumDescriptorRanges = 1;
            roots[root_count].DescriptorTable.pDescriptorRanges = &ranges[1];
            roots[root_count].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            ++root_count;
        }
        roots[root_count].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        roots[root_count].Constants.ShaderRegister = 0;
        roots[root_count].Constants.RegisterSpace = 0;
        roots[root_count].Constants.Num32BitValues = desc.push_constant_bytes / 4u;
        roots[root_count].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        ++root_count;
        D3D12_ROOT_SIGNATURE_DESC sig{};
        sig.NumParameters = root_count;
        sig.pParameters = roots;
        sig.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> errors;
        HRESULT result = D3D12SerializeRootSignature(&sig, D3D_ROOT_SIGNATURE_VERSION_1,
                                                      &blob, &errors);
        if (FAILED(result) && errors)
            throw std::runtime_error("D3D12 root signature: " +
                                     std::string(static_cast<const char*>(errors->GetBufferPointer()),
                                                 errors->GetBufferSize()));
        check(result, "D3D12SerializeRootSignature");
        check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                          IID_PPV_ARGS(&pipeline.signature)), "CreateRootSignature");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = pipeline.signature.Get();
        pso.CS = {desc.shader.bytes.data(), desc.shader.bytes.size()};
        check(device->CreateComputePipelineState(&pso,
                                                 IID_PPV_ARGS(&pipeline.state)),
              "CreateComputePipelineState");
        const PipelineId id = next_pipeline++;
        pipelines.emplace(id, std::move(pipeline));
        return id;
    }

    void destroy_pipeline(PipelineId id) {
        get_pipeline(id);
        // A removed D3D12 device cannot complete outstanding fences.
        // Pipeline COM resources can still be released; do not let a
        // cleanup wait throw during unwinding and hide the dispatch fault.
        if (SUCCEEDED(device->GetDeviceRemovedReason()))
            wait_all();
        pipelines.erase(id);
    }

    void validate_dispatch(const DispatchCommand& cmd) {
        const Pipeline& pipe = get_pipeline(cmd.pipeline);
        if (!cmd.groups.x || !cmd.groups.y || !cmd.groups.z ||
            cmd.groups.x > info.max_workgroup_count[0] ||
            cmd.groups.y > info.max_workgroup_count[1] ||
            cmd.groups.z > info.max_workgroup_count[2] ||
            cmd.buffers.size() != pipe.bindings ||
            cmd.push_constants.size() != pipe.parameters)
            throw std::invalid_argument("Invalid D3D12 dispatch shape or bindings");
        for (std::size_t i = 0; i < cmd.buffers.size(); ++i) {
            const auto& binding = cmd.buffers[i];
            if (binding.slot != i || !binding.bytes ||
                binding.offset % 4 || binding.bytes % 4 ||
                binding.bytes / 4 > std::numeric_limits<UINT>::max())
                throw std::invalid_argument("Invalid D3D12 structured buffer view");
            const auto kind = pipe.view_kinds[i];
            // D3D12 requires RAW buffer UAV/SRV FirstElement to start
            // on a 16-byte boundary. Reject an invalid descriptor before
            // the debug layer removes the device on command execution.
            if ((kind == StorageBufferViewKind::RawRead ||
                 kind == StorageBufferViewKind::RawReadWrite) &&
                binding.offset % 16u != 0u)
                throw std::invalid_argument(
                    "D3D12 RAW buffer view base must be 16-byte aligned");
            const bool srv_slot = kind == StorageBufferViewKind::RawRead ||
                                  kind == StorageBufferViewKind::Structured32Read;
            if (binding.access != BufferAccess::Unspecified &&
                binding.access != BufferAccess::Read &&
                binding.access != BufferAccess::Write &&
                binding.access != BufferAccess::ReadWrite)
                throw std::invalid_argument("Invalid D3D12 compute buffer access intent");
            if (srv_slot && binding.access != BufferAccess::Unspecified &&
                binding.access != BufferAccess::Read)
                throw std::invalid_argument("D3D12 SRV cannot be declared writable");
            const auto& buf = get_buffer(binding.buffer);
            validate_range(buf.bytes, binding.offset, binding.bytes);
        }
        // The entire D3D12 buffer is one stateful subresource. A dispatch
        // cannot simultaneously bind it as an SRV and UAV, even when the
        // views name disjoint byte ranges. Several UAV views may alias.
        for (std::size_t a = 0; a < cmd.buffers.size(); ++a) {
            const auto a_kind = pipe.view_kinds[a];
            const bool a_read = a_kind == StorageBufferViewKind::RawRead ||
                                a_kind == StorageBufferViewKind::Structured32Read;
            for (std::size_t b = a + 1u; b < cmd.buffers.size(); ++b) {
                const auto b_kind = pipe.view_kinds[b];
                const bool b_read = b_kind == StorageBufferViewKind::RawRead ||
                                    b_kind == StorageBufferViewKind::Structured32Read;
                if (a_read != b_read &&
                    cmd.buffers[a].buffer == cmd.buffers[b].buffer)
                    throw std::invalid_argument(
                        "D3D12 cannot bind one resource as SRV and UAV simultaneously");
            }
        }
    }

    SubmissionId submit_batch(const std::vector<BatchCommand>& commands,
                              bool profile = false) {
        if (commands.empty()) throw std::invalid_argument("Empty D3D12 command batch");
        std::size_t descriptor_count = 0;
        std::vector<PipelineId> pipeline_sequence;
        for (const BatchCommand& op : commands) {
            if (const auto* copy = std::get_if<CopyCommand>(&op)) {
                const auto& source = get_buffer(copy->source);
                const auto& dest = get_buffer(copy->destination);
                validate_range(source.bytes, copy->source_offset, copy->bytes);
                validate_range(dest.bytes, copy->destination_offset, copy->bytes);
                const bool overlap = copy->source == copy->destination &&
                    copy->source_offset < copy->destination_offset + copy->bytes &&
                    copy->destination_offset < copy->source_offset + copy->bytes;
                if (!copy->bytes || overlap ||
                    copy->source_offset % 4 || copy->destination_offset % 4 ||
                    copy->bytes % 4)
                    throw std::invalid_argument("Invalid D3D12 buffer copy");
            } else {
                const auto& dispatch = std::get<DispatchCommand>(op);
                validate_dispatch(dispatch);
                descriptor_count += get_pipeline(dispatch.pipeline).bindings;
                pipeline_sequence.push_back(dispatch.pipeline);
            }
        }
        if (descriptor_count > std::numeric_limits<UINT>::max())
            throw std::out_of_range("D3D12 descriptor heap capacity exceeded");
        Submission submission = new_commands(&pipeline_sequence, descriptor_count);
        if (profile) {
            D3D12_QUERY_HEAP_DESC desc{};
            desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
            desc.Count = 2;
            if (!submission.query_heap)
                check(device->CreateQueryHeap(&desc, IID_PPV_ARGS(&submission.query_heap)),
                      "CreateQueryHeap(timestamp)");
            if (!submission.query_readback)
                submission.query_readback = allocate_resource(
                    device.Get(), 2 * sizeof(std::uint64_t),
                    D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST);
            submission.commands->EndQuery(submission.query_heap.Get(),
                                           D3D12_QUERY_TYPE_TIMESTAMP, 0);
        }
        UINT increment = device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        if (descriptor_count) {
            const auto available = submission.heap
                ? submission.heap->GetDesc().NumDescriptors : 0u;
            if (available < descriptor_count) {
                // Geometric capacity growth avoids reallocation when scene
                // work fluctuates around the previous dispatch count.
                // 1,000,000 descriptors are guaranteed for all resource
                // binding tiers. Tier 3 may accept more, but must not fail
                // due solely to an overambitious geometric growth request.
                // Above that threshold, request only the actual batch size.
                constexpr std::size_t kGuaranteedHeapSize = 1000000;
                const std::size_t growth = std::min<std::size_t>(
                    kGuaranteedHeapSize, static_cast<std::size_t>(available) * 2);
                const std::size_t desired = std::max(descriptor_count, growth);
                D3D12_DESCRIPTOR_HEAP_DESC heap{};
                heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
                heap.NumDescriptors = static_cast<UINT>(desired);
                heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
                ComPtr<ID3D12DescriptorHeap> replacement;
                check(device->CreateDescriptorHeap(&heap,
                                                   IID_PPV_ARGS(&replacement)),
                      "CreateDescriptorHeap");
                submission.heap = std::move(replacement);
                submission.cached_views.clear();
                submission.cached_pipelines.clear();
            }
            if (submission.cached_pipelines != pipeline_sequence ||
                submission.cached_views.size() != descriptor_count) {
                submission.cached_pipelines = pipeline_sequence;
                submission.cached_views.assign(descriptor_count, {});
            }
            ID3D12DescriptorHeap* heaps[] = {submission.heap.Get()};
            submission.commands->SetDescriptorHeaps(1, heaps);
        }

        // Track states for the entire batch instead of returning each buffer
        // to COMMON after every operation. Keep real RAW/WAW ordering: a
        // transition orders different uses and a UAV barrier orders
        // consecutive dependent UAV writes. Legacy copy operations have
        // implicit ordering. End at COMMON for the next submission.
        struct Access {
            D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
            bool pending_write = false;
        };
        std::unordered_map<BufferId, Access> resource_access;
        resource_access.reserve(std::min<std::size_t>(commands.size() * 2, 512));
        auto require_state = [&](BufferId id, D3D12_RESOURCE_STATES target) {
            Access& access = resource_access[id];
            auto* resource = get_buffer(id).resource.Get();
            if (access.state != target) {
                transition(submission.commands.Get(), resource, access.state, target);
                access.state = target;
                access.pending_write = false;
            } else if (access.pending_write) {
                if (target == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                    D3D12_RESOURCE_BARRIER barrier{};
                    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                    barrier.UAV.pResource = resource;
                    submission.commands->ResourceBarrier(1, &barrier);
                }
                // Legacy D3D12 CopyBufferRegion commands are synchronous
                // with an implicit copy/resolve flush. Successive COPY_DEST
                // writes therefore need no fabricated COMMON round-trip
                // (DirectX Enhanced Barriers specification, legacy section).
                access.pending_write = false;
            }
        };
        UINT descriptor_index = 0;
        for (const BatchCommand& op : commands) {
            if (const auto* copy = std::get_if<CopyCommand>(&op)) {
                auto& src = get_buffer(copy->source);
                auto& dst = get_buffer(copy->destination);
                if (copy->source == copy->destination) {
                    // Under legacy resource-state barriers a buffer cannot
                    // have COPY_SOURCE and COPY_DEST simultaneously.
                    // Stage through a device-LOCAL scratch copy, never a
                    // host upload/readback or a new queue submission.
                    // Source and destination ranges were checked disjoint.
                    auto scratch = allocate_resource(device.Get(), copy->bytes,
                        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATE_COPY_DEST);
                    // Hold GPU scratch through the submission fence before
                    // executing either copy, including exceptions during
                    // command recording.
                    submission.owned_copy_scratch.push_back(scratch);
                    require_state(copy->source, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    submission.commands->CopyBufferRegion(scratch.Get(), 0u,
                        src.resource.Get(), copy->source_offset, copy->bytes);
                    transition(submission.commands.Get(), scratch.Get(),
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COPY_SOURCE);
                    require_state(copy->destination, D3D12_RESOURCE_STATE_COPY_DEST);
                    submission.commands->CopyBufferRegion(dst.resource.Get(),
                        copy->destination_offset, scratch.Get(), 0u,
                        copy->bytes);
                } else {
                    require_state(copy->source, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    require_state(copy->destination, D3D12_RESOURCE_STATE_COPY_DEST);
                    submission.commands->CopyBufferRegion(dst.resource.Get(),
                        copy->destination_offset, src.resource.Get(),
                        copy->source_offset, copy->bytes);
                }
                resource_access[copy->destination].pending_write = true;
                continue;
            }
            const auto& cmd = std::get<DispatchCommand>(op);
            Pipeline& pipeline = get_pipeline(cmd.pipeline);
            for (std::size_t slot = 0; slot < cmd.buffers.size(); ++slot) {
                const auto& binding = cmd.buffers[slot];
                const auto kind = pipeline.view_kinds[slot];
                const bool read_only = kind == StorageBufferViewKind::RawRead ||
                                       kind == StorageBufferViewKind::Structured32Read;
                const auto desired = read_only
                    ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                    : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                require_state(binding.buffer, desired);
            }

            const auto cpu_start = submission.heap->GetCPUDescriptorHandleForHeapStart();
            const auto gpu_start = submission.heap->GetGPUDescriptorHandleForHeapStart();
            for (UINT slot = 0; slot < pipeline.bindings; ++slot) {
                const auto& binding = cmd.buffers[slot];
                auto& buffer = get_buffer(binding.buffer);
                const auto kind = pipeline.view_kinds[slot];
                const bool read_only = kind == StorageBufferViewKind::RawRead ||
                                       kind == StorageBufferViewKind::Structured32Read;
                const bool raw = kind == StorageBufferViewKind::RawRead ||
                                 kind == StorageBufferViewKind::RawReadWrite;
                const auto physical_slot = descriptor_index +
                    pipeline.descriptor_slots[slot];
                DescriptorView& cached = submission.cached_views[physical_slot];
                const DescriptorView next{
                    binding.buffer, binding.offset, binding.bytes,
                    buffer.resource.Get()};
                if (cached.buffer == next.buffer &&
                    cached.offset == next.offset &&
                    cached.bytes == next.bytes &&
                    cached.resource == next.resource)
                    continue;
                D3D12_CPU_DESCRIPTOR_HANDLE cpu{
                    cpu_start.ptr + physical_slot * increment
                };
                if (!read_only) {
                    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
                    view.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
                    view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                    view.Buffer.FirstElement = binding.offset / 4;
                    view.Buffer.NumElements = static_cast<UINT>(binding.bytes / 4);
                    view.Buffer.StructureByteStride = raw ? 0u : 4u;
                    view.Buffer.Flags = raw ? D3D12_BUFFER_UAV_FLAG_RAW :
                                               D3D12_BUFFER_UAV_FLAG_NONE;
                    device->CreateUnorderedAccessView(buffer.resource.Get(), nullptr, &view, cpu);
                } else {
                    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
                    view.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
                    view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    view.Buffer.FirstElement = binding.offset / 4;
                    view.Buffer.NumElements = static_cast<UINT>(binding.bytes / 4);
                    view.Buffer.StructureByteStride = raw ? 0u : 4u;
                    view.Buffer.Flags = raw ? D3D12_BUFFER_SRV_FLAG_RAW :
                                              D3D12_BUFFER_SRV_FLAG_NONE;
                    device->CreateShaderResourceView(buffer.resource.Get(), &view, cpu);
                }
                cached = next;
            }
            submission.commands->SetComputeRootSignature(pipeline.signature.Get());
            submission.commands->SetPipelineState(pipeline.state.Get());
            UINT table_root = 0u;
            if (pipeline.srv_count) {
                submission.commands->SetComputeRootDescriptorTable(table_root++, {
                    gpu_start.ptr + descriptor_index * increment
                });
            }
            if (pipeline.uav_count) {
                submission.commands->SetComputeRootDescriptorTable(table_root++, {
                    gpu_start.ptr + (descriptor_index + pipeline.srv_count) * increment
                });
            }
            submission.commands->SetComputeRoot32BitConstants(
                table_root, pipeline.parameters / 4u, cmd.push_constants.data(), 0);
            submission.commands->Dispatch(cmd.groups.x, cmd.groups.y, cmd.groups.z);
            for (std::size_t slot = 0; slot < cmd.buffers.size(); ++slot) {
                const auto kind = pipeline.view_kinds[slot];
                const bool writable = kind == StorageBufferViewKind::RawReadWrite ||
                                      kind == StorageBufferViewKind::Structured32ReadWrite;
                if (writable && cmd.buffers[slot].access != BufferAccess::Read)
                    resource_access[cmd.buffers[slot].buffer].pending_write = true;
            }
            descriptor_index += pipeline.bindings;
        }
        for (const auto& entry : resource_access) {
            transition(submission.commands.Get(), get_buffer(entry.first).resource.Get(),
                       entry.second.state, D3D12_RESOURCE_STATE_COMMON);
        }
        if (profile) {
            submission.commands->EndQuery(submission.query_heap.Get(),
                                           D3D12_QUERY_TYPE_TIMESTAMP, 1);
            submission.commands->ResolveQueryData(
                submission.query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                0, 2, submission.query_readback.Get(), 0);
        }
        return enqueue(std::move(submission));
    }

    std::optional<double> profile_batch_gpu_nanoseconds(
        const std::vector<BatchCommand>& operations) {
        std::uint64_t frequency = 0;
        if (FAILED(queue->GetTimestampFrequency(&frequency)) || !frequency)
            return std::nullopt;
        const auto id = submit_batch(operations, true);
        auto it = submissions.find(id);
        wait_value(it->second.fence_value);
        void* mapped = nullptr;
        const D3D12_RANGE read{0, 2 * sizeof(std::uint64_t)};
        check(it->second.query_readback->Map(0, &read, &mapped),
              "Map timestamp readback");
        std::uint64_t ticks[2]{};
        std::memcpy(ticks, mapped, sizeof(ticks));
        const D3D12_RANGE written{0, 0};
        it->second.query_readback->Unmap(0, &written);
        const auto completed = it->second.fence_value;
        recycle_completed(std::move(it->second));
        submissions.erase(it);
        release_retired_through(completed);
        if (ticks[1] < ticks[0])
            throw std::runtime_error("D3D12 GPU timestamp counter regressed");
        return static_cast<double>(ticks[1] - ticks[0]) *
               (1000000000.0 / static_cast<double>(frequency));
    }
};

Context::Context(Backend backend, std::uint32_t ordinal)
    : impl_(std::make_unique<Impl>(backend, ordinal)) {}
Context::~Context() = default;
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;
const DeviceInfo& Context::device_info() const { return impl_->info; }
BufferId Context::create_buffer(std::size_t bytes, MemoryPlacement placement) {
    return impl_->create_buffer(bytes, placement);
}
void Context::destroy_buffer(BufferId id) { impl_->destroy_buffer(id); }
void Context::retire_buffer(BufferId id) { impl_->retire_buffer(id); }
bool Context::buffer_is_host_visible(BufferId id) const {
    return impl_->get_buffer(id).host_visible;
}
void* Context::map_buffer(BufferId id, HostMapAccess access) {
    return impl_->map_buffer(id, access);
}
void Context::unmap_buffer(BufferId id) { impl_->unmap_buffer(id); }
void Context::upload(BufferId id, std::size_t offset, const void* data, std::size_t bytes) {
    impl_->upload(id, offset, data, bytes);
}

SubmissionId Context::submit_upload_buffer(BufferId id, std::size_t offset,
                                           const void* data, std::size_t bytes) {
    return impl_->submit_upload(id, offset, data, bytes);
}

SubmissionId Context::submit_upload_batch(const std::vector<UploadSpan>& uploads) {
    return impl_->submit_uploads(uploads);
}
void Context::download(BufferId id, std::size_t offset, void* data, std::size_t bytes) const {
    impl_->download(id, offset, data, bytes);
}
void Context::copy_buffer(BufferId source, std::size_t so, BufferId dest,
                          std::size_t dst, std::size_t count) {
    if (!count) return;
    impl_->wait_submission(impl_->submit_batch({CopyCommand{source, so, dest, dst, count}}));
}
SubmissionId Context::submit_copy_buffer(BufferId source, std::size_t so, BufferId dest,
                                         std::size_t dst, std::size_t count) {
    return impl_->submit_batch({CopyCommand{source, so, dest, dst, count}});
}
PipelineId Context::create_compute_pipeline(const ComputePipelineDesc& desc) {
    return impl_->create_pipeline(desc);
}
void Context::destroy_compute_pipeline(PipelineId id) { impl_->destroy_pipeline(id); }
void Context::dispatch(PipelineId pipeline, DispatchSize groups,
                       const std::vector<BufferBinding>& buffers,
                       const void* constants, std::size_t count) {
    const auto id = submit_dispatch(pipeline, groups, buffers, constants, count);
    impl_->wait_submission(id);
}
SubmissionId Context::submit_dispatch(PipelineId pipeline, DispatchSize groups,
                                      const std::vector<BufferBinding>& buffers,
                                      const void* constants, std::size_t count) {
    if (count && !constants) throw std::invalid_argument("Missing D3D12 dispatch constants");
    DispatchCommand cmd;
    cmd.pipeline = pipeline;
    cmd.groups = groups;
    cmd.buffers = buffers;
    cmd.push_constants.resize(count);
    if (count) std::memcpy(cmd.push_constants.data(), constants, count);
    return impl_->submit_batch({std::move(cmd)});
}
bool Context::submission_complete(SubmissionId id) const {
    auto it = impl_->submissions.find(id);
    if (it == impl_->submissions.end())
        throw std::invalid_argument("Unknown D3D12 submission");
    return impl_->fence->GetCompletedValue() >= it->second.fence_value;
}
void Context::wait_submission(SubmissionId id) { impl_->wait_submission(id); }
SubmissionId Context::submit_batch(const std::vector<BatchCommand>& commands) {
    return impl_->submit_batch(commands);
}

std::optional<double> Context::profile_batch_gpu_nanoseconds(
    const std::vector<BatchCommand>& commands) {
    return impl_->profile_batch_gpu_nanoseconds(commands);
}
void Context::execute_batch(const std::vector<BatchCommand>& commands) {
    impl_->wait_submission(impl_->submit_batch(commands));
}
void Context::wait_all() { impl_->wait_all(); }

} // namespace fyzz::compute
