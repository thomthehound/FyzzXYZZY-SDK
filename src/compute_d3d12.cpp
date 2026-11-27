// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// D3D12 implementation of the common native compute contract.
// Pipeline-declared storage view types determine SRV/UAV registers and RAW
// versus structured descriptors. The PhysX host selects resources at launch
// setup; this layer never interprets CUDA virtual addresses.

#include "fyzzxyzzy/compute.h"
#include "compute_readback_cache.h"
#include "compute_range_hazards.h"
#if defined(FYZZXYZZY_PROFILE_GPU)
#include "hardware_profile.h"
#endif

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
#include <exception>
#include <functional>
#if defined(FYZZXYZZY_PROFILE_GPU)
#include <cstdio>
#endif
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

// Read single-character build/test switches without CRT environment copies.
bool environment_equals(const char* name, char expected) {
    char value[2]{};
    return GetEnvironmentVariableA(name,value,sizeof(value))==1u &&
           value[0]==expected;
}

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
        info.software_adapter =
            (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
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
        ComPtr<ID3D12Device10> probe;
        if(FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                   IID_PPV_ARGS(&probe))))return;
        D3D12_FEATURE_DATA_D3D12_OPTIONS12 enhanced{};
        if(FAILED(probe->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12,
                &enhanced,sizeof(enhanced))) || !enhanced.EnhancedBarriersSupported)
            return;
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
        D3D12_FEATURE_DATA_D3D12_OPTIONS floatOptions{};
        info.supports_float64_shader =
            SUCCEEDED(probe->CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                &floatOptions,sizeof(floatOptions))) &&
            floatOptions.DoublePrecisionFloatShaderOps;
        info.supports_int64_shader = opts.Int64ShaderOps != FALSE;
        info.supports_int64_wave = info.supports_int64_shader &&
                                   info.supports_wave_arithmetic;
        info.subgroup_size = waveOps ? opts.WaveLaneCountMin : 0u;
        info.min_subgroup_size = info.subgroup_size;
        info.max_subgroup_size = waveOps ? opts.WaveLaneCountMax : 0u;
        info.supports_required_subgroup_size =
            info.supports_wave_arithmetic && info.min_subgroup_size != 0u &&
            info.max_subgroup_size >= info.min_subgroup_size;
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
                    D3D12_RESOURCE_DESC1 desc{};
                    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                    desc.Width = 256;
                    desc.Height = 1;
                    desc.DepthOrArraySize = 1;
                    desc.MipLevels = 1;
                    desc.SampleDesc.Count = 1;
                    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                    ComPtr<ID3D12Resource> trial;
                    if (SUCCEEDED(probe->CreateCommittedResource3(
                            &heap,D3D12_HEAP_FLAG_NONE,&desc,
                            D3D12_BARRIER_LAYOUT_UNDEFINED,nullptr,nullptr,
                            0u,nullptr,IID_PPV_ARGS(&trial)))) {
                        void* pointer = nullptr;
                        D3D12_RANGE read{0, 0};
                        if (SUCCEEDED(trial->Map(0, &read, &pointer)) && pointer) {
                            D3D12_RANGE written{0, 0};
                            trial->Unmap(0, &written);
                            info.supports_gpu_writable_host_visible = true;
                            info.has_device_local_host_visible_memory = true;
                            info.supports_shared_frame_input = true;
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

ComPtr<ID3D12Resource> allocate_resource(ID3D12Device10* device, UINT64 bytes,
                                         D3D12_HEAP_TYPE heap_type,
                                         D3D12_RESOURCE_FLAGS flags) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = heap_type;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC1 desc{};
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
    check(device->CreateCommittedResource3(&heap,D3D12_HEAP_FLAG_NONE,&desc,
              D3D12_BARRIER_LAYOUT_UNDEFINED,nullptr,nullptr,0u,nullptr,
              IID_PPV_ARGS(&resource)), "CreateCommittedResource3");
    return resource;
}

void emit_buffer_barriers(ID3D12GraphicsCommandList7* list,
                          std::span<const D3D12_BUFFER_BARRIER> barriers) {
    if(barriers.empty())return;
    D3D12_BARRIER_GROUP group{};
    group.Type=D3D12_BARRIER_TYPE_BUFFER;
    group.NumBarriers=static_cast<UINT>(barriers.size());
    group.pBufferBarriers=barriers.data();
    list->Barrier(1u,&group);
}

void order_copy_writes(ID3D12GraphicsCommandList7* list,ID3D12Resource* resource) {
    const D3D12_BUFFER_BARRIER barrier{
        D3D12_BARRIER_SYNC_COPY,D3D12_BARRIER_SYNC_COPY,
        D3D12_BARRIER_ACCESS_COPY_DEST,D3D12_BARRIER_ACCESS_COPY_DEST,
        resource,0u,UINT64_MAX};
    emit_buffer_barriers(list,std::span(&barrier,1u));
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
        bool upload_only = false;
        std::uint64_t last_reference_fence=0u;
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
#if defined(FYZZXYZZY_PROFILE_GPU)
    struct ProfileOp {
        PipelineId pipeline=0;
        std::size_t bytes=0u;
        std::uint32_t x=0u,y=0u,z=0u;
        bool indirect=false;
    };
#endif
    struct Submission {
        // A descriptor heap belongs to its completed command submission.
        // These identities describe the live entries in that exact heap.
        std::vector<PipelineId> cached_pipelines;
        std::vector<DescriptorView> cached_views;
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList7> commands;
        ComPtr<ID3D12DescriptorHeap> heap;
        ComPtr<ID3D12QueryHeap> query_heap;
        ComPtr<ID3D12Resource> query_readback;
        // One-off HtoD batch staging belongs to the submission itself.
        // It is dropped on fence completion, not retained in command cache.
        ComPtr<ID3D12Resource> owned_upload;
        ComPtr<ID3D12Resource> owned_readback;
        std::uint64_t fence_value = 0;
#if defined(FYZZXYZZY_PROFILE_GPU)
        std::vector<ProfileOp> profile_ops;
        std::uint64_t profile_trace_id=0u;
        std::uint32_t query_capacity=0u;
#endif
    };

#if defined(FYZZXYZZY_PROFILE_GPU)
    std::uint64_t profile_context_id=hardware_profile::allocate_context_id();
#endif
    ComPtr<ID3D12Device10> device;
    ComPtr<ID3D12InfoQueue> debug_messages;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12CommandSignature> dispatch_signature;
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
    CompletedReadbackCache<ComPtr<ID3D12Resource>> completed_readbacks_;
    struct PendingReadback {
        std::size_t bytes=0;
        ComPtr<ID3D12Resource> staging;
        std::vector<DownloadSpan> outputs;
    };
    std::unordered_map<SubmissionId,PendingReadback> pending_readbacks_;
    static constexpr std::size_t kMaxRecycled = 8;
    BufferId next_buffer = 1;
    PipelineId next_pipeline = 1;
    SubmissionId next_submission = 1;
    std::uint64_t next_fence = 1;
    std::uint64_t completed_fence = 0;
#if defined(FYZZXYZZY_PROFILE_GPU)
    std::uint64_t trace_frequency = 0;
#endif
    // ExecuteCommandLists can succeed even if queue->Signal fails. This
    // fence value must never be used as a completion target.
    std::uint64_t first_unsignaled_fence = 0;

    explicit Impl(Backend backend, std::uint32_t ordinal) {
        if (backend != Backend::D3D12) throw std::invalid_argument("Expected D3D12 backend");
        auto adapters = discover_adapters();
        if (ordinal >= adapters.size())
            throw std::out_of_range("D3D12 adapter with enhanced barriers does not exist");
        info = adapters[ordinal].info;
        if (environment_equals("FYZZXYZZY_D3D12_REQUIRE_HARDWARE",'1')) {
            DXGI_ADAPTER_DESC1 description{};
            check(adapters[ordinal].adapter->GetDesc1(&description),
                  "GetDesc1 for D3D12 hardware verification");
            if (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                throw std::runtime_error(
                    "D3D12 hardware verification requires a physical GPU");
        }
        if (environment_equals("FYZZXYZZY_D3D12_DEBUG",'1')) {
            ComPtr<ID3D12Debug> debug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
                debug->EnableDebugLayer();
                const bool enable_gpu_validation =
                    !environment_equals("FYZZXYZZY_D3D12_GPU_VALIDATION",'0');
                if (enable_gpu_validation) {
                    ComPtr<ID3D12Debug1> gpu_debug;
                    if (SUCCEEDED(debug.As(&gpu_debug)))
                        gpu_debug->SetEnableGPUBasedValidation(TRUE);
                }
                std::cerr << "Enabled D3D12 debug layer"
                          << (enable_gpu_validation ? " and GPU-based validation" : "")
                          << "\n";
            } else {
                std::cerr << "D3D12 debug layer not available on this runner\n";
            }
        }
        check(D3D12CreateDevice(adapters[ordinal].adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                IID_PPV_ARGS(&device)), "D3D12CreateDevice");
        if (environment_equals("FYZZXYZZY_D3D12_DEBUG",'1'))
            device.As(&debug_messages);
        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
#if defined(FYZZXYZZY_PROFILE_GPU)
        if(FAILED(queue->GetTimestampFrequency(&trace_frequency)))
            trace_frequency=0u;
#endif
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                  IID_PPV_ARGS(&fence)), "CreateFence");
        recycled.reserve(kMaxRecycled);
        completion_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!completion_event)
            throw std::runtime_error("CreateEvent failed for D3D12 fence");
    }

    ~Impl() {
        if (!submissions.empty() && SUCCEEDED(device->GetDeviceRemovedReason())) {
            const auto last=next_fence-1u;
            try {
                if(first_unsignaled_fence) {
                    // A teardown-only signal drains the poisoned queue without
                    // publishing the failed submission or admitting more work.
                    check(queue->Signal(fence.Get(),last),"Signal D3D12 teardown fence");
                    first_unsignaled_fence=0;
                    wait_value(last);
                }else wait_all();
            }catch(...) {
                if(completed_fence<last && SUCCEEDED(device->GetDeviceRemovedReason()))
                    std::terminate();
            }
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
#if defined(FYZZXYZZY_PROFILE_GPU)
            submission.profile_ops.clear();
            submission.profile_trace_id=0u;
#endif
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
        submission.owned_readback.Reset();
        // The caller must first wait for the submission fence. A cached heap
        // keeps its descriptors only until the next recording overwrites them.
        if (recycled.size() < kMaxRecycled)
            recycled.push_back(std::move(submission));
    }

    SubmissionId enqueue(Submission&& submission,
                         std::span<Buffer* const> referenced) {
        if (first_unsignaled_fence)
            throw std::runtime_error(
                "D3D12 queue cannot submit after a failed completion signal");
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
        for(auto* buffer:referenced)buffer->last_reference_fence=signal;
        ID3D12CommandList* lists[] = {it->second.commands.Get()};
        queue->ExecuteCommandLists(1, lists);
        // The command list has executed even if the signal fails. Keep
        // its resources owned by submissions, but never wait on this
        // unsignaled value or submit further work on the poisoned queue.
        const HRESULT signal_result = queue->Signal(fence.Get(), signal);
        if (FAILED(signal_result)) {
            first_unsignaled_fence = signal;
            check(signal_result, "Signal D3D12 fence");
        }
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
        if (first_unsignaled_fence && value >= first_unsignaled_fence)
            throw std::runtime_error(
                "D3D12 submission has no completion fence after queue signal failure");
        if (completed >= value) {
            observe_completion(completed);
            return;
        }
        check(fence->SetEventOnCompletion(value, completion_event),
              "SetEventOnCompletion");
        for (;;) {
            const DWORD result=WaitForSingleObject(completion_event,1000u);
            if(result==WAIT_OBJECT_0) {
                // An event can retain a signal from an interrupted wait.
                const auto observed=fence->GetCompletedValue();
                if(observed>=value)break;
            }else if(result!=WAIT_TIMEOUT)
                throw std::runtime_error("D3D12 fence wait failed");
            // Device removal does not guarantee that an outstanding queue
            // fence signals. Check the device while the event is pending.
            const HRESULT reason=device->GetDeviceRemovedReason();
            if(FAILED(reason)) {
                print_debug_messages();
                throw std::runtime_error(
                    "D3D12 device removed during GPU submission (HRESULT "+
                    std::to_string(static_cast<unsigned long>(reason))+")");
            }
        }
        const auto observed=fence->GetCompletedValue();
        if (observed == std::numeric_limits<std::uint64_t>::max()) {
            const auto reason = device->GetDeviceRemovedReason();
            print_debug_messages();
            throw std::runtime_error(
                "D3D12 device removed during GPU submission (HRESULT " +
                std::to_string(static_cast<unsigned long>(reason)) + ")");
        }
        observe_completion(observed);
    }

    void observe_completion(std::uint64_t value) noexcept {
        if(value<=completed_fence)return;
        completed_fence=value;
        std::erase_if(retired_buffers,[&](const RetiredBuffer& retired) {
            return retired.release_after_fence<=completed_fence;
        });
    }

#if defined(FYZZXYZZY_PROFILE_GPU)
    void report_gpu_profile(SubmissionId id,Submission& submission) {
        if(submission.profile_ops.empty() || !trace_frequency)return;
        const auto count=submission.profile_ops.size()+1u;
        void* mapped=nullptr;
        const D3D12_RANGE read{0,count*sizeof(std::uint64_t)};
        std::vector<std::uint64_t> ticks(count);
        check(submission.query_readback->Map(0,&read,&mapped),
              "Map hardware GPU timestamp query");
        std::memcpy(ticks.data(),mapped,read.End);
        const D3D12_RANGE written{0,0};
        submission.query_readback->Unmap(0,&written);
        for(std::size_t i=0u;i<submission.profile_ops.size();++i) {
            const auto& operation=submission.profile_ops[i];
            if(ticks[i+1u]<ticks[i])
                throw std::runtime_error("D3D12 hardware timestamp counter regressed");
            const auto elapsed=ticks[i+1u]-ticks[i];
            hardware_profile::gpu_event({
                "d3d12",operation.pipeline?"dispatch":"copy",
                profile_context_id,submission.profile_trace_id,
                id,i,operation.pipeline,operation.bytes,
                operation.x,operation.y,operation.z,operation.indirect,
                static_cast<double>(elapsed)*1.0e9/static_cast<double>(trace_frequency)});
        }
    }
#endif
    void wait_submission(SubmissionId id) {
        auto it = submissions.find(id);
        if (it == submissions.end()) throw std::invalid_argument("Unknown D3D12 submission");
        const auto completed = it->second.fence_value;
        wait_value(completed);
#if defined(FYZZXYZZY_PROFILE_GPU)
        report_gpu_profile(id,it->second);
#endif
        recycle_completed(std::move(it->second));
        submissions.erase(it);
        recycle_cancelled_readback(id);
    }

    void wait_submissions(std::span<const SubmissionId> ordered) {
        if(ordered.empty())return;
        SubmissionId previous=0;
        for(const auto id:ordered) {
            if(id<=previous || submissions.find(id)==submissions.end())
                throw std::invalid_argument(
                    "D3D12 GPU retire batch requires live ordered submissions");
            previous=id;
        }
        const auto last_fence=submissions.at(ordered.back()).fence_value;
        wait_value(last_fence);
        for(const auto id:ordered) {
            auto it=submissions.find(id);
#if defined(FYZZXYZZY_PROFILE_GPU)
            report_gpu_profile(id,it->second);
#endif
            recycle_completed(std::move(it->second));
            submissions.erase(it);
            recycle_cancelled_readback(id);
        }
    }

    void wait_all() {
        if (submissions.empty()) return;
        const auto completed = next_fence - 1;
        wait_value(completed);
        while(!submissions.empty()) {
            auto it=submissions.begin();
#if defined(FYZZXYZZY_PROFILE_GPU)
            report_gpu_profile(it->first,it->second);
#endif
            const auto id=it->first;
            recycle_completed(std::move(it->second));
            submissions.erase(it);
            recycle_cancelled_readback(id);
        }
    }

    BufferId create_buffer(std::size_t bytes, MemoryPlacement placement) {
        if (!bytes) throw std::invalid_argument("Compute buffer size must be positive");
        Buffer buffer;
        if(placement==MemoryPlacement::Upload) {
            buffer.resource=allocate_resource(device.Get(),bytes,D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE);
            buffer.host_visible=true;
            buffer.upload_only=true;
        }else if (placement == MemoryPlacement::HostVisible ||
            placement == MemoryPlacement::SharedLocal) {
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
            // GPU UAV and buffer access barriers; a regular READBACK heap
            // does not. The admission probe exercises precisely this pairing.
            const auto heap = device->GetCustomHeapProperties(0, D3D12_HEAP_TYPE_READBACK);
            if (heap.Type != D3D12_HEAP_TYPE_CUSTOM ||
                heap.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_WRITE_BACK ||
                heap.MemoryPoolPreference != D3D12_MEMORY_POOL_L0)
                throw UnsupportedMemoryPlacement("D3D12 UMA did not expose a CPU-mappable L0 custom heap");
            D3D12_RESOURCE_DESC1 desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = static_cast<UINT64>(bytes);
            desc.Height = 1;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.SampleDesc.Count = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            check(device->CreateCommittedResource3(
                &heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_BARRIER_LAYOUT_UNDEFINED,
                nullptr,nullptr,0u,nullptr,IID_PPV_ARGS(&buffer.resource)),
                "Create GPU-writable CPU-visible D3D12 UMA buffer");
            buffer.host_visible = true;
        } else {
            // Auto preserves the established D3D12 default, while explicit
            // DeviceLocal remains a GPU-preferred DEFAULT allocation.
            buffer.resource = allocate_resource(device.Get(), bytes, D3D12_HEAP_TYPE_DEFAULT,
                                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
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
        if(buffer.upload_only && access!=HostMapAccess::Write)
            throw std::invalid_argument("D3D12 upload storage requires write mapping");
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
        if(it->second.last_reference_fence>completed_fence)
            retired_buffers.push_back({it->second.last_reference_fence,it->second});
        // The handle becomes unusable immediately. Its backing resource
        // lives in retired_buffers until the last dependent queue fence.
        buffers.erase(it);
    }

    void upload(BufferId id, std::size_t offset, const void* data, std::size_t bytes) {
        const auto& buffer=get_buffer(id);
        if(buffer.upload_only)
            throw std::invalid_argument("Upload storage requires write mapping");
        validate_range(buffer.bytes,offset,bytes);
        if(bytes)wait_submission(submit_upload(id,offset,data,bytes));
    }

    SubmissionId submit_upload(BufferId id,std::size_t offset,
                               const void* data,std::size_t bytes) {
        return submit_uploads({{id,offset,data,bytes}});
    }

    SubmissionId submit_uploads(const std::vector<UploadSpan>& uploads) {
        if (uploads.empty())
            throw std::invalid_argument("Async D3D12 upload batch cannot be empty");
        std::size_t total = 0;
        std::unordered_set<BufferId> destinations;
        destinations.reserve(uploads.size());
        for (const auto& upload : uploads) {
            const auto& target = get_buffer(upload.destination);
            if(target.upload_only)
                throw std::invalid_argument("Upload storage cannot be a GPU destination");
            validate_range(target.bytes, upload.destination_offset, upload.bytes);
            if (!upload.data || !upload.bytes ||
                (upload.destination_offset % 4) || (upload.bytes % 4) ||
                upload.bytes > std::numeric_limits<std::size_t>::max() - total)
                throw std::invalid_argument("Invalid D3D12 upload span or staging capacity");
            total += upload.bytes;
            destinations.insert(upload.destination);
        }

        auto staging = allocate_resource(device.Get(), total, D3D12_HEAP_TYPE_UPLOAD,
                                         D3D12_RESOURCE_FLAG_NONE);
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

        std::vector<Buffer*> referenced;
        referenced.reserve(destinations.size());
        Submission submission = new_commands();
        for (const auto id : destinations)
            referenced.push_back(&get_buffer(id));
        position = 0;
        destinations.clear();
        for (const auto& upload : uploads) {
            if(!destinations.insert(upload.destination).second)
                order_copy_writes(submission.commands.Get(),
                    get_buffer(upload.destination).resource.Get());
            submission.commands->CopyBufferRegion(
                get_buffer(upload.destination).resource.Get(),
                upload.destination_offset, staging.Get(), position, upload.bytes);
            position += upload.bytes;
        }
        // Ownership is attached BEFORE queue submission. A later allocation
        // failure cannot free the memory while the GPU still reads it.
        submission.owned_upload = std::move(staging);
        return enqueue(std::move(submission),referenced);
    }

    void download(BufferId id,std::size_t offset,void* data,std::size_t bytes) {
        const auto& buffer=get_buffer(id);
        if(buffer.upload_only)
            throw std::invalid_argument("Upload storage requires write mapping");
        validate_range(buffer.bytes,offset,bytes);
        if(!bytes)return;
        const std::array<DownloadSpan,1u> spans{{{id,offset,data,bytes}}};
        finish_download_batch(submit_download_batch(spans),true);
    }

    struct ReadbackTail {
        ID3D12Resource* staging;
        std::span<const DownloadSpan> sources;
    };

    ComPtr<ID3D12Resource> acquire_readback(std::size_t bytes) {
        auto cached=completed_readbacks_.acquire(bytes,
            [](ComPtr<ID3D12Resource>) noexcept {});
        if(!cached.capacity)
            return allocate_resource(device.Get(),bytes,
                D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_FLAG_NONE);
        return std::move(cached.resource);
    }
    void recycle_readback(ComPtr<ID3D12Resource> staging) noexcept {
        const auto capacity=static_cast<std::size_t>(staging->GetDesc().Width);
        completed_readbacks_.recycle(std::move(staging),capacity,
            [](ComPtr<ID3D12Resource>) noexcept {});
    }
    SubmissionId submit_download_batch(
            std::span<const DownloadSpan> downloads,
            const std::vector<BatchCommand>& producer = {}) {
        if(downloads.empty())
            throw std::invalid_argument("Empty D3D12 GPU readback batch");
        std::size_t bytes=0;
        for(const auto& item:downloads) {
            const auto& source=get_buffer(item.source);
            if(source.upload_only && source.cpu_mapped)
                throw std::invalid_argument("Mapped upload storage cannot be submitted");
            validate_range(source.bytes,item.source_offset,item.bytes);
            if(!item.data || !item.bytes ||
               (item.source_offset&3u) || (item.bytes&3u) ||
               item.bytes>std::numeric_limits<std::size_t>::max()-bytes)
                throw std::invalid_argument("Invalid D3D12 GPU readback span");
            bytes+=item.bytes;
        }
        std::vector<DownloadSpan> outputs(downloads.begin(),downloads.end());
        auto staging=acquire_readback(bytes);
        const ReadbackTail tail{staging.Get(),downloads};
        const auto id=next_submission;
        const auto [entry,inserted]=pending_readbacks_.emplace(
            id,PendingReadback{bytes,staging,std::move(outputs)});
        if(!inserted)
            throw std::logic_error("D3D12 readback ID already registered");
        try {
            submit_batch(producer,false,&tail);
        }catch(...) {
            pending_readbacks_.erase(entry);
            // Submission owns its staging if execution has begun.
            if(!first_unsignaled_fence)recycle_readback(std::move(staging));
            throw;
        }
        return id;
    }

    // A cancelled readback keeps its staging until the submission retires.
    bool cancel_download_batch(SubmissionId id) {
        const auto it=pending_readbacks_.find(id);
        if(it==pending_readbacks_.end())
            throw std::logic_error("Unknown GPU readback batch");
        it->second.outputs.clear();
        if(submissions.find(id)!=submissions.end())return true;
        recycle_cancelled_readback(id);
        return false;
    }

    void recycle_cancelled_readback(SubmissionId id) noexcept {
        const auto it=pending_readbacks_.find(id);
        if(it==pending_readbacks_.end() || !it->second.outputs.empty())return;
        auto staging=std::move(it->second.staging);
        pending_readbacks_.erase(it);
        recycle_readback(std::move(staging));
    }

    void finish_download_batch(SubmissionId id,bool publish) {
        const auto it=pending_readbacks_.find(id);
        if(it==pending_readbacks_.end())
            throw std::logic_error("Unknown D3D12 GPU readback batch");
        if(it->second.outputs.empty())
            throw std::logic_error("Cancelled GPU readback cannot be finished");
        if(submissions.find(id)!=submissions.end())
            wait_submission(id);
        auto& pending=it->second;
        if(publish) {
            void* mapped=nullptr;
            D3D12_RANGE read{0,pending.bytes};
            check(pending.staging->Map(0,&read,&mapped),
                  "Map asynchronous GPU readback");
            std::size_t offset=0;
            for(const auto& item:pending.outputs) {
                std::memcpy(item.data,
                    static_cast<const std::uint8_t*>(mapped)+offset,
                    item.bytes);
                offset+=item.bytes;
            }
            D3D12_RANGE written{0,0};
            pending.staging->Unmap(0,&written);
        }
        auto staging=std::move(pending.staging);
        pending_readbacks_.erase(it);
        recycle_readback(std::move(staging));
    }

    void download_batch(std::span<const DownloadSpan> downloads) {
        if(downloads.empty())return;
        finish_download_batch(submit_download_batch(downloads),true);
    }

    PipelineId create_pipeline(const ComputePipelineDesc& desc) {
        if (desc.shader.format != ShaderFormat::Dxil || desc.shader.bytes.empty())
            throw std::invalid_argument("Native D3D12 requires compiled DXIL");
        if (const auto width = desc.required_subgroup_size) {
            if ((width & (width-1u)) != 0u ||
                !info.supports_required_subgroup_size ||
                width < info.min_subgroup_size ||
                width > info.max_subgroup_size)
                throw std::invalid_argument(
                    "Required D3D12 wave width is unsupported by this adapter");
            // SM6.6 DXIL [WaveSize] is authoritative at PSO creation.
            // Host-side admission cannot override absent DXIL metadata.
        }
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
        // A removed D3D12 device or a queue whose completion signal
        // failed cannot complete outstanding fences. Preserve the PSO
        // and root signature in the Context until final COM teardown;
        // no stage destructor may retry that impossible fence wait.
        if(first_unsignaled_fence ||
           FAILED(device->GetDeviceRemovedReason()))
            return;
        wait_all();
        pipelines.erase(id);
    }

    void validate_dispatch_shape(const Pipeline& pipe,const DispatchCommand& cmd) const {
        if(!cmd.groups.x || !cmd.groups.y || !cmd.groups.z ||
           cmd.groups.x>info.max_workgroup_count[0] ||
           cmd.groups.y>info.max_workgroup_count[1] ||
           cmd.groups.z>info.max_workgroup_count[2] ||
           cmd.push_constants.size()!=pipe.parameters)
            throw std::invalid_argument("Invalid D3D12 dispatch shape or parameters");
        if(cmd.indirect && (cmd.groups.x!=1u || cmd.groups.y!=1u || cmd.groups.z!=1u))
            throw std::invalid_argument("Invalid D3D12 indirect dispatch groups");
    }

    SubmissionId submit_batch(const std::vector<BatchCommand>& commands,
                              bool profile = false,
                              const ReadbackTail* readback_tail = nullptr) {
#if defined(FYZZXYZZY_PROFILE_GPU)
        const bool hardware_trace=!profile && trace_frequency!=0u;
        profile=profile || hardware_trace;
#endif
        if (commands.empty() && !readback_tail)
            throw std::invalid_argument("Empty D3D12 command batch");
        // Each distinct pipeline/binding view needs one descriptor table
        // for this submission. Repeated dispatches may change constants or
        // execution dimensions without duplicating identical descriptors.
        struct DescriptorTable {
            PipelineId pipeline;
            const Pipeline* native_pipeline;
            std::vector<DescriptorView> views;
            UINT first_descriptor;
        };
        std::vector<DescriptorTable> descriptor_tables;
        std::unordered_map<std::size_t,std::vector<std::size_t>> descriptor_lookup;
        std::vector<std::size_t> dispatch_table_indices,dispatch_range_plans;
        DispatchRangePlans range_plans;
        std::size_t descriptor_count = 0;
        bool uses_indirect = false;
        std::size_t upload_total_bytes = 0;
        std::vector<PipelineId> pipeline_sequence;
        const auto mix_key=[](std::size_t& hash,std::size_t word) {
            hash^=std::hash<std::size_t>{}(word)+0x9e3779b9u+
                  (hash<<6u)+(hash>>2u);
        };
        struct Access {
            Buffer* buffer;
            PendingBufferRanges ranges;
            RangeAccessScope scope{};
            std::size_t incoming=std::numeric_limits<std::size_t>::max();
        };
        // Range plans refer to dense batch-local resource ordinals.
        std::unordered_map<BufferId,std::size_t> resource_ordinals;
        resource_ordinals.reserve(std::min<std::size_t>(commands.size()*2u,512u));
        std::vector<Access> resource_access;
        std::vector<Buffer*> referenced;
        const auto reference=[&](BufferId id) -> Access& {
            const auto found=resource_ordinals.find(id);
            if(found!=resource_ordinals.end())return resource_access[found->second];
            auto& buffer=get_buffer(id);
            resource_ordinals.emplace(id,resource_access.size());
            referenced.push_back(&buffer);
            resource_access.push_back(Access{&buffer,{},{}});
            return resource_access.back();
        };
        const auto same_bindings=[](
                const std::vector<DescriptorView>& a,
                const std::vector<BufferBinding>& b) {
            if(a.size()!=b.size())return false;
            for(std::size_t i=0u;i<a.size();++i)
                if(b[i].slot!=i ||
                   a[i].buffer!=b[i].buffer ||
                   a[i].offset!=b[i].offset ||
                   a[i].bytes!=b[i].bytes)
                    return false;
            // Access intent affects barriers, never descriptor contents.
            return true;
        };
        for (const BatchCommand& op : commands) {
            if (const auto* copy = std::get_if<CopyCommand>(&op)) {
                const auto& source=*reference(copy->source).buffer;
                const auto& dest=*reference(copy->destination).buffer;
                if(dest.upload_only || (source.upload_only && source.cpu_mapped))
                    throw std::invalid_argument("Invalid D3D12 upload storage copy");
                validate_range(source.bytes, copy->source_offset, copy->bytes);
                validate_range(dest.bytes, copy->destination_offset, copy->bytes);
                const bool overlap = copy->source == copy->destination &&
                    copy->source_offset < copy->destination_offset + copy->bytes &&
                    copy->destination_offset < copy->source_offset + copy->bytes;
                if (!copy->bytes || overlap ||
                    copy->source_offset % 4 || copy->destination_offset % 4 ||
                    copy->bytes % 4)
                    throw std::invalid_argument("Invalid D3D12 buffer copy");
            } else if(const auto* upload=std::get_if<UploadCommand>(&op)) {
                const auto& target=*reference(upload->destination).buffer;
                if(target.upload_only)
                    throw std::invalid_argument("Upload storage cannot be a GPU destination");
                validate_range(target.bytes,upload->destination_offset,
                               upload->bytes.size());
                if(upload->bytes.empty() || upload->destination_offset%4u ||
                   upload->bytes.size()%4u ||
                   upload->bytes.size()>
                       std::numeric_limits<std::size_t>::max()-upload_total_bytes)
                    throw std::invalid_argument("Invalid D3D12 ordered upload");
                upload_total_bytes+=upload->bytes.size();
            } else {
                const auto& dispatch = std::get<DispatchCommand>(op);
                uses_indirect |= dispatch.indirect.has_value();
                std::size_t key=std::hash<PipelineId>{}(dispatch.pipeline);
                for(const auto& binding:dispatch.buffers) {
                    mix_key(key,binding.slot);
                    mix_key(key,binding.buffer);
                    mix_key(key,binding.offset);
                    mix_key(key,binding.bytes);
                }
                auto& candidates=descriptor_lookup[key];
                std::size_t table=descriptor_tables.size();
                for(const auto candidate:candidates) {
                    const auto& existing=descriptor_tables[candidate];
                    if(existing.pipeline==dispatch.pipeline &&
                       same_bindings(existing.views,dispatch.buffers)) {
                        table=candidate;
                        break;
                    }
                }
                if(table==descriptor_tables.size()) {
                    const auto& pipeline=get_pipeline(dispatch.pipeline);
                    const auto binding_count=pipeline.bindings;
                    if(dispatch.buffers.size()!=binding_count)
                        throw std::invalid_argument("Incorrect number of D3D12 compute bindings");
                    if(binding_count>
                       std::numeric_limits<UINT>::max()-descriptor_count)
                        throw std::out_of_range("D3D12 descriptor heap capacity exceeded");
                    DescriptorTable prepared{dispatch.pipeline,&pipeline,{},
                                             static_cast<UINT>(descriptor_count)};
                    prepared.views.reserve(binding_count);
                    for(std::size_t i=0u;i<binding_count;++i) {
                        const auto& binding=dispatch.buffers[i];
                        if(binding.slot!=i || !binding.bytes ||
                           binding.offset%4u || binding.bytes%4u ||
                           binding.bytes/4u>std::numeric_limits<UINT>::max())
                            throw std::invalid_argument("Invalid D3D12 structured buffer view");
                        const auto kind=pipeline.view_kinds[i];
                        if((kind==StorageBufferViewKind::RawRead ||
                            kind==StorageBufferViewKind::RawReadWrite) &&
                           binding.offset%16u)
                            throw std::invalid_argument("D3D12 RAW buffer view base must be 16-byte aligned");
                        const auto& buffer=*reference(binding.buffer).buffer;
                        if(buffer.upload_only)
                            throw std::invalid_argument("Upload storage cannot be a shader binding");
                        validate_range(buffer.bytes,binding.offset,binding.bytes);
                        // One native resource cannot mix SRV and UAV views.
                        for(std::size_t prior=0u;prior<i;++prior)
                            if(prepared.views[prior].buffer==binding.buffer &&
                               read_only_view(pipeline.view_kinds[prior])!=read_only_view(kind))
                                throw std::invalid_argument("D3D12 cannot bind one resource as SRV and UAV simultaneously");
                        prepared.views.push_back({binding.buffer,binding.offset,binding.bytes,
                                                  buffer.resource.Get()});
                    }
                    descriptor_tables.push_back(std::move(prepared));
                    candidates.push_back(table);
                    pipeline_sequence.push_back(dispatch.pipeline);
                    descriptor_count+=binding_count;
                }
                const auto& pipeline=*descriptor_tables[table].native_pipeline;
                validate_dispatch_shape(pipeline,dispatch);
                dispatch_table_indices.push_back(table);
                dispatch_range_plans.push_back(range_plans.prepare(table,dispatch,
                    [&](std::vector<OperationRangeUse>& ranges) {
                        for(const auto& binding:dispatch.buffers)
                            validate_binding_access(binding,
                                read_only_view(pipeline.view_kinds[binding.slot]));
                        if(dispatch.indirect) {
                            const auto& indirect=*dispatch.indirect;
                            const auto& args=*reference(indirect.buffer).buffer;
                            if(indirect.offset%4u || args.upload_only)
                                throw std::invalid_argument("Invalid D3D12 indirect dispatch arguments");
                            validate_range(args.bytes,indirect.offset,sizeof(D3D12_DISPATCH_ARGUMENTS));
                            for(const auto& binding:dispatch.buffers)
                                if(binding.buffer==indirect.buffer)
                                    throw std::invalid_argument("D3D12 indirect arguments cannot alias shader bindings");
                            ranges.push_back({resource_ordinals.at(dispatch.indirect->buffer),
                                dispatch.indirect->offset,
                                dispatch.indirect->offset+sizeof(D3D12_DISPATCH_ARGUMENTS),
                                {D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                                 D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT},false});
                        }
                        dispatch.for_each_access([&](const BufferBinding& binding,
                                                BufferAccessExtent extent,BufferAccess access) {
                            const auto kind=pipeline.view_kinds[binding.slot];
                            const bool srv=kind==StorageBufferViewKind::RawRead ||
                                           kind==StorageBufferViewKind::Structured32Read;
                            if(srv && access!=BufferAccess::Read &&
                               access!=BufferAccess::Unspecified)
                                throw std::invalid_argument("Dispatch access writes SRV");
                            const auto begin=binding.offset+extent.offset;
                            ranges.push_back({resource_ordinals.at(binding.buffer),
                                begin,begin+extent.bytes,
                                {D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                                 static_cast<std::uint64_t>(srv
                                    ?D3D12_BARRIER_ACCESS_SHADER_RESOURCE
                                    :D3D12_BARRIER_ACCESS_UNORDERED_ACCESS)},
                                !srv && access!=BufferAccess::Read});
                        });
                        std::sort(ranges.begin(),ranges.end(),[](const auto& a,const auto& b) {
                            return a.buffer<b.buffer;
                        });
                    }));
            }
        }
        if (descriptor_count > std::numeric_limits<UINT>::max())
            throw std::out_of_range("D3D12 descriptor heap capacity exceeded");
#if defined(FYZZXYZZY_PROFILE_GPU)
        if(hardware_trace) {
            FYZZ_PROFILE_COUNTER("d3d12.descriptor_tables",
                                 descriptor_tables.size());
            FYZZ_PROFILE_COUNTER("d3d12.descriptor_table_reuses",
                                 dispatch_table_indices.size()-
                                 descriptor_tables.size());
            FYZZ_PROFILE_COUNTER("d3d12.descriptor_slots",descriptor_count);
            FYZZ_PROFILE_COUNTER("d3d12.range_plans",range_plans.size());
            FYZZ_PROFILE_COUNTER("d3d12.range_plan_reuses",
                                 dispatch_table_indices.size()-range_plans.size());
        }
#endif
        if(uses_indirect && !dispatch_signature) {
            D3D12_INDIRECT_ARGUMENT_DESC argument{};
            argument.Type=D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
            D3D12_COMMAND_SIGNATURE_DESC signature{};
            signature.ByteStride=sizeof(D3D12_DISPATCH_ARGUMENTS);
            signature.NumArgumentDescs=1u;
            signature.pArgumentDescs=&argument;
            check(device->CreateCommandSignature(
                      &signature,nullptr,IID_PPV_ARGS(&dispatch_signature)),
                  "CreateCommandSignature");
        }
        Submission submission = new_commands(&pipeline_sequence, descriptor_count);
        if(readback_tail)submission.owned_readback=readback_tail->staging;
        if(upload_total_bytes) {
            auto staging=allocate_resource(
                device.Get(),upload_total_bytes,D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE);
            void* mapped=nullptr;
            const D3D12_RANGE read{0,0};
            check(staging->Map(0,&read,&mapped),"Map ordered upload batch");
            std::size_t offset=0;
            for(const auto& op:commands)
                if(const auto* upload=std::get_if<UploadCommand>(&op)) {
                    std::memcpy(static_cast<std::uint8_t*>(mapped)+offset,
                                upload->bytes.data(),upload->bytes.size());
                    offset+=upload->bytes.size();
                }
            const D3D12_RANGE written{0,upload_total_bytes};
            staging->Unmap(0,&written);
            submission.owned_upload=std::move(staging);
        }
#if defined(FYZZXYZZY_PROFILE_GPU)
        if(hardware_trace &&
           submission.query_capacity<=commands.size()) {
            submission.query_heap.Reset();
            submission.query_readback.Reset();
            submission.query_capacity=0u;
        }
#endif
        if (profile) {
            D3D12_QUERY_HEAP_DESC desc{};
            desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
            desc.Count = 2;
#if defined(FYZZXYZZY_PROFILE_GPU)
            if(hardware_trace) {
                if(commands.size()>=std::numeric_limits<UINT>::max())
                    throw std::out_of_range("Hardware trace query count exceeds 32-bit range");
                desc.Count=static_cast<UINT>(commands.size()+1u);
            }
#endif
            if (!submission.query_heap) {
                check(device->CreateQueryHeap(&desc, IID_PPV_ARGS(&submission.query_heap)),
                      "CreateQueryHeap(timestamp)");
#if defined(FYZZXYZZY_PROFILE_GPU)
                submission.query_capacity=desc.Count;
#endif
            }
            if (!submission.query_readback)
                submission.query_readback = allocate_resource(
                    device.Get(),
#if defined(FYZZXYZZY_PROFILE_GPU)
                    static_cast<std::size_t>(hardware_trace?desc.Count:2u) *
#else
                    2u *
#endif
                    sizeof(std::uint64_t),
                    D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_FLAG_NONE);
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

        // Whole-buffer barriers separate conflicting access epochs.
        // Each barrier's destination stages cover the complete next epoch.
        std::vector<OperationRangeUse> uses;
        std::vector<D3D12_BUFFER_BARRIER> barriers;
        std::vector<std::size_t> barrier_offsets;
        constexpr RangeAccessScope copy_read{D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_ACCESS_COPY_SOURCE},copy_write{D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_ACCESS_COPY_DEST};
        const auto use=[&](BufferId id,std::size_t offset,std::size_t size,
                           RangeAccessScope scope,bool writes) {
            if(!reference(id).buffer->upload_only)
                uses.push_back({resource_ordinals.at(id),offset,offset+size,scope,writes});
        };
        const auto plan_operation=[&](std::span<const OperationRangeUse> uses) {
            barrier_offsets.push_back(barriers.size());
            for(std::size_t first=0u;first<uses.size();) {
                auto& state=resource_access[uses[first].buffer];
                auto last=first;
                RangeAccessScope scope{};
                bool conflict=false;
                for(;last<uses.size() && uses[last].buffer==uses[first].buffer;++last) {
                    const auto& item=uses[last];
                    scope.include(item.scope);
                    if(!conflict)
                        conflict=state.ranges.conflicts(item.begin,item.end,item.writes);
                }
                // UAV and copy destinations are distinct write access classes.
                constexpr auto write_classes=std::uint64_t(D3D12_BARRIER_ACCESS_UNORDERED_ACCESS) |
                                             D3D12_BARRIER_ACCESS_COPY_DEST;
                conflict|=((state.scope.access|scope.access)&write_classes)==write_classes;
                if(conflict) {
                    state.incoming=barriers.size();
                    barriers.push_back({
                        static_cast<D3D12_BARRIER_SYNC>(state.scope.stages),
                        static_cast<D3D12_BARRIER_SYNC>(scope.stages),
                        static_cast<D3D12_BARRIER_ACCESS>(state.scope.access),
                        D3D12_BARRIER_ACCESS_COMMON,
                        state.buffer->resource.Get(),0u,UINT64_MAX});
                    state.ranges.clear();
                    state.scope={};
                }else if(state.incoming!=std::numeric_limits<std::size_t>::max()) {
                    auto& incoming=barriers[state.incoming];
                    incoming.SyncAfter=static_cast<D3D12_BARRIER_SYNC>(
                        std::uint64_t(incoming.SyncAfter)|scope.stages);
                }
                state.scope.include(scope);
                for(auto i=first;i<last;++i)
                    state.ranges.record(uses[i].begin,uses[i].end,uses[i].writes);
                first=last;
            }
        };
        std::size_t planned_dispatch=0u;
        for(const auto& op:commands) {
            uses.clear();
            if(const auto* copy=std::get_if<CopyCommand>(&op)) {
                use(copy->source,copy->source_offset,copy->bytes,copy_read,false);
                use(copy->destination,copy->destination_offset,copy->bytes,copy_write,true);
                if(uses.size()==2u && uses[0].buffer>uses[1].buffer)
                    std::swap(uses[0],uses[1]);
            }else if(const auto* upload=std::get_if<UploadCommand>(&op)) {
                use(upload->destination,upload->destination_offset,upload->bytes.size(),copy_write,true);
            }else {
                plan_operation(range_plans.ranges(dispatch_range_plans[planned_dispatch++]));
                continue;
            }
            plan_operation(uses);
        }
        if(readback_tail)
            for(const auto& item:readback_tail->sources) {
                uses.clear();
                use(item.source,item.source_offset,item.bytes,copy_read,false);
                plan_operation(uses);
            }
        barrier_offsets.push_back(barriers.size());
        std::size_t operation_index=0u,barrier_calls=0u;
        const auto before_operation=[&]() {
            const auto begin=barrier_offsets[operation_index];
            const auto end=barrier_offsets[++operation_index];
            if(begin!=end) {
                emit_buffer_barriers(submission.commands.Get(),
                    std::span<const D3D12_BUFFER_BARRIER>(barriers).subspan(begin,end-begin));
                ++barrier_calls;
            }
        };
        std::vector<bool> descriptor_table_written(descriptor_tables.size(),false);
        std::size_t dispatch_index=0u;
        std::size_t ordered_upload_offset=0u;
        PipelineId bound_pipeline=0u;
        std::size_t bound_table=std::numeric_limits<std::size_t>::max();
#if defined(FYZZXYZZY_PROFILE_GPU)
        UINT hardware_query_index=1u;
#endif
        for (const BatchCommand& op : commands) {
            before_operation();
            if (const auto* copy = std::get_if<CopyCommand>(&op)) {
                auto& src=*resource_access[resource_ordinals.at(copy->source)].buffer;
                auto& dst=*resource_access[resource_ordinals.at(copy->destination)].buffer;
                submission.commands->CopyBufferRegion(dst.resource.Get(),
                    copy->destination_offset,src.resource.Get(),
                    copy->source_offset,copy->bytes);
#if defined(FYZZXYZZY_PROFILE_GPU)
                if(hardware_trace)
                    submission.commands->EndQuery(submission.query_heap.Get(),
                        D3D12_QUERY_TYPE_TIMESTAMP,hardware_query_index++);
#endif
                continue;
            }
            if(const auto* upload=std::get_if<UploadCommand>(&op)) {
                auto& dst=*resource_access[resource_ordinals.at(upload->destination)].buffer;
                submission.commands->CopyBufferRegion(
                    dst.resource.Get(),upload->destination_offset,
                    submission.owned_upload.Get(),ordered_upload_offset,
                    upload->bytes.size());
                ordered_upload_offset+=upload->bytes.size();
#if defined(FYZZXYZZY_PROFILE_GPU)
                if(hardware_trace)
                    submission.commands->EndQuery(submission.query_heap.Get(),
                        D3D12_QUERY_TYPE_TIMESTAMP,hardware_query_index++);
#endif
                continue;
            }
            const auto& cmd = std::get<DispatchCommand>(op);
            const auto table=dispatch_table_indices[dispatch_index++];
            const auto& prepared=descriptor_tables[table];
            const auto& pipeline=*prepared.native_pipeline;
            const auto descriptor_index=prepared.first_descriptor;
            const auto gpu_start=submission.heap->GetGPUDescriptorHandleForHeapStart();
            if(!descriptor_table_written[table]) {
                descriptor_table_written[table]=true;
                const auto cpu_start=submission.heap->GetCPUDescriptorHandleForHeapStart();
                for (UINT slot = 0; slot < pipeline.bindings; ++slot) {
                    const auto& next=prepared.views[slot];
                    const auto kind = pipeline.view_kinds[slot];
                    const bool read_only = kind == StorageBufferViewKind::RawRead ||
                                           kind == StorageBufferViewKind::Structured32Read;
                    const bool raw = kind == StorageBufferViewKind::RawRead ||
                                     kind == StorageBufferViewKind::RawReadWrite;
                    const auto physical_slot = descriptor_index +
                        pipeline.descriptor_slots[slot];
                    DescriptorView& cached = submission.cached_views[physical_slot];
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
                        view.Buffer.FirstElement = next.offset / 4;
                        view.Buffer.NumElements = static_cast<UINT>(next.bytes / 4);
                        view.Buffer.StructureByteStride = raw ? 0u : 4u;
                        view.Buffer.Flags = raw ? D3D12_BUFFER_UAV_FLAG_RAW :
                                                   D3D12_BUFFER_UAV_FLAG_NONE;
                        device->CreateUnorderedAccessView(next.resource, nullptr, &view, cpu);
                    } else {
                        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
                        view.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
                        view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                        view.Buffer.FirstElement = next.offset / 4;
                        view.Buffer.NumElements = static_cast<UINT>(next.bytes / 4);
                        view.Buffer.StructureByteStride = raw ? 0u : 4u;
                        view.Buffer.Flags = raw ? D3D12_BUFFER_SRV_FLAG_RAW :
                                                  D3D12_BUFFER_SRV_FLAG_NONE;
                        device->CreateShaderResourceView(next.resource, &view, cpu);
                    }
                    cached = next;
                }
            }
            if(bound_pipeline!=cmd.pipeline) {
                submission.commands->SetComputeRootSignature(pipeline.signature.Get());
                submission.commands->SetPipelineState(pipeline.state.Get());
                bound_pipeline=cmd.pipeline;
                bound_table=std::numeric_limits<std::size_t>::max();
            }
            UINT table_root=0u;
            if(pipeline.srv_count) {
                if(bound_table!=table)
                    submission.commands->SetComputeRootDescriptorTable(table_root,{
                        gpu_start.ptr+descriptor_index*increment});
                ++table_root;
            }
            if(pipeline.uav_count) {
                if(bound_table!=table)
                    submission.commands->SetComputeRootDescriptorTable(table_root,{
                        gpu_start.ptr+(descriptor_index+pipeline.srv_count)*increment});
                ++table_root;
            }
            bound_table=table;
            submission.commands->SetComputeRoot32BitConstants(
                table_root,pipeline.parameters/4u,cmd.push_constants.data(),0);
            if(cmd.indirect) {
                auto* arguments=resource_access[resource_ordinals.at(cmd.indirect->buffer)].buffer->resource.Get();
                // The X group count also predicates the single indirect
                // command, so an empty GPU worklist launches no dispatch.
                submission.commands->ExecuteIndirect(
                    dispatch_signature.Get(),1u,
                    arguments,cmd.indirect->offset,
                    arguments,cmd.indirect->offset);
            }else
                submission.commands->Dispatch(
                    cmd.groups.x,cmd.groups.y,cmd.groups.z);
#if defined(FYZZXYZZY_PROFILE_GPU)
            if(hardware_trace)
                submission.commands->EndQuery(submission.query_heap.Get(),
                    D3D12_QUERY_TYPE_TIMESTAMP,hardware_query_index++);
#endif
        }
        if(readback_tail) {
            std::size_t offset=0u;
            for(const auto& item:readback_tail->sources) {
                before_operation();
                if(offset!=0u)
                    order_copy_writes(submission.commands.Get(),readback_tail->staging);
                submission.commands->CopyBufferRegion(
                    readback_tail->staging,offset,
                    resource_access[resource_ordinals.at(item.source)].buffer->resource.Get(),item.source_offset,
                    item.bytes);
                offset+=item.bytes;
            }
        }
        if (profile) {
#if defined(FYZZXYZZY_PROFILE_GPU)
            if(!hardware_trace)
#endif
            submission.commands->EndQuery(submission.query_heap.Get(),
                                           D3D12_QUERY_TYPE_TIMESTAMP, 1);
            submission.commands->ResolveQueryData(
                submission.query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
#if defined(FYZZXYZZY_PROFILE_GPU)
                0,hardware_trace?static_cast<UINT>(commands.size()+1u):2u,
#else
                0, 2,
#endif
                submission.query_readback.Get(), 0);
        }
#if defined(FYZZXYZZY_PROFILE_GPU)
        if(hardware_trace) {
            FYZZ_PROFILE_COUNTER("d3d12.barrier_calls",barrier_calls);
            FYZZ_PROFILE_COUNTER("d3d12.buffer_barriers",barriers.size());
            submission.profile_trace_id=hardware_profile::current_trace_id();
            submission.profile_ops.reserve(commands.size());
            for(const auto& operation:commands) {
                if(const auto* copy=std::get_if<CopyCommand>(&operation))
                    submission.profile_ops.push_back(
                        {0u,copy->bytes,0u,0u,0u,false});
                else if(const auto* upload=std::get_if<UploadCommand>(&operation))
                    submission.profile_ops.push_back(
                        {0u,upload->bytes.size(),0u,0u,0u,false});
                else {
                    const auto& dispatch=std::get<DispatchCommand>(operation);
                    submission.profile_ops.push_back({
                        dispatch.pipeline,0u,dispatch.groups.x,dispatch.groups.y,
                        dispatch.groups.z,dispatch.indirect.has_value()});
                }
            }
        }
#endif
        return enqueue(std::move(submission),referenced);
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
        recycle_completed(std::move(it->second));
        submissions.erase(it);
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
    const auto id=impl_->create_buffer(bytes, placement);
#if defined(FYZZXYZZY_PROFILE_GPU)
    hardware_profile::buffer_event({
        "d3d12","create",impl_->profile_context_id,
        hardware_profile::current_trace_id(),id,bytes,unsigned(placement)});
#endif
    return id;
}
void Context::destroy_buffer(BufferId id) {
#if defined(FYZZXYZZY_PROFILE_GPU)
    const auto bytes=impl_->get_buffer(id).bytes;
#endif
    impl_->destroy_buffer(id);
#if defined(FYZZXYZZY_PROFILE_GPU)
    hardware_profile::buffer_event({
        "d3d12","destroy",impl_->profile_context_id,
        hardware_profile::current_trace_id(),id,bytes,0u});
#endif
}
void Context::retire_buffer(BufferId id) {
#if defined(FYZZXYZZY_PROFILE_GPU)
    const auto bytes=impl_->get_buffer(id).bytes;
#endif
    impl_->retire_buffer(id);
#if defined(FYZZXYZZY_PROFILE_GPU)
    hardware_profile::buffer_event({
        "d3d12","retire",impl_->profile_context_id,
        hardware_profile::current_trace_id(),id,bytes,0u});
#endif
}
bool Context::buffer_is_host_visible(BufferId id) const {
    return impl_->get_buffer(id).host_visible;
}
bool Context::buffer_in_use(BufferId id) const {
    return impl_->get_buffer(id).last_reference_fence>impl_->completed_fence;
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

SubmissionId Context::submit_download_batch(
        std::span<const DownloadSpan> downloads) {
    return impl_->submit_download_batch(downloads);
}
SubmissionId Context::submit_batch_with_readback(
        std::vector<BatchCommand>& commands,
        std::span<const DownloadSpan> downloads) {
    return impl_->submit_download_batch(downloads,commands);
}
void Context::finish_download_batch(SubmissionId id,bool publish) {
    impl_->finish_download_batch(id,publish);
}
bool Context::cancel_download_batch(SubmissionId id) {
    return impl_->cancel_download_batch(id);
}
void Context::download_batch(std::span<const DownloadSpan> downloads) const {
    impl_->download_batch(downloads);
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
#if defined(FYZZXYZZY_PROFILE_GPU)
void Context::set_pipeline_profile_label(PipelineId id,const char* label) const noexcept {
    hardware_profile::pipeline_event({
        "d3d12",label,impl_->profile_context_id,id});
}
#endif
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
    const auto completed=impl_->fence->GetCompletedValue();
    if(completed==std::numeric_limits<std::uint64_t>::max())
        throw std::runtime_error("D3D12 device removed during completion query");
    if(impl_->first_unsignaled_fence &&
       it->second.fence_value>=impl_->first_unsignaled_fence)
        throw std::runtime_error("D3D12 submission has no completion fence");
    impl_->observe_completion(completed);
    return completed>=it->second.fence_value;
}
void Context::wait_submission(SubmissionId id) { impl_->wait_submission(id); }
void Context::wait_submissions(std::span<const SubmissionId> ordered) {
    impl_->wait_submissions(ordered);
}
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
