// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Native D3D12 bring-up of the same Context contract used by the PhysX AABB
// stage on Vulkan. This backend currently supports 3 structured SRVs + 1
// structured UAV and eight 32-bit root constants; it is deliberately not yet
// a general lowering of PhysX's CUDA pointer-rich GPU descriptors.

#include "fyzzxyzzy/compute.h"

#ifndef _WIN32
#error D3D12 compute is Windows-only
#endif

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
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
    };
    struct Pipeline {
        ComPtr<ID3D12RootSignature> signature;
        ComPtr<ID3D12PipelineState> state;
        std::uint32_t bindings = 0;
        std::uint32_t parameters = 0;
    };
    struct Submission {
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> commands;
        ComPtr<ID3D12DescriptorHeap> heap;
        std::uint64_t fence_value = 0;
    };

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    DeviceInfo info;
    std::unordered_map<BufferId, Buffer> buffers;
    std::unordered_map<PipelineId, Pipeline> pipelines;
    std::unordered_map<SubmissionId, Submission> submissions;
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
        check(D3D12CreateDevice(adapters[ordinal].adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                IID_PPV_ARGS(&device)), "D3D12CreateDevice");
        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                  IID_PPV_ARGS(&fence)), "CreateFence");
    }

    ~Impl() {
        if (queue && fence) {
            try { wait_all(); } catch (...) {}
        }
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

    Submission new_commands() {
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

    SubmissionId enqueue(Submission&& submission) {
        check(submission.commands->Close(), "Close command list");
        ID3D12CommandList* lists[] = {submission.commands.Get()};
        queue->ExecuteCommandLists(1, lists);
        const std::uint64_t signal = next_fence++;
        check(queue->Signal(fence.Get(), signal), "Signal D3D12 fence");
        const SubmissionId id = next_submission++;
        submission.fence_value = signal;
        submissions.emplace(id, std::move(submission));
        return id;
    }

    void wait_value(std::uint64_t value) {
        if (fence->GetCompletedValue() >= value) return;
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) throw std::runtime_error("CreateEvent failed for D3D12 fence");
        HRESULT status = fence->SetEventOnCompletion(value, event);
        if (FAILED(status)) {
            CloseHandle(event);
            check(status, "SetEventOnCompletion");
        }
        DWORD result = WaitForSingleObject(event, INFINITE);
        CloseHandle(event);
        if (result != WAIT_OBJECT_0)
            throw std::runtime_error("D3D12 fence wait failed");
    }

    void wait_submission(SubmissionId id) {
        auto it = submissions.find(id);
        if (it == submissions.end()) throw std::invalid_argument("Unknown D3D12 submission");
        wait_value(it->second.fence_value);
        submissions.erase(it);
    }

    void wait_all() {
        if (submissions.empty()) return;
        wait_value(next_fence - 1);
        submissions.clear();
    }

    BufferId create_buffer(std::size_t bytes, MemoryPlacement /*placement*/) {
        if (!bytes) throw std::invalid_argument("Compute buffer size must be positive");
        Buffer buffer;
        buffer.resource = allocate_resource(device.Get(), bytes, D3D12_HEAP_TYPE_DEFAULT,
                                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                             D3D12_RESOURCE_STATE_COMMON);
        buffer.bytes = bytes;
        const BufferId id = next_buffer++;
        buffers.emplace(id, std::move(buffer));
        return id;
    }

    void destroy_buffer(BufferId id) {
        get_buffer(id);
        wait_all(); // resources cannot die while a GPU command refers to them
        buffers.erase(id);
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
            (desc.storage_buffer_binding_count != 1 &&
             desc.storage_buffer_binding_count != 4) || desc.push_constant_bytes != 32)
            throw std::invalid_argument("D3D12 supports single-UAV arena or 3 SRV + 1 UAV PhysX stages with eight root constants");
        const bool arena = desc.storage_buffer_binding_count == 1;

        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = 3;
        ranges[0].BaseShaderRegister = 0;
        ranges[0].RegisterSpace = 0;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = 1;
        ranges[1].BaseShaderRegister = 0;
        ranges[1].RegisterSpace = 0;
        ranges[1].OffsetInDescriptorsFromTableStart = 0;
        D3D12_ROOT_PARAMETER roots[3]{};
        roots[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        roots[0].DescriptorTable.NumDescriptorRanges = 1;
        roots[0].DescriptorTable.pDescriptorRanges = arena ? &ranges[1] : &ranges[0];
        roots[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        roots[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        roots[1].DescriptorTable.NumDescriptorRanges = 1;
        roots[1].DescriptorTable.pDescriptorRanges = &ranges[1];
        roots[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        roots[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        roots[2].Constants.ShaderRegister = 0;
        roots[2].Constants.RegisterSpace = 0;
        roots[2].Constants.Num32BitValues = 8;
        roots[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        // Arena: UAV table at root 0, 8x32-bit constants at root 1.
        // AABB: SRV table 0, UAV table 1, constants at root 2.
        if (arena) roots[1] = roots[2];
        D3D12_ROOT_SIGNATURE_DESC sig{};
        sig.NumParameters = arena ? 2 : 3;
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
        Pipeline pipeline;
        check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                          IID_PPV_ARGS(&pipeline.signature)), "CreateRootSignature");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = pipeline.signature.Get();
        pso.CS = {desc.shader.bytes.data(), desc.shader.bytes.size()};
        check(device->CreateComputePipelineState(&pso,
                                                 IID_PPV_ARGS(&pipeline.state)),
              "CreateComputePipelineState");
        pipeline.bindings = desc.storage_buffer_binding_count;
        pipeline.parameters = 32;
        const PipelineId id = next_pipeline++;
        pipelines.emplace(id, std::move(pipeline));
        return id;
    }

    void destroy_pipeline(PipelineId id) {
        get_pipeline(id);
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
            const auto& buf = get_buffer(binding.buffer);
            validate_range(buf.bytes, binding.offset, binding.bytes);
        }
        if (pipe.bindings == 4 &&
            (cmd.buffers[3].buffer == cmd.buffers[0].buffer ||
             cmd.buffers[3].buffer == cmd.buffers[1].buffer ||
             cmd.buffers[3].buffer == cmd.buffers[2].buffer))
            throw std::invalid_argument("D3D12 AABB shader cannot alias its SRVs and UAV");
    }

    SubmissionId submit_batch(const std::vector<BatchCommand>& commands) {
        if (commands.empty()) throw std::invalid_argument("Empty D3D12 command batch");
        std::size_t descriptor_count = 0;
        for (const BatchCommand& op : commands) {
            if (const auto* copy = std::get_if<CopyCommand>(&op)) {
                const auto& source = get_buffer(copy->source);
                const auto& dest = get_buffer(copy->destination);
                validate_range(source.bytes, copy->source_offset, copy->bytes);
                validate_range(dest.bytes, copy->destination_offset, copy->bytes);
                if (!copy->bytes || copy->source == copy->destination ||
                    copy->source_offset % 4 || copy->destination_offset % 4 ||
                    copy->bytes % 4)
                    throw std::invalid_argument("Invalid D3D12 buffer copy");
            } else {
                validate_dispatch(std::get<DispatchCommand>(op));
                descriptor_count += get_pipeline(std::get<DispatchCommand>(op).pipeline).bindings;
            }
        }
        if (descriptor_count > std::numeric_limits<UINT>::max())
            throw std::out_of_range("D3D12 descriptor heap capacity exceeded");
        Submission submission = new_commands();
        UINT increment = device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        if (descriptor_count) {
            D3D12_DESCRIPTOR_HEAP_DESC heap{};
            heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            heap.NumDescriptors = static_cast<UINT>(descriptor_count);
            heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            check(device->CreateDescriptorHeap(&heap,
                                               IID_PPV_ARGS(&submission.heap)),
                  "CreateDescriptorHeap");
            ID3D12DescriptorHeap* heaps[] = {submission.heap.Get()};
            submission.commands->SetDescriptorHeaps(1, heaps);
        }
        UINT descriptor_index = 0;
        for (const BatchCommand& op : commands) {
            if (const auto* copy = std::get_if<CopyCommand>(&op)) {
                auto& src = get_buffer(copy->source);
                auto& dst = get_buffer(copy->destination);
                transition(submission.commands.Get(), src.resource.Get(),
                           D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
                transition(submission.commands.Get(), dst.resource.Get(),
                           D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                submission.commands->CopyBufferRegion(dst.resource.Get(), copy->destination_offset,
                                                       src.resource.Get(), copy->source_offset,
                                                       copy->bytes);
                transition(submission.commands.Get(), src.resource.Get(),
                           D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
                transition(submission.commands.Get(), dst.resource.Get(),
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
                continue;
            }
            const auto& cmd = std::get<DispatchCommand>(op);
            Pipeline& pipeline = get_pipeline(cmd.pipeline);
            std::unordered_set<BufferId> transitioned;
            for (std::size_t slot = 0; slot < cmd.buffers.size(); ++slot) {
                const auto& binding = cmd.buffers[slot];
                if (!transitioned.insert(binding.buffer).second) continue;
                auto desired = (pipeline.bindings == 1 || slot == 3) ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                         : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                transition(submission.commands.Get(), get_buffer(binding.buffer).resource.Get(),
                           D3D12_RESOURCE_STATE_COMMON, desired);
            }

            const auto cpu_start = submission.heap->GetCPUDescriptorHandleForHeapStart();
            const auto gpu_start = submission.heap->GetGPUDescriptorHandleForHeapStart();
            for (UINT slot = 0; slot < pipeline.bindings; ++slot) {
                const auto& binding = cmd.buffers[slot];
                auto& buffer = get_buffer(binding.buffer);
                D3D12_CPU_DESCRIPTOR_HANDLE cpu{
                    cpu_start.ptr + (descriptor_index + slot) * increment
                };
                if (pipeline.bindings == 1 || slot == 3) {
                    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
                    // RWByteAddressBuffer requires a RAW/typeless UAV. The
                    // original AABB RWStructuredBuffer<uint> uses a structured
                    // UAV instead. WARP may tolerate the wrong view type, but
                    // real D3D12 hardware need not.
                    const bool raw_arena = pipeline.bindings == 1;
                    view.Format = raw_arena ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
                    view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                    view.Buffer.FirstElement = binding.offset / 4;
                    view.Buffer.NumElements = static_cast<UINT>(binding.bytes / 4);
                    view.Buffer.StructureByteStride = raw_arena ? 0 : 4;
                    view.Buffer.Flags = raw_arena ? D3D12_BUFFER_UAV_FLAG_RAW :
                                                   D3D12_BUFFER_UAV_FLAG_NONE;
                    device->CreateUnorderedAccessView(buffer.resource.Get(), nullptr, &view, cpu);
                } else {
                    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
                    view.Format = DXGI_FORMAT_UNKNOWN;
                    view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    view.Buffer.FirstElement = binding.offset / 4;
                    view.Buffer.NumElements = static_cast<UINT>(binding.bytes / 4);
                    view.Buffer.StructureByteStride = 4;
                    device->CreateShaderResourceView(buffer.resource.Get(), &view, cpu);
                }
            }
            submission.commands->SetComputeRootSignature(pipeline.signature.Get());
            submission.commands->SetPipelineState(pipeline.state.Get());
            submission.commands->SetComputeRootDescriptorTable(0, {
                gpu_start.ptr + descriptor_index * increment
            });
            if (pipeline.bindings == 4) {
                submission.commands->SetComputeRootDescriptorTable(1, {
                    gpu_start.ptr + (descriptor_index + 3) * increment
                });
            }
            submission.commands->SetComputeRoot32BitConstants(
                pipeline.bindings == 1 ? 1 : 2, 8, cmd.push_constants.data(), 0);
            submission.commands->Dispatch(cmd.groups.x, cmd.groups.y, cmd.groups.z);
            D3D12_RESOURCE_BARRIER uav{};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource = get_buffer(cmd.buffers[pipeline.bindings - 1].buffer).resource.Get();
            submission.commands->ResourceBarrier(1, &uav);
            transitioned.clear();
            for (std::size_t slot = 0; slot < cmd.buffers.size(); ++slot) {
                auto id = cmd.buffers[slot].buffer;
                if (!transitioned.insert(id).second) continue;
                transition(submission.commands.Get(), get_buffer(id).resource.Get(),
                           (pipeline.bindings == 1 || slot == 3) ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                     : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_COMMON);
            }
            descriptor_index += pipeline.bindings;
        }
        return enqueue(std::move(submission));
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
void Context::upload(BufferId id, std::size_t offset, const void* data, std::size_t bytes) {
    impl_->upload(id, offset, data, bytes);
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
void Context::execute_batch(const std::vector<BatchCommand>& commands) {
    impl_->wait_submission(impl_->submit_batch(commands));
}
void Context::wait_all() { impl_->wait_all(); }

} // namespace fyzz::compute
