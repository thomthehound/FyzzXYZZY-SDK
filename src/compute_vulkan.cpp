// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"

#include <vulkan/vulkan.h>

#include <array>

#include <cstring>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fyzz::compute {
namespace {

void require(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed (VkResult " +
                                 std::to_string(static_cast<int>(result)) + ")");
    }
}

VkInstance create_instance() {
    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "FyzzXYZZY";
    application.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create.pApplicationInfo = &application;
    VkInstance instance = VK_NULL_HANDLE;
    require(vkCreateInstance(&create, nullptr, &instance), "vkCreateInstance");
    return instance;
}

struct DeviceCandidate {
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    DeviceInfo info;
};

std::vector<DeviceCandidate> find_compute_devices(VkInstance instance) {
    std::uint32_t count = 0;
    require(vkEnumeratePhysicalDevices(instance, &count, nullptr),
            "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> devices(count);
    if (count > 0) {
        require(vkEnumeratePhysicalDevices(instance, &count, devices.data()),
                "vkEnumeratePhysicalDevices");
    }
    std::vector<DeviceCandidate> result;
    for (VkPhysicalDevice physical : devices) {
        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        if (family_count > 0) {
            vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
        }
        for (std::uint32_t i = 0; i < family_count; ++i) {
            if (families[i].queueCount == 0 ||
                !(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                continue;
            }
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(physical, &properties);
            const auto& limits = properties.limits;
            DeviceInfo info;
            info.ordinal = static_cast<std::uint32_t>(result.size());
            info.name = properties.deviceName;
            info.vendor_id = properties.vendorID;
            info.device_id = properties.deviceID;
            info.api_version = properties.apiVersion;
            info.max_storage_buffer_range = limits.maxStorageBufferRange;
            info.max_workgroup_invocations = limits.maxComputeWorkGroupInvocations;
            for (std::size_t j = 0; j < 3; ++j) {
                info.max_workgroup_count[j] = limits.maxComputeWorkGroupCount[j];
                info.max_workgroup_size[j] = limits.maxComputeWorkGroupSize[j];
            }
            result.push_back({physical, i, std::move(info)});
            break;
        }
    }
    return result;
}

void validate_range(VkDeviceSize size, std::size_t offset, std::size_t length) {
    if (offset > size || length > size - offset) {
        throw std::out_of_range("Compute buffer range exceeds allocation");
    }
}

} // namespace

std::vector<DeviceInfo> enumerate_devices(Backend backend) {
    if (backend != Backend::Vulkan) {
        throw std::invalid_argument("Unsupported compute backend");
    }
    VkInstance instance = create_instance();
    try {
        auto candidates = find_compute_devices(instance);
        std::vector<DeviceInfo> result;
        result.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            result.push_back(candidate.info);
        }
        vkDestroyInstance(instance, nullptr);
        return result;
    } catch (...) {
        vkDestroyInstance(instance, nullptr);
        throw;
    }
}

struct Context::Impl {
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize bytes = 0;
        MemoryPlacement placement = MemoryPlacement::HostVisible;
        // Transfer-only staging may exceed maxStorageBufferRange, which
        // constrains a bound shader descriptor, not a transfer source.
        bool storage_eligible = true;
    };
    struct Pipeline {
        VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        std::uint32_t binding_count = 0;
        std::uint32_t push_bytes = 0;
    };
    struct Submission {
        VkCommandBuffer commands = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkQueryPool timestamps = VK_NULL_HANDLE;
        bool cacheable_batch = false;
        std::uint32_t descriptor_capacity = 0;
        std::uint32_t set_capacity = 0;
    };
    struct CachedBatch {
        VkCommandBuffer commands = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        std::uint32_t descriptor_capacity = 0;
        std::uint32_t set_capacity = 0;
    };
    static constexpr std::size_t kMaxCachedBatches = 8;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    DeviceInfo info;
    std::unordered_map<BufferId, Buffer> buffers;
    std::unordered_map<PipelineId, Pipeline> pipelines;
    std::unordered_map<SubmissionId, Submission> submissions;
    struct RetiredBuffer {
        SubmissionId release_after;
        Buffer buffer;
    };
    std::vector<RetiredBuffer> retired_buffers;
    // A fixed-capacity, non-allocating cache. Retirement happens after GPU
    // completion and must never fail from host allocation pressure.
    std::array<CachedBatch, kMaxCachedBatches> cached_batches{};
    std::size_t cached_batch_count = 0;
    BufferId next_buffer = 1;
    PipelineId next_pipeline = 1;
    SubmissionId next_submission = 1;
    VkPhysicalDeviceLimits limits{};
    std::uint32_t timestamp_valid_bits = 0;

    explicit Impl(Backend backend, std::uint32_t ordinal) {
        if (backend != Backend::Vulkan) {
            throw std::invalid_argument("Unsupported compute backend");
        }
        instance = create_instance();
        try {
            auto candidates = find_compute_devices(instance);
            if (ordinal >= candidates.size()) {
                throw std::out_of_range("Requested Vulkan compute device does not exist");
            }
            const auto& selected = candidates[ordinal];
            physical = selected.physical;
            info = selected.info;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(physical, &properties);
            limits = properties.limits;
            std::uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
            timestamp_valid_bits = families.at(selected.queue_family).timestampValidBits;

            float priority = 1.0f;
            VkDeviceQueueCreateInfo queue_info{};
            queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_info.queueFamilyIndex = selected.queue_family;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &priority;

            VkDeviceCreateInfo create{};
            create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            create.queueCreateInfoCount = 1;
            create.pQueueCreateInfos = &queue_info;
            require(vkCreateDevice(physical, &create, nullptr, &device), "vkCreateDevice");
            vkGetDeviceQueue(device, selected.queue_family, 0, &queue);

            VkCommandPoolCreateInfo pool{};
            pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                         VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool.queueFamilyIndex = selected.queue_family;
            require(vkCreateCommandPool(device, &pool, nullptr, &command_pool),
                    "vkCreateCommandPool");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    void cleanup() noexcept {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            for (const auto& item : submissions) {
                release_submission(item.second);
            }
            submissions.clear();
            // vkDeviceWaitIdle above guarantees the last submission has
            // completed; no retired GPU allocation may survive device teardown.
            for (const auto& retired : retired_buffers)
                release_buffer(retired.buffer);
            retired_buffers.clear();
            for (std::size_t i = 0; i < cached_batch_count; ++i) {
                const auto& cache = cached_batches[i];
                if (cache.descriptor_pool != VK_NULL_HANDLE)
                    vkDestroyDescriptorPool(device, cache.descriptor_pool, nullptr);
                if (cache.commands != VK_NULL_HANDLE)
                    vkFreeCommandBuffers(device, command_pool, 1, &cache.commands);
            }
            cached_batch_count = 0;
            for (auto& item : pipelines) {
                destroy_pipeline_handles(item.second);
            }
            pipelines.clear();
            for (auto& item : buffers) {
                vkDestroyBuffer(device, item.second.buffer, nullptr);
                vkFreeMemory(device, item.second.memory, nullptr);
            }
            buffers.clear();
            if (command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, command_pool, nullptr);
                command_pool = VK_NULL_HANDLE;
            }
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    }

    ~Impl() { cleanup(); }

    void release_submission(const Submission& submission) noexcept {
        if (submission.timestamps != VK_NULL_HANDLE)
            vkDestroyQueryPool(device, submission.timestamps, nullptr);
        if (submission.fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, submission.fence, nullptr);
        }
        if (submission.commands != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, command_pool, 1, &submission.commands);
        }
        if (submission.descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, submission.descriptor_pool, nullptr);
        }
    }

    void retire_submission(const Submission& submission) noexcept {
        // No host allocation, and Vulkan destroy/free APIs return void.
        // This cannot leave a live submission retaining destroyed handles
        // because retirement itself can no longer throw.
        if (submission.timestamps != VK_NULL_HANDLE)
            vkDestroyQueryPool(device, submission.timestamps, nullptr);
        if (submission.fence != VK_NULL_HANDLE)
            vkDestroyFence(device, submission.fence, nullptr);
        if (!submission.cacheable_batch ||
            cached_batch_count >= kMaxCachedBatches) {
            if (submission.commands != VK_NULL_HANDLE)
                vkFreeCommandBuffers(device, command_pool, 1, &submission.commands);
            if (submission.descriptor_pool != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(device, submission.descriptor_pool, nullptr);
            return;
        }
        cached_batches[cached_batch_count++] = {
            submission.commands, submission.descriptor_pool,
            submission.descriptor_capacity, submission.set_capacity};
    }

    SubmissionId enqueue(VkCommandBuffer commands, VkDescriptorPool descriptor_pool) {
        if (next_submission == std::numeric_limits<SubmissionId>::max()) {
            throw std::overflow_error("Compute submission ID range exhausted");
        }
        VkFenceCreateInfo fence_create{};
        fence_create.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        require(vkCreateFence(device, &fence_create, nullptr, &fence), "vkCreateFence");
        const SubmissionId id = next_submission;
        try {
            submissions.emplace(id, Submission{commands, descriptor_pool, fence});
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &commands;
            const VkResult status = vkQueueSubmit(queue, 1, &submit, fence);
            if (status != VK_SUCCESS) {
                submissions.erase(id);
                require(status, "vkQueueSubmit");
            }
        } catch (...) {
            vkDestroyFence(device, fence, nullptr);
            throw;
        }
        ++next_submission;
        return id;
    }

    void release_retired_through(SubmissionId completed_id) noexcept {
        // The Context uses one ordered Vulkan queue. Completion of N
        // therefore makes every submission <= N safe to release, even if a
        // caller waits on N before earlier submission IDs.
        auto it = retired_buffers.begin();
        while (it != retired_buffers.end()) {
            if (it->release_after <= completed_id) {
                release_buffer(it->buffer);
                it = retired_buffers.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool submission_complete(SubmissionId id) const {
        const auto it = submissions.find(id);
        if (it == submissions.end()) {
            throw std::out_of_range("Unknown compute submission ID");
        }
        const VkResult status = vkGetFenceStatus(device, it->second.fence);
        if (status == VK_NOT_READY) {
            return false;
        }
        require(status, "vkGetFenceStatus");
        return true;
    }

    void wait_submission(SubmissionId id) {
        auto it = submissions.find(id);
        if (it == submissions.end()) {
            throw std::out_of_range("Unknown compute submission ID");
        }
        const VkFence fence = it->second.fence;
        require(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
                "vkWaitForFences");
        retire_submission(it->second);
        submissions.erase(it);
        release_retired_through(id);
    }

    void wait_all() {
        while (!submissions.empty()) {
            wait_submission(submissions.begin()->first);
        }
    }

    void destroy_pipeline_handles(Pipeline& pipeline) noexcept {
        if (pipeline.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline.pipeline, nullptr);
            pipeline.pipeline = VK_NULL_HANDLE;
        }
        if (pipeline.layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipeline.layout, nullptr);
            pipeline.layout = VK_NULL_HANDLE;
        }
        if (pipeline.descriptors != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, pipeline.descriptors, nullptr);
            pipeline.descriptors = VK_NULL_HANDLE;
        }
    }

    Buffer& get(BufferId id) {
        auto it = buffers.find(id);
        if (it == buffers.end()) {
            throw std::out_of_range("Unknown compute buffer ID");
        }
        return it->second;
    }

    const Buffer& get(BufferId id) const {
        auto it = buffers.find(id);
        if (it == buffers.end()) {
            throw std::out_of_range("Unknown compute buffer ID");
        }
        return it->second;
    }

    void release_buffer(const Buffer& resource) const noexcept {
        if (resource.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, resource.buffer, nullptr);
        }
        if (resource.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, resource.memory, nullptr);
        }
    }

    Buffer make_buffer(std::size_t bytes, MemoryPlacement placement,
                       bool storage_eligible = true) const {
        Buffer resource;
        resource.placement = placement;
        resource.storage_eligible = storage_eligible;
        try {
            VkBufferCreateInfo create{};
            create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            create.size = static_cast<VkDeviceSize>(bytes);
            create.usage = storage_eligible
                ? (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                : VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            require(vkCreateBuffer(device, &create, nullptr, &resource.buffer),
                    "vkCreateBuffer");

            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device, resource.buffer, &requirements);
            VkPhysicalDeviceMemoryProperties properties{};
            vkGetPhysicalDeviceMemoryProperties(physical, &properties);
            const VkMemoryPropertyFlags required = placement == MemoryPlacement::HostVisible
                ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            std::uint32_t type_index = properties.memoryTypeCount;
            // Prefer non-mappable device-local memory when available.
            for (std::uint32_t pass = 0; pass < 2; ++pass) {
                for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
                    const VkMemoryPropertyFlags flags = properties.memoryTypes[i].propertyFlags;
                    if ((requirements.memoryTypeBits & (1u << i)) == 0 ||
                        (flags & required) != required) {
                        continue;
                    }
                    if (placement == MemoryPlacement::DeviceLocal && pass == 0 &&
                        (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
                        continue;
                    }
                    type_index = i;
                    break;
                }
                if (type_index != properties.memoryTypeCount) {
                    break;
                }
            }
            if (type_index == properties.memoryTypeCount) {
                throw std::runtime_error("No compatible Vulkan memory type for buffer placement");
            }

            VkMemoryAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = type_index;
            require(vkAllocateMemory(device, &allocation, nullptr, &resource.memory),
                    "vkAllocateMemory");
            require(vkBindBufferMemory(device, resource.buffer, resource.memory, 0),
                    "vkBindBufferMemory");
            resource.bytes = bytes;
            return resource;
        } catch (...) {
            release_buffer(resource);
            throw;
        }
    }

    BufferId create_buffer(std::size_t bytes, MemoryPlacement placement,
                           bool storage_eligible = true) {
        if (!bytes || (storage_eligible && bytes > info.max_storage_buffer_range)) {
            throw std::invalid_argument("Compute buffer size is zero or exceeds device limits");
        }
        // maxStorageBufferRange applies only to descriptors. Internal
        // transfer-only buffers are instead bounded by physical VkBuffer/
        // allocation support (vkCreateBuffer/vkAllocateMemory failures).
        if (static_cast<std::uint64_t>(bytes) != static_cast<VkDeviceSize>(bytes))
            throw std::out_of_range("Vulkan transfer byte count exceeds VkDeviceSize");
        if (next_buffer == std::numeric_limits<BufferId>::max()) {
            throw std::overflow_error("Compute buffer ID range exhausted");
        }
        const Buffer buffer = make_buffer(bytes, placement, storage_eligible);
        try {
            const BufferId id = next_buffer;
            buffers.emplace(id, buffer);
            ++next_buffer;
            return id;
        } catch (...) {
            release_buffer(buffer);
            throw;
        }
    }

    void destroy_buffer(BufferId id) {
        wait_all();
        const auto it = buffers.find(id);
        if (it == buffers.end()) {
            throw std::out_of_range("Unknown compute buffer ID");
        }
        release_buffer(it->second);
        buffers.erase(it);
    }

    void retire_buffer(BufferId id) {
        const auto it = buffers.find(id);
        if (it == buffers.end())
            throw std::out_of_range("Unknown compute buffer ID");
        if (submissions.empty()) {
            // No tracked in-flight work. Any earlier submitted work was
            // already waited on the ordered queue.
            release_buffer(it->second);
        } else {
            SubmissionId last = 0;
            for (const auto& pending : submissions)
                last = std::max(last, pending.first);
            // Queue-retirement record must be secured *before* invalidating
            // the public BufferId, preserving ownership on allocation failure.
            retired_buffers.push_back({last, it->second});
        }
        buffers.erase(it);
    }

    void map_transfer(const Buffer& buffer, std::size_t offset,
                      void* data, std::size_t bytes, bool upload) const {
        void* mapped = nullptr;
        // Map from offset zero to avoid minMemoryMapAlignment restrictions.
        require(vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &mapped),
                "vkMapMemory");
        unsigned char* location = static_cast<unsigned char*>(mapped) + offset;
        if (upload) {
            std::memcpy(location, data, bytes);
        } else {
            std::memcpy(data, location, bytes);
        }
        vkUnmapMemory(device, buffer.memory);
    }

    SubmissionId submit_copy(const Buffer& source, std::size_t source_offset,
                     const Buffer& destination, std::size_t destination_offset,
                     std::size_t bytes) {
        if (source_offset % 4 != 0 || destination_offset % 4 != 0 || bytes % 4 != 0) {
            throw std::invalid_argument("Vulkan buffer copies require 4-byte alignment");
        }
        VkCommandBufferAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocation.commandPool = command_pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkCommandBuffer commands = VK_NULL_HANDLE;
        require(vkAllocateCommandBuffers(device, &allocation, &commands),
                "vkAllocateCommandBuffers");
        try {
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            require(vkBeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");

            // Make prior shader/transfer writes visible to the transfer read.
            VkMemoryBarrier before{};
            before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 1, &before, 0, nullptr, 0, nullptr);

            VkBufferCopy region{};
            region.srcOffset = source_offset;
            region.dstOffset = destination_offset;
            region.size = bytes;
            vkCmdCopyBuffer(commands, source.buffer, destination.buffer, 1, &region);

            // The destination may next be accessed by the host or a shader.
            VkMemoryBarrier after{};
            after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_HOST_READ_BIT |
                                  VK_ACCESS_SHADER_READ_BIT |
                                  VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &after, 0, nullptr, 0, nullptr);
            require(vkEndCommandBuffer(commands), "vkEndCommandBuffer");

            return enqueue(commands, VK_NULL_HANDLE);
        } catch (...) {
            vkFreeCommandBuffers(device, command_pool, 1, &commands);
            throw;
        }
    }

    SubmissionId submit_uploads(const std::vector<UploadSpan>& uploads) {
        if (uploads.empty())
            throw std::invalid_argument("Async Vulkan upload batch cannot be empty");
        std::size_t total = 0;
        // Validate every caller-owned source and GPU destination BEFORE
        // recording GPU work or invalidating any public resource handle.
        for (const auto& upload : uploads) {
            const Buffer& destination = get(upload.destination);
            validate_range(destination.bytes, upload.destination_offset, upload.bytes);
            if (!upload.data || !upload.bytes ||
                (upload.destination_offset % 4) || (upload.bytes % 4) ||
                upload.bytes > std::numeric_limits<std::size_t>::max() - total)
                throw std::invalid_argument("Invalid Vulkan upload span or staging capacity");
            total += upload.bytes;
        }
        // Prevent post-submit allocation failure during deferred staging
        // ownership transfer. In the single-queue model, the last submitted
        // fence captures the lifetime of this entire upload batch.
        retired_buffers.reserve(retired_buffers.size() + 1);
        // The packed staging buffer is never bound to a shader. It is legal
        // for its aggregate size to exceed maxStorageBufferRange even when
        // every public destination remains within that descriptor limit.
        const BufferId staging_id = create_buffer(total, MemoryPlacement::HostVisible, false);
        bool submitted = false;
        try {
            const Buffer& staging = get(staging_id);
            void* mapped = nullptr;
            require(vkMapMemory(device, staging.memory, 0, VK_WHOLE_SIZE, 0, &mapped),
                    "vkMapMemory batched upload");
            std::size_t position = 0;
            for (const auto& upload : uploads) {
                std::memcpy(static_cast<unsigned char*>(mapped) + position,
                            upload.data, upload.bytes);
                position += upload.bytes;
            }
            vkUnmapMemory(device, staging.memory);

            std::vector<BatchCommand> copies;
            copies.reserve(uploads.size());
            position = 0;
            for (const auto& upload : uploads) {
                copies.emplace_back(CopyCommand{
                    staging_id, position, upload.destination,
                    upload.destination_offset, upload.bytes});
                position += upload.bytes;
            }
            const SubmissionId id = submit_batch(copies);
            submitted = true;
            // Guaranteed room in retired_buffers. Retires only the public
            // staging ID, not the memory still referenced by the GPU.
            retire_buffer(staging_id);
            return id;
        } catch (...) {
            // If a GPU command was actually enqueued, the staging handle
            // remains live until its fence; never release backing memory.
            if (!submitted)
                destroy_buffer(staging_id);
            throw;
        }
    }

    SubmissionId submit_upload(BufferId destination_id, std::size_t offset,
                               const void* data, std::size_t bytes) {
        // Snapshot into coherent pinned/mappable memory before submitting;
        // the user's pointer is never retained beyond this call.
        const Buffer& destination = get(destination_id);
        validate_range(destination.bytes, offset, bytes);
        if (!bytes || !data || (offset % 4) || (bytes % 4))
            throw std::invalid_argument("Async Vulkan upload requires nonempty aligned data");
        // Reserve the retirement slot BEFORE queue submission. Otherwise a
        // bad_alloc while retaining staging after submit could free memory
        // that a live GPU copy still references.
        retired_buffers.reserve(retired_buffers.size() + 1);
        const Buffer staging = make_buffer(bytes, MemoryPlacement::HostVisible);
        try {
            map_transfer(staging, 0, const_cast<void*>(data), bytes, true);
            const SubmissionId id = submit_copy(staging, 0, destination, offset, bytes);
            // Buffer is trivially copyable; reserved capacity prevents
            // vector relocation/allocation after the GPU is submitted.
            retired_buffers.push_back({id, staging});
            return id;
        } catch (...) {
            release_buffer(staging);
            throw;
        }
    }

    void transfer(BufferId id, std::size_t offset, void* data,
                  std::size_t bytes, bool upload) {
        wait_all();
        const Buffer& buffer = get(id);
        validate_range(buffer.bytes, offset, bytes);
        if (bytes == 0) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("Null compute transfer data");
        }
        if (buffer.placement == MemoryPlacement::HostVisible) {
            map_transfer(buffer, offset, data, bytes, upload);
            return;
        }
        // Staging uses Vulkan's four-byte buffer-copy alignment.
        if (offset % 4 != 0 || bytes % 4 != 0) {
            throw std::invalid_argument("Device-local transfers require 4-byte alignment");
        }
        const Buffer staging = make_buffer(bytes, MemoryPlacement::HostVisible);
        try {
            if (upload) {
                map_transfer(staging, 0, data, bytes, true);
                wait_submission(submit_copy(staging, 0, buffer, offset, bytes));
            } else {
                wait_submission(submit_copy(buffer, offset, staging, 0, bytes));
                map_transfer(staging, 0, data, bytes, false);
            }
        } catch (...) {
            release_buffer(staging);
            throw;
        }
        release_buffer(staging);
    }

    const Pipeline& get_pipeline(PipelineId id) const {
        const auto it = pipelines.find(id);
        if (it == pipelines.end()) {
            throw std::out_of_range("Unknown compute pipeline ID");
        }
        return it->second;
    }

    PipelineId create_pipeline(const ComputePipelineDesc& desc) {
        if (desc.shader.format != ShaderFormat::Spirv) {
            throw std::invalid_argument("Vulkan compute pipelines require SPIR-V");
        }
        if (desc.shader.bytes.size() < 20 ||
            (desc.shader.bytes.size() % sizeof(std::uint32_t)) != 0) {
            throw std::invalid_argument("Invalid SPIR-V bytecode size");
        }
        std::vector<std::uint32_t> words(desc.shader.bytes.size() / 4);
        std::memcpy(words.data(), desc.shader.bytes.data(), desc.shader.bytes.size());
        if (words[0] != 0x07230203u) {
            throw std::invalid_argument("Invalid SPIR-V magic");
        }
        if (desc.shader.entry_point.empty() ||
            desc.shader.entry_point.find('\0') != std::string::npos) {
            throw std::invalid_argument("Invalid compute shader entry point");
        }
        if (desc.storage_buffer_binding_count > limits.maxPerStageDescriptorStorageBuffers ||
            desc.storage_buffer_binding_count > limits.maxDescriptorSetStorageBuffers) {
            throw std::out_of_range("Storage-buffer binding count exceeds device limits");
        }
        if ((desc.push_constant_bytes % 4u) != 0 ||
            desc.push_constant_bytes > limits.maxPushConstantsSize) {
            throw std::out_of_range("Push constant size exceeds device limits or alignment");
        }
        if (next_pipeline == std::numeric_limits<PipelineId>::max()) {
            throw std::overflow_error("Compute pipeline ID range exhausted");
        }

        Pipeline item;
        VkShaderModule shader = VK_NULL_HANDLE;
        try {
            VkShaderModuleCreateInfo shader_create{};
            shader_create.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shader_create.codeSize = desc.shader.bytes.size();
            shader_create.pCode = words.data();
            require(vkCreateShaderModule(device, &shader_create, nullptr, &shader),
                    "vkCreateShaderModule");

            std::vector<VkDescriptorSetLayoutBinding> slots(
                desc.storage_buffer_binding_count);
            for (std::uint32_t i = 0; i < slots.size(); ++i) {
                slots[i].binding = i;
                slots[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                slots[i].descriptorCount = 1;
                slots[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo descriptor_create{};
            descriptor_create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptor_create.bindingCount = static_cast<std::uint32_t>(slots.size());
            descriptor_create.pBindings = slots.data();
            require(vkCreateDescriptorSetLayout(device, &descriptor_create, nullptr,
                                                &item.descriptors),
                    "vkCreateDescriptorSetLayout");

            VkPushConstantRange push_range{};
            push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push_range.offset = 0;
            push_range.size = desc.push_constant_bytes;

            VkPipelineLayoutCreateInfo layout_create{};
            layout_create.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout_create.setLayoutCount = 1;
            layout_create.pSetLayouts = &item.descriptors;
            layout_create.pushConstantRangeCount = desc.push_constant_bytes ? 1u : 0u;
            layout_create.pPushConstantRanges = desc.push_constant_bytes ? &push_range : nullptr;
            require(vkCreatePipelineLayout(device, &layout_create, nullptr, &item.layout),
                    "vkCreatePipelineLayout");

            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = desc.shader.entry_point.c_str();

            VkComputePipelineCreateInfo create{};
            create.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            create.stage = stage;
            create.layout = item.layout;
            require(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create, nullptr,
                                             &item.pipeline),
                    "vkCreateComputePipelines");
            item.binding_count = desc.storage_buffer_binding_count;
            item.push_bytes = desc.push_constant_bytes;
            const PipelineId id = next_pipeline;
            pipelines.emplace(id, item);
            ++next_pipeline;
            vkDestroyShaderModule(device, shader, nullptr);
            return id;
        } catch (...) {
            if (shader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, shader, nullptr);
            }
            destroy_pipeline_handles(item);
            throw;
        }
    }

    void destroy_pipeline(PipelineId id) {
        wait_all();
        auto it = pipelines.find(id);
        if (it == pipelines.end()) {
            throw std::out_of_range("Unknown compute pipeline ID");
        }
        destroy_pipeline_handles(it->second);
        pipelines.erase(it);
    }

    std::vector<VkDescriptorBufferInfo> validate_dispatch(
            PipelineId id, DispatchSize groups,
            const std::vector<BufferBinding>& bindings,
            const void* push_constants, std::size_t push_bytes) const {
        const Pipeline& pipeline = get_pipeline(id);
        const std::uint32_t counts[3] = {groups.x, groups.y, groups.z};
        for (std::uint32_t i = 0; i < 3; ++i) {
            if (counts[i] == 0 || counts[i] > info.max_workgroup_count[i]) {
                throw std::out_of_range("Compute workgroup count exceeds device limits");
            }
        }
        if (bindings.size() != pipeline.binding_count) {
            throw std::invalid_argument("Incorrect number of compute buffer bindings");
        }
        if (push_bytes != pipeline.push_bytes || (push_bytes && !push_constants)) {
            throw std::invalid_argument("Push constants do not match pipeline layout");
        }

        std::vector<VkDescriptorBufferInfo> descriptors(pipeline.binding_count);
        std::vector<bool> assigned(pipeline.binding_count, false);
        for (const auto& binding : bindings) {
            if (binding.slot >= pipeline.binding_count || assigned[binding.slot]) {
                throw std::invalid_argument("Invalid or duplicate compute binding slot");
            }
            if (binding.access != BufferAccess::Unspecified &&
                binding.access != BufferAccess::Read &&
                binding.access != BufferAccess::Write &&
                binding.access != BufferAccess::ReadWrite)
                throw std::invalid_argument("Invalid compute buffer access intent");
            const Buffer& buffer = get(binding.buffer);
            if (!buffer.storage_eligible)
                throw std::invalid_argument("Transfer-only Vulkan staging cannot bind as shader storage");
            validate_range(buffer.bytes, binding.offset, binding.bytes);
            if (binding.bytes == 0 || binding.bytes > limits.maxStorageBufferRange) {
                throw std::invalid_argument("Invalid compute descriptor range");
            }
            if (binding.offset % limits.minStorageBufferOffsetAlignment != 0) {
                throw std::invalid_argument("Compute descriptor offset is misaligned");
            }
            assigned[binding.slot] = true;
            descriptors[binding.slot] = {buffer.buffer, binding.offset, binding.bytes};
        }
        return descriptors;
    }

    SubmissionId dispatch(PipelineId id, DispatchSize groups,
                  const std::vector<BufferBinding>& bindings,
                  const void* push_constants, std::size_t push_bytes) {
        const Pipeline& pipeline = get_pipeline(id);
        std::vector<VkDescriptorBufferInfo> descriptors =
            validate_dispatch(id, groups, bindings, push_constants, push_bytes);

        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkCommandBuffer commands = VK_NULL_HANDLE;
        try {
            VkDescriptorPoolSize pool_size{};
            pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            pool_size.descriptorCount = std::max(1u, pipeline.binding_count);
            VkDescriptorPoolCreateInfo pool_create{};
            pool_create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_create.maxSets = 1;
            pool_create.poolSizeCount = 1;
            pool_create.pPoolSizes = &pool_size;
            require(vkCreateDescriptorPool(device, &pool_create, nullptr, &pool),
                    "vkCreateDescriptorPool");

            VkDescriptorSetAllocateInfo set_allocate{};
            set_allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            set_allocate.descriptorPool = pool;
            set_allocate.descriptorSetCount = 1;
            set_allocate.pSetLayouts = &pipeline.descriptors;
            VkDescriptorSet set = VK_NULL_HANDLE;
            require(vkAllocateDescriptorSets(device, &set_allocate, &set),
                    "vkAllocateDescriptorSets");

            std::vector<VkWriteDescriptorSet> writes(pipeline.binding_count);
            for (std::uint32_t i = 0; i < writes.size(); ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = set;
                writes[i].dstBinding = i;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].descriptorCount = 1;
                writes[i].pBufferInfo = &descriptors[i];
            }
            if (!writes.empty()) {
                vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                                       writes.data(), 0, nullptr);
            }

            VkCommandBufferAllocateInfo allocate{};
            allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocate.commandPool = command_pool;
            allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate.commandBufferCount = 1;
            require(vkAllocateCommandBuffers(device, &allocate, &commands),
                    "vkAllocateCommandBuffers");

            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            require(vkBeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");
            // Order prior GPU writes before reading or updating shader buffers.
            VkMemoryBarrier before{};
            before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &before, 0, nullptr, 0, nullptr);
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline.layout, 0, 1, &set, 0, nullptr);
            if (push_bytes) {
                vkCmdPushConstants(commands, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, static_cast<std::uint32_t>(push_bytes), push_constants);
            }
            vkCmdDispatch(commands, groups.x, groups.y, groups.z);

            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT |
                                    VK_ACCESS_TRANSFER_READ_BIT |
                                    VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &barrier, 0, nullptr, 0, nullptr);
            require(vkEndCommandBuffer(commands), "vkEndCommandBuffer");

            return enqueue(commands, pool);
        } catch (...) {
            if (commands != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, command_pool, 1, &commands);
            }
            if (pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, pool, nullptr);
            }
            throw;
        }
    }

    SubmissionId submit_batch(const std::vector<BatchCommand>& operations,
                              bool profile = false) {
        if (operations.empty()) {
            throw std::invalid_argument("Compute batch must contain an operation");
        }

        // Validate the complete batch before recording any GPU commands.
        // This prevents a malformed later operation from partially executing.
        std::uint64_t dispatch_count = 0;
        std::uint64_t descriptor_count = 0;
        for (const BatchCommand& operation : operations) {
            if (const auto* copy = std::get_if<CopyCommand>(&operation)) {
                const Buffer& source = get(copy->source);
                const Buffer& destination = get(copy->destination);
                validate_range(source.bytes, copy->source_offset, copy->bytes);
                validate_range(destination.bytes, copy->destination_offset, copy->bytes);
                if (copy->bytes == 0 || copy->source == copy->destination ||
                    (copy->source_offset % 4) != 0 ||
                    (copy->destination_offset % 4) != 0 ||
                    (copy->bytes % 4) != 0) {
                    throw std::invalid_argument("Invalid compute batch buffer copy");
                }
            } else {
                const auto& dispatch = std::get<DispatchCommand>(operation);
                const Pipeline& pipeline = get_pipeline(dispatch.pipeline);
                validate_dispatch(dispatch.pipeline, dispatch.groups,
                                  dispatch.buffers, dispatch.push_constants.data(),
                                  dispatch.push_constants.size());
                ++dispatch_count;
                descriptor_count += pipeline.binding_count;
            }
        }
        if (dispatch_count > std::numeric_limits<std::uint32_t>::max() ||
            descriptor_count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::out_of_range("Compute batch descriptor capacity exceeded");
        }

        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkCommandBuffer commands = VK_NULL_HANDLE;
        VkQueryPool timestamps = VK_NULL_HANDLE;
        std::uint32_t descriptor_capacity = 0;
        std::uint32_t set_capacity = 0;
        try {
            if (cached_batch_count != 0) {
                CachedBatch cached = cached_batches[--cached_batch_count];
                cached_batches[cached_batch_count] = {};
                pool = cached.descriptor_pool;
                commands = cached.commands;
                descriptor_capacity = cached.descriptor_capacity;
                set_capacity = cached.set_capacity;
            }
            if (dispatch_count != 0) {
                if (pool == VK_NULL_HANDLE ||
                    descriptor_capacity < descriptor_count ||
                    set_capacity < dispatch_count) {
                    if (pool != VK_NULL_HANDLE)
                        vkDestroyDescriptorPool(device, pool, nullptr);
                    pool = VK_NULL_HANDLE;
                    VkDescriptorPoolSize pool_size{};
                    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    pool_size.descriptorCount = static_cast<std::uint32_t>(
                        std::max<std::uint64_t>(1, descriptor_count));
                    VkDescriptorPoolCreateInfo create{};
                    create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                    create.maxSets = static_cast<std::uint32_t>(dispatch_count);
                    create.poolSizeCount = 1;
                    create.pPoolSizes = &pool_size;
                    require(vkCreateDescriptorPool(device, &create, nullptr, &pool),
                            "vkCreateDescriptorPool");
                    descriptor_capacity = pool_size.descriptorCount;
                    set_capacity = create.maxSets;
                } else {
                    require(vkResetDescriptorPool(device, pool, 0),
                            "vkResetDescriptorPool");
                }
            }
            if (commands != VK_NULL_HANDLE) {
                require(vkResetCommandBuffer(commands, 0), "vkResetCommandBuffer");
            } else {
                VkCommandBufferAllocateInfo allocate{};
                allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                allocate.commandPool = command_pool;
                allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocate.commandBufferCount = 1;
                require(vkAllocateCommandBuffers(device, &allocate, &commands),
                        "vkAllocateCommandBuffers");
            }

            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            require(vkBeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");
            if (profile) {
                VkQueryPoolCreateInfo query{};
                query.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                query.queryType = VK_QUERY_TYPE_TIMESTAMP;
                query.queryCount = 2;
                require(vkCreateQueryPool(device, &query, nullptr, &timestamps),
                        "vkCreateQueryPool");
                vkCmdResetQueryPool(commands, timestamps, 0, 2);
                vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps, 0);
            }

            // A single inter-submission dependency preserves the original
            // queue contract. Vulkan queue ordering alone does not publish
            // shader writes as memory dependencies between submissions.
            VkMemoryBarrier from_previous{};
            from_previous.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            from_previous.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            from_previous.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                                          VK_ACCESS_TRANSFER_WRITE_BIT |
                                          VK_ACCESS_SHADER_READ_BIT |
                                          VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &from_previous, 0, nullptr, 0, nullptr);

            struct Access {
                VkPipelineStageFlags stage = 0;
                VkAccessFlags mask = 0;
                bool writes = false;
            };
            std::unordered_map<BufferId, Access> last_access;
            last_access.reserve(std::min<std::size_t>(operations.size() * 2, 512));
            auto before_operation = [&](const std::vector<std::pair<BufferId, Access>>& uses) {
                std::vector<VkBufferMemoryBarrier> barriers;
                VkPipelineStageFlags sources = 0;
                VkPipelineStageFlags targets = 0;
                for (const auto& use : uses) {
                    const auto prev = last_access.find(use.first);
                    if (prev == last_access.end()) continue;
                    if (!prev->second.writes && !use.second.writes) continue;
                    VkBufferMemoryBarrier barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    barrier.srcAccessMask = prev->second.mask;
                    barrier.dstAccessMask = use.second.mask;
                    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.buffer = get(use.first).buffer;
                    barrier.offset = 0;
                    barrier.size = VK_WHOLE_SIZE;
                    barriers.push_back(barrier);
                    sources |= prev->second.stage;
                    targets |= use.second.stage;
                }
                if (!barriers.empty()) {
                    vkCmdPipelineBarrier(commands, sources, targets, 0,
                                         0, nullptr,
                                         static_cast<std::uint32_t>(barriers.size()),
                                         barriers.data(), 0, nullptr);
                }
                for (const auto& use : uses) {
                    auto& current = last_access[use.first];
                    if (!current.stage || current.writes || use.second.writes) {
                        current = use.second;
                    } else {
                        // Read after read is unordered: retain both source
                        // stages in case a later operation writes the buffer.
                        current.stage |= use.second.stage;
                        current.mask |= use.second.mask;
                    }
                }
            };

            for (const BatchCommand& operation : operations) {
                if (const auto* copy = std::get_if<CopyCommand>(&operation)) {
                    before_operation({
                        {copy->source, {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, false}},
                        {copy->destination, {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, true}}
                    });
                    VkBufferCopy region{};
                    region.srcOffset = copy->source_offset;
                    region.dstOffset = copy->destination_offset;
                    region.size = copy->bytes;
                    vkCmdCopyBuffer(commands, get(copy->source).buffer,
                                    get(copy->destination).buffer, 1, &region);
                    continue;
                }

                const auto& dispatch = std::get<DispatchCommand>(operation);
                const Pipeline& pipeline = get_pipeline(dispatch.pipeline);
                // Access intent comes from the PhysX launch contract, not
                // an assumption that each storage buffer is writable. The
                // conservative Unspecified default remains read/write.
                // Multiple slots aliasing one allocation are merged before
                // the dependency graph is updated, regardless of view ranges.
                std::vector<std::pair<BufferId, Access>> shader_uses;
                shader_uses.reserve(dispatch.buffers.size());
                for (const auto& binding : dispatch.buffers) {
                    const bool reads = binding.access != BufferAccess::Write;
                    const bool writes = binding.access != BufferAccess::Read;
                    const VkAccessFlags mask =
                        (reads ? VK_ACCESS_SHADER_READ_BIT : 0u) |
                        (writes ? VK_ACCESS_SHADER_WRITE_BIT : 0u);
                    auto found = std::find_if(shader_uses.begin(), shader_uses.end(),
                        [&](const auto& entry) { return entry.first == binding.buffer; });
                    if (found == shader_uses.end()) {
                        shader_uses.push_back({
                            binding.buffer,
                            {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, mask, writes}});
                    } else {
                        found->second.mask |= mask;
                        found->second.writes |= writes;
                    }
                }
                before_operation(shader_uses);
                auto descriptors = validate_dispatch(
                    dispatch.pipeline, dispatch.groups, dispatch.buffers,
                    dispatch.push_constants.data(), dispatch.push_constants.size());

                VkDescriptorSetAllocateInfo descriptor_allocate{};
                descriptor_allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptor_allocate.descriptorPool = pool;
                descriptor_allocate.descriptorSetCount = 1;
                descriptor_allocate.pSetLayouts = &pipeline.descriptors;
                VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
                require(vkAllocateDescriptorSets(device, &descriptor_allocate,
                                                 &descriptor_set), "vkAllocateDescriptorSets");

                std::vector<VkWriteDescriptorSet> writes(descriptors.size());
                for (std::uint32_t slot = 0; slot < writes.size(); ++slot) {
                    writes[slot].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writes[slot].dstSet = descriptor_set;
                    writes[slot].dstBinding = slot;
                    writes[slot].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[slot].descriptorCount = 1;
                    writes[slot].pBufferInfo = &descriptors[slot];
                }
                if (!writes.empty()) {
                    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                                           writes.data(), 0, nullptr);
                }

                vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline.layout, 0, 1, &descriptor_set, 0, nullptr);
                if (!dispatch.push_constants.empty()) {
                    vkCmdPushConstants(commands, pipeline.layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       static_cast<std::uint32_t>(dispatch.push_constants.size()),
                                       dispatch.push_constants.data());
                }
                vkCmdDispatch(commands, dispatch.groups.x, dispatch.groups.y,
                              dispatch.groups.z);
            }

            VkMemoryBarrier to_host{};
            to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            to_host.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host,
                                 0, nullptr, 0, nullptr);
            if (timestamps != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 1);
            require(vkEndCommandBuffer(commands), "vkEndCommandBuffer");
            const SubmissionId id = enqueue(commands, pool);
            auto& submitted = submissions.at(id);
            submitted.timestamps = timestamps;
            submitted.cacheable_batch = true;
            submitted.descriptor_capacity = descriptor_capacity;
            submitted.set_capacity = set_capacity;
            return id;
        } catch (...) {
            if (timestamps != VK_NULL_HANDLE)
                vkDestroyQueryPool(device, timestamps, nullptr);
            if (commands != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, command_pool, 1, &commands);
            }
            if (pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, pool, nullptr);
            }
            throw;
        }
    }


    std::optional<double> profile_batch_gpu_nanoseconds(
        const std::vector<BatchCommand>& operations) {
        if (timestamp_valid_bits == 0 || limits.timestampPeriod <= 0.0f)
            return std::nullopt;
        const auto id = submit_batch(operations, true);
        auto it = submissions.find(id);
        const auto fence = it->second.fence;
        require(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
                "vkWaitForFences timestamp batch");
        std::uint64_t ticks[2]{};
        const VkResult result = vkGetQueryPoolResults(
            device, it->second.timestamps, 0, 2, sizeof(ticks),
            ticks, sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (result != VK_SUCCESS) {
            retire_submission(it->second);
            submissions.erase(it);
            release_retired_through(id);
            require(result, "vkGetQueryPoolResults");
        }
        const auto mask = timestamp_valid_bits == 64 ? ~std::uint64_t{0} :
                          ((std::uint64_t{1} << timestamp_valid_bits) - 1);
        const std::uint64_t delta = (ticks[1] - ticks[0]) & mask;
        retire_submission(it->second);
        submissions.erase(it);
        release_retired_through(id);
        return static_cast<double>(delta) * limits.timestampPeriod;
    }

    SubmissionId copy(BufferId source_id, std::size_t source_offset,
                      BufferId destination_id, std::size_t destination_offset,
                      std::size_t bytes, bool allow_empty = false) {
        const Buffer& source = get(source_id);
        const Buffer& destination = get(destination_id);
        validate_range(source.bytes, source_offset, bytes);
        validate_range(destination.bytes, destination_offset, bytes);
        if (bytes == 0) {
            if (allow_empty) {
                return 0;
            }
            throw std::invalid_argument("Asynchronous compute copy cannot be empty");
        }
        if (source_id == destination_id) {
            throw std::invalid_argument("In-place GPU buffer copies are not supported yet");
        }
        return submit_copy(source, source_offset, destination, destination_offset, bytes);
    }
};

Context::Context(Backend backend, std::uint32_t device_ordinal)
    : impl_(std::make_unique<Impl>(backend, device_ordinal)) {}

Context::~Context() = default;
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;

const DeviceInfo& Context::device_info() const { return impl_->info; }

BufferId Context::create_buffer(std::size_t bytes, MemoryPlacement placement) {
    return impl_->create_buffer(bytes, placement);
}

void Context::destroy_buffer(BufferId id) { impl_->destroy_buffer(id); }
void Context::retire_buffer(BufferId id) { impl_->retire_buffer(id); }

void Context::upload(BufferId id, std::size_t offset, const void* data, std::size_t bytes) {
    impl_->transfer(id, offset, const_cast<void*>(data), bytes, true);
}

SubmissionId Context::submit_upload_buffer(BufferId id, std::size_t offset,
                                           const void* data, std::size_t bytes) {
    return impl_->submit_upload(id, offset, data, bytes);
}

SubmissionId Context::submit_upload_batch(const std::vector<UploadSpan>& uploads) {
    return impl_->submit_uploads(uploads);
}

void Context::download(BufferId id, std::size_t offset, void* data, std::size_t bytes) const {
    impl_->transfer(id, offset, data, bytes, false);
}

void Context::copy_buffer(BufferId source, std::size_t source_offset,
                          BufferId destination, std::size_t destination_offset,
                          std::size_t bytes) {
    const SubmissionId id = impl_->copy(source, source_offset,
                                        destination, destination_offset, bytes, true);
    if (id != 0) {
        impl_->wait_submission(id);
    }
}

SubmissionId Context::submit_copy_buffer(BufferId source, std::size_t source_offset,
                                         BufferId destination, std::size_t destination_offset,
                                         std::size_t bytes) {
    return impl_->copy(source, source_offset, destination, destination_offset, bytes);
}

PipelineId Context::create_compute_pipeline(const ComputePipelineDesc& description) {
    return impl_->create_pipeline(description);
}

void Context::destroy_compute_pipeline(PipelineId id) { impl_->destroy_pipeline(id); }

void Context::dispatch(PipelineId pipeline, DispatchSize groups,
                       const std::vector<BufferBinding>& buffers,
                       const void* push_constants, std::size_t push_constant_bytes) {
    impl_->wait_submission(
        impl_->dispatch(pipeline, groups, buffers, push_constants, push_constant_bytes));
}

SubmissionId Context::submit_dispatch(PipelineId pipeline, DispatchSize groups,
                                      const std::vector<BufferBinding>& buffers,
                                      const void* push_constants, std::size_t push_constant_bytes) {
    return impl_->dispatch(pipeline, groups, buffers, push_constants, push_constant_bytes);
}

bool Context::submission_complete(SubmissionId id) const {
    return impl_->submission_complete(id);
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
