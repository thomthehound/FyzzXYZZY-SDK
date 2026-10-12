// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fizzx/compute.h"

#include <vulkan/vulkan.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fizzx::compute {
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
    application.pApplicationName = "FizzX";
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
    };

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    DeviceInfo info;
    std::unordered_map<BufferId, Buffer> buffers;
    BufferId next_buffer = 1;

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
            pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
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

    BufferId create_buffer(std::size_t bytes) {
        if (bytes == 0 || bytes > info.max_storage_buffer_range) {
            throw std::invalid_argument("Compute buffer size is zero or exceeds device limits");
        }
        if (next_buffer == std::numeric_limits<BufferId>::max()) {
            throw std::overflow_error("Compute buffer ID range exhausted");
        }
        Buffer buffer;
        try {
            VkBufferCreateInfo create{};
            create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            create.size = static_cast<VkDeviceSize>(bytes);
            create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            require(vkCreateBuffer(device, &create, nullptr, &buffer.buffer), "vkCreateBuffer");

            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
            VkPhysicalDeviceMemoryProperties memory_properties{};
            vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
            std::uint32_t type_index = memory_properties.memoryTypeCount;
            for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
                constexpr VkMemoryPropertyFlags needed =
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
                    (memory_properties.memoryTypes[i].propertyFlags & needed) == needed) {
                    type_index = i;
                    break;
                }
            }
            if (type_index == memory_properties.memoryTypeCount) {
                throw std::runtime_error("No host-visible coherent Vulkan memory type");
            }

            VkMemoryAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = type_index;
            require(vkAllocateMemory(device, &allocation, nullptr, &buffer.memory),
                    "vkAllocateMemory");
            require(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0),
                    "vkBindBufferMemory");
            buffer.bytes = bytes;
            const BufferId id = next_buffer++;
            buffers.emplace(id, buffer);
            return id;
        } catch (...) {
            if (buffer.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, buffer.buffer, nullptr);
            }
            if (buffer.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, buffer.memory, nullptr);
            }
            throw;
        }
    }

    void destroy_buffer(BufferId id) {
        const auto it = buffers.find(id);
        if (it == buffers.end()) {
            throw std::out_of_range("Unknown compute buffer ID");
        }
        vkDestroyBuffer(device, it->second.buffer, nullptr);
        vkFreeMemory(device, it->second.memory, nullptr);
        buffers.erase(it);
    }

    void transfer(BufferId id, std::size_t offset, void* data,
                  std::size_t bytes, bool upload) const {
        const Buffer& buffer = get(id);
        validate_range(buffer.bytes, offset, bytes);
        if (bytes == 0) {
            return;
        }
        if (!data) {
            throw std::invalid_argument("Null compute transfer data");
        }
        // Mapping begins at zero to satisfy minMemoryMapAlignment even for
        // arbitrary byte offsets in the public transfer API.
        void* mapped = nullptr;
        require(vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &mapped),
                "vkMapMemory");
        auto* bytes_base = static_cast<unsigned char*>(mapped) + offset;
        if (upload) {
            std::memcpy(bytes_base, data, bytes);
        } else {
            std::memcpy(data, bytes_base, bytes);
        }
        vkUnmapMemory(device, buffer.memory);
    }

    void copy(BufferId source_id, std::size_t source_offset,
              BufferId destination_id, std::size_t destination_offset,
              std::size_t bytes) {
        const Buffer& source = get(source_id);
        const Buffer& destination = get(destination_id);
        validate_range(source.bytes, source_offset, bytes);
        validate_range(destination.bytes, destination_offset, bytes);
        if (bytes == 0) {
            return;
        }
        if (source_id == destination_id) {
            throw std::invalid_argument("In-place GPU buffer copies are not supported yet");
        }
        if ((source_offset % 4) != 0 || (destination_offset % 4) != 0 ||
            (bytes % 4) != 0) {
            throw std::invalid_argument("Vulkan buffer-copy offsets and sizes must be 4-byte aligned");
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
            VkBufferCopy region{};
            region.srcOffset = source_offset;
            region.dstOffset = destination_offset;
            region.size = bytes;
            vkCmdCopyBuffer(commands, source.buffer, destination.buffer, 1, &region);
            require(vkEndCommandBuffer(commands), "vkEndCommandBuffer");

            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &commands;
            require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
            require(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
        } catch (...) {
            vkFreeCommandBuffers(device, command_pool, 1, &commands);
            throw;
        }
        vkFreeCommandBuffers(device, command_pool, 1, &commands);
    }
};

Context::Context(Backend backend, std::uint32_t device_ordinal)
    : impl_(std::make_unique<Impl>(backend, device_ordinal)) {}

Context::~Context() = default;
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;

const DeviceInfo& Context::device_info() const { return impl_->info; }

BufferId Context::create_buffer(std::size_t bytes) {
    return impl_->create_buffer(bytes);
}

void Context::destroy_buffer(BufferId id) { impl_->destroy_buffer(id); }

void Context::upload(BufferId id, std::size_t offset, const void* data, std::size_t bytes) {
    impl_->transfer(id, offset, const_cast<void*>(data), bytes, true);
}

void Context::download(BufferId id, std::size_t offset, void* data, std::size_t bytes) const {
    impl_->transfer(id, offset, data, bytes, false);
}

void Context::copy_buffer(BufferId source, std::size_t source_offset,
                          BufferId destination, std::size_t destination_offset,
                          std::size_t bytes) {
    impl_->copy(source, source_offset, destination, destination_offset, bytes);
}

} // namespace fizzx::compute
