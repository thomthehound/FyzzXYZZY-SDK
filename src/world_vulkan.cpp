// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/floor_world.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::compute {
namespace {

struct BodyEntry {
    bool is_floor = false;
    std::size_t dynamic_index = 0;
    float floor_height = 0.0f;
    Vec3 center{};
    Vec3 half_extents{};
};

struct IslandRange {
    std::uint32_t begin;
    std::uint32_t count;
};

struct ContactParameters {
    std::uint32_t island_count;
    float floor_y;
    float restitution;
    float resting_speed;
    std::uint32_t iterations;
};
static_assert(sizeof(IslandRange) == 8, "GPU islands use std430 uvec2");
static_assert(sizeof(ContactParameters) == 20, "Contact shader push-constant layout");

class FloorGpuWorld final : public WorldBackend {
public:
    FloorGpuWorld(ShaderBinary floor_shader, ShaderBinary contact_shader,
                  std::uint32_t device_ordinal)
        : context_(Backend::Vulkan, device_ordinal),
          physics_(context_, std::move(floor_shader)),
          contacts_pipeline_(context_.create_compute_pipeline(ComputePipelineDesc{
              std::move(contact_shader), 5, sizeof(ContactParameters)})) {}

    ~FloorGpuWorld() override {
        // Context owns remaining handles if the Vulkan device has been lost.
        try {
            context_.destroy_compute_pipeline(contacts_pipeline_);
            for (BufferId id : {positions_, velocities_, properties_, members_, islands_}) {
                if (id) context_.destroy_buffer(id);
            }
        } catch (...) {
        }
    }

    BodyId add_static_box(Vec3, Vec3) override {
        throw std::logic_error(
            "Vulkan floor backend does not support static boxes; use add_static_floor");
    }

    BodyId add_static_floor(float height) override {
        if (floor_y_.has_value()) {
            throw std::logic_error("Vulkan floor backend supports only one horizontal floor");
        }
        const BodyId id = next_id();
        bodies_.push_back(BodyEntry{true, 0, height});
        floor_y_ = height;
        return id;
    }

    BodyId add_dynamic_box(Vec3 half_extents, Vec3 position, float density) override {
        const double mass = 8.0 * static_cast<double>(half_extents.x) *
            half_extents.y * half_extents.z * density;
        const float inverse_mass = static_cast<float>(1.0 / mass);
        if (!std::isfinite(inverse_mass) || inverse_mass <= 0.0f) {
            throw std::invalid_argument("Box density and extents produce invalid mass");
        }
        const BodyId id = next_id();
        const std::size_t slot = dynamic_count_;
        reserve_dynamic(slot + 1);
        const FloorBodyVector p{position.x, position.y, position.z, half_extents.y};
        const FloorBodyVector v{0.0f, 0.0f, 0.0f, 0.2f};
        const FloorBodyVector props{half_extents.x, half_extents.z, inverse_mass, 0.0f};
        const std::size_t byte_offset = slot * sizeof(FloorBodyVector);
        context_.upload(positions_, byte_offset, &p, sizeof(p));
        context_.upload(velocities_, byte_offset, &v, sizeof(v));
        context_.upload(properties_, byte_offset, &props, sizeof(props));
        bodies_.push_back(BodyEntry{false, slot, 0.0f, position, half_extents});
        ++dynamic_count_;
        topology_dirty_ = true;
        return id;
    }

    void step(float seconds) override {
        if (dynamic_count_ == 0) return;
        if (!floor_y_.has_value()) {
            throw std::logic_error("Vulkan floor backend requires a static floor before stepping");
        }
        if (topology_dirty_) rebuild_islands();

        FloorStepParameters parameters;
        parameters.count = static_cast<std::uint32_t>(dynamic_count_);
        parameters.dt = seconds;
        parameters.floor_y = *floor_y_;
        parameters.gravity_y = -9.81f;
        parameters.tangent_retention = 0.5f;
        parameters.resting_speed = 0.12f;

        // Integration/floor contact and inter-body impulses share a Vulkan
        // command buffer and a completion fence. No CPU collision response.
        std::vector<BatchCommand> commands;
        physics_.append_step(commands, positions_, velocities_, parameters);
        const ContactParameters contacts{
            island_count_, *floor_y_, 0.0f, 0.12f, 8u
        };
        std::vector<std::uint8_t> constants(sizeof(contacts));
        std::memcpy(constants.data(), &contacts, constants.size());
        const std::size_t state_bytes = dynamic_count_ * sizeof(FloorBodyVector);
        commands.emplace_back(DispatchCommand{
            contacts_pipeline_, {(island_count_ + 63u) / 64u, 1, 1},
            {{0, positions_, 0, state_bytes},
             {1, velocities_, 0, state_bytes},
             {2, properties_, 0, state_bytes},
             {3, members_, 0, dynamic_count_ * sizeof(std::uint32_t)},
             {4, islands_, 0, island_count_ * sizeof(IslandRange)}},
            std::move(constants)
        });
        context_.execute_batch(commands);
    }

    BodyState body_state(BodyId id) const override {
        if (id >= bodies_.size()) {
            throw std::out_of_range("Unknown FyzzXYZZY body ID");
        }
        const BodyEntry& body = bodies_[id];
        if (body.is_floor) {
            return {{0.0f, body.floor_height, 0.0f}, {0.0f, 0.0f, 0.0f}};
        }
        FloorBodyVector p{}, v{};
        const std::size_t offset = body.dynamic_index * sizeof(FloorBodyVector);
        context_.download(positions_, offset, &p, sizeof(p));
        context_.download(velocities_, offset, &v, sizeof(v));
        return {{p.x, p.y, p.z}, {v.x, v.y, v.z}};
    }

private:
    BodyId next_id() const {
        if (bodies_.size() >= std::numeric_limits<BodyId>::max()) {
            throw std::overflow_error("World body ID capacity exceeded");
        }
        return static_cast<BodyId>(bodies_.size());
    }

    void rebuild_islands() {
        // X/Z positions never change in this restricted backend. Connected
        // components of overlapping footprints cannot contact across islands.
        const std::uint32_t n = static_cast<std::uint32_t>(dynamic_count_);
        std::vector<const BodyEntry*> entries(n);
        for (const BodyEntry& entry : bodies_) {
            if (!entry.is_floor) entries[entry.dynamic_index] = &entry;
        }
        std::vector<std::uint32_t> parent(n);
        for (std::uint32_t i = 0; i < n; ++i) parent[i] = i;
        auto root = [&](std::uint32_t i) {
            while (i != parent[i]) {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        for (std::uint32_t i = 0; i < n; ++i) {
            for (std::uint32_t j = i + 1; j < n; ++j) {
                const BodyEntry& a = *entries[i];
                const BodyEntry& b = *entries[j];
                if (std::abs(a.center.x - b.center.x) <
                        a.half_extents.x + b.half_extents.x &&
                    std::abs(a.center.z - b.center.z) <
                        a.half_extents.z + b.half_extents.z) {
                    parent[root(j)] = root(i);
                }
            }
        }
        std::vector<std::vector<std::uint32_t>> groups(n);
        for (std::uint32_t i = 0; i < n; ++i) groups[root(i)].push_back(i);

        std::vector<std::uint32_t> members;
        std::vector<IslandRange> islands;
        members.reserve(n);
        islands.reserve(n);
        for (const auto& group : groups) {
            if (group.empty()) continue;
            islands.push_back(IslandRange{
                static_cast<std::uint32_t>(members.size()),
                static_cast<std::uint32_t>(group.size())});
            members.insert(members.end(), group.begin(), group.end());
        }
        context_.upload(members_, 0, members.data(),
                        members.size() * sizeof(std::uint32_t));
        context_.upload(islands_, 0, islands.data(),
                        islands.size() * sizeof(IslandRange));
        island_count_ = static_cast<std::uint32_t>(islands.size());
        topology_dirty_ = false;
    }

    void reserve_dynamic(std::size_t count) {
        if (count <= capacity_) return;
        if (count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Vulkan floor world body count exceeds shader limits");
        }
        const std::size_t limit = static_cast<std::size_t>(
            context_.device_info().max_storage_buffer_range / sizeof(FloorBodyVector));
        if (count > limit) {
            throw std::out_of_range("Vulkan floor world exceeds storage-buffer limit");
        }
        std::size_t new_capacity = std::max<std::size_t>(8, capacity_);
        while (new_capacity < count) {
            if (new_capacity > limit / 2) {
                new_capacity = limit;
                break;
            }
            new_capacity *= 2;
        }
        if (new_capacity > limit) new_capacity = limit;
        const std::size_t bytes = new_capacity * sizeof(FloorBodyVector);
        // Allocate replacement state first: failures leave existing buffers
        // intact. The island tables are rebuilt after each body insertion.
        BufferId next[5]{};
        const BufferId old[5] = {positions_, velocities_, properties_, members_, islands_};
        const std::size_t sizes[5] = {
            bytes, bytes, bytes, new_capacity * sizeof(std::uint32_t),
            new_capacity * sizeof(IslandRange)
        };
        try {
            for (int i = 0; i < 5; ++i) {
                next[i] = context_.create_buffer(sizes[i], MemoryPlacement::DeviceLocal);
            }
            if (dynamic_count_ != 0) {
                const std::size_t active = dynamic_count_ * sizeof(FloorBodyVector);
                for (int i = 0; i < 3; ++i) {
                    context_.copy_buffer(old[i], 0, next[i], 0, active);
                }
            }
        } catch (...) {
            for (BufferId id : next) if (id) context_.destroy_buffer(id);
            throw;
        }
        for (BufferId id : old) if (id) context_.destroy_buffer(id);
        positions_ = next[0];
        velocities_ = next[1];
        properties_ = next[2];
        members_ = next[3];
        islands_ = next[4];
        capacity_ = new_capacity;
        topology_dirty_ = true;
    }

    Context context_;
    FloorContactKernel physics_;
    PipelineId contacts_pipeline_ = 0;
    BufferId positions_ = 0;
    BufferId velocities_ = 0;
    BufferId properties_ = 0;
    BufferId members_ = 0;
    BufferId islands_ = 0;
    std::size_t dynamic_count_ = 0;
    std::size_t capacity_ = 0;
    std::uint32_t island_count_ = 0;
    bool topology_dirty_ = true;
    std::optional<float> floor_y_;
    std::vector<BodyEntry> bodies_;
};

} // namespace

World create_floor_world(ShaderBinary floor_shader, ShaderBinary contact_shader,
                         std::uint32_t device_ordinal) {
    return World(std::make_unique<FloorGpuWorld>(
        std::move(floor_shader), std::move(contact_shader), device_ordinal));
}

} // namespace fyzz::compute
