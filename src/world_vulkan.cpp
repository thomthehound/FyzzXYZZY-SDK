// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/floor_world.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

class FloorGpuWorld final : public WorldBackend {
public:
    FloorGpuWorld(ShaderBinary shader, std::uint32_t device_ordinal)
        : context_(Backend::Vulkan, device_ordinal),
          physics_(context_, std::move(shader)) {}

    ~FloorGpuWorld() override {
        // The Context owns any remaining resources if the device was lost.
        try {
            if (positions_) context_.destroy_buffer(positions_);
            if (velocities_) context_.destroy_buffer(velocities_);
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
        (void)density; // Gravity is mass-independent for this limited workload.
        // Horizontal velocities start at zero and this kernel applies no X/Z
        // acceleration. Disjoint horizontal footprints therefore remain
        // disjoint. Reject any configuration that could require body-body
        // contact, which this backend cannot resolve.
        for (const BodyEntry& existing : bodies_) {
            if (!existing.is_floor &&
                std::abs(existing.center.x - position.x) <
                    existing.half_extents.x + half_extents.x &&
                std::abs(existing.center.z - position.z) <
                    existing.half_extents.z + half_extents.z) {
                throw std::logic_error(
                    "Vulkan floor backend does not support potential body-to-body contact");
            }
        }
        const BodyId id = next_id();
        const std::size_t slot = dynamic_count_;
        reserve_dynamic(slot + 1);
        const FloorBodyVector p{position.x, position.y, position.z, half_extents.y};
        const FloorBodyVector v{0.0f, 0.0f, 0.0f, 0.2f};
        const std::size_t byte_offset = slot * sizeof(FloorBodyVector);
        context_.upload(positions_, byte_offset, &p, sizeof(p));
        context_.upload(velocities_, byte_offset, &v, sizeof(v));
        bodies_.push_back(BodyEntry{false, slot, 0.0f, position, half_extents});
        ++dynamic_count_;
        return id;
    }

    void step(float seconds) override {
        if (dynamic_count_ == 0) return;
        if (!floor_y_.has_value()) {
            throw std::logic_error("Vulkan floor backend requires a static floor before stepping");
        }
        FloorStepParameters parameters;
        parameters.count = static_cast<std::uint32_t>(dynamic_count_);
        parameters.dt = seconds;
        parameters.floor_y = *floor_y_;
        parameters.gravity_y = -9.81f;
        parameters.tangent_retention = 0.5f;
        parameters.resting_speed = 0.12f;
        physics_.step(positions_, velocities_, parameters);
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
        BufferId new_positions = 0;
        BufferId new_velocities = 0;
        try {
            new_positions = context_.create_buffer(bytes, MemoryPlacement::DeviceLocal);
            new_velocities = context_.create_buffer(bytes, MemoryPlacement::DeviceLocal);
            if (dynamic_count_ != 0) {
                const std::size_t active_bytes = dynamic_count_ * sizeof(FloorBodyVector);
                context_.copy_buffer(positions_, 0, new_positions, 0, active_bytes);
                context_.copy_buffer(velocities_, 0, new_velocities, 0, active_bytes);
            }
        } catch (...) {
            if (new_positions) context_.destroy_buffer(new_positions);
            if (new_velocities) context_.destroy_buffer(new_velocities);
            throw;
        }
        if (positions_) context_.destroy_buffer(positions_);
        if (velocities_) context_.destroy_buffer(velocities_);
        positions_ = new_positions;
        velocities_ = new_velocities;
        capacity_ = new_capacity;
    }

    Context context_;
    FloorContactKernel physics_;
    BufferId positions_ = 0;
    BufferId velocities_ = 0;
    std::size_t dynamic_count_ = 0;
    std::size_t capacity_ = 0;
    std::optional<float> floor_y_;
    std::vector<BodyEntry> bodies_;
};

} // namespace

World create_floor_world(ShaderBinary shader, std::uint32_t device_ordinal) {
    return World(std::make_unique<FloorGpuWorld>(std::move(shader), device_ordinal));
}

} // namespace fyzz::compute
