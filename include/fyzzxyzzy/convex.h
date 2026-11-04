// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/world.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace fyzz {

using ConvexTriangle = std::array<std::uint32_t,3>;

// Outward planar polygon, ordered around its boundary. Triangulated
// coplanar faces share one polygon and therefore one contact feature.
struct ConvexFace {
    Vec3 normal{};
    float distance=0.f;
    std::uint32_t first_index=0;
    std::uint32_t index_count=0;
};

// Density-one properties in the mesh's authoring frame. Inertia columns
// are about the actual center of mass, not the authoring origin.
struct ConvexMassProperties {
    double volume=0.0;
    Vec3 center{};
    std::array<Vec3,3> inertia_columns{};
};

class ConvexMesh final {
public:
    std::span<const Vec3> vertices() const noexcept { return vertices_; }
    std::span<const ConvexTriangle> triangles() const noexcept { return triangles_; }
    std::span<const ConvexFace> faces() const noexcept { return faces_; }
    std::span<const std::uint32_t> face_indices() const noexcept {
        return face_indices_;
    }
    const ConvexMassProperties& mass_properties() const noexcept { return mass_; }
    Vec3 bounds_min() const noexcept { return bounds_min_; }
    Vec3 bounds_max() const noexcept { return bounds_max_; }
private:
    friend std::shared_ptr<const ConvexMesh> cook_convex_mesh(
        std::span<const Vec3>,std::span<const ConvexTriangle>);
    std::vector<Vec3> vertices_;
    std::vector<ConvexTriangle> triangles_;
    std::vector<ConvexFace> faces_;
    std::vector<std::uint32_t> face_indices_;
    ConvexMassProperties mass_{};
    Vec3 bounds_min_{},bounds_max_{};
};

// Input triangles must be outward-wound and describe a closed, convex,
// manifold surface. Cooking rejects nonconvex, degenerate and open meshes.
std::shared_ptr<const ConvexMesh> cook_convex_mesh(
    std::span<const Vec3> vertices,
    std::span<const ConvexTriangle> triangles);

} // namespace fyzz
