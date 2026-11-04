// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/convex.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace fyzz {
namespace {
using DVec=std::array<double,3>;
DVec sub(DVec a,DVec b) {
    return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
}
DVec cross(DVec a,DVec b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]};
}
double dot(DVec a,DVec b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
DVec as_double(Vec3 v) { return {v.x,v.y,v.z}; }
Vec3 as_float(DVec v) {
    for(const double x:v)
        if(!std::isfinite(x) ||
           std::abs(x)>std::numeric_limits<float>::max())
            throw std::invalid_argument("Convex mass property outside float range");
    return {static_cast<float>(v[0]),static_cast<float>(v[1]),
            static_cast<float>(v[2])};
}
bool finite(Vec3 v) {
    return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z);
}
}

std::shared_ptr<const ConvexMesh> cook_convex_mesh(
    std::span<const Vec3> vertices,
    std::span<const ConvexTriangle> triangles) {
    if(vertices.size()<4u || triangles.size()<4u ||
       vertices.size()>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Convex hull requires a closed volume");
    auto mesh=std::shared_ptr<ConvexMesh>(new ConvexMesh());
    mesh->vertices_.assign(vertices.begin(),vertices.end());
    mesh->triangles_.assign(triangles.begin(),triangles.end());

    DVec lower=as_double(vertices.front()),upper=lower,origin{};
    for(const Vec3 v:vertices) {
        if(!finite(v))
            throw std::invalid_argument("Convex vertex must be finite");
        const DVec p=as_double(v);
        for(unsigned k=0;k<3u;++k) {
            lower[k]=std::min(lower[k],p[k]);
            upper[k]=std::max(upper[k],p[k]);
            origin[k]+=p[k]/static_cast<double>(vertices.size());
        }
    }
    DVec extent=sub(upper,lower);
    const double scale=std::max({extent[0],extent[1],extent[2]});
    if(!(scale>0.0) || !std::isfinite(scale))
        throw std::invalid_argument("Convex hull has no spatial extent");
    mesh->bounds_min_=as_float(lower);
    mesh->bounds_max_=as_float(upper);

    // Signed edge incidence tests both manifold closure and outward
    // orientation; an edge must appear once in each direction.
    std::unordered_map<std::uint64_t,std::uint8_t> edges;
    edges.reserve(triangles.size()*3u);
    std::vector<std::uint8_t> referenced(vertices.size(),0u);
    double volume=0.0;
    DVec first{};
    std::array<DVec,3> second{};
    for(const auto& face:triangles) {
        for(unsigned k=0;k<3u;++k)
            if(face[k]>=vertices.size() ||
               face[k]==face[(k+1u)%3u] ||
               face[k]==face[(k+2u)%3u])
                throw std::invalid_argument("Convex face has invalid indices");
        const DVec a=sub(as_double(vertices[face[0]]),origin);
        const DVec b=sub(as_double(vertices[face[1]]),origin);
        const DVec c=sub(as_double(vertices[face[2]]),origin);
        const DVec normal=cross(sub(b,a),sub(c,a));
        const double normal_length=std::sqrt(dot(normal,normal));
        if(!(normal_length>scale*scale*1.e-12) ||
           !std::isfinite(normal_length))
            throw std::invalid_argument("Convex hull has degenerate triangles");
        // Every oriented triangle plane must support the entire volume.
        const double tolerance=scale*normal_length*1.e-6;
        for(const Vec3 point:vertices)
            if(dot(normal,sub(as_double(point),as_double(vertices[face[0]])))>
               tolerance)
                throw std::invalid_argument("Convex hull is nonconvex or inward-wound");

        for(unsigned k=0;k<3u;++k) {
            const auto u=face[k],v=face[(k+1u)%3u];
            referenced[u]=1u;
            const auto low=std::min(u,v),high=std::max(u,v);
            const auto key=(std::uint64_t(low)<<32u)|high;
            auto& winding=edges[key];
            const auto direction=std::uint8_t(u==low?1u:2u);
            if(winding&direction)
                throw std::invalid_argument("Convex surface has nonmanifold edges");
            winding|=direction;
        }

        // Signed tetrahedra relative to the vertex mean avoid the large
        // origin-dependent terms that destabilize translated hulls.
        const double tetra=dot(a,cross(b,c))/6.0;
        volume+=tetra;
        for(unsigned i=0;i<3u;++i)
            first[i]+=tetra*(a[i]+b[i]+c[i])*0.25;
        for(unsigned i=0;i<3u;++i)
            for(unsigned j=0;j<3u;++j) {
                const double sum_i=a[i]+b[i]+c[i];
                const double sum_j=a[j]+b[j]+c[j];
                second[i][j]+=tetra*0.05*(
                    sum_i*sum_j+a[i]*a[j]+b[i]*b[j]+c[i]*c[j]);
            }
    }
    if(std::any_of(referenced.begin(),referenced.end(),
                   [](std::uint8_t used){return !used;}) ||
       std::any_of(edges.begin(),edges.end(),
                   [](const auto& e){return e.second!=3u;}))
        throw std::invalid_argument("Convex surface must be closed and use every vertex");
    if(!std::isfinite(volume) || !(volume>0.0))
        throw std::invalid_argument("Convex mesh has nonpositive volume");
    DVec center{};
    for(unsigned i=0;i<3u;++i)
        center[i]=first[i]/volume;
    DVec world_center{};
    for(unsigned i=0;i<3u;++i)
        world_center[i]=center[i]+origin[i];
    // Shift the second-moment tensor to the actual centroid before
    // converting to rigid-body inertia; off-diagonal terms are essential
    // for asymmetric hulls and rotated compound shapes.
    std::array<DVec,3> covariance{};
    for(unsigned i=0;i<3u;++i)
        for(unsigned j=0;j<3u;++j)
            covariance[i][j]=second[i][j]-volume*center[i]*center[j];
    const double trace=covariance[0][0]+covariance[1][1]+covariance[2][2];
    std::array<DVec,3> inertia{};
    for(unsigned i=0;i<3u;++i)
        for(unsigned j=0;j<3u;++j)
            inertia[i][j]=(i==j?trace:0.0)-covariance[i][j];
    // Positive-definite inertia is a physical admission requirement.
    const double leading=inertia[0][0];
    const double second_minor=inertia[0][0]*inertia[1][1]-
                              inertia[0][1]*inertia[1][0];
    const double determinant=
        inertia[0][0]*(inertia[1][1]*inertia[2][2]-inertia[1][2]*inertia[2][1])-
        inertia[0][1]*(inertia[1][0]*inertia[2][2]-inertia[1][2]*inertia[2][0])+
        inertia[0][2]*(inertia[1][0]*inertia[2][1]-inertia[1][1]*inertia[2][0]);
    if(!(leading>0.0) || !(second_minor>0.0) ||
       !(determinant>0.0) || !std::isfinite(determinant))
        throw std::invalid_argument("Convex hull has invalid rigid-body inertia");

    mesh->mass_.volume=volume;
    mesh->mass_.center=as_float(world_center);
    for(unsigned j=0;j<3u;++j)
        mesh->mass_.inertia_columns[j]=as_float({
            inertia[0][j],inertia[1][j],inertia[2][j]});
    return mesh;
}

} // namespace fyzz
