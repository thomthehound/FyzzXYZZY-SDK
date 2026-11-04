// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/convex.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
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
    struct EdgeIncidence {
        std::uint8_t winding=0u;
        std::size_t face=0u;
    };
    std::unordered_map<std::uint64_t,EdgeIncidence> edges;
    edges.reserve(triangles.size()*3u);
    // Two closed surfaces can each satisfy edge incidence while
    // describing the same support hull twice. Faces must share one
    // edge-connected boundary before their mass integrals are combined.
    std::vector<std::size_t> component(triangles.size());
    std::iota(component.begin(),component.end(),std::size_t{0});
    const auto root=[&](std::size_t i) {
        while(component[i]!=i) {
            component[i]=component[component[i]];
            i=component[i];
        }
        return i;
    };
    std::vector<std::uint8_t> referenced(vertices.size(),0u);
    double volume=0.0;
    DVec first{};
    std::array<DVec,3> second{};
    for(std::size_t face_index=0;face_index<triangles.size();++face_index) {
        const auto& face=triangles[face_index];
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
            auto& incidence=edges[key];
            const auto direction=std::uint8_t(u==low?1u:2u);
            if(incidence.winding&direction)
                throw std::invalid_argument("Convex surface has nonmanifold edges");
            if(incidence.winding==0u) {
                incidence.face=face_index;
            }else {
                component[root(face_index)]=root(incidence.face);
            }
            incidence.winding|=direction;
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
                   [](const auto& e){return e.second.winding!=3u;}))
        throw std::invalid_argument("Convex surface must be closed and use every vertex");
    const std::size_t boundary_root=root(0u);
    for(std::size_t i=1u;i<triangles.size();++i)
        if(root(i)!=boundary_root)
            throw std::invalid_argument("Convex surface has disconnected shells");
    // Preserve the actual support facets, not the input triangulation.
    // A planar polygon supplies one stable face feature regardless of
    // how its author chose diagonals for the mass integral.
    struct PlanarGroup {
        DVec normal{},point{};
        std::vector<std::size_t> triangles;
    };
    std::vector<PlanarGroup> groups;
    const double plane_tolerance=scale*1.e-6;
    for(std::size_t i=0;i<triangles.size();++i) {
        const auto& tri=triangles[i];
        const DVec p0=as_double(vertices[tri[0]]);
        const DVec p1=as_double(vertices[tri[1]]);
        const DVec p2=as_double(vertices[tri[2]]);
        DVec n=cross(sub(p1,p0),sub(p2,p0));
        const double length=std::sqrt(dot(n,n));
        for(auto& component:n)component/=length;
        auto group=std::find_if(groups.begin(),groups.end(),
            [&](const PlanarGroup& candidate) {
                return dot(n,candidate.normal)>1.0-1.e-7 &&
                    std::abs(dot(candidate.normal,
                                 sub(p0,candidate.point)))<=plane_tolerance &&
                    std::abs(dot(candidate.normal,
                                 sub(p1,candidate.point)))<=plane_tolerance &&
                    std::abs(dot(candidate.normal,
                                 sub(p2,candidate.point)))<=plane_tolerance;
            });
        if(group==groups.end()) {
            groups.push_back({n,p0,{i}});
        }else {
            group->triangles.push_back(i);
        }
    }
    for(const auto& group:groups) {
        struct DirectedEdge {
            std::uint32_t from=0,to=0;
        };
        std::unordered_map<std::uint64_t,DirectedEdge> boundary;
        boundary.reserve(group.triangles.size()*3u);
        for(const auto triangle_index:group.triangles) {
            const auto& tri=triangles[triangle_index];
            for(unsigned k=0u;k<3u;++k) {
                const auto from=tri[k],to=tri[(k+1u)%3u];
                const auto low=std::min(from,to),high=std::max(from,to);
                const auto key=(std::uint64_t(low)<<32u)|high;
                const auto edge=boundary.find(key);
                if(edge==boundary.end()) {
                    boundary.emplace(key,DirectedEdge{from,to});
                }else {
                    if(edge->second.from!=to || edge->second.to!=from)
                        throw std::invalid_argument(
                            "Convex polygon has inconsistent triangulation");
                    boundary.erase(edge);
                }
            }
        }
        if(boundary.size()<3u)
            throw std::invalid_argument("Convex polygon has no boundary");
        constexpr auto invalid=std::numeric_limits<std::uint32_t>::max();
        std::vector<std::uint32_t> next(vertices.size(),invalid);
        std::vector<std::uint32_t> previous(vertices.size(),invalid);
        std::uint32_t first=invalid;
        for(const auto& [key,edge]:boundary) {
            if(next[edge.from]!=invalid || previous[edge.to]!=invalid)
                throw std::invalid_argument(
                    "Convex planar facet has branching boundary");
            next[edge.from]=edge.to;
            previous[edge.to]=edge.from;
            first=std::min(first,edge.from);
        }
        if(boundary.size()>std::numeric_limits<std::uint32_t>::max() ||
           mesh->face_indices_.size()>
               std::numeric_limits<std::uint32_t>::max()-boundary.size())
            throw std::overflow_error("Convex face index stream too large");
        const auto offset=static_cast<std::uint32_t>(
            mesh->face_indices_.size());
        std::uint32_t current=first;
        for(std::size_t i=0u;i<boundary.size();++i) {
            if(current==invalid || next[current]==invalid ||
               (i>0u && current==first))
                throw std::invalid_argument(
                    "Convex planar facet is not a single boundary loop");
            mesh->face_indices_.push_back(current);
            current=next[current];
        }
        if(current!=first)
            throw std::invalid_argument(
                "Convex planar facet has disconnected boundary loops");
        const double distance=dot(group.normal,group.point);
        if(!std::isfinite(distance) ||
           std::abs(distance)>std::numeric_limits<float>::max())
            throw std::invalid_argument("Convex facet plane outside float range");
        mesh->faces_.push_back({
            as_float(group.normal),static_cast<float>(distance),offset,
            static_cast<std::uint32_t>(boundary.size())});
    }
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
