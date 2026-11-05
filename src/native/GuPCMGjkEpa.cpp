// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// GuGJKPenetration / GuEPA box support continuation for the PCM
// box-box generator. Source GJK termination distances and EPA tolerance
// are retained; fixed local working sets avoid per-contact allocations.

#include "native/scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>

namespace fyzz::native {
namespace {
Vec3 add(Vec3 a,Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
Vec3 sub(Vec3 a,Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
Vec3 mul(Vec3 a,float s) { return {a.x*s,a.y*s,a.z*s}; }
Vec3 neg(Vec3 a) { return {-a.x,-a.y,-a.z}; }
float dot(Vec3 a,Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
Vec3 cross(Vec3 a,Vec3 b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
float length2(Vec3 v) { return dot(v,v); }
float box_min_margin(Vec3 half) {
    // GuVecBox.h::CalculateBoxMargin, BOX_MIN_MARGIN_RATIO. This is
    // not the larger BOX_MARGIN_RATIO used for PCM cache refresh.
    return 0.05f*std::min({half.x,half.y,half.z});
}
Vec3 rotate(Quat q,Vec3 v) {
    const Vec3 u{q.x,q.y,q.z};
    const Vec3 t=mul(cross(u,v),2.0f);
    return add(v,add(mul(t,q.w),cross(u,t)));
}
Vec3 inverse_rotate(Quat q,Vec3 v) {
    return rotate({-q.x,-q.y,-q.z,q.w},v);
}
struct Support {
    Vec3 a{},b{},q{};
    std::uint32_t ai=0,bi=0;
};
struct BoxPair {
    Vec3 ha{},hb{},a_in_b{};
    Quat qa{},qb{},relative{};
    std::span<const Vec3> vertices_a{},vertices_b{};
    float radius_a=0.f,radius_b=0.f;
    // The cooked-hull support is a direct extreme-vertex search, not
    // the AABB proxy. Polyhedral feature IDs remain full width.
    static Vec3 vertex(Vec3 extent,Vec3 direction,
                       std::span<const Vec3> hull,float radius,
                       std::uint32_t& index) {
        index=0u;
        if(radius>0.f)return {};
        if(!hull.empty()) {
            float best=dot(hull.front(),direction);
            for(std::uint32_t i=1u;i<hull.size();++i) {
                const float projection=dot(hull[i],direction);
                if(projection>best) {
                    best=projection;
                    index=i;
                }
            }
            return hull[index];
        }
        const bool x=direction.x>0.f,y=direction.y>0.f,z=direction.z>0.f;
        index=(x?1u:0u)|(y?2u:0u)|(z?4u:0u);
        return {x?extent.x:-extent.x,
                y?extent.y:-extent.y,z?extent.z:-extent.z};
    }
    Support support(Vec3 direction) const {
        const Vec3 dir_a=inverse_rotate(relative,direction);
        Support s;
        const Vec3 pa=vertex(ha,dir_a,vertices_a,radius_a,s.ai);
        s.b=vertex(hb,neg(direction),vertices_b,radius_b,s.bi);
        s.a=add(a_in_b,rotate(relative,pa));
        s.q=sub(s.a,s.b);
        return s;
    }
    Vec3 local_a(Vec3 point_in_b) const {
        return inverse_rotate(relative,sub(point_in_b,a_in_b));
    }
};
// Dense (<=3) pivoted affine solve for simplex barycentric coordinates.
// The source GJKCPairDoSimplex selects the nearest Voronoi feature; the
// subset solve performs the same constrained closest-origin problem.
bool solve(int n,float a[3][4],float result[3]) {
    // The Gram matrix carries squared lengths. An absolute pivot cutoff
    // would reject the same valid simplex when the boxes are small.
    float matrix_scale=0.0f;
    for(int i=0;i<n;++i)
        matrix_scale=std::max(matrix_scale,std::abs(a[i][i]));
    if(!(matrix_scale>0.0f) || !std::isfinite(matrix_scale))
        return false;
    const float pivot_tolerance=
        std::numeric_limits<float>::epsilon()*matrix_scale;
    for(int col=0;col<n;++col) {
        int pivot=col;
        for(int row=col+1;row<n;++row)
            if(std::abs(a[row][col])>std::abs(a[pivot][col]))pivot=row;
        if(!(std::abs(a[pivot][col])>pivot_tolerance))
            return false;
        for(int j=col;j<=n;++j)
            std::swap(a[col][j],a[pivot][j]);
        const float div=a[col][col];
        for(int j=col;j<=n;++j)a[col][j]/=div;
        for(int row=0;row<n;++row)if(row!=col) {
            const float factor=a[row][col];
            for(int j=col;j<=n;++j)
                a[row][j]=std::fma(-factor,a[col][j],a[row][j]);
        }
    }
    for(int i=0;i<n;++i)result[i]=a[i][n];
    return true;
}
struct Closest {
    Vec3 point{},a{},b{};
    std::array<float,4> weights{};
    unsigned mask=0;
    float dist2=std::numeric_limits<float>::max();
};
Closest closest_simplex(const std::array<Support,4>& simplex,int count) {
    Closest best;
    for(unsigned mask=1;mask<(1u<<count);++mask) {
        int ids[4]{},n=0;
        for(int i=0;i<count;++i)
            if(mask&(1u<<i))ids[n++]=i;
        std::array<float,4> weights{};
        weights[ids[0]]=1.0f;
        if(n>1) {
            const Vec3 base=simplex[ids[0]].q;
            Vec3 directions[3]{};
            float matrix[3][4]{};
            for(int i=1;i<n;++i)
                directions[i-1]=sub(simplex[ids[i]].q,base);
            for(int i=1;i<n;++i) {
                for(int j=1;j<n;++j)
                    matrix[i-1][j-1]=dot(directions[i-1],
                                         directions[j-1]);
                matrix[i-1][n-1]=-dot(directions[i-1],base);
            }
            float coefficients[3]{};
            if(!solve(n-1,matrix,coefficients))continue;
            for(int i=1;i<n;++i) {
                weights[ids[i]]=coefficients[i-1];
                weights[ids[0]]-=coefficients[i-1];
            }
        }
        bool feasible=true;
        for(int i=0;i<n;++i)
            if(weights[ids[i]]< -1.0e-5f)feasible=false;
        if(!feasible)continue;
        Vec3 q{},pa{},pb{};
        for(int i=0;i<count;++i) {
            q=add(q,mul(simplex[i].q,weights[i]));
            pa=add(pa,mul(simplex[i].a,weights[i]));
            pb=add(pb,mul(simplex[i].b,weights[i]));
        }
        const float d=length2(q);
        if(d<best.dist2) {
            best={q,pa,pb,weights,mask,d};
        }
    }
    return best;
}
void compact(std::array<Support,4>& simplex,int& count,
             const Closest& closest) {
    int write=0;
    for(int read=0;read<count;++read)
        if((closest.mask&(1u<<read)) &&
           closest.weights[read]>1.0e-6f)
            simplex[write++]=simplex[read];
    if(write==0 && count>0) {
        simplex[0]=simplex[0];
        write=1;
    }
    count=write;
}
bool contact_from_gjk(const BoxPair& pair,Closest const& nearest,
                      PcmContactCandidate& output) {
    const float distance=std::sqrt(nearest.dist2);
    if(!(distance>0.0f) || !std::isfinite(distance))return false;
    const Vec3 normal=mul(nearest.point,1.0f/distance);
    output={pair.local_a(nearest.a),nearest.b,normal,distance};
    return true;
}
struct Facet {
    int i=0,j=0,k=0;
    Vec3 normal{};
    float distance=0.0f;
    bool valid=false;
    bool in_heap=true; // source addFacet queues only <= current upper bound
};
// GuEPAFacet::isValid2 retains the signed distance and input winding.
// The source EPA accepts a negative plane distance when an initial
// point/segment/triangle has not yet enclosed the origin; flipping
// every face to positive would erase exactly that contact branch.
Facet make_facet(const std::array<Support,64>& vertices,
                 int i,int j,int k) {
    const Vec3 a=vertices[i].q,b=vertices[j].q,c=vertices[k].q;
    const Vec3 e0=sub(b,a),e1=sub(c,a);
    Vec3 n=cross(e0,e1);
    const float norm2=length2(n);
    // GuEPA.cpp::Facet::isValid2 compares the squared cross product
    // directly with FEps(), not with a scale-relative area tolerance.
    // Tiny triangles may therefore fail source EPA facet admission.
    if(!(norm2>std::numeric_limits<float>::epsilon()))
        return {};
    n=mul(n,1.0f/std::sqrt(norm2));
    return {i,j,k,n,dot(n,a),true};
}
// GuEPA.cpp::expandPoint, expandSegment, expandTriangle and the source
// four-vertex tetrahedral admission. All support points remain in B's
// coordinate frame. Unlike a synthetic origin-containing tetra search,
// this reproduces the valid back-to-back initial facet pair.
bool initialize_epa(const BoxPair& pair,
                    std::array<Support,64>& vertices,
                    int& count,std::array<Facet,64>& facets,
                    int& facet_count) {
    if(count<=0 || count>4)return false;
    if(count==1) {
        const Support next=pair.support({1,0,0});
        if(length2(sub(next.q,vertices[0].q))==0.0f)return false;
        vertices[count++]=next;
    }
    if(count==2) {
        const Vec3 edge=sub(vertices[1].q,vertices[0].q);
        const Vec3 ab{std::abs(edge.x),std::abs(edge.y),std::abs(edge.z)};
        Vec3 axis{1,0,0};
        if(ab.x>ab.y && ab.z>ab.y)
            axis={0,1,0};
        else if(ab.x>ab.z)
            axis={0,0,1};
        Vec3 perpendicular=cross(axis,edge);
        const float sq=length2(perpendicular);
        if(!(sq>0.0f))return false;
        perpendicular=mul(perpendicular,1.0f/std::sqrt(sq));
        vertices[count++]=pair.support(perpendicular);
    }
    if(count==3) {
        const Facet f0=make_facet(vertices,0,1,2);
        const Facet f1=make_facet(vertices,1,0,2);
        if(!f0.valid || !f1.valid)return false;
        facets[facet_count++]=f0;
        facets[facet_count++]=f1;
        return true;
    }
    const Vec3 p0=vertices[0].q;
    const Vec3 v1=sub(vertices[1].q,p0);
    const Vec3 v2=sub(vertices[2].q,p0);
    if(dot(cross(v1,v2),sub(vertices[3].q,p0))>0.0f)
        std::swap(vertices[1],vertices[2]);
    constexpr std::array<std::array<int,3>,4> tetra_faces{{
        {0,1,2},{0,3,1},{0,2,3},{1,3,2}
    }};
    for(const auto& face:tetra_faces) {
        const Facet facet=make_facet(
            vertices,face[0],face[1],face[2]);
        if(!facet.valid)return false;
        facets[facet_count++]=facet;
    }
    return true;
}
bool contact_from_facet(const BoxPair& pair,
                        const std::array<Support,64>& vertices,
                        Facet const& facet,PcmContactCandidate& output) {
    const auto& p=vertices[facet.i];
    const auto& q=vertices[facet.j];
    const auto& r=vertices[facet.k];
    // EPA Facet::getClosestPoint: barycentric projection of the origin
    // onto its nearest Minkowski facet, with the shape witnesses
    // interpolated by identical weights.
    const Vec3 e0=sub(q.q,p.q),e1=sub(r.q,p.q);
    const float a=dot(e0,e0),b=dot(e0,e1),c=dot(e1,e1);
    const float denominator=a*c-b*b;
    // The barycentric Gram determinant also scales as length^4.
    // Reject near-collinear facets by relative area, not world units.
    if(!(denominator>std::numeric_limits<float>::epsilon()*a*c))
        return false;
    const float x=-dot(p.q,e0),y=-dot(p.q,e1);
    const float u=(x*c-y*b)/denominator;
    const float v=(y*a-x*b)/denominator;
    const float w=1.0f-u-v;
    const Vec3 pa=add(add(mul(p.a,w),mul(q.a,u)),mul(r.a,v));
    const Vec3 pb=add(add(mul(p.b,w),mul(q.b,u)),mul(r.b,v));
    if(!std::isfinite(pa.x) || !std::isfinite(pb.x) ||
       !std::isfinite(facet.distance))return false;
    // Source EPA::calculateContactInformation negates the polytope
    // outward facet normal and writes negative penetration depth.
    const Vec3 normal=neg(facet.normal);
    const float depth=std::abs(facet.distance);
    output={pair.local_a(pa),pb,normal,-depth};
    const Vec3 difference=sub(pa,pb);
    return std::sqrt(length2(difference))<=depth+1.0e-3f;
}
bool epa(const BoxPair& pair,
         const std::array<Support,4>& simplex,int simplex_count,
         PcmContactCandidate& output) {
    std::array<Support,64> vertices{};
    int vertices_count=simplex_count;
    for(int i=0;i<simplex_count;++i)vertices[i]=simplex[i];
    // GuEPAFacet.h: MaxFacets=64, MaxEdges=32,
    // MaxSupportPoints=64. Retired facets return to the fixed ID pool.
    std::array<Facet,64> facets{};
    int facet_count=0;
    if(!initialize_epa(
            pair,vertices,vertices_count,facets,facet_count))
        return false;
    const float epsilon=0.1f*std::min(
        box_min_margin(pair.ha),box_min_margin(pair.hb));
    struct Edge {int from,to;};
    std::array<Edge,32> boundary{};
    float upper_bound=std::numeric_limits<float>::max();
    for(int iteration=0;iteration<60;++iteration) {
        int closest=-1;
        float distance=std::numeric_limits<float>::max();
        for(int i=0;i<facet_count;++i)
            if(facets[i].valid && facets[i].in_heap &&
               facets[i].distance<distance) {
                closest=i;
                distance=facets[i].distance;
            }
        if(closest<0)return false;
        const Facet selected=facets[closest];
        // GJK's BoxV support is expressed in the second box's frame.
        const Support new_point=pair.support(selected.normal);
        const float projection=dot(new_point.q,selected.normal);
        if(std::abs(projection-distance)<=epsilon)
            return contact_from_facet(pair,vertices,selected,output);
        if(vertices_count==64)return false;
        upper_bound=std::min(upper_bound,projection);
        bool duplicate=false;
        for(int i=0;i<vertices_count;++i)
            if(vertices[i].ai==new_point.ai &&
               vertices[i].bi==new_point.bi) {
                duplicate=true;
                break;
            }
        if(duplicate)
            return contact_from_facet(pair,vertices,selected,output);
        const int new_index=vertices_count++;
        vertices[new_index]=new_point;
        int boundary_count=0;
        for(int i=0;i<facet_count;++i) {
            Facet& f=facets[i];
            if(!f.valid)continue;
            // GuEPAFacet::silhouette only retains facets whose signed
            // point-to-plane distance is strictly negative.
            if(dot(f.normal,sub(new_point.q,vertices[f.i].q))<0.0f)
                continue;
            f.valid=false;
            for(const Edge e:std::array<Edge,3>{{
                    {f.i,f.j},{f.j,f.k},{f.k,f.i}}}) {
                int reverse=-1;
                for(int j=0;j<boundary_count;++j)
                    if(boundary[j].from==e.to &&
                       boundary[j].to==e.from) {
                        reverse=j;
                        break;
                    }
                if(reverse>=0)
                    boundary[reverse]=boundary[--boundary_count];
                else {
                    if(boundary_count==static_cast<int>(boundary.size()))
                        return false;
                    boundary[boundary_count++]=e;
                }
            }
        }
        if(boundary_count==0)return false;
        // EPAFacetManager returns obsolete facet IDs to its pool
        // before new horizon triangles are admitted. This preserves
        // the source's 64-slot fixed working set across iterations.
        int available_slots=static_cast<int>(facets.size())-facet_count;
        for(int i=0;i<facet_count;++i)
            available_slots+=!facets[i].valid;
        if(boundary_count>available_slots)return false;
        for(int i=0;i<boundary_count;++i) {
            const auto e=boundary[i];
            // These horizon edges come from removed (visible) facets.
            // The surviving neighbor has opposite winding, so this
            // orientation already matches source addFacet(target,source).
            Facet f=make_facet(vertices,e.from,e.to,new_index);
            if(!f.valid)return false;
            // GuEPA.cpp::addFacet accepts triangles outside the
            // current upper bound as topology, but not heap entries.
            f.in_heap=f.distance<=upper_bound;
            int slot=-1;
            for(int j=0;j<facet_count;++j)
                if(!facets[j].valid) {
                    slot=j;
                    break;
                }
            if(slot<0)slot=facet_count++;
            facets[slot]=f;
        }
    }
    return false;
}
} // namespace

static bool gjk_epa(
    const BoxPair& pair,float contact_distance,
    PcmContactCandidate& contact,PcmGjkWarmstart* warmstart) {
    const float epsilon=0.1f*std::min(
        box_min_margin(pair.ha),box_min_margin(pair.hb));
    constexpr float progress_factor=1.0f-0.000225f;
    const bool point_core=pair.radius_a>0.f || pair.radius_b>0.f;
    std::array<Support,4> simplex{};
    int count=0;
    if(warmstart)*warmstart={};
    const auto store=[&] {
        if(!warmstart)return;
        warmstart->count=static_cast<std::uint8_t>(count);
        for(int i=0;i<count;++i) {
            warmstart->a[i]=simplex[i].ai;
            warmstart->b[i]=simplex[i].bi;
        }
    };
    Vec3 closest=length2(pair.a_in_b)>0.0f
        ?pair.a_in_b:Vec3{1,0,0};
    float previous=std::numeric_limits<float>::max();
    for(int iteration=0;iteration<32;++iteration) {
        const float dist=std::sqrt(length2(closest));
        // Polyhedral cores retain the source EPA admission margin.
        // Sphere point cores must converge their separated distance
        // before the analytical radius is applied; their near-face
        // simplex is not a valid EPA penetration polytope.
        if(count>0 && dist<=epsilon && !point_core) {
            store();
            return epa(pair,simplex,count,contact);
        }
        if(dist<=0.0f) {
            if(!point_core || count==0)return false;
            store();
            return epa(pair,simplex,count,contact);
        }
        const Vec3 normal=mul(closest,1.0f/dist);
        const Support support=pair.support(neg(closest));
        const float projection=dot(normal,support.q);
        if(projection>contact_distance) {
            store();
            return false; // GJK_NON_INTERSECT
        }
        // A positive supporting plane establishes geometric separation
        // even when the closest simplex is below the GJK/EPA margin
        // threshold. Source convergence must precede EPA admission:
        // proximity to the origin is not evidence that it lies inside
        // the Minkowski difference. This matters for speculative
        // rotated edge/edge contacts with a sub-epsilon positive gap.
        if(count>0 && projection>dist*progress_factor) {
            const Closest current=closest_simplex(simplex,count);
            store();
            return current.mask &&
                contact_from_gjk(pair,current,contact);
        }

        for(int i=0;i<count;++i)
            if(support.ai==simplex[i].ai &&
               support.bi==simplex[i].bi) {
                // A repeated support cannot improve the simplex. With a
                // positive supporting plane, the separation lies between
                // that projection and the simplex distance. A witness
                // inside contact distance remains geometrically admissible.
                store();
                if(projection>0.f) {
                    const Closest current=closest_simplex(simplex,count);
                    return current.mask &&
                        contact_from_gjk(pair,current,contact) &&
                        contact.separation<=contact_distance;
                }
                return point_core && epa(pair,simplex,count,contact);
            }
        if(count==4) {
            store();
            return epa(pair,simplex,count,contact);
        }
        simplex[count++]=support;
        const Closest next=closest_simplex(simplex,count);
        if(!next.mask) {
            store();
            return false;
        }
        closest=next.point;
        const float next_dist=std::sqrt(next.dist2);
        compact(simplex,count,next);
        if(next_dist>=previous) {
            // GuGJKPenetration returns GJK_DEGENERATE when its simplex
            // distance fails to decrease. In the admitted box/box
            // caller only GJK_CONTACT and EPA_CONTACT produce witnesses.
            // A positive projected gap does not override that status.
            store();
            if(point_core) {
                if(projection>0.f)
                    return contact_from_gjk(pair,next,contact);
                return epa(pair,simplex,count,contact);
            }
            return false;
        }
        previous=next_dist;
    }
    if(point_core && count>0) {
        store();
        return epa(pair,simplex,count,contact);
    }
    return false;
}
bool source_pcm_box_box_gjk_epa(
    Vec3 half0,Vec3 position0,Quat rotation0,
    Vec3 half1,Vec3 position1,Quat rotation1,
    float contact_distance,PcmContactCandidate& contact,
    PcmGjkWarmstart* warmstart) {
    const PcmFiniteShape a{half0,position0,rotation0};
    const PcmFiniteShape b{half1,position1,rotation1};
    return source_pcm_convex_gjk_epa(
        a,b,contact_distance,contact,warmstart);
}

bool source_pcm_convex_gjk_epa(
    const PcmFiniteShape& first,const PcmFiniteShape& second,
    float contact_distance,PcmContactCandidate& contact,
    PcmGjkWarmstart* warmstart) {
    const Vec3 relative=inverse_rotate(second.orientation,
        sub(first.position,second.position));
    const Quat inverse_b{-second.orientation.x,-second.orientation.y,
                         -second.orientation.z,second.orientation.w};
    const Quat& rotation0=first.orientation;
    const Quat qa{
        inverse_b.w*rotation0.x+rotation0.w*inverse_b.x+
            inverse_b.y*rotation0.z-inverse_b.z*rotation0.y,
        inverse_b.w*rotation0.y+rotation0.w*inverse_b.y+
            inverse_b.z*rotation0.x-inverse_b.x*rotation0.z,
        inverse_b.w*rotation0.z+rotation0.w*inverse_b.z+
            inverse_b.x*rotation0.y-inverse_b.y*rotation0.x,
        inverse_b.w*rotation0.w-inverse_b.x*rotation0.x-
            inverse_b.y*rotation0.y-inverse_b.z*rotation0.z};
    const BoxPair pair{first.half_extents,second.half_extents,
        relative,first.orientation,second.orientation,qa,
        first.hull_vertices,second.hull_vertices,
        first.sphere_radius,second.sphere_radius};
    if(!gjk_epa(pair,contact_distance+
                    first.sphere_radius+second.sphere_radius,
                contact,warmstart))
        return false;
    // The simplex sees spheres as point supports. Analytical radius
    // offsets recover the exact sphere surface along the final B-local
    // contact normal without approximating the sphere with a polytope.
    if(first.sphere_radius>0.f) {
        const Vec3 offset=inverse_rotate(qa,
            mul(contact.normal_b,-first.sphere_radius));
        contact.local_a=add(contact.local_a,offset);
        contact.separation-=first.sphere_radius;
    }
    if(second.sphere_radius>0.f) {
        contact.local_b=add(contact.local_b,
            mul(contact.normal_b,second.sphere_radius));
        contact.separation-=second.sphere_radius;
    }
    return contact.separation<=contact_distance;
}

std::uint8_t source_pcm_convex_face_contacts(
    const PcmFiniteShape& first,const PcmFiniteShape& second,
    float contact_distance,const PcmContactCandidate& witness,
    std::span<PcmContactCandidate> output) {
    if(output.empty() || first.sphere_radius>0.f ||
       second.sphere_radius>0.f ||
       (!first.hull_vertices.empty() && first.hull_faces.empty()) ||
       (!second.hull_vertices.empty() && second.hull_faces.empty()))
        return 0u;

    const auto axis=[](unsigned i) -> Vec3 {
        switch(i) {
            case 0: return {1.f,0.f,0.f};
            case 1: return {0.f,1.f,0.f};
            default: return {0.f,0.f,1.f};
        }
    };
    const auto component=[](Vec3 v,unsigned i) {
        return i==0u?v.x:(i==1u?v.y:v.z);
    };
    const auto face_count=[](const PcmFiniteShape& shape) {
        return shape.hull_vertices.empty()?6u:
            static_cast<unsigned>(shape.hull_faces.size());
    };
    const auto face_normal=[&](const PcmFiniteShape& shape,
                               unsigned face) {
        if(!shape.hull_vertices.empty())
            return shape.hull_faces[face].normal;
        return mul(axis(face/2u),(face&1u)?1.f:-1.f);
    };
    const auto vertex_count=[&](const PcmFiniteShape& shape,
                                unsigned face) {
        return shape.hull_vertices.empty()?4u:
            shape.hull_faces[face].index_count;
    };
    const auto vertex_local=[&](const PcmFiniteShape& shape,
                                unsigned face,unsigned ordinal) {
        if(!shape.hull_vertices.empty()) {
            const auto& f=shape.hull_faces[face];
            return shape.hull_vertices[
                shape.hull_face_indices[f.first_index+ordinal]];
        }
        const Vec3 n=face_normal(shape,face);
        const unsigned coordinate=face/2u;
        const Vec3 u=axis((coordinate+1u)%3u);
        const Vec3 v=cross(n,u);
        constexpr std::array<std::array<float,2>,4> corners{{
            {1.f,1.f},{-1.f,1.f},{-1.f,-1.f},{1.f,-1.f}}};
        const float hu=component(shape.half_extents,(coordinate+1u)%3u);
        const float hv=component(shape.half_extents,(coordinate+2u)%3u);
        return add(mul(n,component(shape.half_extents,coordinate)),
                   add(mul(u,corners[ordinal][0]*hu),
                       mul(v,corners[ordinal][1]*hv)));
    };
    const auto to_b=[&](Vec3 p) {
        return inverse_rotate(second.orientation,
            sub(add(first.position,rotate(first.orientation,p)),
                second.position));
    };
    const auto face_vertex=[&](const PcmFiniteShape& shape,
                               bool shape_is_a,unsigned face,
                               unsigned ordinal) {
        const Vec3 local=vertex_local(shape,face,ordinal);
        return shape_is_a?to_b(local):local;
    };
    const auto face_normal_b=[&](const PcmFiniteShape& shape,
                                bool shape_is_a,unsigned face) {
        const Vec3 normal=face_normal(shape,face);
        return shape_is_a
            ?inverse_rotate(second.orientation,
                rotate(first.orientation,normal))
            :normal;
    };
    const Vec3 normal=witness.normal_b;
    if(!(length2(normal)>0.f))return 0u;
    unsigned face_a=0u,face_b=0u;
    float alignment_a=-2.f,alignment_b=-2.f;
    for(unsigned i=0;i<face_count(first);++i) {
        const float d=dot(neg(normal),face_normal_b(first,true,i));
        if(d>alignment_a){alignment_a=d;face_a=i;}
    }
    for(unsigned i=0;i<face_count(second);++i) {
        const float d=dot(normal,face_normal_b(second,false,i));
        if(d>alignment_b){alignment_b=d;face_b=i;}
    }
    const bool reference_b=alignment_b>=alignment_a;
    if(std::max(alignment_a,alignment_b)<0.95f)
        return 0u; // Edge/edge: retain the exact GJK/EPA witness.
    const auto& reference=reference_b?second:first;
    const auto& incident=reference_b?first:second;
    const bool reference_is_a=!reference_b;
    const bool incident_is_a=reference_b;
    const unsigned reference_face=reference_b?face_b:face_a;
    const Vec3 reference_normal=face_normal_b(
        reference,reference_is_a,reference_face);
    unsigned incident_face=0u;
    float incident_alignment=-2.f;
    for(unsigned i=0;i<face_count(incident);++i) {
        const float d=dot(neg(reference_normal),
            face_normal_b(incident,incident_is_a,i));
        if(d>incident_alignment){incident_alignment=d;incident_face=i;}
    }
    const Vec3 incident_normal=face_normal_b(
        incident,incident_is_a,incident_face);
    const float denominator=dot(incident_normal,reference_normal);
    if(!(denominator< -0.25f))return 0u;

    const unsigned nr=vertex_count(reference,reference_face);
    const unsigned ni=vertex_count(incident,incident_face);
    if(nr<3u || ni<3u)return 0u;
    const auto r=[&](unsigned i) {
        return face_vertex(reference,reference_is_a,reference_face,i%nr);
    };
    const auto q=[&](unsigned i) {
        return face_vertex(incident,incident_is_a,incident_face,i%ni);
    };
    const Vec3 anchor=r(0u);
    const Vec3 incident_anchor=q(0u);
    // Clip-plane tolerance depends on the incident polygon's physical
    // dimensions, not on the hull's arbitrary authoring origin.
    float edge_length_sq=0.f;
    for(unsigned i=0u;i<nr;++i)
        edge_length_sq=std::max(edge_length_sq,
            length2(sub(r(i+1u),r(i))));
    for(unsigned i=0u;i<ni;++i)
        edge_length_sq=std::max(edge_length_sq,
            length2(sub(q(i+1u),q(i))));
    const float scale=std::sqrt(edge_length_sq);
    const float tolerance=std::max(scale*1.e-5f,1.e-6f);
    const float tolerance_sq=tolerance*tolerance;
    const auto cross2=[&](Vec3 a,Vec3 b) {
        return dot(reference_normal,cross(a,b));
    };
    const auto inside_reference=[&](Vec3 point) {
        for(unsigned i=0u;i<nr;++i)
            if(cross2(sub(r(i+1u),r(i)),sub(point,r(i)))<
               -tolerance*scale)
                return false;
        return true;
    };
    const auto inside_incident=[&](Vec3 point) {
        for(unsigned i=0u;i<ni;++i)
            if(cross2(sub(q(i+1u),q(i)),sub(point,q(i)))>
               tolerance*scale)
                return false;
        return true;
    };
    std::uint8_t count=0u;
    const auto emit=[&](Vec3 incident_point) {
        const float separation=dot(reference_normal,
            sub(incident_point,anchor));
        if(!std::isfinite(separation) || separation>contact_distance)
            return;
        const Vec3 projected=sub(incident_point,
            mul(reference_normal,separation));
        const Vec3 a=reference_b?incident_point:projected;
        const Vec3 b=reference_b?projected:incident_point;
        const Vec3 a_local=inverse_rotate(first.orientation,
            sub(add(second.position,rotate(second.orientation,a)),
                first.position));
        const PcmContactCandidate candidate{
            a_local,b,reference_b?reference_normal:neg(reference_normal),
            separation};
        if(!std::isfinite(a_local.x) || !std::isfinite(b.x))
            return;
        for(unsigned i=0u;i<count;++i) {
            if(length2(sub(output[i].local_b,candidate.local_b))<=
               tolerance_sq) {
                if(candidate.separation<output[i].separation)
                    output[i]=candidate;
                return;
            }
        }
        const auto capacity=std::min<std::size_t>(output.size(),4u);
        if(count<capacity) {
            output[count++]=candidate;
            return;
        }
        if(capacity<4u)return;
        // Greedy five-point reduction retains the deepest contact and
        // maximizes the projected manifold footprint. It uses the same
        // bounded four-witness storage as the persistent Sc EdgeIndex.
        std::array<PcmContactCandidate,5> choices{
            output[0],output[1],output[2],output[3],candidate};
        unsigned deepest=0u;
        for(unsigned i=1u;i<5u;++i)
            if(choices[i].separation<choices[deepest].separation)
                deepest=i;
        float best=-1.f;
        unsigned omit=4u;
        for(unsigned excluded=0u;excluded<5u;++excluded) {
            if(excluded==deepest)continue;
            float score=0.f;
            for(unsigned i=0u;i<5u;++i) {
                if(i==excluded)continue;
                for(unsigned j=i+1u;j<5u;++j)
                    if(j!=excluded)
                        score+=length2(sub(
                            choices[i].local_b,choices[j].local_b));
            }
            if(score>best) {best=score;omit=excluded;}
        }
        if(omit!=4u)output[omit]=candidate;
    };
    for(unsigned i=0u;i<ni;++i) {
        const Vec3 v=q(i);
        if(inside_reference(v))emit(v);
    }
    for(unsigned i=0u;i<nr;++i) {
        const Vec3 v=r(i);
        if(!inside_incident(v))continue;
        const float height=dot(incident_normal,
            sub(incident_anchor,v))/denominator;
        if(std::isfinite(height))
            emit(add(v,mul(reference_normal,height)));
    }
    for(unsigned i=0u;i<ni;++i) {
        const Vec3 a=q(i),e=sub(q(i+1u),a);
        for(unsigned j=0u;j<nr;++j) {
            const Vec3 b=r(j),f=sub(r(j+1u),b);
            const float determinant=cross2(e,f);
            if(std::abs(determinant)<=tolerance_sq)continue;
            const Vec3 d=sub(b,a);
            const float t=cross2(d,f)/determinant;
            const float u=cross2(d,e)/determinant;
            if(t>=-tolerance && t<=1.f+tolerance &&
               u>=-tolerance && u<=1.f+tolerance)
                emit(add(a,mul(e,std::clamp(t,0.f,1.f))));
        }
    }
    return count;
}

} // namespace fyzz::native
