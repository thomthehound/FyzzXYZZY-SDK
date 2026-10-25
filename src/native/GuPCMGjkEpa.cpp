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
    std::uint8_t ai=0,bi=0;
};
struct BoxPair {
    Vec3 ha{},hb{},a_in_b{};
    Quat qa{},qb{},relative{};
    // Original BoxV supportLocal chooses positive only for > 0.
    // The index encodes the support corner, not an arbitrary witness.
    Support support(Vec3 direction) const {
        const Vec3 dir_a=inverse_rotate(relative,direction);
        const auto vertex=[](Vec3 e,Vec3 d,std::uint8_t& index) {
            const bool x=d.x>0.0f,y=d.y>0.0f,z=d.z>0.0f;
            index=static_cast<std::uint8_t>(
                (x?1:0)|(y?2:0)|(z?4:0));
            return Vec3{x?e.x:-e.x,y?e.y:-e.y,z?e.z:-e.z};
        };
        Support s;
        const Vec3 pa=vertex(ha,dir_a,s.ai);
        s.b=vertex(hb,neg(direction),s.bi);
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
    for(int col=0;col<n;++col) {
        int pivot=col;
        for(int row=col+1;row<n;++row)
            if(std::abs(a[row][col])>std::abs(a[pivot][col]))pivot=row;
        if(std::abs(a[pivot][col])<1.0e-12f)return false;
        for(int j=col;j<=n;++j)
            std::swap(a[col][j],a[pivot][j]);
        const float div=a[col][col];
        for(int j=col;j<=n;++j)a[col][j]/=div;
        for(int row=0;row<n;++row)if(row!=col) {
            const float factor=a[row][col];
            for(int j=col;j<=n;++j)
                a[row][j]-=factor*a[col][j];
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
    Vec3 n=cross(sub(b,a),sub(c,a));
    const float norm2=length2(n);
    if(!(norm2>std::numeric_limits<float>::epsilon()))return {};
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
        const Support next=pair.support({-1,0,0});
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
        vertices[count++]=pair.support(neg(perpendicular));
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
    if(denominator<=1.0e-16f)return false;
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

bool source_pcm_box_box_gjk_epa(
    Vec3 half0,Vec3 position0,Quat rotation0,
    Vec3 half1,Vec3 position1,Quat rotation1,
    float contact_distance,PcmContactCandidate& contact,
    PcmGjkWarmstart* warmstart) {
    // The second shape is the common GJK/EPA coordinate frame.
    const Vec3 relative=inverse_rotate(rotation1,
                                      sub(position0,position1));
    const Quat inverse_b{-rotation1.x,-rotation1.y,
                         -rotation1.z,rotation1.w};
    const Quat qa{
        inverse_b.w*rotation0.x+rotation0.w*inverse_b.x+
            inverse_b.y*rotation0.z-inverse_b.z*rotation0.y,
        inverse_b.w*rotation0.y+rotation0.w*inverse_b.y+
            inverse_b.z*rotation0.x-inverse_b.x*rotation0.z,
        inverse_b.w*rotation0.z+rotation0.w*inverse_b.z+
            inverse_b.x*rotation0.y-inverse_b.y*rotation0.x,
        inverse_b.w*rotation0.w-inverse_b.x*rotation0.x-
            inverse_b.y*rotation0.y-inverse_b.z*rotation0.z
    };
    const BoxPair pair{half0,half1,relative,rotation0,rotation1,qa};
    const float epsilon=0.1f*std::min(
        box_min_margin(half0),box_min_margin(half1));
    constexpr float progress_factor=1.0f-0.000225f;
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
    Vec3 closest=length2(relative)>0.0f?relative:Vec3{1,0,0};
    float previous=std::numeric_limits<float>::max();
    for(int iteration=0;iteration<32;++iteration) {
        const float dist=std::sqrt(length2(closest));
        // Check the support plane before accepting a near-origin
        // simplex for EPA: a small positive edge gap is not overlap.
        // A zero-length simplex still needs EPA's penetration branch.
        if(dist<=0.0f) {
            if(count>0) {
                store();
                return epa(pair,simplex,count,contact);
            }
            return false;
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

        // A positive support plane proves that the Minkowski origin
        // is outside the shapes, even before the nearest simplex has
        // converged. Continue refining that separated simplex instead
        // of admitting it to EPA solely on the margin tolerance.
        // A nonpositive plane at this distance retains EPA admission.
        if(count>0 && dist<=epsilon && projection<=0.0f) {
            store();
            return epa(pair,simplex,count,contact);
        }

        for(int i=0;i<count;++i)
            if(support.ai==simplex[i].ai &&
               support.bi==simplex[i].bi) {
                // Reusing an extreme vertex closes the support search.
                // If its plane still separates the shapes, the current
                // closest witnesses are an admissible speculative pair.
                const Closest current=closest_simplex(simplex,count);
                store();
                return projection>0.0f && current.mask &&
                    current.dist2>0.0f &&
                    current.dist2<=contact_distance*contact_distance &&
                    contact_from_gjk(pair,current,contact);
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
            // A non-improving closest-point iteration cannot justify an
            // EPA overlap when a positive support plane proves a gap.
            // Use the converged simplex witness only within the actual
            // contact-distance envelope.
            store();
            return projection>0.0f && next.mask &&
                next.dist2>0.0f &&
                next.dist2<=contact_distance*contact_distance &&
                contact_from_gjk(pair,next,contact);
        }
        previous=next_dist;
    }
    return false;
}
} // namespace fyzz::native
