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
// GuGJKSimplex.h::GJKCPairDoSimplex chooses Voronoi regions with
// specialized segment/triangle tests and the four tetrahedron faces.
// Do not enumerate all simplex subsets and solve Gram matrices here:
// that changes degenerate-feature tie order and costs considerably more
// than the original PhysX GJK inner loop.
struct Closest {
    Vec3 point{},a{},b{};
    std::array<float,4> weights{};
    std::array<int,4> order{};
    int count=0;
    unsigned mask=0;
    float dist2=std::numeric_limits<float>::max();
};
Closest make_closest(const std::array<Support,4>& simplex,
                     std::array<int,4> selected,int count,
                     std::array<float,4> weights) {
    Closest result;
    result.order=selected;
    result.count=count;
    result.weights=weights;
    for(int j=0;j<count;++j) {
        const int i=selected[j];
        result.mask|=1u<<i;
        result.point=add(result.point,mul(simplex[i].q,weights[i]));
        result.a=add(result.a,mul(simplex[i].a,weights[i]));
        result.b=add(result.b,mul(simplex[i].b,weights[i]));
    }
    result.dist2=length2(result.point);
    return result;
}
Closest closest_segment(const std::array<Support,4>& simplex,
                        int a,int b) {
    const Vec3 p=simplex[a].q,q=simplex[b].q;
    const Vec3 ab=sub(q,p);
    const float denom=dot(ab,ab);
    // GuGJKSimplex.h::closestPtPointSegment collapses only a truly
    // degenerate segment; an endpoint Voronoi witness can still retain
    // two support points in the source simplex.
    if(denom<=std::numeric_limits<float>::epsilon())
        return make_closest(simplex,{a,0,0,0},1,
                            [a]{std::array<float,4> w{};w[a]=1;return w;}());
    const float t=std::clamp(-dot(p,ab)/denom,0.0f,1.0f);
    std::array<float,4> w{};
    w[a]=1.0f-t;
    w[b]=t;
    return make_closest(simplex,{a,b,0,0},2,w);
}
Closest closest_triangle(const std::array<Support,4>& simplex,
                         int ia,int ib,int ic,bool allow_degenerate=true) {
    const Vec3 a=simplex[ia].q,b=simplex[ib].q,c=simplex[ic].q;
    const Vec3 ab=sub(b,a),ac=sub(c,a);
    const Vec3 n=cross(ab,ac);
    const float nn=length2(n);
    // Original closestPtPointTriangle uses PX_EPS_REAL^2, while its
    // tetrahedron face chooser delegates directly to barycentric regions.
    constexpr float eps=std::numeric_limits<float>::epsilon();
    if(allow_degenerate && nn<=eps*eps)
        return closest_segment(simplex,ia,ib);
    if(nn==0.0f)return {};
    const float va=dot(n,cross(b,c));
    const float vb=dot(n,cross(c,a));
    const float vc=dot(n,cross(a,b));
    const auto one=[&](int i) {
        std::array<float,4> w{};w[i]=1.0f;
        return make_closest(simplex,{i,0,0,0},1,w);
    };
    const auto edge=[&](int i,int j,float t) {
        std::array<float,4> w{};w[i]=1.0f-t;w[j]=t;
        return make_closest(simplex,{i,j,0,0},2,w);
    };
    if(va>=0.0f && vb>=0.0f && vc>=0.0f) {
        // Face projection onto the source n=AB x AC plane.
        std::array<float,4> w{};
        w[ia]=va/nn;w[ib]=vb/nn;w[ic]=vc/nn;
        Closest face=make_closest(simplex,{ia,ib,ic,0},3,w);
        face.point=mul(n,dot(n,a)/nn);
        face.dist2=length2(face.point);
        return face;
    }
    const Vec3 ap=neg(a),bp=neg(b),cp=neg(c);
    const float d1=dot(ab,ap),d2=dot(ac,ap);
    const float d3=dot(ab,bp),d4=dot(ac,bp);
    const float d5=dot(ab,cp),d6=dot(ac,cp);
    // Preserve GuGJKSimplex.h's region precedence: AB, BC, AC,
    // vertex A, vertex B, and finally vertex C.
    if(vc<=0.0f && d1>=0.0f && d3<=0.0f) {
        const float denom=d1-d3;
        return edge(ia,ib,std::abs(denom)>eps?d1/denom:0.0f);
    }
    if(va<=0.0f && d4>=d3 && d5>=d6) {
        const float num=d4-d3,denom=num+d5-d6;
        return edge(ib,ic,std::abs(denom)>eps?num/denom:0.0f);
    }
    if(vb<=0.0f && d2>=0.0f && d6<=0.0f) {
        const float denom=d2-d6;
        return edge(ia,ic,std::abs(denom)>eps?d2/denom:0.0f);
    }
    if(d1<=0.0f && d2<=0.0f)return one(ia);
    if(d3>=0.0f && d3>=d4)return one(ib);
    return one(ic);
}
Closest closest_tetrahedron(const std::array<Support,4>& simplex) {
    const Vec3 a=simplex[0].q,b=simplex[1].q;
    const Vec3 c=simplex[2].q,d=simplex[3].q;
    const Vec3 ab=sub(b,a),ac=sub(c,a),ad=sub(d,a);
    const Vec3 n=cross(ab,ac);
    const float n2=length2(n);
    // GuGJKSimplex.cpp's tetrahedron degeneracy check is a
    // normalized signed point-to-plane distance with a 1e-4 cutoff.
    if(!(n2>0.0f) ||
       std::abs(dot(n,sub(d,a))/std::sqrt(n2))<1.0e-4f)
        return closest_triangle(simplex,0,1,2);
    const Vec3 v0=n,v1=cross(ac,ad),v2=cross(ad,ab);
    const Vec3 v3=cross(sub(d,b),sub(c,b));
    const std::array<bool,4> outside{{
        dot(v0,a)*dot(v0,d)>=0.0f,
        dot(v1,a)*dot(v1,b)>=0.0f,
        dot(v2,a)*dot(v2,c)>=0.0f,
        dot(v3,b)*dot(v3,a)>=0.0f
    }};
    if(!outside[0] && !outside[1] &&
       !outside[2] && !outside[3]) {
        // Source leaves the full tetrahedron untouched when its
        // Minkowski simplex encloses the origin; EPA owns penetration.
        std::array<float,4> w{{0.25f,0.25f,0.25f,0.25f}};
        Closest inside=make_closest(simplex,{0,1,2,3},4,w);
        inside.point={};inside.dist2=0.0f;
        return inside;
    }
    constexpr std::array<std::array<int,3>,4> faces{{
        {{0,1,2}},{{0,2,3}},{{0,3,1}},{{1,3,2}}
    }};
    Closest best;
    for(int i=0;i<4;++i)if(outside[i]) {
        const auto face=faces[i];
        const Closest candidate=closest_triangle(
            simplex,face[0],face[1],face[2],false);
        // getClosestPtPointTriangle selects strictly smaller distances;
        // ties retain the preceding source face in original order.
        if(candidate.mask && candidate.dist2<best.dist2)
            best=candidate;
    }
    return best;
}
Closest closest_simplex(const std::array<Support,4>& simplex,int count) {
    switch(count) {
    case 1: {
        std::array<float,4> w{{1.0f,0.0f,0.0f,0.0f}};
        return make_closest(simplex,{0,0,0,0},1,w);
    }
    case 2: return closest_segment(simplex,0,1);
    case 3: return closest_triangle(simplex,0,1,2);
    case 4: return closest_tetrahedron(simplex);
    default: return {};
    }
}
void compact(std::array<Support,4>& simplex,int& count,
             const Closest& nearest) {
    // Preserve the exact source feature order, including zero-weight
    // endpoints retained by its nondegenerate segment routine.
    std::array<Support,4> next{};
    for(int i=0;i<nearest.count;++i)
        next[i]=simplex[nearest.order[i]];
    for(int i=0;i<nearest.count;++i)simplex[i]=next[i];
    count=nearest.count;
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
        // GuGJKPenetration terminates once the closest simplex lies
        // inside the minimum-margin epsilon. The source then admits
        // the simplex to EPA even if a small geometric gap remains.
        // EPA's signed facet distance, not a speculative GJK witness,
        // determines the final PCM contact normal and penetration.
        if(count>0 && dist<=epsilon) {
            store();
            return epa(pair,simplex,count,contact);
        }
        if(dist<=0.0f)return false;
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
                // Repeating an extreme support vertex cannot advance
                // the simplex. Source GJK's non-improving-distance
                // status is GJK_DEGENERATE, not GJK_CONTACT; the PCM
                // box-box caller does not publish that status.
                store();
                return false;
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
            return false;
        }
        previous=next_dist;
    }
    return false;
}
} // namespace fyzz::native
