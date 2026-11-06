// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// The pending edge's unused contact slots hold the compact GJK recurrence.
// The finalizer overwrites these slots with the persistent Gu witness.
struct GjkGpuState {
    GjkSimplex simplex;
    float3 closest;
    float previous;
    float4 weights;
    float dist2;
    uint mask;
    uint phase;              // 0: advance, 1: reduce, 2: terminal
    uint outcome;            // 0: reject, 1: contact, 3: EPA
    float projection;
    float contactLimit;
    uint pointCore;
};
GjkGpuState gjkInitialState(GjkBoxPair pair,float distance) {
    GjkGpuState state=(GjkGpuState)0;
    state.closest=gjkLength2(pair.aInB)>0.f
        ?pair.aInB:float3(1.f,0.f,0.f);
    state.previous=3.402823466e+38f;
    state.dist2=state.previous;
    state.contactLimit=distance+pair.radiusA+pair.radiusB;
    state.pointCore=pair.radiusA>0.f || pair.radiusB>0.f?1u:0u;
    return state;
}
GjkCompactPoint gjkReadGpuPoint(uint offset) {
    GjkCompactPoint feature=(GjkCompactPoint)0;
    feature.q=asfloat(edgeCaches.Load3(offset));
    feature.ai=edgeCaches.Load(offset+12u);
    feature.bi=edgeCaches.Load(offset+16u);
    return feature;
}
void gjkWriteGpuPoint(uint offset,GjkCompactPoint feature) {
    edgeCaches.Store3(offset,asuint(feature.q));
    edgeCaches.Store(offset+12u,feature.ai);
    edgeCaches.Store(offset+16u,feature.bi);
}
GjkGpuState gjkLoadGpuState(uint cache) {
    GjkGpuState state=(GjkGpuState)0;
    state.simplex.point0=gjkReadGpuPoint(cache+80u);
    state.simplex.point1=gjkReadGpuPoint(cache+100u);
    state.simplex.point2=gjkReadGpuPoint(cache+120u);
    state.simplex.point3=gjkReadGpuPoint(cache+140u);
    state.simplex.count=edgeCaches.Load(cache+160u);
    state.closest=asfloat(edgeCaches.Load3(cache+164u));
    state.previous=asfloat(edgeCaches.Load(cache+176u));
    state.weights=asfloat(edgeCaches.Load4(cache+180u));
    state.dist2=asfloat(edgeCaches.Load(cache+196u));
    state.mask=edgeCaches.Load(cache+200u);
    state.phase=edgeCaches.Load(cache+204u);
    state.outcome=edgeCaches.Load(cache+208u);
    state.projection=asfloat(edgeCaches.Load(cache+212u));
    state.contactLimit=asfloat(edgeCaches.Load(cache+216u));
    state.pointCore=edgeCaches.Load(cache+220u);
    return state;
}
void gjkStoreGpuState(uint cache,GjkGpuState state) {
    gjkWriteGpuPoint(cache+80u,state.simplex.point0);
    gjkWriteGpuPoint(cache+100u,state.simplex.point1);
    gjkWriteGpuPoint(cache+120u,state.simplex.point2);
    gjkWriteGpuPoint(cache+140u,state.simplex.point3);
    edgeCaches.Store(cache+160u,state.simplex.count);
    edgeCaches.Store3(cache+164u,asuint(state.closest));
    edgeCaches.Store(cache+176u,asuint(state.previous));
    edgeCaches.Store4(cache+180u,asuint(state.weights));
    edgeCaches.Store(cache+196u,asuint(state.dist2));
    edgeCaches.Store(cache+200u,state.mask);
    edgeCaches.Store(cache+204u,state.phase);
    edgeCaches.Store(cache+208u,state.outcome);
    edgeCaches.Store(cache+212u,asuint(state.projection));
    edgeCaches.Store(cache+216u,asuint(state.contactLimit));
    edgeCaches.Store(cache+220u,state.pointCore);
}
GjkClosest gjkCachedClosest(GjkGpuState state) {
    GjkClosest nearest=(GjkClosest)0;
    nearest.closestPoint=state.closest;
    nearest.weights=state.weights;
    nearest.mask=state.mask;
    nearest.dist2=state.dist2;
    return nearest;
}
