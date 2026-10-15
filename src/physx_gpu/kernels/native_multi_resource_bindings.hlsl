// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Native compute binding conformance: three independent RWByteAddressBuffer
// views, including two nonoverlapping views into one physical GPU resource.
// This tests binding/ordering only, not a replacement physics algorithm.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer sourceA : register(u0, space0);
#ifdef __spirv__
[[vk::binding(1, 0)]]
#endif
RWByteAddressBuffer sourceB : register(u1, space0);
#ifdef __spirv__
[[vk::binding(2, 0)]]
#endif
RWByteAddressBuffer resultBuffer : register(u2, space0);

struct Parameters {
    uint count; uint r0; uint r1; uint r2;
    uint r3; uint r4; uint r5; uint r6;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= params.count) return;
    const uint offset = id.x * 4u;
    const uint a = sourceA.Load(offset);
    const uint b = sourceB.Load(offset);
    resultBuffer.Store(offset, a * 3u + b);
    sourceA.Store(offset, a + 1u);
    sourceB.Store(offset, b + 2u);
}
