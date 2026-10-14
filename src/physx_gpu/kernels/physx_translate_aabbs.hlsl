// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Shared HLSL translation of PhysX broadphase.cu::translateAABBsLaunch.
// Compiled by DXC to Vulkan SPIR-V and D3D12 DXIL from this one algorithm.
// The conditional annotations lower only resource bindings and dispatch
// parameters into the different API interfaces; they do not change physics.
//
// PhysX logical storage-buffer slots are 0..3. D3D12 uses distinct SRV and
// UAV register classes, while Vulkan uses one descriptor set of storage
// buffers. PhysX data layout remains six float/uint coordinates per box.

#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
StructuredBuffer<float> input_bounds : register(t0, space0);
#ifdef __spirv__
[[vk::binding(1, 0)]]
#endif
StructuredBuffer<float> contact_distances : register(t1, space0);
#ifdef __spirv__
[[vk::binding(2, 0)]]
#endif
StructuredBuffer<uint> environment_ids : register(t2, space0);
#ifdef __spirv__
[[vk::binding(3, 0)]]
#endif
RWStructuredBuffer<uint> output_bounds : register(u0, space0);

struct Parameters {
    uint count;
    uint shift_x;
    uint shift_y;
    uint shift_z;
    uint env_bits_x;
    uint env_bits_y;
    uint env_bits_z;
    uint env_enabled;
};
#ifdef __spirv__
// PhysX stage::append() supplies 32 bytes of Vulkan push constants.
[[vk::push_constant]] Parameters parameters;
#else
// The D3D12 lowering will supply the same 32-byte payload as root constants
// or a CBV. No separate D3D12 simulation parameters are introduced.
ConstantBuffer<Parameters> parameters : register(b0, space0);
#endif

[numthreads(128, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
    uint thread_index = dispatch_thread_id.x;
    uint box_index = thread_index >> 3;
    uint component = thread_index & 7;
    if (box_index >= parameters.count || component >= 6)
        return;

    uint axis = component < 3 ? component : component - 3;
    uint source_index = box_index * 6 + component;
    float value = input_bounds[source_index];
    float distance = contact_distances[box_index];
    float projected = component < 3 ? value - distance : value + distance;

    uint bits = asuint(projected);
    uint encoded = (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
    uint shift = axis == 0 ? parameters.shift_x :
                 (axis == 1 ? parameters.shift_y : parameters.shift_z);
    uint env_shift = axis == 0 ? parameters.env_bits_x :
                     (axis == 1 ? parameters.env_bits_y : parameters.env_bits_z);
    uint mask = component >= 3 ? 1u : 0u;
    uint result = ((encoded >> shift) & ~1u) | mask;

    if (parameters.env_enabled != 0 && env_shift != 0) {
        uint env_id = environment_ids[box_index];
        if (env_id != 0xffffffffu)
            result |= env_id << (32u - env_shift);
        else
            result = mask != 0 ? 0xfe7fffffu : 0x01800000u;
    }

    output_bounds[source_index] = result;
}
