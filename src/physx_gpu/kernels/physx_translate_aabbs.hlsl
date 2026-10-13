// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// HLSL implementation of PhysX broadphase.cu::translateAABBsLaunch.
// This is the D3D12/DXIL compilation proof for the same PhysX GPU stage
// currently exercised through Vulkan GLSL. The PhysX algorithm and data
// layout are unchanged. Once Vulkan HLSL->SPIR-V is validated, converge
// on this shared source rather than maintain two shader implementations.
//
// Logical slots 0..3 are the four buffers also exposed by the Vulkan stage.
// D3D12 register spaces are distinct for SRV and UAV resources; the future
// execution backend maps these logical slots through shader reflection.

StructuredBuffer<float> input_bounds : register(t0, space0);
StructuredBuffer<float> contact_distances : register(t1, space0);
StructuredBuffer<uint> environment_ids : register(t2, space0);
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
ConstantBuffer<Parameters> parameters : register(b0, space0);

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
