// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// GPU lowering of the memsetD32Async operations issued on
// PxgCudaBroadPhaseSap::mStream. This is a device fill, not a host copy
// or a newly scheduled simulation pass. The source call supplies the
// original CUdeviceptr, 32-bit fill value, and number of 32-bit words.
#ifdef __spirv__
[[vk::binding(0, 0)]]
#endif
RWByteAddressBuffer words : register(u0, space0);

struct Parameters {
    uint offset;
    uint count;
    uint value;
    uint stride;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0, space0);
#endif

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    uint index = group.x * 256u + thread.x;
    // Use a single bounded GPU operation even when source buffers grow
    // beyond the device's maximum number of dispatch groups.
    // Subtraction-based termination avoids a 32-bit index overflow.
    while (index < params.count) {
        words.Store(params.offset + index * 4u, params.value);
        if (params.count - 1u - index < params.stride)
            break;
        index += params.stride;
    }
}
