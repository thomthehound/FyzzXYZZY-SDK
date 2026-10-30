// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer output:register(u0,space0);

struct Parameters {
    uint byteOffset;
    uint r0; uint r1; uint r2;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params:register(b0,space0);
[WaveSize(32)]
#endif
[numthreads(32,1,1)]
void main(uint3 thread:SV_GroupThreadID) {
    if(thread.x==0u)
        output.Store(params.byteOffset,WaveGetLaneCount());
}
