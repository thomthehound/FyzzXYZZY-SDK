// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// PhysX BP_INITIALIZE_RANKS 64x256. Use source projection ranks.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer rankX:register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer rankY:register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer rankZ:register(u2,space0);
struct Params {
    uint biasX;
    uint biasY;
    uint biasZ;
    uint boxes;
    uint pad0;
    uint pad1;
    uint pad2;
    uint pad3;
};
#ifdef __spirv__
[[vk::push_constant]] Params p;
#else
ConstantBuffer<Params> p:register(b0,space0);
#endif

[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    const uint vectors=(2u*p.boxes+3u)/4u;
    for(uint v=tid.x;v<vectors;v+=16384u) {
        const uint index=v*4u;
        [unroll] for(uint lane=0u;lane<4u;++lane) {
            rankX.Store(p.biasX+(index+lane)*4u,index+lane);
            rankY.Store(p.biasY+(index+lane)*4u,index+lane);
            rankZ.Store(p.biasZ+(index+lane)*4u,index+lane);
        }
    }
}
