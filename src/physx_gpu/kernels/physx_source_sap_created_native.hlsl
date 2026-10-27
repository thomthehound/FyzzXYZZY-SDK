// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
// Source broadphase.cu::markCreatedPairsLaunch (BP_UPDATE_CREATEDPAIRS).
// Original source buffers, 32x256, previous+created projection placement,
// new endpoint flags, old-bounds invalidation and odd-population padding.
#ifdef __spirv__
[[vk::binding(0,0)]]
#endif
RWByteAddressBuffer created : register(u0,space0);
#ifdef __spirv__
[[vk::binding(1,0)]]
#endif
RWByteAddressBuffer newBounds : register(u1,space0);
#ifdef __spirv__
[[vk::binding(2,0)]]
#endif
RWByteAddressBuffer oldBounds : register(u2,space0);
#ifdef __spirv__
[[vk::binding(3,0)]]
#endif
RWByteAddressBuffer projX : register(u3,space0);
#ifdef __spirv__
[[vk::binding(4,0)]]
#endif
RWByteAddressBuffer projY : register(u4,space0);
#ifdef __spirv__
[[vk::binding(5,0)]]
#endif
RWByteAddressBuffer projZ : register(u5,space0);
#ifdef __spirv__
[[vk::binding(6,0)]]
#endif
RWByteAddressBuffer handlesX : register(u6,space0);
#ifdef __spirv__
[[vk::binding(7,0)]]
#endif
RWByteAddressBuffer handlesY : register(u7,space0);
#ifdef __spirv__
[[vk::binding(8,0)]]
#endif
RWByteAddressBuffer handlesZ : register(u8,space0);
struct Parameters {
    uint createdBias, newBoundsBias, oldBoundsBias, projXBias;
    uint projYBias, projZBias, handlesXBias, handlesYBias;
    uint handlesZBias, previousCount, createdCount, boxCapacity;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters p;
#else
ConstantBuffer<Parameters> p : register(b0,space0);
#endif
[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    for(uint i=tid.x;i<p.createdCount;i+=8192u) {
        const uint handle=created.Load(p.createdBias+i*4u);
        const uint bounds=p.newBoundsBias+handle*24u;
        const uint old=p.oldBoundsBias+handle*24u;
        const uint index=(p.previousCount+i)*8u;
        const uint startHandle=(handle<<3u)|3u;
        const uint endHandle=(handle<<3u)|2u;
        projX.Store(p.projXBias+index,newBounds.Load(bounds));
        projX.Store(p.projXBias+index+4u,newBounds.Load(bounds+12u));
        projY.Store(p.projYBias+index,newBounds.Load(bounds+4u));
        projY.Store(p.projYBias+index+4u,newBounds.Load(bounds+16u));
        projZ.Store(p.projZBias+index,newBounds.Load(bounds+8u));
        projZ.Store(p.projZBias+index+4u,newBounds.Load(bounds+20u));
        handlesX.Store(p.handlesXBias+index,startHandle);
        handlesX.Store(p.handlesXBias+index+4u,endHandle);
        handlesY.Store(p.handlesYBias+index,startHandle);
        handlesY.Store(p.handlesYBias+index+4u,endHandle);
        handlesZ.Store(p.handlesZBias+index,startHandle);
        handlesZ.Store(p.handlesZBias+index+4u,endHandle);
        [unroll] for(uint axis=0u;axis<3u;++axis) {
            oldBounds.Store(old+axis*4u,0xff7fffffu);
            oldBounds.Store(old+12u+axis*4u,0x00800000u);
        }
    }
    const uint total=p.previousCount+p.createdCount;
    if(tid.x<(total&1u)) {
        const uint byteIndex=total*8u;
        projX.Store2(p.projXBias+byteIndex,uint2(0xffffffffu,0xffffffffu));
        projY.Store2(p.projYBias+byteIndex,uint2(0xffffffffu,0xffffffffu));
        projZ.Store2(p.projZBias+byteIndex,uint2(0xffffffffu,0xffffffffu));
        handlesX.Store2(p.handlesXBias+byteIndex,uint2(0x3fffffffu,0x3fffffffu));
        handlesY.Store2(p.handlesYBias+byteIndex,uint2(0x3fffffffu,0x3fffffffu));
        handlesZ.Store2(p.handlesZBias+byteIndex,uint2(0x3fffffffu,0x3fffffffu));
    }
}
