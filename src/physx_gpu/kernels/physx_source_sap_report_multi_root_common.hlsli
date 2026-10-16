// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Original PhysX reports, bound to separate native heap roots. Each byte
// address is shader-view local; Host CudaDeviceAddressMap owns resolution.
struct Parameters {
    uint bias0;
    uint bias1;
    uint bias2;
    uint bias3;
    uint bias4;
    uint bias5;
    uint bias6;
    uint bias7;
    uint bias8;
    uint bias9;
    uint bias10;
    uint bias11;
    uint bias12;
    uint rawCapacity;
    uint aggCapacity;
    uint volumeCapacity;
    uint volumeStride;
    uint aggregateOffset;
    uint hasVolumes;
    uint unused0;
    uint unused1;
    uint unused2;
    uint unused3;
    uint unused4;
    uint unused5;
    uint unused6;
    uint unused7;
    uint unused8;
    uint unused9;
    uint unused10;
    uint unused11;
    uint unused12;
};
#ifdef __spirv__
[[vk::push_constant]] Parameters params;
#else
ConstantBuffer<Parameters> params : register(b0,space0);
#endif
