// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Validate the physical Sc task ABI through a GPU-produced indirect dispatch.
ByteAddressBuffer work:register(t0,space0);
RWByteAddressBuffer result:register(u0,space0);
cbuffer Arguments:register(b0,space0) {
    uint taskBias;
};

[numthreads(128,1,1)]
void main(uint3 group:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    if(lane.x!=0u)return;
    const uint task=taskBias+group.x*36u;
    result.Store4(group.x*16u,
                  work.Load4(task+16u));
}
