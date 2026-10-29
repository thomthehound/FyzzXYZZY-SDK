// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_foundation_lifetime.h"

#include "foundation/PxPhysicsVersion.h"

#include <cstddef>
#include <cstdio>
#include <mutex>
#include <stdexcept>

namespace fyzz::physx_runtime {
namespace {

struct FoundationState {
    std::mutex mutex;
    std::shared_ptr<Foundation> instance;
    std::size_t leases=0;
};

FoundationState& foundation_state() {
    static FoundationState state;
    return state;
}

// Each acquired Foundation shares a control block with a lease. A strong
// process-wide reference stays live until the last lease destructor holds
// the mutex and releases the underlying PhysX singleton. This avoids the
// expired-weak-reference window between last shared_ptr release and the
// Foundation destructor that could race PxCreateFoundation.
struct FoundationLease {
    ~FoundationLease() noexcept {
        auto& state=foundation_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        if(--state.leases==0)
            state.instance.reset();
    }
};

} // namespace

void Foundation::Errors::reportError(
    physx::PxErrorCode::Enum,const char* message,
    const char* file,int line) {
    std::fprintf(stderr,"PhysX: %s (%s:%d)\n",
                 message ? message : "",file ? file : "",line);
}

Foundation::Foundation()
    : instance_(PxCreateFoundation(PX_PHYSICS_VERSION,allocator_,errors_)) {
    if(!instance_)
        throw std::runtime_error("Failed to create PhysX Foundation");
}

Foundation::~Foundation() {
    instance_->release();
}

std::shared_ptr<Foundation> acquire_foundation() {
    auto& state=foundation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if(!state.instance)
        state.instance=std::make_shared<Foundation>();
    std::shared_ptr<FoundationLease> token;
    try {
        token=std::make_shared<FoundationLease>();
    } catch (...) {
        if(state.leases==0)
            state.instance.reset();
        throw;
    }
    ++state.leases;
    // The alias retains the lease control block. The actual Foundation is
    // owned by FoundationState until the final lease releases it under lock.
    return std::shared_ptr<Foundation>(
        std::move(token),state.instance.get());
}
} // namespace fyzz::physx_runtime
