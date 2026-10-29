// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_foundation_lifetime.h"

#include "foundation/PxPhysicsVersion.h"

#include <cstdio>
#include <mutex>
#include <stdexcept>

namespace fyzz::physx_runtime {
namespace {
std::mutex& foundation_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::weak_ptr<Foundation>& shared_foundation() {
    static std::weak_ptr<Foundation> foundation;
    return foundation;
}
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
    // Serialize destruction with acquisition: a new PxCreateFoundation
    // must not race the previous singleton's final release.
    std::lock_guard<std::mutex> lock(foundation_mutex());
    instance_->release();
}

std::shared_ptr<Foundation> acquire_foundation() {
    std::lock_guard<std::mutex> lock(foundation_mutex());
    if(auto existing=shared_foundation().lock())
        return existing;
    auto created=std::make_shared<Foundation>();
    shared_foundation()=created;
    return created;
}
} // namespace fyzz::physx_runtime
