// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/broadphase_completion.h"
#include "physx_gpu/sap_host_frame_roles.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_sap_fetch_native.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace fyzz::compute;
using fyzz::physx_gpu::BroadPhaseCompletionGate;
using fyzz::physx_gpu::SapFrameBufferRoles;
using fyzz::physx_gpu::SapUpdateInputs;
using fyzz::physx_gpu::decide_sap_update;

struct PairTask final : physx::PxBaseTask {
    explicit PairTask(const char* label, std::vector<std::string>& events)
        : label_(label), events_(events) {}

    void run() override {}
    const char* getName() const override { return label_; }
    void addReference() override { ++refs_; }
    void removeReference() override {
        if (refs_ != 1)
            throw std::runtime_error("PhysX task released twice");
        --refs_;
        ++releases_;
        events_.emplace_back(label_);
        if (throw_on_remove)
            throw std::runtime_error("Original task release failed after state change");
    }
    bool throw_on_remove = false;
    int32_t getReference() const override { return refs_; }
    void release() override {}

    int releases() const { return releases_; }

    const char* label_;
    std::vector<std::string>& events_;
    int refs_ = 1;
    int releases_ = 0;
};

template<class F>
void require_throws(F&& fn, const char* what) {
    try { fn(); } catch (const std::exception&) { return; }
    throw std::runtime_error(what);
}

void check_event_order(const std::vector<std::string>& events) {
    if (events != std::vector<std::string>{"publish", "continuations", "found", "lost"})
        throw std::runtime_error("PhysX pair tasks ran before result publication or out of order");
}

void exercise(Backend backend) {
    Context context(backend);
    BroadPhaseCompletionGate gate(context);
    std::vector<std::string> events;

    std::array<std::uint32_t, 73> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0xca110000u + static_cast<std::uint32_t>(i);
    const auto buffer = context.create_buffer(sizeof(input), MemoryPlacement::DeviceLocal);

    // A normal frame: GPU work finishes at the fetch boundary. The CPU
    // result-publication callback verifies the copied data, and pair tasks
    // become runnable only after that operation has succeeded.
    auto submission = context.submit_upload_buffer(buffer, 0, input.data(), sizeof(input));
    gate.arm(submission);
    require_throws([&] { gate.arm(submission); },
                   "Overlapping broadphase GPU frames were accepted");
    PairTask found("found", events), lost("lost", events);
    if (found.releases() || lost.releases())
        throw std::runtime_error("PhysX task released before GPU completion");
    gate.fetch_and_release([&] {
        std::array<std::uint32_t, 73> observed{};
        context.download(buffer, 0, observed.data(), sizeof(observed));
        if (observed != input)
            throw std::runtime_error("PhysX broadphase result visibility failed");
        events.emplace_back("publish");
    }, [&] { events.emplace_back("continuations"); }, found, lost);
    check_event_order(events);
    if (gate.armed() || found.releases() != 1 || lost.releases() != 1)
        throw std::runtime_error("PhysX broadphase publication was not one-shot");
    require_throws([&] { gate.fetch_and_release([] {}, [] {}, found, lost); },
                   "Duplicate pair-task release was permitted");

    // Publication failure after GPU completion must not release tasks.
    // The caller can apply PhysX's own abort/empty-pairs policy and retry.
    events.clear();
    gate.arm(context.submit_upload_buffer(buffer, 0, input.data(), sizeof(input)));
    PairTask fail_found("found", events), fail_lost("lost", events);
    require_throws([&] {
        gate.fetch_and_release([&] { throw std::runtime_error("Invalid result count"); },
                               [&] { events.emplace_back("continuations"); },
                               fail_found, fail_lost);
    }, "Publication failure was swallowed");
    if (!gate.armed() || fail_found.releases() || fail_lost.releases() ||
        !gate.gpu_complete())
        throw std::runtime_error("Tasks released against failed PhysX publication");
    if (!events.empty())
        throw std::runtime_error("PhysX continuation installed before result validation");
    gate.fetch_and_release([&] { events.emplace_back("publish"); },
                           [&] { events.emplace_back("continuations"); },
                           fail_found, fail_lost);
    check_event_order(events);

    // Upstream no-state-change fast path may have no new GPU command, but
    // the original postBroadPhase still fetches and releases pair tasks.
    events.clear();
    gate.arm(0);
    PairTask empty_found("found", events), empty_lost("lost", events);
    gate.fetch_and_release([&] { events.emplace_back("publish"); },
                           [&] { events.emplace_back("continuations"); },
                           empty_found, empty_lost);
    check_event_order(events);

    // Original PxgAABBManager::postBroadPhase(nullptr) executes its two
    // concrete Cm::Task::runInternal functions inline, after GPU results
    // and resize. It must NOT remove their task references.
    events.clear();
    gate.arm(context.submit_upload_buffer(buffer, 0, input.data(), sizeof(input)));
    PairTask inline_found("found", events), inline_lost("lost", events);
    gate.fetch_and_run_inline([&] {
        std::array<std::uint32_t, 73> observed{};
        context.download(buffer, 0, observed.data(), sizeof(observed));
        if (observed != input)
            throw std::runtime_error("Original inline path ran before GPU completion");
        events.emplace_back("publish");
    }, [&] { events.emplace_back("found"); },
       [&] { events.emplace_back("lost"); });
    if (events != std::vector<std::string>{"publish", "found", "lost"} ||
        gate.armed() || inline_found.releases() || inline_lost.releases())
        throw std::runtime_error("PhysX null-continuation branch changed task semantics");
    require_throws([&] {
        gate.fetch_and_run_inline([] {}, [] {}, [] {});
    }, "Inline source pair tasks ran twice");

    // Publication failure is retryable before either inline task runs.
    events.clear();
    gate.arm(0);
    require_throws([&] {
        gate.fetch_and_run_inline([&] {
            throw std::runtime_error("Original GPU pair resize failed");
        }, [&] { events.emplace_back("found"); },
           [&] { events.emplace_back("lost"); });
    }, "Inline fetch failure was swallowed");
    if (!gate.armed() || !events.empty())
        throw std::runtime_error("Inline source tasks ran against failed publication");
    gate.fetch_and_run_inline([&] { events.emplace_back("publish"); },
                              [&] { events.emplace_back("found"); },
                              [&] { events.emplace_back("lost"); });
    if (events != std::vector<std::string>{"publish", "found", "lost"})
        throw std::runtime_error("Inline retry did not preserve source publication order");

    // If runInternal mutates scene state and throws, the second original
    // task cannot safely be replayed in the same or a future frame.
    BroadPhaseCompletionGate inline_terminal(context);
    inline_terminal.arm(0);
    events.clear();
    require_throws([&] {
        inline_terminal.fetch_and_run_inline(
            [&] { events.emplace_back("publish"); },
            [&] {
                events.emplace_back("found");
                throw std::runtime_error("Original found-pair callback aborted");
            },
            [&] { events.emplace_back("lost"); });
    }, "Partial inline source task failure was swallowed");
    if (events != std::vector<std::string>{"publish", "found"} ||
        inline_terminal.armed())
        throw std::runtime_error("Original inline tasks replayed after partial execution");
    require_throws([&] { inline_terminal.arm(0); },
                   "Partially run source task accepted another frame");

    // No callback, aliased task inputs or unarmed state may never silently
    // release a reference.
    gate.arm(0);
    PairTask invalid_found("found", events);
    require_throws([&] {
        gate.fetch_and_release({}, {}, invalid_found, invalid_found);
    }, "Null callback and alias task were accepted");
    require_throws([&] {
        gate.fetch_and_release([] {}, [] {}, invalid_found, invalid_found);
    }, "Aliased PhysX pair tasks were accepted");
    if (invalid_found.releases())
        throw std::runtime_error("Malformed PhysX publication released a task");
    PairTask valid_lost("lost", events);
    gate.fetch_and_release([] {}, [] {}, invalid_found, valid_lost);

    // A failed setContinuation on the second task might have already
    // incremented references on the first. Neither task may be released,
    // and rearming/reinstalling a continuation must be rejected.
    BroadPhaseCompletionGate terminal(context);
    terminal.arm(0);
    PairTask terminal_found("found", events), terminal_lost("lost", events);
    require_throws([&] {
        terminal.fetch_and_release([] {},
            [&] { throw std::runtime_error("Second task continuation unavailable"); },
            terminal_found, terminal_lost);
    }, "Continuation failure was swallowed");
    if (terminal_found.releases() || terminal_lost.releases())
        throw std::runtime_error("Tasks released after failed PhysX continuation setup");
    require_throws([&] { terminal.arm(0); },
                   "Partially installed PhysX continuations were retried");

    // PhysX task release can enter a dispatcher immediately. If the
    // first release fails after changing task state, neither replay nor
    // rearming may be allowed: doing either can dispatch old work twice.
    BroadPhaseCompletionGate released_partial(context);
    released_partial.arm(0);
    events.clear();
    PairTask partial_found("found", events), partial_lost("lost", events);
    partial_found.throw_on_remove = true;
    require_throws([&] {
        released_partial.fetch_and_release(
            [&] { events.emplace_back("publish"); },
            [&] { events.emplace_back("continuations"); },
            partial_found, partial_lost);
    }, "Partial source task release failure was swallowed");
    if (events != std::vector<std::string>{
            "publish", "continuations", "found"} ||
        released_partial.armed() ||
        partial_found.releases() != 1 || partial_lost.releases() != 0)
        throw std::runtime_error("Partial PhysX continuation release replayed task work");
    require_throws([&] { released_partial.arm(0); },
                   "Broadphase accepted new frame after failed pair-task release");
    require_throws([&] {
        released_partial.fetch_and_release([] {}, [] {}, partial_found, partial_lost);
    }, "Broadphase retried a partially released pair task");
    context.destroy_buffer(buffer);
}


void verify_source_update_gates() {
    // Actual PxgCudaBroadPhaseSap constructor sets mForceUpdate=true.
    // The first idle update therefore runs once to complete a valid
    // old/new-bound swap; the following idle call returns early.
    bool force = true;
    auto idle = decide_sap_update(SapUpdateInputs{12u, 0u, 0u, false, false, force});
    if (!idle.dispatch_update || !idle.forced_idle_update ||
        idle.force_update_next || idle.current_boxes != 12u ||
        idle.perform_incremental_sap || idle.repair_removed_projections ||
        idle.sort_created_removed_projections || idle.clear_created_handle_flags)
        throw std::runtime_error("PhysX first idle forced-update gate changed");
    force = idle.force_update_next;
    idle = decide_sap_update(SapUpdateInputs{12u, 0u, 0u, false, false, force});
    if (idle.dispatch_update || idle.force_update_next || idle.forced_idle_update)
        throw std::runtime_error("PhysX repeated idle call should skip GPU work");

    // Changed GPU state runs the incremental SAP comparison, whether or
    // not the host bounds are reported as changed.
    const auto gpuChanged =
        decide_sap_update(SapUpdateInputs{12u, 0u, 0u, false, true, false});
    if (!gpuChanged.dispatch_update || !gpuChanged.perform_incremental_sap ||
        !gpuChanged.force_update_next || gpuChanged.forced_idle_update)
        throw std::runtime_error("PhysX gpuStateChanged decision changed");
    const auto afterGpu =
        decide_sap_update(SapUpdateInputs{12u, 0u, 0u, false, false,
                                          gpuChanged.force_update_next});
    if (!afterGpu.dispatch_update || !afterGpu.forced_idle_update ||
        afterGpu.force_update_next)
        throw std::runtime_error("PhysX one forced frame after activity missing");

    // Removed handles repair the old positions. Created *or* removed
    // handles cause a full WOR sort; only created handles need new-flag
    // clearing. Preserve the original previous-box count for rank setup.
    const auto removeAndCreate =
        decide_sap_update(SapUpdateInputs{12u, 5u, 3u, false, false, false});
    if (!removeAndCreate.dispatch_update || removeAndCreate.previous_boxes != 12u ||
        removeAndCreate.current_boxes != 14u ||
        !removeAndCreate.repair_removed_projections ||
        !removeAndCreate.sort_created_removed_projections ||
        !removeAndCreate.clear_created_handle_flags ||
        removeAndCreate.perform_incremental_sap)
        throw std::runtime_error("PhysX created/removed host branches changed");
    const auto removeOnly =
        decide_sap_update(SapUpdateInputs{12u, 0u, 3u, false, false, false});
    if (!removeOnly.repair_removed_projections ||
        !removeOnly.sort_created_removed_projections ||
        removeOnly.clear_created_handle_flags ||
        removeOnly.current_boxes != 9u)
        throw std::runtime_error("PhysX removed-only branch changed");
    const auto cpuChange =
        decide_sap_update(SapUpdateInputs{12u, 0u, 0u, true, false, false});
    if (!cpuChange.dispatch_update || cpuChange.perform_incremental_sap ||
        !cpuChange.force_update_next)
        throw std::runtime_error("PhysX stateChanged branch changed");

    require_throws([] {
        decide_sap_update(SapUpdateInputs{1u, 0u, 2u, true, false, false});
    }, "PhysX removed more SAP boxes than available");
    require_throws([] {
        decide_sap_update(SapUpdateInputs{0xffffffffu, 1u, 0u, true, false, false});
    }, "PhysX overflowing box population accepted");
}

void verify_source_frame_swaps(Context& context) {
    using Member = SapFrameBufferRoles::History SapFrameBufferRoles::*;
    // Exactly the seven original PxgCudaBuffer histories, each with
    // current [0..2] and previous [3..5] slots. A swap changes
    // allocation identities, never any allocation's contents.
    const std::array<Member, 7> families = {{
        &SapFrameBufferRoles::box_handles,
        &SapFrameBufferRoles::block_end_histograms,
        &SapFrameBufferRoles::end_histograms,
        &SapFrameBufferRoles::ordered_end_handles,
        &SapFrameBufferRoles::block_start_histograms,
        &SapFrameBufferRoles::start_histograms,
        &SapFrameBufferRoles::ordered_start_handles
    }};

    SapFrameBufferRoles roles;
    for (std::uint64_t f = 0u; f < families.size(); ++f)
        for (std::uint64_t i = 0u; i < 6u; ++i)
            (roles.*families[f])[i] = 1000u + 16u * f + i;
    roles.old_integer_bounds = 4001u;
    roles.new_integer_bounds = 4002u;
    const auto initial = roles;
    const auto check = [&](bool historiesSwapped, bool boundsSwapped) {
        for (const auto member : families)
            for (std::uint32_t i = 0u; i < 6u; ++i) {
                const auto source = historiesSwapped ?
                    (i < 3u ? i + 3u : i - 3u) : i;
                if ((roles.*member)[i] != (initial.*member)[source])
                    throw std::runtime_error("PhysX SAP six-buffer history identity mismatch");
            }
        const auto expectedOld = boundsSwapped ?
            initial.new_integer_bounds : initial.old_integer_bounds;
        const auto expectedNew = boundsSwapped ?
            initial.old_integer_bounds : initial.new_integer_bounds;
        if (roles.old_integer_bounds != expectedOld ||
            roles.new_integer_bounds != expectedNew)
            throw std::runtime_error("PhysX integer bounds and history swap were conflated");
    };

    // update() swaps the integer AABBs *before* the report kernel trio.
    roles.swap_bounds_after_generate_new_pairs();
    check(false, true);

    // Fetch can only rotate histories after its GPU result transfer;
    // the existing completion gate releases original PhysX pair tasks
    // strictly after that callback. This exercise uses a real GPU upload.
    std::array<std::uint32_t, 17u> payload{};
    for (std::uint32_t i = 0; i < payload.size(); ++i)
        payload[i] = 0xbeef0000u + i;
    const auto buffer =
        context.create_buffer(sizeof(payload), MemoryPlacement::DeviceLocal);
    BroadPhaseCompletionGate gate(context);
    gate.arm(context.submit_upload_buffer(
        buffer, 0u, payload.data(), sizeof(payload)));
    std::vector<std::string> events;
    PairTask found("found", events), lost("lost", events);
    check(false, true);
    gate.fetch_and_release([&] {
        std::array<std::uint32_t, 17u> actual{};
        context.download(buffer, 0u, actual.data(), sizeof(actual));
        if (actual != payload)
            throw std::runtime_error("SAP swap occurred before GPU fetch");
        events.emplace_back("fetch");
        roles.swap_histories_after_fetch();
        events.emplace_back("swap");
    }, [&] { events.emplace_back("continuations"); }, found, lost);
    if (events != std::vector<std::string>{
            "fetch", "swap", "continuations", "found", "lost"})
        throw std::runtime_error("Source SAP fetch, swap and task order changed");
    check(true, true);

    // An empty/early-return update still enters the original
    // fetchBroadPhaseResults path and swaps all seven histories,
    // without an integer-bound swap or a new GPU submission.
    gate.arm(0u);
    events.clear();
    PairTask quietFound("found", events), quietLost("lost", events);
    require_throws([&] {
        gate.fetch_and_release([&] {
            events.emplace_back("fetch-failed");
            throw std::runtime_error("GPU report transfer error");
        }, [&] { events.emplace_back("continuations"); },
        quietFound, quietLost);
    }, "Failed PhysX fetch unexpectedly succeeded");
    check(true, true);
    if (!gate.armed() || quietFound.releases() || quietLost.releases())
        throw std::runtime_error("Failed fetch published partial SAP frame");
    events.clear();
    gate.fetch_and_release([&] {
        events.emplace_back("fetch");
        roles.swap_histories_after_fetch();
        events.emplace_back("swap");
    }, [&] { events.emplace_back("continuations"); },
    quietFound, quietLost);
    if (events != std::vector<std::string>{
            "fetch", "swap", "continuations", "found", "lost"})
        throw std::runtime_error("Quiet-frame history swap must precede task release");
    check(false, true);
    context.destroy_buffer(buffer);
}


void verify_source_native_fetch(Backend backend) {
    using fyzz::physx_gpu::CudaDeviceAddressMap;
    using fyzz::physx_gpu::SourceSapDmaNative;
    using fyzz::physx_gpu::SourceSapFetchedCounts;
    using fyzz::physx_gpu::fetch_source_sap_results;
    using Pair = physx::PxgBroadPhasePair;
    Context context(backend);
    CudaDeviceAddressMap addresses;
    constexpr std::size_t bytes=4096u;
    const auto root=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    const auto descriptor=addresses.register_slice(root,256u,sizeof(physx::PxgBroadPhaseDesc));
    const auto foundAddress=addresses.register_slice(root,2048u,8u*sizeof(Pair));
    const auto lostAddress=addresses.register_slice(root,3072u,8u*sizeof(Pair));
    const auto cpuVirtual=[](CudaDeviceAddressMap::Address pointer) {
        return reinterpret_cast<Pair*>(static_cast<std::uintptr_t>(pointer));
    };
    physx::PxgBroadPhaseDesc gpu{};
    gpu.max_found_lost_pairs=4u;
    gpu.sharedFoundPairIndex=4u; gpu.sharedFoundAggPairIndex=1u;
    gpu.sharedLostPairIndex=3u; gpu.sharedLostAggPairIndex=1u;
    gpu.overlapChecksTotalRegion=0x100000039ull; // Original PxU64 counter.
    gpu.startRegionAccumTotal=23u;
    gpu.regionAccumTotal=61u;
    gpu.foundPairReportMap=cpuVirtual(foundAddress);
    gpu.lostPairReportMap=cpuVirtual(lostAddress);
    std::array<Pair,8> foundGpu{},lostGpu{},foundHost{},lostHost{};
    for(std::uint32_t i=0u;i<8u;++i) {
        foundGpu[i]=Pair(100u+i,200u+i);
        lostGpu[i]=Pair(300u+i,400u+i);
    }
    SourceSapDmaNative source(context,addresses);
    // All three original updateData and GPU descriptor-like upload groups
    // share a source ordered stream; no early host wait is permitted.
    source.queue_uploads({
        {descriptor,&gpu,sizeof(gpu)},
        {foundAddress,foundGpu.data(),sizeof(foundGpu)},
        {lostAddress,lostGpu.data(),sizeof(lostGpu)}
    });
    SapFrameBufferRoles roles;
    for(std::uint32_t i=0;i<6u;++i) {
        roles.box_handles[i]=1000u+i;
        roles.end_histograms[i]=2000u+i;
        roles.start_histograms[i]=3000u+i;
        roles.block_start_histograms[i]=4000u+i;
        roles.block_end_histograms[i]=5000u+i;
        roles.ordered_start_handles[i]=6000u+i;
        roles.ordered_end_handles[i]=7000u+i;
    }
    roles.old_integer_bounds=8001u;
    roles.new_integer_bounds=8002u;
    const auto prior=roles;
    physx::PxgBroadPhaseDesc host{};
    BroadPhaseCompletionGate gate(context);
    gate.arm_source_stream(source);
    require_throws([&] {gate.arm_source_stream(source);},
                   "Original stream completion gate accepted overlapping frame");
    std::vector<std::string> events;
    PairTask foundTask("found",events),lostTask("lost",events);

    SourceSapFetchedCounts result{};
    gate.fetch_and_release([&] {
        result=fetch_source_sap_results(context,addresses,descriptor,host,
            {foundHost.data(),foundHost.size()},
            {lostHost.data(),lostHost.size()},false,roles);
        events.emplace_back("publish");
    },[&]{events.emplace_back("continuations");},foundTask,lostTask);
    check_event_order(events);
    if(result.found_actors!=3u || result.lost_actors!=2u ||
       result.pairs_needed!=4u || result.overflow ||
       result.overlap_checks_total_region!=0x100000039ull ||
       result.start_region_accum_total!=23u ||
       result.region_accum_total!=61u ||
       host.sharedFoundPairIndex!=4u)
        throw std::runtime_error("Original source descriptor fetch counters changed");
    for(std::uint32_t i=0u;i<8u;++i) {
        if(i<3u && (foundHost[i].mVolA!=foundGpu[i].mVolA ||
                     foundHost[i].mVolB!=foundGpu[i].mVolB))
            throw std::runtime_error("Original found-pair mapped readback changed");
        if(i<2u && (lostHost[i].mVolA!=lostGpu[i].mVolA ||
                     lostHost[i].mVolB!=lostGpu[i].mVolB))
            throw std::runtime_error("Original lost-pair mapped readback changed");
    }
    if(roles.box_handles[0]!=prior.box_handles[3] ||
       roles.end_histograms[1]!=prior.end_histograms[4] ||
       roles.start_histograms[2]!=prior.start_histograms[5] ||
       roles.block_end_histograms[0]!=prior.block_end_histograms[3] ||
       roles.block_start_histograms[1]!=prior.block_start_histograms[4] ||
       roles.ordered_end_handles[2]!=prior.ordered_end_handles[5] ||
       roles.ordered_start_handles[0]!=prior.ordered_start_handles[3] ||
       roles.old_integer_bounds!=prior.old_integer_bounds ||
       roles.new_integer_bounds!=prior.new_integer_bounds)
        throw std::runtime_error("Source fetch swapped wrong history buffers");

    // Failed source DtoH publication is retryable BEFORE continuations;
    // the seven history families must NOT be double-swapped by that retry.
    events.clear();
    const auto snapshot=roles;
    physx::PxgBroadPhaseDesc next=gpu;
    next.sharedFoundPairIndex=8u; // capped by the source to four records.
    next.sharedFoundAggPairIndex=1u;
    next.sharedLostPairIndex=5u;
    next.sharedLostAggPairIndex=2u;
    next.found_lost_pairs_overflow_flags=true;
    source.queue_uploads({{descriptor,&next,sizeof(next)}});
    gate.arm_source_stream(source);
    PairTask retryFound("found",events),retryLost("lost",events);
    require_throws([&] {
        gate.fetch_and_release([&] {
            (void)fetch_source_sap_results(context,addresses,descriptor,host,
                {foundHost.data(),1u}, {lostHost.data(),lostHost.size()},false,roles);
            events.emplace_back("publish");
        },[&]{events.emplace_back("continuations");},retryFound,retryLost);
    },"Source published a frame despite undersized found-pair host buffer");
    if(!gate.armed() || !events.empty() ||
       retryFound.releases() || retryLost.releases() ||
       roles.box_handles!=snapshot.box_handles ||
       host.sharedFoundPairIndex!=gpu.sharedFoundPairIndex)
        throw std::runtime_error("Failed source fetch mutated publication state");

    gate.fetch_and_release([&]{
        result=fetch_source_sap_results(context,addresses,descriptor,host,
            {foundHost.data(),foundHost.size()},
            {lostHost.data(),lostHost.size()},false,roles);
        events.emplace_back("publish");
    },[&]{events.emplace_back("continuations");},retryFound,retryLost);
    check_event_order(events);
    if(result.found_actors!=3u || result.lost_actors!=2u ||
       result.pairs_needed!=8u || !result.overflow ||
       roles.box_handles!=prior.box_handles)
        throw std::runtime_error("Source overflow clamp or retry swap changed");

    // Original isInAbortMode zeros both actor lists but retains descriptor
    // totals and flips history even if no new kernel was launched.
    events.clear();
    next.sharedFoundPairIndex=100u;
    next.sharedLostPairIndex=100u;
    next.sharedFoundAggPairIndex=0u;
    next.sharedLostAggPairIndex=0u;
    source.queue_uploads({{descriptor,&next,sizeof(next)}});
    gate.arm_source_stream(source);
    PairTask abortFound("found",events),abortLost("lost",events);
    gate.fetch_and_release([&]{
        result=fetch_source_sap_results(context,addresses,descriptor,host,
            {nullptr,0u},{nullptr,0u},true,roles);
        events.emplace_back("publish");
    },[&]{events.emplace_back("continuations");},abortFound,abortLost);
    check_event_order(events);
    if(result.found_actors || result.lost_actors ||
       result.pairs_needed!=100u ||
       roles.box_handles!=snapshot.box_handles)
        throw std::runtime_error("Source abort path changed pair sizes or swaps");

    // The source non-updated frame still fetches descriptor and swaps
    // histories; a source stream with no submissions is valid.
    events.clear();
    gate.arm_source_stream(source);
    PairTask idleFound("found",events),idleLost("lost",events);
    gate.fetch_and_release([&]{
        result=fetch_source_sap_results(context,addresses,descriptor,host,
            {nullptr,0u},{nullptr,0u},true,roles);
        events.emplace_back("publish");
    },[&]{events.emplace_back("continuations");},idleFound,idleLost);
    check_event_order(events);
    if(roles.box_handles!=prior.box_handles)
        throw std::runtime_error("Original idle fetch omitted buffer history rotation");

    // Invalid GPU report counters must not underflow host pair counts or
    // publish an invalid history swap; fixes require PhysX abort handling.
    events.clear();
    next.sharedFoundPairIndex=1u;
    next.sharedFoundAggPairIndex=2u;
    source.queue_uploads({{descriptor,&next,sizeof(next)}});
    gate.arm_source_stream(source);
    PairTask invalidFound("found",events),invalidLost("lost",events);
    require_throws([&]{
        gate.fetch_and_release([&]{
            (void)fetch_source_sap_results(context,addresses,descriptor,host,
                {nullptr,0u},{nullptr,0u},false,roles);
            events.emplace_back("publish");
        },[&]{events.emplace_back("continuations");},
        invalidFound,invalidLost);
    },"Invalid GPU descriptor counters were published");
    if(!gate.armed() || !events.empty() ||
       invalidFound.releases() || invalidLost.releases() ||
       roles.box_handles!=prior.box_handles)
        throw std::runtime_error("Invalid GPU counters released original tasks");
    // Do not destroy the source resource while gate remains armed;
    // in the real scene the original abort/teardown path handles that.
    context.destroy_buffer(root);
}

} // namespace

int main() {
    try {
#if defined(FYZZXYZZY_TEST_D3D12)
        exercise(Backend::D3D12);
        verify_source_update_gates();
        { Context context(Backend::D3D12); verify_source_frame_swaps(context); }
        verify_source_native_fetch(Backend::D3D12);
#else
        exercise(Backend::Vulkan);
        verify_source_update_gates();
        { Context context(Backend::Vulkan); verify_source_frame_swaps(context); }
        verify_source_native_fetch(Backend::Vulkan);
#endif
        std::cout << "PhysX broadphase completion gate PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX broadphase completion gate FAILED: " << e.what() << '\n';
        return 1;
    }
}
