// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/sap_newbounds_sequence_arena.h"
#include "physx_gpu/sap_host_frame_roles.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

namespace {
constexpr std::uint32_t kCanary = 0xcdcdcdcd;
constexpr std::uint32_t kProjections = 6u; // Three current PhysX boxes.

std::vector<std::uint8_t> load(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("New-bounds sequence shader missing");
    std::vector<char> bytes{std::istreambuf_iterator<char>{file},
                            std::istreambuf_iterator<char>{}};
    return {bytes.begin(), bytes.end()};
}
ShaderBinary shader(const char* path) {
    ShaderBinary result;
#if defined(FYZZXYZZY_TEST_D3D12)
    result.format = ShaderFormat::Dxil;
#else
    result.format = ShaderFormat::Spirv;
#endif
    result.bytes = load(path);
    return result;
}
struct Case {
    bool created = true;
    bool empty_work = false;
    bool two_regions = false;
    bool new_projection = true;
    bool group_exclusion = false;
    bool aabb_separation = false;
    bool environment = false;
    bool wildcard_environment = false;
    std::uint32_t initial_report_count = 0u;
    std::uint32_t report_capacity = 5u;
};
struct Fixture {
    std::vector<std::uint32_t> bytes;
    SapNewBoundsSequenceDescriptorOffsets descriptors{};
    SapNewBoundsSequenceArenaDescriptors mapping{};
    std::uint32_t work_items = 0u;
    std::uint32_t block_totals = 0u;
    std::uint32_t region_work_total = 0u;
    std::uint32_t overlap_counts = 0u;
    std::uint32_t overlap_total = 0u;
    std::uint32_t handle_regions = 0u;
    std::uint32_t found_count = 0u;
    std::uint32_t reports = 0u;
    std::uint32_t report_slots = 5u;

    std::uint32_t allocate(std::uint32_t words) {
        while (bytes.size() % 4u) bytes.push_back(kCanary);
        const auto offset = static_cast<std::uint32_t>(bytes.size() * 4u);
        bytes.resize(bytes.size() + words, kCanary);
        return offset;
    }
    void put(std::uint32_t offset, std::uint32_t value) {
        bytes.at(offset / 4u) = value;
    }
    std::uint32_t get(std::uint32_t offset) const {
        return bytes.at(offset / 4u);
    }
    void put64(std::uint32_t offset, std::uint64_t value) {
        put(offset, static_cast<std::uint32_t>(value));
        put(offset + 4u, static_cast<std::uint32_t>(value >> 32u));
    }
    std::uint64_t get64(std::uint32_t offset) const {
        return std::uint64_t(get(offset)) |
               (std::uint64_t(get(offset + 4u)) << 32u);
    }
    template <typename T> void store_desc(std::uint32_t offset, const T& value) {
        std::memcpy(bytes.data() + offset / 4u, &value, sizeof(value));
    }
};

Fixture prepare(const Case& c) {
    Fixture f;
    f.descriptors.work = f.allocate(sizeof(SapOverlapWorkArenaDesc) / 4u);
    f.descriptors.prefix = f.allocate(sizeof(SapOverlapPrefixArenaDesc) / 4u);
    f.descriptors.pairs = f.allocate(sizeof(SapNewBoundsPairsArenaDesc) / 4u);

    auto sorted = f.allocate(kProjections);
    auto sap = f.allocate(3u * 2u);
    auto ranges = f.allocate(3u * 8u);
    auto region_prefix = f.allocate(kProjections);
    f.region_work_total = f.allocate(1u);
    auto active_hist = f.allocate(64u * kProjections);
    auto start_hist = f.allocate(64u * kProjections);
    f.overlap_counts = f.allocate(2u * 2u); // two u64 values
    f.handle_regions = f.allocate(2u * 2u);
    f.block_totals = f.allocate(32u * 2u);
    f.overlap_total = f.allocate(2u);
    auto active_order = f.allocate(64u * kProjections);
    auto start_order = f.allocate(64u * kProjections);
    auto bounds = f.allocate(3u * 6u);
    auto group_ids = f.allocate(3u);
    auto env_ids = f.allocate(3u);
    f.report_slots = 5u;
    f.reports = f.allocate(f.report_slots * 2u);
    f.found_count = f.allocate(1u);
    f.allocate(16u); // guard padding

    SapNewBoundsPairsArenaDesc p{};
    p.new_integer_bounds_offset = bounds;
    p.group_ids_offset = group_ids;
    p.environment_ids_offset = env_ids;
    p.sorted_x_handles_offset = sorted;
    p.x_sap_positions_offset = sap;
    p.region_ranges_offset = ranges;
    p.overlap_prefix_offset = f.overlap_counts;
    p.region_work_total_offset = f.region_work_total;
    p.overlap_total_offset = f.overlap_total;
    p.handle_regions_offset = f.handle_regions;
    p.active_histograms_offset = active_hist;
    p.start_histograms_offset = start_hist;
    p.ordered_active_handles_offset = active_order;
    p.ordered_start_handles_offset = start_order;
    p.found_reports_offset = f.reports;
    p.found_counter_offset = f.found_count;
    p.max_found_lost_pairs = c.report_capacity;
    p.current_boxes = 3u;
    p.box_capacity = 3u;
    p.has_environment_ids = c.environment ? 1u : 0u;

    f.mapping = make_sap_newbounds_sequence_descriptors(
        p, region_prefix, f.block_totals);
    f.store_desc(f.descriptors.work, f.mapping.work);
    f.store_desc(f.descriptors.prefix, f.mapping.prefix);
    f.store_desc(f.descriptors.pairs, f.mapping.pairs);

    if (f.mapping.work.overlap_counts_offset != f.mapping.prefix.overlap_counts_offset ||
        f.mapping.prefix.overlap_counts_offset != f.mapping.pairs.overlap_prefix_offset ||
        f.mapping.work.handle_regions_offset != f.mapping.pairs.handle_regions_offset ||
        f.mapping.work.region_total_offset != f.mapping.prefix.region_work_total_offset ||
        f.mapping.prefix.region_work_total_offset != f.mapping.pairs.region_work_total_offset ||
        f.mapping.prefix.total_offset != f.mapping.pairs.overlap_total_offset ||
        f.mapping.work.start_histograms_offset != f.mapping.pairs.start_histograms_offset)
        throw std::runtime_error("PhysX sequence descriptor aliases diverged");

    const std::uint32_t n = c.empty_work ? 0u : (c.two_regions ? 2u : 1u);
    f.work_items = n;
    f.put(f.region_work_total, n);
    f.put(f.found_count, c.initial_report_count);

    // X starts for three boxes, followed by their ends; owner 0 spans
    // projections [0,3]. Only that projection contributes region work.
    for (std::uint32_t handle = 0; handle < 3u; ++handle) {
        f.put(sorted + handle * 4u,
              (handle << 3u) | 1u |
              ((handle == 0u && c.new_projection) ? 2u : 0u));
        f.put(sorted + (handle + 3u) * 4u, handle << 3u);
        f.put(sap + handle * 8u, handle);
        f.put(sap + handle * 8u + 4u, handle + 3u);
        f.put(ranges + handle * 32u, c.two_regions && handle == 2u ? 1u : 0u);
        f.put(ranges + handle * 32u + 4u, 0u);
        f.put(ranges + handle * 32u + 8u, 0u);
        f.put(ranges + handle * 32u + 16u,
              c.two_regions ? 1u : 0u);
        f.put(ranges + handle * 32u + 20u, 0u);
        f.put(ranges + handle * 32u + 24u, 0u);

        const auto bound = bounds + handle * 24u;
        f.put(bound, handle); // min X
        f.put(bound + 4u, handle == 2u && c.aabb_separation ? 20u : 0u);
        f.put(bound + 8u, 0u);
        f.put(bound + 12u, 10u);
        f.put(bound + 16u, handle == 2u && c.aabb_separation ? 30u : 10u);
        f.put(bound + 20u, 10u);

        f.put(group_ids + handle * 4u,
              handle == 2u && c.group_exclusion ? 10u : handle + 10u);
        f.put(env_ids + handle * 4u,
              handle == 2u ?
                  (c.wildcard_environment ? 0xffffffffu : 8u) : 7u);
    }

    // Source producer's regionAccum: the owner start has N possible
    // X regions, every other projection has zero work items.
    for (std::uint32_t x = 0u; x < kProjections; ++x)
        f.put(region_prefix + x * 4u, x == 0u ? 0u : n);

    // Valid source-style region memberships:
    // - one-region case: both partners occur in region 0;
    // - two-region case: partner 1 spans both regions, partner 2
    //   belongs only to region 1. This produces raw counts [1,2].
    //   The canonical-region rule emits (0,1) in region 0 and
    //   (0,2) in region 1, rejecting the repeated (0,1).
    const auto selected_hist = c.new_projection ? start_hist : active_hist;
    const auto selected_order = c.new_projection ? start_order : active_order;
    for (std::uint32_t region = 0u; region < (c.two_regions ? 2u : 1u); ++region) {
        const auto start = selected_hist + region * kProjections * 4u;
        const auto both = !c.two_regions || region == 1u;
        for (std::uint32_t x = 0u; x < kProjections; ++x) {
            const std::uint32_t count =
                x < 2u ? 0u : (x == 2u ? 1u : (both ? 2u : 1u));
            f.put(start + x * 4u, count);
        }
        const auto ordered = selected_order + region * kProjections * 4u;
        f.put(ordered, 1u);
        if (both) f.put(ordered + 4u, 2u);
    }
    return f;
}

std::set<std::pair<std::uint32_t, std::uint32_t>>
reference_pairs(const Case& c) {
    std::set<std::pair<std::uint32_t, std::uint32_t>> pairs;
    if (!c.created || c.empty_work) return pairs;
    pairs.emplace(0u, 1u);
    if (!c.group_exclusion && !c.aabb_separation &&
        (!c.environment || c.wildcard_environment))
        pairs.emplace(0u, 2u);
    return pairs;
}

void check(Context& context, const SapNewBoundsSequenceArenaStage& stage,
           const Case& c) {
    auto f = prepare(c);
    auto expected = f;
    const auto valid = reference_pairs(c);
    if (c.created) {
        // Independently enumerate per-region candidate membership,
        // then form the true uint64 exclusive overlap offsets. Unlike the
        // shader, the oracle does not binary-search region/work prefixes.
        std::uint64_t comparisonTotal = 0u;
        for (std::uint32_t i = 0; i < f.work_items; ++i) {
            expected.put64(f.overlap_counts + i * 8u, comparisonTotal);
            expected.put(f.handle_regions + i * 8u, 0u);
            expected.put(f.handle_regions + i * 8u + 4u, i * kProjections);
            comparisonTotal += c.two_regions && i == 0u ? 1u : 2u;
        }
        for (std::uint32_t block = 0; block < 32u; ++block)
            expected.put64(f.block_totals + block * 8u,
                           block == 0u ? comparisonTotal : 0u);
        expected.put64(f.overlap_total, comparisonTotal);
        expected.put(f.found_count, c.initial_report_count +
                                    static_cast<std::uint32_t>(valid.size()));
    }

    const std::uint32_t size =
        static_cast<std::uint32_t>(f.bytes.size() * sizeof(std::uint32_t));
    const auto arena = context.create_buffer(size, MemoryPlacement::DeviceLocal);
    const auto mirror = context.create_buffer(size, MemoryPlacement::DeviceLocal);
    context.upload(arena, 0u, f.bytes.data(), size);
    // Feed this source stage the box count and created-handle gate derived
    // from the existing PxgCudaBroadPhaseSap::update host role, rather than
    // inventing a second frame-counting scheme inside the sequence.
    SapUpdateInputs frame{};
    frame.previous_boxes = c.created ? 2u : 3u;
    frame.created_handles = c.created ? 1u : 0u;
    frame.force_update = true;
    const auto decision = decide_sap_update(frame);
    if (decision.current_boxes != 3u ||
        !decision.dispatch_update ||
        decision.sort_created_removed_projections != c.created)
        throw std::runtime_error("PhysX source frame decision changed");
    std::vector<BatchCommand> commands;
    stage.append(commands, arena, size, f.descriptors,
                 decision.current_boxes, frame.created_handles);
    if (commands.size() != (c.created ? 4u : 0u))
        throw std::runtime_error("PhysX four-dispatch created-handle gate changed");
    if (c.created) {
        const std::uint32_t groups[] = {32u, 32u, 32u, 64u};
        const std::uint32_t offsets[] = {
            f.descriptors.work, f.descriptors.prefix,
            f.descriptors.prefix, f.descriptors.pairs};
        for (std::size_t i = 0u; i < 4u; ++i) {
            const auto& d = std::get<DispatchCommand>(commands[i]);
            if (d.groups.x != groups[i] || d.groups.y != 1u ||
                d.groups.z != 1u || d.buffers.size() != 1u ||
                d.buffers[0].buffer != arena ||
                d.buffers[0].access != BufferAccess::ReadWrite ||
                d.push_constants.size() != 32u)
                throw std::runtime_error("PhysX source dispatch sequence mismatch");
            std::uint32_t suppliedOffset = ~0u;
            std::memcpy(&suppliedOffset, d.push_constants.data(), 4u);
            if (suppliedOffset != offsets[i])
                throw std::runtime_error("PhysX sequence descriptor order changed");
        }
    }
    // One ordered GPU batch. No CPU intermediate downloads or invented
    // per-stage synchronization; the backend tracks the common RW arena.
    commands.emplace_back(CopyCommand{arena, 0u, mirror, 0u, size});
    context.wait_submission(context.submit_batch(commands));
    std::vector<std::uint32_t> actual(f.bytes.size());
    context.download(mirror, 0u, actual.data(), size);
    context.destroy_buffer(mirror);
    context.destroy_buffer(arena);

    const std::uint32_t count = actual.at(f.found_count / 4u);
    if (count != expected.get(f.found_count))
        throw std::runtime_error(
            "PhysX integrated found count mismatch: observed " +
            std::to_string(count) + " expected " +
            std::to_string(expected.get(f.found_count)));

    // Report reservations across workgroups have no global ordering
    // promise. Check an exact set when uncapped; otherwise a unique subset.
    const std::uint32_t slots =
        c.report_capacity > c.initial_report_count ?
        c.report_capacity - c.initial_report_count : 0u;
    const auto written = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(valid.size()), slots);
    std::set<std::pair<std::uint32_t, std::uint32_t>> observed;
    for (std::uint32_t i = 0u; i < written; ++i) {
        const auto byte = f.reports + (c.initial_report_count + i) * 8u;
        const auto pair = std::make_pair(
            actual.at(byte / 4u), actual.at(byte / 4u + 1u));
        if (!valid.count(pair) || !observed.insert(pair).second)
            throw std::runtime_error("Unexpected or duplicated PhysX report");
        expected.put(byte, pair.first);
        expected.put(byte + 4u, pair.second);
    }
    if (written == valid.size() && observed != valid)
        throw std::runtime_error("Integrated PhysX found-pair set differs");
    if (actual != expected.bytes) {
        const auto mismatch = std::mismatch(
            actual.begin(), actual.end(), expected.bytes.begin());
        throw std::runtime_error(
            "Integrated PhysX intermediate/canary mismatch at word " +
            std::to_string(std::distance(actual.begin(), mismatch.first)));
    }
}

void test_invalid(const SapNewBoundsSequenceArenaStage& stage) {
    std::vector<BatchCommand> commands;
    stage.append(commands, 0u, 0u, {1u, 2u, 3u}, 0u, 0u);
    if (!commands.empty()) throw std::runtime_error("Quiet frame dispatched");
    // The first stage must not leak commands if a later stage rejects the
    // descriptor; caller can safely abandon/repair the frame.
    bool rejected = false;
    try { stage.append(commands, 1u, 128u, {0u, 112u, 0u}, 3u, 1u); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected || !commands.empty())
        throw std::runtime_error("Partially appended rejected PhysX frame");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument(
                "Expected overlap producer, two prefix, and pair shader binaries");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto& dev = context.device_info();
        if (!dev.supports_int64_shader || !dev.supports_int64_wave ||
            !dev.supports_wave_arithmetic ||
            dev.subgroup_size < 4u || dev.subgroup_size > 128u ||
            256u % dev.subgroup_size != 0u ||
            dev.max_workgroup_invocations < 256u ||
            dev.max_workgroup_size[0] < 256u ||
            dev.max_workgroup_count[0] < 64u) {
            std::cout << "SKIP: source new-bounds GPU wave/int64/grid unavailable\n";
            return 77;
        }
        SapOverlapWorkArenaStage work(context, shader(argv[1]));
        SapOverlapPrefixArenaStage prefix(
            context, shader(argv[2]), shader(argv[3]));
        SapNewBoundsPairsArenaStage pairs(context, shader(argv[4]));
        SapNewBoundsSequenceArenaStage sequence(work, prefix, pairs);

        for (bool multi : {false, true}) {
            Case c;
            c.two_regions = multi;
            check(context, sequence, c);
            c.new_projection = false;
            check(context, sequence, c);
            c.new_projection = true;
            c.group_exclusion = true;
            check(context, sequence, c);
            c.group_exclusion = false;
            c.aabb_separation = true;
            check(context, sequence, c);
            c.aabb_separation = false;
            c.environment = true;
            check(context, sequence, c);
            c.wildcard_environment = true;
            check(context, sequence, c);
        }
        Case c;
        c.two_regions = true;
        c.initial_report_count = 1u;
        c.report_capacity = 2u;
        check(context, sequence, c);
        c.report_capacity = 1u;
        check(context, sequence, c);
        c.report_capacity = 5u;
        c.empty_work = true;
        check(context, sequence, c);
        c.empty_work = false;
        c.created = false;
        check(context, sequence, c);
        test_invalid(sequence);
        std::cout << "PhysX integrated new-bounds four-kernel sequence PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PhysX integrated new-bounds sequence FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
