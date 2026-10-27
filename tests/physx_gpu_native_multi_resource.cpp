// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

using namespace fyzz::compute;

namespace {
constexpr std::size_t kElements = 64u;
constexpr std::size_t kViewBytes = kElements * sizeof(std::uint32_t);
constexpr std::uint32_t kCanary = 0xcdcdcdcd;
constexpr std::uint32_t kOutputCanary = 0xbabababa;

void check(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

ShaderBinary read_shader(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Native multi-resource shader not found");
    std::vector<char> bytes{std::istreambuf_iterator<char>(file),
                            std::istreambuf_iterator<char>()};
    ShaderBinary shader;
#ifdef FYZZXYZZY_TEST_D3D12
    shader.format = ShaderFormat::Dxil;
#else
    shader.format = ShaderFormat::Spirv;
#endif
    shader.bytes.assign(bytes.begin(), bytes.end());
    return shader;
}

void run_case(Context& context, PipelineId pipeline, bool shared_backing) {
    const std::size_t alignment = std::max<std::size_t>(
        kViewBytes, static_cast<std::size_t>(
            context.device_info().storage_buffer_offset_alignment));
    check(alignment % sizeof(std::uint32_t) == 0u,
          "Native descriptor alignment not word-aligned");
    const std::size_t second_offset = shared_backing ? alignment : 0u;
    const std::size_t first_bytes = shared_backing
        ? second_offset + 2u * kViewBytes : 2u * kViewBytes;

    std::vector<std::uint32_t> first(first_bytes / 4u, kCanary);
    std::vector<std::uint32_t> second(2u * kElements, kCanary);
    std::vector<std::uint32_t> output(2u * kElements, kOutputCanary);
    std::vector<std::uint32_t> mirror(2u * kElements, kOutputCanary);

    for (std::size_t i = 0; i < kElements; ++i) {
        first[i] = static_cast<std::uint32_t>(10u + i);
        if (shared_backing)
            first[second_offset / 4u + i] =
                static_cast<std::uint32_t>(200u + 2u * i);
        else
            second[i] = static_cast<std::uint32_t>(200u + 2u * i);
    }

    const auto source_a = context.create_buffer(first_bytes, MemoryPlacement::DeviceLocal);
    const auto source_b = shared_backing
        ? source_a : context.create_buffer(second.size() * 4u, MemoryPlacement::DeviceLocal);
    const auto destination = context.create_buffer(output.size() * 4u, MemoryPlacement::DeviceLocal);
    const auto snapshot = context.create_buffer(mirror.size() * 4u, MemoryPlacement::DeviceLocal);
    context.upload(source_a, 0u, first.data(), first_bytes);
    if (!shared_backing)
        context.upload(source_b, 0u, second.data(), second.size() * 4u);
    context.upload(destination, 0u, output.data(), output.size() * 4u);
    context.upload(snapshot, 0u, mirror.data(), mirror.size() * 4u);

    DispatchCommand dispatch;
    dispatch.pipeline = pipeline;
    dispatch.groups = {1u, 1u, 1u};
    dispatch.buffers = {
        {0u, source_a, 0u, kViewBytes, BufferAccess::ReadWrite},
        {1u, source_b, second_offset, kViewBytes, BufferAccess::ReadWrite},
        {2u, destination, 0u, kViewBytes, BufferAccess::Write}
    };
    const std::array<std::uint32_t, 8> parameters{{kElements, 0u, 0u, 0u, 0u, 0u, 0u, 0u}};
    dispatch.push_constants.resize(sizeof(parameters));
    std::memcpy(dispatch.push_constants.data(), parameters.data(), sizeof(parameters));

    // Validate the entire batch before any dispatch can execute. A
    // malformed later kernel must not mutate the buffers from the
    // preceding otherwise-valid kernel.
    auto invalid=dispatch;
    invalid.buffers[2].slot=1u;
    bool rejected=false;
    try {(void)context.submit_batch({dispatch,invalid});}
    catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"Malformed later dispatch entered native GPU batch");

    // Two dependent dispatches and a GPU copy must execute in source order
    // with no host synchronization between them. Both source arrays are
    // modified by the shader; the second dispatch reads their new contents.
    auto second_dispatch=dispatch;
#ifndef FYZZXYZZY_TEST_D3D12
    // Vulkan slot order is explicitly specified by BufferBinding::slot.
    // Its flattened prevalidated descriptor array must preserve the slots
    // even if a later dispatch supplies bindings in a different order.
    std::rotate(second_dispatch.buffers.begin(),
                second_dispatch.buffers.begin()+1,
                second_dispatch.buffers.end());
#endif
    const std::vector<BatchCommand> commands{
        dispatch, second_dispatch,
        CopyCommand{destination, 0u, snapshot, 0u, kViewBytes}
    };
    const auto submission = context.submit_batch(commands);
    context.wait_submission(submission);

    std::vector<std::uint32_t> actual_first(first.size());
    std::vector<std::uint32_t> actual_second(second.size());
    std::vector<std::uint32_t> actual_output(output.size());
    std::vector<std::uint32_t> actual_mirror(mirror.size());
    context.download(source_a, 0u, actual_first.data(), first_bytes);
    if (!shared_backing)
        context.download(source_b, 0u, actual_second.data(), second.size() * 4u);
    context.download(destination, 0u, actual_output.data(), output.size() * 4u);
    context.download(snapshot, 0u, actual_mirror.data(), mirror.size() * 4u);

    for (std::size_t i = 0; i < kElements; ++i) {
        const auto a = static_cast<std::uint32_t>(10u + i);
        const auto b = static_cast<std::uint32_t>(200u + 2u * i);
        first[i] = a + 2u;
        if (shared_backing)
            first[second_offset / 4u + i] = b + 4u;
        else
            second[i] = b + 4u;
        output[i] = 3u * (a + 1u) + (b + 2u);
        mirror[i] = output[i];
    }
    check(actual_first == first,
          "Native multi-resource source A changed outside its view or has incorrect results");
    if (!shared_backing)
        check(actual_second == second,
              "Native multi-resource source B changed outside its view or has incorrect results");
    check(actual_output == output,
          "Native multi-resource output or canary mismatch");
    check(actual_mirror == mirror,
          "Native multi-resource ordered GPU copy or canary mismatch");

    context.destroy_buffer(snapshot);
    context.destroy_buffer(destination);
    if (!shared_backing) context.destroy_buffer(source_b);
    context.destroy_buffer(source_a);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("Expected compiled native multi-resource shader");
        Context context(
#ifdef FYZZXYZZY_TEST_D3D12
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto& device = context.device_info();
        if (device.max_workgroup_invocations < 64u ||
            device.max_workgroup_size[0] < 64u ||
            device.max_workgroup_count[0] < 1u) {
            std::cout << "SKIP: native 64-thread compute unavailable\n";
            return 77;
        }
        ComputePipelineDesc desc;
        desc.shader = read_shader(argv[1]);
        desc.storage_buffer_binding_count = 3u;
        desc.storage_buffer_views = {
            StorageBufferViewKind::RawReadWrite,
            StorageBufferViewKind::RawReadWrite,
            StorageBufferViewKind::RawReadWrite
        };
        desc.push_constant_bytes = 32u;
        const auto pipeline = context.create_compute_pipeline(desc);
        run_case(context, pipeline, false);
        run_case(context, pipeline, true);
        context.destroy_compute_pipeline(pipeline);
        std::cout << "Native multi-resource binding and ordering PASS\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Native multi-resource binding FAILED: " << ex.what() << '\n';
        return 1;
    }
}
