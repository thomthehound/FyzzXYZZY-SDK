// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {

struct Parameters {
    std::uint32_t count;
    std::uint32_t multiplier;
    std::uint32_t addend;
};

std::vector<std::uint8_t> read_shader(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Compiled compute shader was not found");
    }
    std::vector<char> input(std::istreambuf_iterator<char>{stream},
                            std::istreambuf_iterator<char>{});
    return {input.begin(), input.end()};
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("Expected the compiled shader file path");
        }
        using namespace fyzz::compute;
        constexpr std::uint32_t size = 259;
        constexpr std::uint32_t array_size = 320;
        constexpr std::uint32_t sentinel = 0xFEEDACEDu;
        std::array<std::uint32_t, array_size> input{};
        std::array<std::uint32_t, array_size> output{};
        for (std::uint32_t i = 0; i < array_size; ++i) {
            input[i] = i * 37u + 11u;
            output[i] = sentinel;
        }

        Context context(Backend::Vulkan, 0);
        const BufferId src = context.create_buffer(sizeof(input), MemoryPlacement::DeviceLocal);
        const BufferId dst = context.create_buffer(sizeof(output), MemoryPlacement::DeviceLocal);
        context.upload(src, 0, input.data(), sizeof(input));
        context.upload(dst, 0, output.data(), sizeof(output));

        ComputePipelineDesc description;
        description.shader.format = ShaderFormat::Spirv;
        description.shader.bytes = read_shader(argv[1]);
        description.shader.entry_point = "main";
        description.storage_buffer_binding_count = 2;
        description.push_constant_bytes = sizeof(Parameters);
        const PipelineId pipeline = context.create_compute_pipeline(description);

        Parameters parameters{size, 3, 7};
        const std::vector<BufferBinding> bindings = {
            {1, dst, 0, sizeof(output), BufferAccess::Write},
            {0, src, 0, sizeof(input), BufferAccess::Read}
        };
        context.dispatch(pipeline, {(size + 63u) / 64u, 1, 1},
                         bindings, &parameters, sizeof(parameters));
        context.download(dst, 0, output.data(), sizeof(output));

        for (std::uint32_t i = 0; i < array_size; ++i) {
            const std::uint32_t expected =
                i < size ? input[i] * parameters.multiplier + parameters.addend : sentinel;
            if (output[i] != expected) {
                throw std::runtime_error("Compute shader produced incorrect output at " +
                                         std::to_string(i));
            }
        }

        bool rejected = false;
        try {
            context.dispatch(pipeline, {0, 1, 1}, bindings, &parameters, sizeof(parameters));
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Zero-workgroup dispatch was accepted");
        }
        rejected = false;
        try {
            context.dispatch(pipeline, {1, 1, 1},
                             {{0, src, 0, sizeof(input)}, {0, dst, 0, sizeof(output)}},
                             &parameters, sizeof(parameters));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Duplicate descriptor slot was accepted");
        }
        // Explicit intent metadata must be validated, not silently
        // interpreted as writable for an invalid enum value.
        rejected = false;
        auto malformed_access = bindings;
        malformed_access[0].access = static_cast<BufferAccess>(123);
        try {
            context.dispatch(pipeline, {1, 1, 1},
                             malformed_access, &parameters, sizeof(parameters));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Invalid access intent was accepted");
        }
        rejected = false;
        try {
            context.dispatch(pipeline, {1, 1, 1}, bindings, &parameters, 0);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Mismatched push constant size was accepted");
        }

        // The shader consumes only the logical prefix of each larger view.
        auto narrow_bindings=bindings;
        for(auto& binding:narrow_bindings)
            binding.access_extent=BufferAccessExtent{0u,size*sizeof(std::uint32_t)};
        context.dispatch(pipeline,{(size+63u)/64u,1u,1u},
                         narrow_bindings,&parameters,sizeof(parameters));
        context.download(dst,0u,output.data(),sizeof(output));
        for(std::uint32_t i=0u;i<array_size;++i) {
            const auto expected=i<size
                ?input[i]*parameters.multiplier+parameters.addend:sentinel;
            if(output[i]!=expected)
                throw std::runtime_error("Access extent changed descriptor addressing");
        }
        for(const auto extent:{BufferAccessExtent{0u,0u},
                               BufferAccessExtent{sizeof(output),4u},
                               BufferAccessExtent{4u,sizeof(output)}}) {
            auto invalid=narrow_bindings;
            invalid[0].access_extent=extent;
            rejected=false;
            try {
                context.dispatch(pipeline,{1u,1u,1u},invalid,
                                 &parameters,sizeof(parameters));
            }catch(const std::invalid_argument&) { rejected=true; }
            if(!rejected)
                throw std::runtime_error("Invalid access extent was accepted");
        }

        // A compute dispatch and its dependent transfer are both enqueued
        // before waiting. The shader writes must be visible to the copy.
        const BufferId mirror = context.create_buffer(
            sizeof(output), MemoryPlacement::DeviceLocal);
        parameters.multiplier = 7;
        parameters.addend = 19;
        const SubmissionId dispatched = context.submit_dispatch(
            pipeline, {(size + 63u) / 64u, 1, 1},
            bindings, &parameters, sizeof(parameters));
        const SubmissionId copied = context.submit_copy_buffer(
            dst, 0, mirror, 0, sizeof(output));
        context.wait_submission(copied);
        if (!context.submission_complete(dispatched)) {
            throw std::runtime_error("Dependent compute submission remains incomplete");
        }
        context.wait_submission(dispatched);
        context.download(mirror, 0, output.data(), sizeof(output));
        for (std::uint32_t i = 0; i < array_size; ++i) {
            const std::uint32_t expected =
                i < size ? input[i] * parameters.multiplier + parameters.addend : sentinel;
            if (output[i] != expected) {
                throw std::runtime_error("Dependent asynchronous dispatch/copy result incorrect");
            }
        }
        // A three-command batch has two dependent dispatches and a copy.
        // It must execute in order under a single completion fence.
        Parameters first_pass{size, 2, 11};
        Parameters second_pass{size, 5, 2};
        auto push_bytes = [](const Parameters& value) {
            std::vector<std::uint8_t> bytes(sizeof(value));
            std::memcpy(bytes.data(), &value, sizeof(value));
            return bytes;
        };
        const DispatchSize groups{(size + 63u) / 64u, 1, 1};
        // Descriptor slot order is independent of the access declaration.
        DispatchCommand precise{pipeline,groups,bindings,push_bytes(first_pass)};
        constexpr std::size_t half=128u*sizeof(std::uint32_t);
        constexpr std::size_t used=size*sizeof(std::uint32_t);
        precise.accesses={
            {0u,{0u,half},BufferAccess::Read},
            {0u,{half,used-half},BufferAccess::Read},
            {1u,{0u,used},BufferAccess::Write}};
        const std::vector<BatchCommand> batch = {
            precise,
            CopyCommand{dst, 0, mirror, 0, sizeof(output)},
            DispatchCommand{pipeline, groups,
                            {{0, mirror, 0, sizeof(output), BufferAccess::Read},
                             {1, dst, 0, sizeof(output), BufferAccess::Write}},
                            push_bytes(second_pass)}
        };
        const SubmissionId batch_id = context.submit_batch(batch);
        context.wait_submission(batch_id);
        context.download(dst, 0, output.data(), sizeof(output));
        for (std::uint32_t i = 0; i < array_size; ++i) {
            const std::uint32_t expected = i < size
                ? (input[i] * first_pass.multiplier + first_pass.addend) *
                       second_pass.multiplier + second_pass.addend
                : sentinel;
            if (output[i] != expected) {
                throw std::runtime_error("Ordered Vulkan compute batch result incorrect");
            }
        }

        for(const auto access:{
                DispatchAccess{2u,{0u,4u},BufferAccess::Read},
                DispatchAccess{0u,{0u,4u},BufferAccess::Write},
                DispatchAccess{1u,{sizeof(output),4u},BufferAccess::Write}}) {
            auto invalid=precise;
            invalid.accesses.push_back(access);
            rejected=false;
            try {context.wait_submission(context.submit_batch({invalid}));}
            catch(const std::invalid_argument&) {rejected=true;}
            if(!rejected)
                throw std::runtime_error("Invalid phase access declaration was accepted");
        }
        bool rejected_batch = false;
        try {
            context.submit_batch({});
        } catch (const std::invalid_argument&) {
            rejected_batch = true;
        }
        if (!rejected_batch) {
            throw std::runtime_error("Empty Vulkan compute batch was accepted");
        }

        // All input validation must precede submission. No earlier operation
        // may execute if a later command has an invalid buffer range.
        rejected_batch = false;
        try {
            context.submit_batch({
                CopyCommand{src, 0, dst, 0, sizeof(input)},
                CopyCommand{src, 0, dst, 0, sizeof(input) + 4}
            });
        } catch (const std::out_of_range&) {
            rejected_batch = true;
        }
        if (!rejected_batch) {
            throw std::runtime_error("Invalid second batch command was accepted");
        }
        context.download(dst, 0, output.data(), sizeof(output));
        for (std::uint32_t i = 0; i < array_size; ++i) {
            const std::uint32_t expected = i < size
                ? (input[i] * first_pass.multiplier + first_pass.addend) *
                       second_pass.multiplier + second_pass.addend
                : sentinel;
            if (output[i] != expected) {
                throw std::runtime_error("Rejected batch unexpectedly changed GPU data");
            }
        }
        // Unrelated buffers must not force global inter-operation memory
        // barriers. Their commands remain in the same ordered batch, while
        // a dependent copy and second dispatch still require RAW barriers.
        const BufferId independent_input = context.create_buffer(
            sizeof(input), MemoryPlacement::DeviceLocal);
        const BufferId independent_output = context.create_buffer(
            sizeof(output), MemoryPlacement::DeviceLocal);
        const BufferId independent_copy = context.create_buffer(
            sizeof(output), MemoryPlacement::DeviceLocal);
        context.upload(independent_input, 0, input.data(), sizeof(input));
        std::array<std::uint32_t, array_size> initial{};
        initial.fill(sentinel);
        context.upload(independent_output, 0, initial.data(), sizeof(initial));
        Parameters alternate{size, 9, 13};
        std::vector<BatchCommand> independent_batch = {
            DispatchCommand{pipeline, groups, bindings, push_bytes(first_pass)},
            DispatchCommand{pipeline, groups,
                {{0, independent_input, 0, sizeof(input), BufferAccess::Read},
                 {1, independent_output, 0, sizeof(output), BufferAccess::Write}},
                push_bytes(alternate)},
            CopyCommand{independent_output, 0, independent_copy, 0, sizeof(output)}
        };
        for (int iteration = 0; iteration < 12; ++iteration) {
            const SubmissionId id = context.submit_batch(independent_batch);
            context.wait_submission(id);
        }
        context.download(independent_copy, 0, output.data(), sizeof(output));
        for (std::uint32_t i = 0; i < array_size; ++i) {
            const std::uint32_t expected = i < size
                ? input[i] * alternate.multiplier + alternate.addend : sentinel;
            if (output[i] != expected)
                throw std::runtime_error("Independent dispatch/copy or recycled Vulkan resources failed");
        }
        // Simultaneously in-flight submissions must not reuse one another's
        // command buffer or descriptor pool before the corresponding fence.
        const SubmissionId first_in_flight = context.submit_batch(independent_batch);
        const SubmissionId second_in_flight = context.submit_batch(independent_batch);
        context.wait_submission(second_in_flight);
        if (!context.submission_complete(first_in_flight))
            throw std::runtime_error("Earlier same-queue submission not completed");
        context.wait_submission(first_in_flight);

        // Uploads, dependent kernels and copies share one queue fence.
        // Changing the source array after snapshot construction cannot
        // affect either GPU upload's owned bytes.
        std::array<std::uint32_t,array_size> later_input{};
        for(std::uint32_t i=0;i<array_size;++i)
            later_input[i]=i*19u+5u;
        auto owned_bytes=[](const auto& values) {
            std::vector<std::uint8_t> bytes(sizeof(values));
            std::memcpy(bytes.data(),values.data(),sizeof(values));
            return bytes;
        };
        context.upload(dst,0,initial.data(),sizeof(initial));
        const std::vector<BatchCommand> ordered_uploads{
            UploadCommand{src,0,owned_bytes(input)},
            DispatchCommand{pipeline,groups,bindings,push_bytes(first_pass)},
            CopyCommand{dst,0,mirror,0,sizeof(output)},
            UploadCommand{src,0,owned_bytes(later_input)},
            DispatchCommand{pipeline,groups,bindings,push_bytes(second_pass)}
        };
        const auto ordered_id=context.submit_batch(ordered_uploads);
        context.wait_submission(ordered_id);
        std::array<std::uint32_t,array_size> earlier_result{};
        context.download(mirror,0,earlier_result.data(),sizeof(earlier_result));
        context.download(dst,0,output.data(),sizeof(output));
        for(std::uint32_t i=0;i<array_size;++i) {
            const auto first_expected=i<size
                ?input[i]*first_pass.multiplier+first_pass.addend:sentinel;
            const auto second_expected=i<size
                ?later_input[i]*second_pass.multiplier+second_pass.addend:sentinel;
            if(earlier_result[i]!=first_expected ||
               output[i]!=second_expected)
                throw std::runtime_error(
                    "Ordered host uploads did not preserve GPU copy/dispatch dependencies");
        }
        // Preserve RAW, WAR and WAW dependencies across submissions,
        // including a completed intermediate token and a rejected batch.
        context.upload(dst,0,initial.data(),sizeof(initial));
        std::vector<SubmissionId> split;
        for(const auto& operation:ordered_uploads)
            split.push_back(context.submit_batch({operation}));
        context.wait_submission(split[2]);
        rejected_batch=false;
        try {
            context.submit_batch({
                CopyCommand{mirror,0,dst,0,sizeof(output)},
                CopyCommand{src,0,dst,0,sizeof(output)+4u}});
        }catch(const std::out_of_range&) {rejected_batch=true;}
        if(!rejected_batch)
            throw std::runtime_error("Invalid split batch was accepted");
        context.download(mirror,0,earlier_result.data(),sizeof(earlier_result));
        context.download(dst,0,output.data(),sizeof(output));
        for(std::uint32_t i=0;i<array_size;++i) {
            const auto first_expected=i<size
                ?input[i]*first_pass.multiplier+first_pass.addend:sentinel;
            const auto second_expected=i<size
                ?later_input[i]*second_pass.multiplier+second_pass.addend:sentinel;
            if(earlier_result[i]!=first_expected || output[i]!=second_expected)
                throw std::runtime_error("Cross-submission range dependency was lost");
        }
        for(std::size_t i=0u;i<split.size();++i)
            if(i!=2u)context.wait_submission(split[i]);
        // Publish only the written host-visible prefix, and retain
        // native resources through their own last referencing fence.
        const auto host_output=context.create_buffer(sizeof(output),
                                                     MemoryPlacement::HostVisible);
        context.upload(host_output,0u,initial.data(),sizeof(initial));
        auto host_bindings=narrow_bindings;
        host_bindings[0].buffer=host_output;
        const auto host_dispatch=context.submit_batch({
            DispatchCommand{pipeline,groups,host_bindings,push_bytes(first_pass)}});
        const auto tail_copy=context.submit_copy_buffer(
            src,size*4u,host_output,size*4u,4u);
        const auto unrelated=context.submit_batch(independent_batch);
        context.wait_submission(tail_copy);
        if(!context.submission_complete(host_dispatch))
            throw std::runtime_error("Completed queue prefix lost its live token");
        const auto* host_values=static_cast<const std::uint32_t*>(
            context.map_buffer(host_output,HostMapAccess::Read));
        bool correct=true;
        for(std::uint32_t i=0u;i<array_size;++i) {
            const auto expected=i<size
                ?later_input[i]*first_pass.multiplier+first_pass.addend
                :i==size?later_input[i]:sentinel;
            correct&=host_values[i]==expected;
        }
        context.unmap_buffer(host_output);
        if(!correct)
            throw std::runtime_error("Host-visible shader or transfer publication failed");
        context.retire_buffer(host_output);
        context.wait_submission(host_dispatch);
        context.wait_submission(unrelated);
        context.destroy_buffer(independent_copy);
        context.destroy_buffer(independent_output);
        context.destroy_buffer(independent_input);

        context.destroy_buffer(mirror);
        context.destroy_compute_pipeline(pipeline);
        context.destroy_buffer(dst);
        context.destroy_buffer(src);
        std::cout << "Portable compute shader acceptance PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Portable compute shader acceptance FAILED: " << e.what() << '\n';
        return 1;
    }
}
