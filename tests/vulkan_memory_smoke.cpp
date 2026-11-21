// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        using namespace fyzz::compute;
        const auto devices = enumerate_devices(Backend::Vulkan);
        if (devices.empty()) {
            throw std::runtime_error("No Vulkan device exposing a compute queue");
        }
        Context context(Backend::Vulkan, 0);
        const auto& info = context.device_info();
        std::cout << "Vulkan compute device: " << info.name
                  << ", vendor: " << info.vendor_id << '\n';

        std::array<std::uint32_t, 64> source{};
        std::array<std::uint32_t, 64> target{};
        std::array<std::uint32_t, 64> result{};
        for (std::size_t i = 0; i < source.size(); ++i) {
            source[i] = static_cast<std::uint32_t>(i * 17 + 3);
            target[i] = 0xC0FFEEu;
        }

        const auto src = context.create_buffer(sizeof(source));
        const auto dst = context.create_buffer(sizeof(target));
        context.upload(src, 0, source.data(), sizeof(source));
        context.upload(dst, 0, target.data(), sizeof(target));

        constexpr std::size_t kStart = 8;
        constexpr std::size_t kCount = 32;
        context.copy_buffer(src, kStart * sizeof(std::uint32_t),
                            dst, 16 * sizeof(std::uint32_t),
                            kCount * sizeof(std::uint32_t));
        context.download(dst, 0, result.data(), sizeof(result));

        for (std::size_t i = 0; i < result.size(); ++i) {
            const std::uint32_t expected =
                i >= 16 && i < 16 + kCount ? source[kStart + i - 16] : 0xC0FFEEu;
            if (result[i] != expected) {
                throw std::runtime_error("Vulkan buffer-copy result incorrect");
            }
        }

        bool checked_bounds = false;
        try {
            context.copy_buffer(src, sizeof(source) - 4, dst, 0, 8);
        } catch (const std::out_of_range&) {
            checked_bounds = true;
        }
        if (!checked_bounds) {
            throw std::runtime_error("Compute buffer bounds check missing");
        }
        // Confirm the same transfer semantics with GPU-preferred memory.
        const auto local_src = context.create_buffer(sizeof(source), MemoryPlacement::DeviceLocal);
        const auto local_dst = context.create_buffer(sizeof(target), MemoryPlacement::DeviceLocal);
        context.upload(local_src, 0, source.data(), sizeof(source));
        context.upload(local_dst, 0, target.data(), sizeof(target));
        context.copy_buffer(local_src, kStart * sizeof(std::uint32_t),
                            local_dst, 16 * sizeof(std::uint32_t),
                            kCount * sizeof(std::uint32_t));
        context.download(local_dst, 0, result.data(), sizeof(result));
        for (std::size_t i = 0; i < result.size(); ++i) {
            const std::uint32_t expected =
                i >= 16 && i < 16 + kCount ? source[kStart + i - 16] : 0xC0FFEEu;
            if (result[i] != expected) {
                throw std::runtime_error("Device-local buffer copy result incorrect");
            }
        }

        // Verify device-local uploads, downloads, and mixed-placement copies.
        const std::array<std::uint32_t, 3> patch_values = {91u, 92u, 93u};
        context.upload(local_src, 4 * sizeof(std::uint32_t),
                       patch_values.data(), sizeof(patch_values));
        std::array<std::uint32_t, 3> patch_result{};
        context.download(local_src, 4 * sizeof(std::uint32_t),
                         patch_result.data(), sizeof(patch_result));
        if (patch_result != patch_values) {
            throw std::runtime_error("Device-local subrange transfer incorrect");
        }
        context.copy_buffer(local_src, 0, dst, 0, sizeof(source));
        context.download(dst, 0, result.data(), sizeof(result));
        for (std::size_t i = 0; i < result.size(); ++i) {
            const auto expected = i >= 4 && i < 7 ? patch_values[i - 4] : source[i];
            if (result[i] != expected) {
                throw std::runtime_error("Mixed-memory buffer copy result incorrect");
            }
        }

        bool checked_alignment = false;
        try {
            context.upload(local_src, 1, patch_values.data(), sizeof(std::uint32_t));
        } catch (const std::invalid_argument&) {
            checked_alignment = true;
        }
        if (!checked_alignment) {
            throw std::runtime_error("Misaligned device-local upload was accepted");
        }
        // Chain device-local and host-visible copies without a queue-idle wait
        // between commands. The second fence must cover both copies.
        const SubmissionId first = context.submit_copy_buffer(
            local_src, 0, local_dst, 0, sizeof(source));
        const SubmissionId second = context.submit_copy_buffer(
            local_dst, 0, dst, 0, sizeof(source));
        context.wait_submission(second);
        if (!context.submission_complete(first)) {
            throw std::runtime_error("Earlier Vulkan submission did not finish");
        }
        context.wait_submission(first);
        context.download(dst, 0, result.data(), sizeof(result));
        for (std::size_t i = 0; i < result.size(); ++i) {
            const auto expected = i >= 4 && i < 7 ? patch_values[i - 4] : source[i];
            if (result[i] != expected) {
                throw std::runtime_error("Chained asynchronous copies produced wrong data");
            }
        }
        bool rejected_submission = false;
        try {
            context.wait_submission(first);
        } catch (const std::out_of_range&) {
            rejected_submission = true;
        }
        if (!rejected_submission) {
            throw std::runtime_error("Consumed submission ID remained valid");
        }
        context.wait_all();

        // Source readback may grow staging while earlier GPU submission
        // tokens remain owned by an independent ordered stream. Reaping
        // those tokens during staging retirement breaks the Sc fetch.
        std::array<std::uint32_t,2> readback{};
        const SubmissionId retained_small=context.submit_copy_buffer(
            local_src,0,local_dst,0,sizeof(source));
        const std::vector<DownloadSpan> small{{
            local_dst,0,readback.data(),sizeof(std::uint32_t)}};
        context.download_batch(small);
        const SubmissionId retained_grow=context.submit_copy_buffer(
            local_src,0,local_dst,0,sizeof(source));
        const std::vector<DownloadSpan> larger{{
            local_dst,0,readback.data(),sizeof(readback)}};
        context.download_batch(larger);
        if(readback[0]!=source[0] || readback[1]!=source[1])
            throw std::runtime_error(
                "Expanded Vulkan readback lost source-ordered copies");
        const std::array<SubmissionId,2> retained{
            retained_small,retained_grow};
        context.wait_submissions(retained);

        // The readback copy is submitted before its host publication.
        // A rejected transaction must discard staging without writing
        // to the caller's borrowed result arena.
        constexpr std::uint32_t sentinel=0x4e7b2391u;
        std::array<std::uint32_t,3> deferred{sentinel,sentinel,sentinel};
        const std::array<DownloadSpan,2> deferred_spans{{
            {local_src,0,deferred.data(),sizeof(std::uint32_t)},
            {local_src,4*sizeof(std::uint32_t),
                deferred.data()+1,2*sizeof(std::uint32_t)}
        }};
        const auto transfer=context.submit_download_batch(deferred_spans);
        if(deferred[0]!=sentinel || deferred[1]!=sentinel ||
           deferred[2]!=sentinel)
            throw std::runtime_error(
                "Asynchronous readback published before source fetch");
        std::array<std::uint32_t,4> later{
            sentinel,sentinel,sentinel,sentinel};
        const std::array<DownloadSpan,1> later_spans{{
            {local_src,3*sizeof(std::uint32_t),
             later.data(),sizeof(later)}
        }};
        const auto later_transfer=context.submit_download_batch(later_spans);
        if(later[0]!=sentinel || deferred[0]!=sentinel)
            throw std::runtime_error(
                "Overlapping GPU readbacks published before their owners");
        // Dy can complete before the earlier Sc witness publication.
        context.finish_download_batch(later_transfer);
        if(later[0]!=source[3] || later[1]!=patch_values[0] ||
           later[2]!=patch_values[1] || later[3]!=patch_values[2] ||
           deferred[0]!=sentinel || deferred[1]!=sentinel)
            throw std::runtime_error(
                "Later GPU readback corrupted the earlier borrowed result");
        context.finish_download_batch(transfer);
        if(deferred[0]!=source[0] ||
           deferred[1]!=patch_values[0] ||
           deferred[2]!=patch_values[1])
            throw std::runtime_error(
                "Asynchronous source readback lost ordered Gu bytes");
        bool rejected_duplicate=false;
        try {
            context.finish_download_batch(transfer);
        } catch(const std::logic_error&) {
            rejected_duplicate=true;
        }
        if(!rejected_duplicate)
            throw std::runtime_error(
                "Completed asynchronous readback retained its transaction");

        deferred.fill(sentinel);
        const auto discarded=context.submit_download_batch(deferred_spans);
        context.finish_download_batch(discarded,false);
        if(deferred[0]!=sentinel || deferred[1]!=sentinel ||
           deferred[2]!=sentinel)
            throw std::runtime_error(
                "Abandoned asynchronous source readback mutated host output");

        // A blocking single-buffer fetch is allowed to complete earlier
        // work, but the originating stream still owns its submission ID.
        const SubmissionId direct_device=context.submit_copy_buffer(
            local_src,0,local_dst,0,sizeof(source));
        context.download(local_dst,0,result.data(),sizeof(result));
        if(result[0]!=source[0] || result[4]!=patch_values[0])
            throw std::runtime_error(
                "Blocking Vulkan device readback lost ordered source data");
        context.wait_submission(direct_device);

        // Source SAP descriptor allocations may be host-visible. A
        // batched readback must still fence an earlier GPU write to
        // such an allocation, without requiring its token to be reaped.
        const auto mapped_descriptor=context.create_buffer(
            sizeof(source),MemoryPlacement::HostVisible);
        const SubmissionId descriptor_write=context.submit_copy_buffer(
            local_src,0,mapped_descriptor,0,sizeof(source));
        std::array<std::uint32_t,2> descriptor_words{};
        const std::array<DownloadSpan,1> descriptor_readback{{
            {mapped_descriptor,0,descriptor_words.data(),
             sizeof(descriptor_words)}
        }};
        context.download_batch(descriptor_readback);
        if(descriptor_words[0]!=source[0] ||
           descriptor_words[1]!=source[1])
            throw std::runtime_error(
                "Ordered host-visible source descriptor readback lost GPU writes");
        if(!context.submission_complete(descriptor_write))
            throw std::runtime_error(
                "Source descriptor readback did not complete earlier GPU commands");
        context.wait_submission(descriptor_write);
        const SubmissionId direct_mapped=context.submit_copy_buffer(
            local_src,0,mapped_descriptor,0,sizeof(source));
        context.download(mapped_descriptor,0,descriptor_words.data(),
                         sizeof(descriptor_words));
        if(descriptor_words[0]!=source[0] ||
           descriptor_words[1]!=source[1])
            throw std::runtime_error(
                "Blocking Vulkan mapped readback lost ordered source data");
        context.wait_submission(direct_mapped);
        context.destroy_buffer(mapped_descriptor);

        // A copy-only batch creates no descriptor pool.
        context.execute_batch({
            CopyCommand{local_src, 0, local_dst, 0, sizeof(source)},
            CopyCommand{local_dst, 0, dst, 0, sizeof(source)}
        });
        context.download(dst, 0, result.data(), sizeof(result));
        for (std::size_t i = 0; i < result.size(); ++i) {
            const auto expected = i >= 4 && i < 7 ? patch_values[i - 4] : source[i];
            if (result[i] != expected) {
                throw std::runtime_error("Copy-only batch result incorrect");
            }
        }
        context.destroy_buffer(local_dst);
        context.destroy_buffer(local_src);
        context.destroy_buffer(dst);
        context.destroy_buffer(src);
        std::cout << "Portable compute memory acceptance PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Portable compute memory acceptance FAILED: " << e.what() << '\n';
        return 1;
    }
}
