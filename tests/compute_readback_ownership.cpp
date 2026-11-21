// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        using namespace fyzz::compute;
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        std::array<std::uint32_t,64u> input{};
        for(std::uint32_t i=0u;i<input.size();++i)
            input[i]=i*29u+13u;
        const auto gpu=context.create_buffer(
            sizeof(input),MemoryPlacement::DeviceLocal);
        context.upload(gpu,0u,input.data(),sizeof(input));

        constexpr std::uint32_t sentinel=0x5a17cd03u;
        std::array<std::uint32_t,3u> first{};
        std::array<std::uint32_t,11u> second{};
        first.fill(sentinel);
        second.fill(sentinel);
        const std::array<DownloadSpan,1u> small{{
            {gpu,0u,first.data(),sizeof(first)}
        }};
        const std::array<DownloadSpan,1u> large{{
            {gpu,20u*sizeof(std::uint32_t),second.data(),sizeof(second)}
        }};
        const auto first_id=context.submit_download_batch(small);
        const auto second_id=context.submit_download_batch(large);
        if(first[0]!=sentinel || second[0]!=sentinel)
            throw std::runtime_error(
                "Queued GPU readbacks published without a host completion");
        context.finish_download_batch(second_id,true);
        for(std::size_t i=0u;i<second.size();++i)
            if(second[i]!=input[20u+i])
                throw std::runtime_error(
                    "Later readback did not preserve its own staging");
        if(first[0]!=sentinel || first[2]!=sentinel)
            throw std::runtime_error(
                "Later readback published the earlier Sc-owned output");
        context.finish_download_batch(first_id,true);
        for(std::size_t i=0u;i<first.size();++i)
            if(first[i]!=input[i])
                throw std::runtime_error(
                    "Earlier readback was corrupted by later staging");

        first.fill(sentinel);
        second.fill(sentinel);
        const auto live_id=context.submit_download_batch(large);
        const auto discarded_id=context.submit_download_batch(small);
        context.finish_download_batch(discarded_id,false);
        if(first[0]!=sentinel || first[2]!=sentinel)
            throw std::runtime_error(
                "Rejected readback mutated the caller-owned output");
        context.finish_download_batch(live_id,true);
        for(std::size_t i=0u;i<second.size();++i)
            if(second[i]!=input[20u+i])
                throw std::runtime_error(
                    "Abandoning one readback lost an independent result");
        for(std::uint32_t round=0u;round<6u;++round) {
            first.fill(sentinel);
            second.fill(sentinel);
            const auto earlier=context.submit_download_batch(small);
            const auto later=context.submit_download_batch(large);
            context.finish_download_batch(later,true);
            context.finish_download_batch(earlier,true);
            for(std::size_t i=0u;i<first.size();++i)
                if(first[i]!=input[i])
                    throw std::runtime_error(
                        "Reused small readback staging lost source data");
            for(std::size_t i=0u;i<second.size();++i)
                if(second[i]!=input[20u+i])
                    throw std::runtime_error(
                        "Reused large readback staging lost source data");
        }
        context.destroy_buffer(gpu);
        std::cout<<"Compute independent readback ownership PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Compute independent readback ownership FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
