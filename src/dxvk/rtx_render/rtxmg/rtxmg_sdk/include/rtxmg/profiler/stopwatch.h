//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

// clang-format off

#include <cassert>
#include <chrono>
#include <cstdint>
#include <optional>

#include <nvrhi/nvrhi.h>

// clang-format on

class StopwatchCPU
{
public:
    void Start();
    void Stop();

    std::optional<float> Elapsed();  // returns dt = stop - start
    std::optional<float> Before(std::chrono::steady_clock::time_point t);
    std::optional<float> After(std::chrono::steady_clock::time_point t);

private:
    using steady_clock = std::chrono::steady_clock;
    using duration = std::chrono::duration<double, std::milli>;

    steady_clock::time_point m_startTime;
    steady_clock::time_point m_stopTime;
};

class StopwatchGPU
{
public:
    void Start(nvrhi::ICommandList* commandList);
    void Stop();

    std::optional<float> Elapsed();      // returns dt = stop - start
    std::optional<float> ElapsedAsync(); // returns dt = stop - start
private:

    void ProcessUnresolvedQueries();

    static constexpr uint32_t kMaxInFlightQueries = 3;

    nvrhi::DeviceHandle m_device;
    std::array<nvrhi::TimerQueryHandle, kMaxInFlightQueries> m_timerQueries;
    nvrhi::CommandListHandle m_commandList;
    int32_t m_queryIndex = -1;
    int32_t m_unresolvedQueryIndex = -1;
    float m_lastDuration = 0.f;
    bool m_hasLastDuration = false;

    enum class State : uint8_t
    {
        uninitialized = 0,
        reset,
        ticking,
        stopped
    } state = State::uninitialized;
};
