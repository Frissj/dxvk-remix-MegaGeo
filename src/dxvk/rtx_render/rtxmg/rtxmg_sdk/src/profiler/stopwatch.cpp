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

// clang-format off

#include "rtxmg/profiler/stopwatch.h"

#include <cassert>
#include <donut/core/log.h>

// clang-format on

//
// StopwatchCPU
//

void StopwatchCPU::Start()
{
    m_startTime = steady_clock::now();
    assert( m_stopTime < m_startTime );
}

void StopwatchCPU::Stop()
{
    assert( m_startTime.time_since_epoch().count() > 0 );
    m_stopTime = steady_clock::now();
}

std::optional<float> StopwatchCPU::Elapsed()
{
    assert( m_stopTime >= m_startTime );

    if( m_startTime == steady_clock::time_point{} )
        return {};

    float elapsed = static_cast<float>( duration( m_stopTime - m_startTime ).count() );
    m_startTime = m_stopTime = {};
    return elapsed;
}

std::optional<float> StopwatchCPU::Before( steady_clock::time_point t )
{
    assert( m_startTime >= t && m_startTime > steady_clock::time_point{});
    return (float)duration( m_startTime - t).count();
}

std::optional<float> StopwatchCPU::After( steady_clock::time_point t )
{
    assert( m_stopTime <= t && m_stopTime > steady_clock::time_point{});
    return (float)duration( t - m_stopTime ).count();
}

//
// StopwatchGPU
//
void StopwatchGPU::ProcessUnresolvedQueries()
{
    if (state == State::uninitialized)
        return;

    // New frame started
    // Check our previous queries
    uint32_t unresolvedQueryIndex = m_unresolvedQueryIndex;
    while(unresolvedQueryIndex != m_queryIndex)
    {
        unresolvedQueryIndex = (unresolvedQueryIndex + 1) % kMaxInFlightQueries;
        if (!m_device->pollTimerQuery(m_timerQueries[unresolvedQueryIndex]))
        {
            break;
        }
        // save the last one
        m_unresolvedQueryIndex = unresolvedQueryIndex;
        m_lastDuration = m_device->getTimerQueryTime(m_timerQueries[m_unresolvedQueryIndex]);
        m_hasLastDuration = true;
        m_device->resetTimerQuery(m_timerQueries[m_unresolvedQueryIndex]);
    }
}

void StopwatchGPU::Start(nvrhi::ICommandList* commandList)
{
    if (state == State::uninitialized)
    {
        m_device = commandList->getDevice();
        for (auto& query : m_timerQueries)
        {
            query = m_device->createTimerQuery();
        }
        m_queryIndex = -1;
        m_unresolvedQueryIndex = -1;
        m_hasLastDuration = false;
        state = State::reset;
    }

    ProcessUnresolvedQueries();
    
    // Start a new query. Assumption is one star/stop pair per frame
    // It's possible we overflow max in flight queries, but we just overwrite
    m_queryIndex = (m_queryIndex + 1) % kMaxInFlightQueries;

    // all but 'stopped' states are valid, so can advance up to 3 times
    assert(state != State::ticking);
    commandList->beginTimerQuery(m_timerQueries[m_queryIndex]);
    m_commandList = commandList;
    m_device = commandList->getDevice();
    state = State::ticking;
}

void StopwatchGPU::Stop()
{
    assert(state == State::ticking);
    m_commandList->endTimerQuery(m_timerQueries[m_queryIndex]);
    state = State::stopped;
}

std::optional<float> StopwatchGPU::Elapsed()
{
    if( state == State::reset || state == State::uninitialized)
        return {};

    assert(state == State::stopped);

    state = State::reset;

    ProcessUnresolvedQueries();

    return m_lastDuration * 1000.0f;
}

std::optional<float> StopwatchGPU::ElapsedAsync()
{
    if (state == State::reset || state == State::uninitialized)
        return {};

    // user is responsible for device sync, so we can't track it
    assert(state != State::ticking);

    state = State::reset;

    ProcessUnresolvedQueries();
    if (m_hasLastDuration)
        return m_lastDuration * 1000.0f;
    return {};
}
