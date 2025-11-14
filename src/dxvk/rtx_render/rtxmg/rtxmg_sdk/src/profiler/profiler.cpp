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


#include "rtxmg/profiler/profiler.h"
#include "rtxmg/profiler/stopwatch.h"


#include <cassert>
#include <type_traits>
#include <variant>

// clang-format on
static Profiler *s_profiler = nullptr;

Profiler& Profiler::Get()
{
    if (!s_profiler)
        s_profiler = new Profiler();
    return *s_profiler;
}

void Profiler::Terminate()
{
    delete s_profiler;
}

using CPUTimer = Profiler::Timer<StopwatchCPU>;
using GPUTimer = Profiler::Timer<StopwatchGPU>;

template <>
CPUTimer& Profiler::Timer<StopwatchCPU>::Resolve()
{
    if( auto e = Elapsed() )
        PushBack( *e );
    return *this;
}
template <>
GPUTimer& Profiler::Timer<StopwatchGPU>::Resolve()
{
    if( auto e = ElapsedAsync() )
        PushBack( *e );
    return *this;
}

template <>
CPUTimer& Profiler::Timer<StopwatchCPU>::Profile()
{
    if (Profiler::Get().IsRecording())
    {
        auto e = Elapsed();
        PushBack( e ? *e : 0.f );
    }
    return *this;
}
template <>
GPUTimer& Profiler::Timer<StopwatchGPU>::Profile()
{
    if (Profiler::Get().IsRecording())
    {
        auto e = ElapsedAsync();
        if (e)
        {
            PushBack(*e);
        }
    }
    return *this;
}

void Profiler::FrameStart( std::chrono::steady_clock::time_point time )
{
    int frequency = recordingFrequency;
    if( frequency > 0 )
    {
        double period = 1000. / double( frequency );

        m_isRecording = std::chrono::duration<double, std::milli>( time - m_prevTime ).count() >= period
                        || m_prevTime == std::chrono::steady_clock::time_point{};
    }
    else if( frequency < 0 )
        m_isRecording = true;
    else
        m_isRecording = false;

    if (m_isRecording)
        m_prevTime = time;
}

void Profiler::FrameEnd()
{

}

void Profiler::FrameResolve()
{
    auto resolveTimers = []( auto& timers ) {
        for( auto& timer : timers )
            timer->Resolve();
    };

    resolveTimers( m_cpuTimers );

    if( !m_gpuTimers.empty() )
    {
        resolveTimers( m_gpuTimers );
    }
}
