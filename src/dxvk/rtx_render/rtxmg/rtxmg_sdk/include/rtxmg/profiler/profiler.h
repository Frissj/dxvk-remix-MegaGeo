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

#pragma once

#include "rtxmg/profiler/stopwatch.h"
#include "rtxmg/profiler/sampler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// clang-format on

struct ImGuiContext;

// Generic execution framework for host/device profiling data.
// 
// * Typical benchmark usage pattern:
//   
//          Timer<> t0 = profiler.InitTimer("timer name");
//          for (frame loop) 
//          {
//              profiler.FrameStart(steady_clock::now());
//   
//              t0.start();
//              // ... excecute profiled task
//              t0.stop();
//           
//              profiler.frameStop(); // all timers have been stopped
//              profiler.FrameResolve(); 
//          }
//          float avg = t0.average();
//          profiler.Terminate();
//   
//    
// * Typical (interactive) profiling usage pattern:
//   
//          Timer<> t0 = profiler.InitTimer("timer name");
//          for (frame loop) 
//          {
//              profiler.FrameStart(steady_clock::now());
//   
//              t0.start();
//              // ... excecute profiled task
//              t0.stop();
//           
//              // ... 
// 
//              profiler.frameStop(); // all timers have been stopped
// 
//              // ...  
// 
//              profiler.FrameSync(); //before any timer is polled
// 
//              float ravg = t0.resolve().runningAverage();
//          }
//          profiler.Terminate();
//
class Profiler
{
  public:
    constexpr static size_t BENCH_FRAME_COUNT = 400;  

    // returns the singleton Profiler
    static Profiler& Get();

    // force the immediate release all device resources
    static void Terminate();

    // frequency < 0 : profile every frame (benchmark mode)
    // frequency == 0 : disable profiling
    // frequency > 0 : records samples at the given pace (in Hz)
    int recordingFrequency = -1;

    // returns true if the Profiler is recording data for the current frame
    bool IsRecording() const { return m_isRecording; }

    // insert at the start of every frame (allows to pace sampling and skip
    // some frames if the frame-rate is too high)
    // 
    // note: the profiler will only monitor events on stream 0 if no dedicated 
    // streams are specified here. This can cause run-time exceptions if device
    // timers are polled without host synchronization
    void FrameStart( std::chrono::steady_clock::time_point time );

    // insert after the last timer is stopped in the frame
    void FrameEnd();

    // benchmarks data for the frame
    void FrameResolve();

    // Generic profiling timer with benchmarking functionality
    template <typename clock_type>
    struct Timer : public Sampler<float, BENCH_FRAME_COUNT>, private clock_type
    {
        Timer( char const* name ) : Sampler( {.name = name} ) { }

        using clock_type::Start;
        using clock_type::Stop;

        // note: user is responsible for device synchronization: use FrameSync()
        Timer& Resolve();  // record duration if the timer was active
        Timer& Profile();  // record duration or 0. if the timer was inactive
    };

    typedef Timer<StopwatchCPU> CPUTimer;
    typedef Timer<StopwatchGPU> GPUTimer;

    template <typename timer_type>
    static inline timer_type& InitTimer( char const* name );

  private:
    Profiler() noexcept         = default;
    Profiler( Profiler const& ) = delete;
    Profiler& operator=( Profiler const& ) = delete;

    std::chrono::steady_clock::time_point m_prevTime;

    bool m_isRecording = false;

  private:
    std::vector<std::unique_ptr<CPUTimer>> m_cpuTimers;
    std::vector<std::unique_ptr<GPUTimer>> m_gpuTimers;
};

template <typename timer_type>
inline timer_type& Profiler::InitTimer( char const* name )
{
    Profiler& profiler = Get();
    assert( profiler.m_prevTime.time_since_epoch().count() == 0 );
    if constexpr( std::is_same_v<timer_type, CPUTimer> )
        return *profiler.m_cpuTimers.emplace_back( std::make_unique<CPUTimer>( name ) );
    else if constexpr( std::is_same_v<timer_type, GPUTimer> )
        return *profiler.m_gpuTimers.emplace_back( std::make_unique<GPUTimer>( name ) );
}

class ScopedGPUTimer
{
public:
    ScopedGPUTimer(Profiler::Timer<StopwatchGPU>& timer, nvrhi::ICommandList *commandlist) : m_timer(timer) 
    {
        m_timer.Start(commandlist);
    }

    ~ScopedGPUTimer()
    {
        m_timer.Stop();
    }

private:
    Profiler::Timer<StopwatchGPU>& m_timer;
};