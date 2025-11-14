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

#include "rtxmg/profiler/profiler.h"
#include <imgui_internal.h>
#include "donut/core/math/math.h"

struct ImPlotContext;
class UserInterface;

// clang-format on

class ProfilerGUI
{
  public:

    // if fps >= 0 displays value in profiler controller window
    int fps = -1;

    // if ntris > 0 displays value in profiler controller window 
    uint32_t desiredTris = 0;
    uint32_t allocatedTris = 0;
    uint32_t desiredClusters = 0;
    uint32_t allocatedClusters = 0;

    struct ControllerWindow
    {
        ImVec2    pos = ImVec2(0, 0);
        ImVec2    pivot = ImVec2(0, 0);
        ImVec2    size = ImVec2(115, 0);
    } controllerWindow;

    struct ProfilerWindow
    {
        ImVec2    pos = ImVec2(0, 0);
        ImVec2    pivot = ImVec2(1, 0);
        ImVec2    size = ImVec2(0, 0);
        ImVec2    screenLayoutSize = ImVec2(0, 0);
    } profilerWindow;

    bool displayGraphWindow = true;

  public:
    template <typename... SamplerGroup>
    void BuildUI( ImFont *iconicFont, ImPlotContext *plotContext, SamplerGroup&... groups );

  private:

    void BuildControllerUI( ImFont *iconicFont, ImPlotContext *plotContext );
    void BuildFrequencySelectorUI();
};

inline ImVec2 MakeImVec2(const dm::float2& v)
{
    return ImVec2{ v.x, v.y };
}

inline ImVec2 MakeImVec2(const dm::int2& v)
{
    return ImVec2{ float(v.x), float(v.y) };
}

inline dm::float2 MakeFloat2(const ImVec2& v)
{
    return dm::float2{ v.x, v.y };
}

inline void SetConstrainedWindowPos(const char *windowName, ImVec2 windowPos, const ImVec2& windowPivot, const ImVec2& screenSize)
{
    ImGuiCond cond = ImGuiCond_FirstUseEver;    
    ImGuiWindow* window = ImGui::FindWindowByName(windowName);

    // Bound the window position to be on screen by a margin
    const float kMinOnscreenLength = 20.0f;
    if (window)
    {
        const dm::float2 kMinOnscreenSize = { kMinOnscreenLength, kMinOnscreenLength };
        dm::float2 currentWindowPos = MakeFloat2(window->Pos);
        dm::float2 currentWindowSize = MakeFloat2(window->Size);
        dm::box2 windowRect{ currentWindowPos, currentWindowPos + currentWindowSize };
        dm::box2 screenLayoutRect{ kMinOnscreenSize, MakeFloat2(screenSize) - kMinOnscreenSize };
        
        if (!screenLayoutRect.intersects(windowRect))
        {
            cond = ImGuiCond_Always;
            dm::float2 minCornerAdjustment = -min(windowRect.m_maxs - screenLayoutRect.m_mins, dm::float2::zero());
            dm::float2 maxCornerAdjustment = -max(windowRect.m_mins - screenLayoutRect.m_maxs, dm::float2::zero());
            dm::float2 adjustment = minCornerAdjustment + maxCornerAdjustment;
            windowRect = windowRect.translate(adjustment);

            windowPos = MakeImVec2(windowRect.m_mins + MakeFloat2(windowPivot) * currentWindowSize);
        }
    }
    ImGui::SetNextWindowPos(windowPos, cond, windowPivot);
}

template <typename... SamplerGroup>
inline void ProfilerGUI::BuildUI( ImFont *iconicFont, ImPlotContext *context, SamplerGroup&... groups )
{
    BuildControllerUI(iconicFont, context);

    if (displayGraphWindow)
    {
        const char* kWindowName = "Profiler";
        SetConstrainedWindowPos(kWindowName, profilerWindow.pos, profilerWindow.pivot, profilerWindow.screenLayoutSize);
        ImGui::SetNextWindowSize(profilerWindow.size, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver); // Collapse the window by default
        ImGui::SetNextWindowBgAlpha(.65f);

        if (ImGui::Begin(kWindowName, &displayGraphWindow, ImGuiWindowFlags_None))
        {
            BuildFrequencySelectorUI();

            if (ImGui::BeginTabBar("MyTabBar", ImGuiTabBarFlags_Reorderable))
            {
                ImVec2 tabSize = profilerWindow.size;
                (
                    [&] {
                        if( ImGui::BeginTabItem( groups.name.c_str() ) )
                        {
                            groups.BuildUI( iconicFont, context );
                            ImGui::EndTabItem();
                        }
                    }(),
                    ... );
                ImGui::EndTabBar();
            }
        }
        ImGui::End();
    }
}
