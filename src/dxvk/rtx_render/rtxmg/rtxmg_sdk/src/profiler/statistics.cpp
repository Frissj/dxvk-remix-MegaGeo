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

#include <numeric>
#include <opensubdiv/tmr/topologyMap.h>

#include <cmath>
#include <locale>
#include <string>
#include <sstream>

#include <imgui_internal.h>
#include <implot.h>

#include "rtxmg/profiler/gui.h"
#include "rtxmg/profiler/statistics.h"
#include "rtxmg/utils/buffer.h"
#include "rtxmg/utils/formatters.h"

namespace stats {

    constexpr ImPlotAxisFlags flags = ImPlotAxisFlags_AutoFit;
    constexpr double          yref = 0;  // std::numeric_limits<double>::lowest();
    constexpr double          xscale = 1.0;
    constexpr double          xstart = 0;

    FrameSamplers        frameSamplers;
    ClusterAccelSamplers clusterAccelSamplers;
    MemUsageSamplers     memUsageSamplers;
    EvaluatorSamplers evaluatorSamplers;

    void FrameSamplers::BuildUI(ImFont *iconicFont, ImPlotContext *plotContext) const
    {
        constexpr int stride = (int)sizeof(float);

        const float fontScale = ImGui::GetIO().FontGlobalScale;

        enum class GraphMode : int {
            Overview = 0,
            HiZ
        };
        static GraphMode mode = GraphMode::Overview;
        ImGui::PushItemWidth(125);
        ImGui::Combo("Graph modes", reinterpret_cast<int*>(&mode), "Overview\0Hierarchical-Z\0");
        ImGui::PopItemWidth();

        std::array<GPUTimer*, 3> timers = { nullptr, nullptr, nullptr };

        switch (mode)
        {
        case GraphMode::Overview: {
            timers[0] = &gpuFrameTime.Profile();
            timers[1] = &gpuRenderTime.Profile();
            timers[2] = &gpuDenoiserTime.Profile();
            computeMotionVectorsTimer.Profile();
        } break;
        case GraphMode::HiZ: {
            timers[0] = &zRenderPassTime.Profile();
            timers[1] = &zReprojectionTime.Profile();
            timers[2] = &hiZRenderTime.Profile();
        } break;
        default:
            return;
        }

        // Implot appears to be having issues if the arrays of data have different sizes & offsets
        assert(timers[0]->size() == timers[1]->size() && timers[0]->size() == timers[2]->size());

        if (ImPlot::BeginPlot("##timers2", ImVec2(-1, 150 * fontScale)))
        {
            ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(timers[0]->size()), ImGuiCond_Always);

            constexpr bool autofit = false;

            if constexpr (autofit)
            {
                // ImPlot autofit is a little wiggly - exploring alternatives below
                ImPlot::SetupAxis(ImAxis_Y1, timers[0]->name.c_str(), ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);
                ImPlot::SetupAxis(ImAxis_Y2, timers[1]->name.c_str(), ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);
                ImPlot::SetupAxis(ImAxis_Y3, timers[2]->name.c_str(), ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);
            }
            else
            {
                ImPlot::SetupAxis(ImAxis_Y1, "Time (ms)");
                ImPlot::SetupAxis(ImAxis_Y2, "##hidden1", ImPlotAxisFlags_NoDecorations);
                ImPlot::SetupAxis(ImAxis_Y3, "##hidden2", ImPlotAxisFlags_NoDecorations);

                float vmax = timers[0]->RunningAverage() * 1.75f;
                if (vmax < 1e-6)
                    vmax = timers[1]->RunningAverage() * 1.75f;

                // Constrain both Y axes to the same range for better readability
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0., vmax, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y2, 0., vmax, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y3, 0., vmax, ImPlotCond_Always);
            }


            for (uint8_t i = 0; i < timers.size(); ++i)
            {
                if (!timers[i])
                    continue;

                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1 + i);

                ImPlot::PlotLine(timers[i]->name.c_str(), timers[i]->data(), (int)timers[i]->size(),
                    xscale, xstart, ImPlotShadedFlags_None, timers[i]->Offset(), stride);
            }

            ImPlot::EndPlot();
            if (ImGui::IsItemHovered())
            {
                switch (mode)
                {
                case GraphMode::Overview:
                    ImGui::SetTooltip(
                        "GPU timers:\n"
                        "  - Frame: %.4fms\n"
                        "  - Pathtrace: %.4fms\n"
                        "  - Motion Vectors: %.4fms\n"
                        "  - Denoiser: %.4fms\n",
                        gpuFrameTime.RunningAverage(),
                        gpuRenderTime.RunningAverage(),
                        computeMotionVectorsTimer.RunningAverage(),
                        gpuDenoiserTime.RunningAverage());
                break;
                default:
                    return;
                }
            }

        }

        // note: only one of the profiler tabs is active at a time, so we have to call
        // profile() on these timers to Update these samplers

        static Sampler<float> trisPerSec;
        if (Profiler::Get().IsRecording())
        {
            auto const& clusterTiling = clusterAccelSamplers.clusterTilingTime.Profile();
            auto const& fillClusters = clusterAccelSamplers.fillClustersTime.Profile();
            auto const& clas = clusterAccelSamplers.buildClasTime.Profile();
            auto const& blas = clusterAccelSamplers.buildBlasTime.Profile();
            float sumTime = (clusterTiling.latest + fillClusters.latest + clas.latest + blas.latest);

            uint32_t ntris = clusterAccelSamplers.numTriangles.latest;

            trisPerSec.PushBack(static_cast<float>(1000. * double(ntris) / double(sumTime)));
        }

        if (ImPlot::BeginPlot("BVH Throughput", ImVec2(-1, 150 * fontScale)))
        {
            ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, (double)trisPerSec.size(), ImGuiCond_Always);

            ImPlot::SetupAxis(ImAxis_Y1, "Tris / Sec", ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisFormat(ImAxis_Y1, HumanFormatter, nullptr);

            ImPlot::PlotShaded(trisPerSec.name.c_str(), trisPerSec.data(), (int)trisPerSec.size(), 0.f, xscale, xstart,
                ImPlotShadedFlags_None, trisPerSec.Offset(), stride);

            ImPlot::EndPlot();

            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Number of triangles processed per second.\n\n"
                    "Processing includes:\n"
                    "  - surface edge-metric evaluation\n"
                    "  - catmark limit surface evaluation\n"
                    "  - displacement\n"
                    "  - tessellation\n"
                    "  - cluster fill\n"
                    "  - BVH build\n");
        }
    }

    void EvaluatorSamplers::BuildUI(ImFont *iconicFont, ImPlotContext *plotContext)
    {
        if (hasBadTopology)
        {
            ImGui::PushFont(iconicFont);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.223f, 0.325f, 0.447f, 1.f));
            ImGui::SeparatorText((char const*)(u8"\ue08F" "## recommendations"));
            ImGui::PopStyleColor();
            ImGui::PopFont();

            ImGui::TextWrapped("Switch to the 'Topology Quality' color mode in the settings "
                "window to visualize problem areas in the mesh. Areas in red are in need of attention");

            m_topologyQualityButtonPressed = ImGui::Button("Topology Quality");
            if (ImGui::IsItemHovered() && ImGui::GetCurrentContext()->HoveredIdTimer > .5f)
                ImGui::SetTooltip("Switches the color mode to 'Toplogy Quality'.");
            ImGui::Spacing();
        }
        char buf[32];

        const float fontScale = ImGui::GetIO().FontGlobalScale;

        auto buildRow = [&buf]<typename T>(char const* name, T value, char const* tooltip = nullptr, bool displayMB = false)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", name);
            ImGui::TableSetColumnIndex(1);
            if constexpr (std::is_same_v<T, size_t>)
            {
                if (displayMB)
                    MegabytesFormatter(static_cast<double>(value), buf, (int) std::size(buf));
                else
                    MemoryFormatter(static_cast<double>(value), buf, (int) std::size(buf));
                ImGui::Text("%s", buf);
            }
            else if constexpr (std::is_same_v<T, float>)
                ImGui::Text("%.1f", value);
            else if constexpr (std::is_integral_v<T>)
                ImGui::Text("%d", int64_t(value));
            if (tooltip && ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tooltip);
        };

        for (uint32_t i = 0; i < (uint32_t)surfaceTableStats.size(); ++i)
            surfaceTableStats[i].BuildUI(iconicFont, plotContext, i);

        ImGui::Spacing();

        if (ImGui::CollapsingHeader("TopologyMap", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::BeginTable("Topology Map", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX);
            {
                static const float kColWidth0 = 200 * fontScale;
                static const float kColWidth1 = 80 * fontScale;

                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, kColWidth0);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, kColWidth1);
                ImGui::TableHeadersRow();

                buildRow("PSL mean", topologyMapStats.pslMean);
                buildRow("Hash count", (int64_t)topologyMapStats.hashCount);
                buildRow("Address space", (int64_t)topologyMapStats.addressCount);
                buildRow("load factor", topologyMapStats.loadFactor);
                ImGui::TableNextRow(ImGuiTableRowFlags_Headers, 2.5f);
                //buildRow( "Patch-points max", (int64_t)topologyMap.patchPointsMax );
                buildRow("Subdivision plans count", (int64_t)topologyMapStats.plansCount);
                buildRow("Stencil matrix row count min", (int64_t)topologyMapStats.stencilCountMin);
                buildRow("Stencil matrix row count max", (int64_t)topologyMapStats.stencilCountMax);
                buildRow("Stencil matrix row count avg", (int64_t)topologyMapStats.stencilCountAvg);
                buildRow("Memory use", topologyMapStats.plansByteSize);
            }
            ImGui::EndTable();

            ImGui::SameLine();

            if (ImPlot::BeginPlot("##TopomapStencilHistogram", ImVec2(-1, 174 * fontScale), ImPlotFlags_NoMouseText))
            {
                auto const& values = topologyMapStats.stencilCountHistogram;

                ImPlotFormatter formatter = [](double value, char* buff, int size, void* user_data) -> int
                    {
                        auto const* samplers = reinterpret_cast<EvaluatorSamplers const*>(user_data);
                        uint32_t    min = samplers->topologyMapStats.stencilCountMin;
                        uint32_t    max = samplers->topologyMapStats.stencilCountMax;
                        uint32_t    count = (uint32_t)samplers->topologyMapStats.stencilCountHistogram.size();
                        if (uint32_t range = max - min; range > 0 && count > 0)
                            value = min + (value / count) * range;
                        else
                            value = min;
                        return snprintf(buff, size_t(size), "%d", (int)value);
                    };

                ImPlot::SetupAxis(ImAxis_X1, "Num patch points", ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisFormat(ImAxis_X1, formatter, (void*)this);
                ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_AutoFit);
                ImPlot::PlotBars("Num Plans", values.data(), static_cast<int>(values.size()), 1.0, 0.5, ImPlotBarsFlags_None);

                ImPlot::EndPlot();
            }
        }
    }


    void ClusterAccelSamplers::BuildUI(ImFont *iconicFont, ImPlotContext *plotContext) const
    {
        constexpr int stride = (int)sizeof(uint32_t);

        const float fontScale = ImGui::GetIO().FontGlobalScale;

        if (ImPlot::BeginPlot("##accel_builder_tess", ImVec2(-1, 150 * fontScale)))
        {
            auto const& buildBlas = buildBlasTime.Profile();
            auto const& fillClusters = fillClustersTime.Profile();
            auto const& clusterTiling = clusterTilingTime.Profile();
            auto const& buildClas = buildClasTime.Profile();
            ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
            float vmax = 3.f;
            if (float ravg = std::max(buildClas.RunningAverage(),
                std::max(buildBlas.RunningAverage(), std::max(clusterTiling.RunningAverage(), fillClusters.RunningAverage())));
                ravg > (vmax * .01f))
                vmax = ravg * 2.f;
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0., vmax, ImPlotCond_Always);

            ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(buildBlas.size()), ImGuiCond_Always);

            ImPlot::SetupAxis(ImAxis_Y1, "Time (ms)", ImPlotAxisFlags_AutoFit);
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
            auto plot = [](auto& series)
                {
                    ImPlot::PlotLine(series.name.c_str(), series.data(), (int)series.size(), xscale, xstart,
                                      ImPlotShadedFlags_None, static_cast<int>(series.Offset()), stride);
                };
            plot(clusterTiling);
            plot(fillClusters);
            plot(buildClas);
            plot(buildBlas);
            ImPlot::EndPlot();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "GPU timers:\n\n"
                    "  - Cluster Tiling: %.4fms tessellation metric\n"
                    "    + limit surface evaluation prep\n\n"
                    "  - Fill Clusters: %.4fms subdivision surface\n"
                    "    limit evaluation + vertex writing.\n\n"
                    "  - CLAS build: %.4fms CLAS build time.\n\n"
                    "  - BLAS build: %.4fms BLAS from CLAS build time",
                    clusterTiling.RunningAverage(),
                    fillClusters.RunningAverage(),
                    buildClas.RunningAverage(),
                    buildBlas.RunningAverage());
        }
        ImGui::Spacing();

        auto const& nt = numTriangles;
        auto const& nc = numClusters;
        if (ImPlot::BeginPlot("##accel_builder_geo", ImVec2(-1, 150 * fontScale)))
        {
            ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(nt.size()), ImGuiCond_Always);
            ImPlot::SetupAxis(ImAxis_Y1, nt.name.c_str(), ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisFormat(ImAxis_Y1, HumanFormatter, nullptr);

            ImPlot::SetupAxis(ImAxis_X2, nullptr, ImPlotAxisFlags_NoDecorations);
            ImPlot::SetupAxisLimits(ImAxis_X2, 0, static_cast<double>(nc.size()), ImGuiCond_Always);
            ImPlot::SetupAxis(ImAxis_Y2, nc.name.c_str(), ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);
            ImPlot::SetupAxisFormat(ImAxis_Y2, HumanFormatter, nullptr);

            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);

            //ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);
            ImPlot::PlotShaded(nt.name.c_str(), nt.data(), (int)nt.size(), yref, xscale, xstart, ImPlotShadedFlags_None,
                                static_cast<int>(nt.Offset()), stride);
            //ImPlot::PlotLine(nt.name.c_str(), nt.samples.data(), (int)nt.samples.m_size(), xscale, xstart, ImPlotShadedFlags_None, nt.offset(), stride);

            ImPlot::SetAxes(ImAxis_X2, ImAxis_Y2);
            //ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);
            //ImPlot::PlotShaded(nc.name.c_str(), nc.samples.data(), (int)nc.samples.m_size(),  yref, xscale, xstart, ImPlotShadedFlags_None, nc.offset(), stride);
            ImPlot::PlotLine(nc.name.c_str(), nc.data(), (int)nc.size(), xscale, xstart, ImPlotShadedFlags_None,
                              static_cast<int>(nc.Offset()), stride);

            ImPlot::EndPlot();
        }
    }
    void MemUsageSamplers::BuildUI(ImFont *iconicFont, ImPlotContext *plotContext) const
    {
        const float fontScale = ImGui::GetIO().FontGlobalScale;

        uint32_t desiredTris = stats::clusterAccelSamplers.numTriangles.latest;
        uint32_t desiredClusters= stats::clusterAccelSamplers.numClusters.latest;
        uint32_t allocatedClusters = stats::clusterAccelSamplers.numClusters.max;
        uint32_t numPixels = stats::clusterAccelSamplers.renderSize.x * stats::clusterAccelSamplers.renderSize.y;

        ImGui::Spacing();

        ImGui::Text("Render Resolution: %d x %d", stats::clusterAccelSamplers.renderSize.x, stats::clusterAccelSamplers.renderSize.y);
        ImGui::Text("Micro-triangles: %zu (%.2f per pixel)", desiredTris, desiredTris / float(numPixels));
        ImGui::Text("Clusters: %zu / %zu", desiredClusters, allocatedClusters);

        ImGui::Spacing();

        ImGui::Separator();

        ImGui::Spacing();
        ImGui::Text("BVH");
        ImGui::Spacing();

        ImGui::BeginTable("Memory Usage", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX);
        {
            static const float kColWidth0 = 200 * fontScale;
            static const float kColWidth1 = 80 * fontScale;
            
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, kColWidth0);
            ImGui::TableSetupColumn("Required", ImGuiTableColumnFlags_WidthFixed, kColWidth1);
            ImGui::TableSetupColumn("Allocated", ImGuiTableColumnFlags_WidthFixed, kColWidth1);
            ImGui::TableSetupColumn("Per-uTri", ImGuiTableColumnFlags_WidthFixed, kColWidth1);
            ImGui::TableSetupColumn("Per Pixel", ImGuiTableColumnFlags_WidthFixed, kColWidth1);
            ImGui::TableSetupColumn("Per Cluster", ImGuiTableColumnFlags_WidthFixed, kColWidth1);

            ImGui::TableHeadersRow();

            char buf[32];
            char bufMax[32];

            auto buildRow = [&buf, &bufMax](char const* name, size_t bytes, size_t maxBytes, uint32_t tris, uint32_t pixels, uint32_t clusters, bool displayMB = true)
                {
                    bool memoryExceeded = bytes > maxBytes;
                    if (memoryExceeded)
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
                    }
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%s", name);
                    ImGui::TableSetColumnIndex(1);
                    if (displayMB)
                    {
                        MegabytesFormatter(static_cast<double>(bytes), buf, (int)std::size(buf));
                        MegabytesFormatter(static_cast<double>(maxBytes), bufMax, (int)std::size(bufMax));
                    }
                    else
                    {
                        MemoryFormatter(static_cast<double>(bytes), buf, (int)std::size(buf));
                        MemoryFormatter(static_cast<double>(maxBytes), bufMax, (int)std::size(bufMax));
                    }
                    
                    ImGui::Text("%s", buf);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%zu bytes", bytes);

                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%s", bufMax);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%zu bytes", maxBytes);

                    ImGui::TableSetColumnIndex(3);
                    if (tris)
                        ImGui::Text("%.1f bits", float(bytes) / float(tris) * 8);
                    else
                        ImGui::Text("n/a");

                    ImGui::TableSetColumnIndex(4);
                    if (pixels)
                        ImGui::Text("%.2f B", float(bytes) / pixels);
                    else
                        ImGui::Text("n/a");


                    ImGui::TableSetColumnIndex(5);
                    if (clusters)
                    {
                        MemoryFormatter(static_cast<double>(bytes / double(clusters)), buf, (int)std::size(buf));
                        ImGui::Text("%s", buf);
                    }
                    else
                        ImGui::Text("n/a");

                    if (memoryExceeded)
                    {
                        ImGui::PopStyleColor();
                    }
                };

            buildRow("Vertex buffer", vertexBufferSize.latest, vertexBufferSize.max, desiredTris, numPixels, desiredClusters, 0);
            buildRow("Vertex normals buffer", vertexNormalsBufferSize.latest, vertexNormalsBufferSize.max, desiredTris, numPixels, desiredClusters, 0);
            buildRow("Cluster AS (CLAS)", clasSize.latest, clasSize.max, desiredTris, numPixels, desiredClusters);
            buildRow("Cluster Data buffer", clusterShadingDataSize.latest, clusterShadingDataSize.max, 0, 0, desiredClusters);
            buildRow("Bottom Level AS (BLAS)", blasSize.latest, blasSize.max, 0, 0, allocatedClusters);
            buildRow("BLAS scratch buffer", blasScratchSize.latest, blasScratchSize.max, 0, 0, allocatedClusters);
            
            size_t total = blasSize.latest + blasScratchSize.latest + clasSize.latest + vertexBufferSize.latest
                + vertexNormalsBufferSize.latest + clusterShadingDataSize.latest;

            size_t totalMax = blasSize.max + blasScratchSize.max + clasSize.max + vertexBufferSize.max
                + vertexNormalsBufferSize.max + clusterShadingDataSize.max;

            buildRow("Total ", total, totalMax, desiredTris, numPixels, desiredClusters, false);
        }
        ImGui::EndTable();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();


        ImGui::Text("Topology & Subdivision");
        ImGui::Spacing();

        ImGui::BeginTable("Subdivision", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX);
        {
            static const float kColWidth0 = 200 * fontScale;
            static const float kColWidth1 = 80 * fontScale;

            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, kColWidth0);
            ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthFixed, kColWidth1);

            ImGui::TableHeadersRow();

            char buf[32];

            auto buildRow = [&buf](char const* name, size_t m_size, bool displayMB = true, char const* tooltip = nullptr)
                {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%s", name);
                    if (tooltip && ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", tooltip);
                    ImGui::TableSetColumnIndex(1);
                    if (displayMB)
                        MegabytesFormatter(static_cast<double>(m_size), buf, (int) std::size(buf));
                    else
                        MemoryFormatter(static_cast<double>(m_size), buf, (int) std::size(buf));
                    ImGui::Text("%s", buf);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%zu bytes.", m_size);
                };

            
            buildRow("Topology map", evaluatorSamplers.topologyMapStats.plansByteSize, false,
                "Total m_size of topology map.\n"
                "note: there should be only 1 topology map shared by all the sub-d meshes in the scene.\n");
            buildRow("Surface tables", evaluatorSamplers.surfaceTablesByteSizeTotal, false,
                "Total m_size of vertex surface table.\n"
                "The 'surface table' replaces the index buffer for each sub-d mesh in the scene.\n"
                "The m_size of a surface table is typically 3x to 5x the m_size of the control cage\n"
                "index buffer (compare above).\n");

            size_t total = evaluatorSamplers.topologyMapStats.plansByteSize
                + evaluatorSamplers.surfaceTablesByteSizeTotal;

            buildRow("Total ", total, false);
        }
        ImGui::EndTable();
    }

    void SurfaceTableStats::BuildTopologyRecommendations()
    {
        float ratio = 0.f;
        if (!IsCatmarkTopology(&ratio))
        {
            std::stringstream ss;
            ss.setf(std::ios::fixed);
            ss.precision(1);
            ss << "High number of irregular (non-quad) faces detected (" << ratio * 100.f << " %). ";
            ss << "Irregular faces impact both performance and memory. Catmark subdivision ";
            ss << "meshes should use mostly quads.";
            topologyRecommendations.push_back(ss.str());
        }

        // note: TopologyRefiner::GetMaxValence() accumulates both face and vertex valence
        // into the same max variable ; on the rare occasion where a model has both a high
        // valence vertex and a face of equal or greater valence, this recommendation will
        // not be triggered. The assumption is that once the high valence faces are removed
        // from the topology, if high valence vertices remain, this recommendation will 
        // then trigger as intended.
        if ((maxValence > 8) && (maxValence > maxFaceSize))
        {
            std::stringstream ss;
            ss << "Some vertices have up to " << maxValence << " incident edges. Ideally max ";
            ss << "valence should be <= 8.";
            topologyRecommendations.push_back(ss.str());
        }

        if (maxFaceSize > 5)
        {
            std::stringstream ss;
            ss << "Some polygons faces have up to " << maxFaceSize << " edges. It is recommended ";
            ss << "to use quads with a few triangles and pentagons in delicate areas.";
            topologyRecommendations.push_back(ss.str());
        }

        if (sharpnessMax > 8.f)
        {
            std::stringstream ss;
            ss << "Some creased edges or vertices have a very high sharpness value (found up ";
            ss << "to " << sharpnessMax << "). Consider replacing those with 'infinitely sharp' ";
            ss << "creases of value 10 for better performance.";
            topologyRecommendations.push_back(ss.str());
        }
        else if (sharpnessMax > 4.f && sharpnessMax <= 8.f)
        {
            std::stringstream ss;
            ss << "Some creased edges or vertices have a high sharpness value (found up to ";
            ss << sharpnessMax << "). Consider adding edge-loops and reducing sharpness creases ";
            ss << "to values <= 4.0 for better control over the surface and better performance.";
            topologyRecommendations.push_back(ss.str());
        }
    }
    
    void SurfaceTableStats::BuildRecommendationsUI(ImFont *iconicFont) const
    {
        for (auto const& rec : topologyRecommendations)
        {
            ImGui::Spacing();

            ImGui::PushFont(iconicFont);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 0.f, 1.f));
            ImGui::SeparatorText((char const*)(u8"\ue0D8" "## env map"));
            ImGui::PopStyleColor();
            ImGui::PopFont();

            ImGui::TextWrapped("%s", rec.c_str());

            ImGui::Spacing();
        }
    }


    void SurfaceTableStats::BuildUI(ImFont* iconicFont, ImPlotContext* plotContext, uint32_t imguiID) const
    {
        const float fontScale = ImGui::GetIO().FontGlobalScale;

        char buf[128];

        auto buildRow = [&buf]<typename T>(char const* name, T value, char const* tooltip = nullptr, bool displayMB = false)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", name);
            ImGui::TableSetColumnIndex(1);
            if constexpr (std::is_same_v<T, size_t>)
            {
                if (displayMB)
                    MegabytesFormatter(static_cast<double>(value), buf, (int) std::size(buf));
                else
                    MemoryFormatter(static_cast<double>(value), buf, (int) std::size(buf));
                ImGui::Text("%s", buf);
            }
            else if constexpr (std::is_same_v<T, float>)
                ImGui::Text("%.1f", value);
            else if constexpr (std::is_integral_v<T>)
                ImGui::Text("%d", int64_t(value));
            if (tooltip && ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tooltip);
        };

        if (ImGui::CollapsingHeader(name.empty() ? "Surface Table" : name.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
        {
            BuildRecommendationsUI(iconicFont);

            ImGui::Spacing();

            snprintf(buf, std::size(buf), "##Surface_Table_%d", imguiID);

            ImGui::BeginTable(buf, 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX);
            {
                static const float kColWidth0 = 200 * fontScale;
                static const float kColWidth1 = 80 * fontScale;

                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, kColWidth0);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, kColWidth1);
                ImGui::TableHeadersRow();

                buildRow("Memory use", byteSize,
                    "Total memory use for Tmr::SurfaceTable.\n"
                    "note: does not account for the texcoord surface table.\n");

                buildRow("Surfaces count", (int64_t)surfaceCount,
                    "Number of surfaces in the table.\n");

                buildRow("Pure regular surface count", bsplineSurfaceCount,
                            "Number of 'pure' regular surfaces in the table that can\n"
                            "be resolved with a single b-spline patch (ie. surfaces\n"
                            "with 16 control points, no boundaries and a subdivision\n"
                            "plan with no stencil matrix.\n");

                buildRow("Irregular face count", irregularFaceCount,
                    "Number of irregular faces in the table (ie. non-quads)\n"
                    "for Catmark subdivision scheme.\n");

                buildRow("Valence max", maxValence,
                    "Maximum vertex valence in the control cage.\n");

                buildRow("Face m_size max", maxFaceSize,
                    "Maximum number of vertices in a face in the control cage.\n");

                buildRow("Sharpness max", sharpnessMax,
                    "Highest sharpness value for edge or vertex in the control cage.\n");

                buildRow("Inf sharp", infSharpCreases,
                    "Number of edges or vertices with an 'infinitely' sharp crease tag.\n");

                buildRow("Stencil matrix row count avg", stencilCountAvg,
                    "Average number of patch-points per surface across the table.\n"
                    "Obtained by iterating over each surface in the table and\n"
                    "summing up the number of patch-points (aka rows in the stencil\n"
                    "matrix) in the subdivision plan associated with that surface.\n"
                    "The aveage is given by dividing this sum by the number of surfaces\n"
                    "in the table.\n"
                    "This average is a proxy measure of the global amount of computations\n"
                    "required to obtain the limit surface. Use of sharp edges or high\n"
                    "valence vertices will increase this average, while prevalance of\n"
                    "'regular' topology lowers this average.\n");
            }
            ImGui::EndTable();

            ImGui::SameLine();

            if (bsplineSurfaceCount > 0)
            {
                snprintf(buf, std::size(buf), "##Surface_Stencil_Pie_Chart_%d", imguiID);

                if (ImPlot::BeginPlot(buf, ImVec2(235 * fontScale, 153 * fontScale), ImPlotFlags_NoMouseText))
                {
                    float count = float(surfaceCount);
                    float bspline_ratio = float(bsplineSurfaceCount) / count;
                    float regular_ratio = float(regularSurfaceCount) / count;
                    float isolation_ratio = float(isolationSurfaceCount) / count;
                    float sharp_ratio = float(sharpSurfaceCount) / count;
                    float holes_ratio = float(holesCount) / count;

                    float values[5] = { bspline_ratio, regular_ratio, isolation_ratio, sharp_ratio, holes_ratio };

                    static char const* labels[std::size(values)] = {
                        "BSpline",
                        "Regular",
                        "Smooth",
                        "Sharp",
                        "Holes",
                    };

                    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
                    ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_NoDecorations);

                    ImPlot::SetupLegend(ImPlotLocation_West, ImPlotLegendFlags_Outside);

                    ImPlot::PlotPieChart(labels, values, (int) std::size(values), 0, 0, 0.4, "%.2f", ImPlotFlags_NoLegend);

                    ImPlot::EndPlot();
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Distribution of surfaces in the table:\n"
                            " - 'B-Spline' surfaces are 'pure' regular surfaces\n"
                            "   (16 control points in 1-ring, no boundaries, no\n"
                            "   patch-points, no stencil matrix).\n"
                            "   The limit of these surfaces can be evaluated\n"
                            "   through a dedicated fast-path.\n"
                            "\n"
                            " - 'Regular' surfaces are still b-spline surfaces, but\n"
                            "   with boundaries (9 or 12 control points in 1-ring,\n"
                            "   no patch-points, no stencil matrix).\n"
                            "\n"
                            " - 'Smooth' surfaces are areas that require feature\n"
                            "    isolation (a stencil matrix is required, but with a\n"
                            "    lower isolation level).\n"
                            "\n"
                            " - 'Sharp' surfaces are surfaces with semi-sharp\n"
                            "   creases (full feature isolation and stencil matrix).\n");
                }
            }
            ImGui::SameLine();

            snprintf(buf, std::size(buf), "##Surface_Stencil_Hisogram_%d", imguiID);

            if (ImPlot::BeginPlot(buf, ImVec2(-1, 153 * fontScale), ImPlotFlags_NoMouseText))
            {
                auto const& values = stencilCountHistogram;

                ImPlotFormatter formatter = [](double value, char* buff, int size, void* user_data) -> int
                    {
                        auto const* samplers = reinterpret_cast<EvaluatorSamplers const*>(user_data);
                        uint32_t    min = samplers->topologyMapStats.stencilCountMin;
                        uint32_t    max = samplers->topologyMapStats.stencilCountMax;
                        uint32_t    count = (uint32_t)samplers->topologyMapStats.stencilCountHistogram.size();
                        if (uint32_t range = max - min; range > 0 && count > 0)
                            value = min + (value / count) * range;
                        else
                            value = min;
                        return snprintf(buff, size_t(size), "%d", (int)value);
                    };

                ImPlot::SetupAxis(ImAxis_X1, "Num patch points", ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisFormat(ImAxis_X1, formatter, reinterpret_cast<void*>(&evaluatorSamplers));
                ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_AutoFit);
                ImPlot::PlotBars("Num Surfaces", values.data(), static_cast<int>(values.size()), 1.0, 0.5, ImPlotBarsFlags_None);

                ImPlot::EndPlot();
            }
        }

        ImGui::Spacing();
    }

}  // end namespace stats
