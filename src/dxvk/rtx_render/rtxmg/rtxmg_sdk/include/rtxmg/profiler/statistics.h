//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#include "rtxmg/profiler/profiler.h"

#include <nvrhi/utils.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/core/math/math.h>

#include <string>
#include <map>
#include <mutex>

#include <implot.h>

class UserInterface;

namespace stats
{
    using CPUTimer = Profiler::CPUTimer;
    using GPUTimer = Profiler::GPUTimer;
    //
    // SubD stats
    //
    struct TopologyMapStats
    {
        // hashmap
        float  pslMean = 0.f;
        size_t hashCount = 0;
        size_t addressCount = 0;
        float  loadFactor = 0.f;

        // plans
        uint32_t plansCount = 0;
        size_t   plansByteSize = 0;

        uint32_t regularFacePlansCount = 0;

        uint32_t maxFaceSize = 0;
        uint32_t sharpnessCount = 0;
        float    sharpnessMax;

        // patch points
        uint32_t stencilCountMin = 0;
        uint32_t stencilCountMax = 0;
        float stencilCountAvg = 0;
        std::vector<uint32_t> stencilCountHistogram;
    };

    struct SurfaceTableStats
    {
        std::string name;

        size_t indexBufferSize = 0;
        size_t vertCountBufferSize = 0;

        size_t byteSize = 0;
        size_t surfaceCount = 0;

        uint32_t irregularFaceCount = 0;
        uint32_t maxValence = 0;
        uint32_t maxFaceSize = 0;
        // | boundaries | stencils  | creases |
        uint32_t holesCount = 0;            // |            |           |         |
        uint32_t bsplineSurfaceCount = 0;   // |            |           |         |
        uint32_t regularSurfaceCount = 0;   // |     x      |           |         |
        uint32_t isolationSurfaceCount = 0; // |     X      |     X     |         |
        uint32_t sharpSurfaceCount = 0;     // |     X      |     X     |    X    |

        float sharpnessMax = 0.f;
        uint32_t infSharpCreases = 0;

        uint32_t stencilCountMin = ~uint32_t(0);
        uint32_t stencilCountMax = 0;
        float stencilCountAvg = 0;
        std::vector<uint32_t> stencilCountHistogram;

        std::vector<std::string> topologyRecommendations;

        bool IsCatmarkTopology(float* ratio = nullptr) const
        {
            // guess if the user passed a triangles mesh (ie. not a subd model)
            float _ratio = float(irregularFaceCount) / float(surfaceCount);
            if (ratio)
                *ratio = _ratio;
            return _ratio < .25f;
        }
        void BuildTopologyRecommendations();

        void BuildRecommendationsUI(ImFont *iconicFont) const;

        void BuildUI(ImFont *iconicFont, ImPlotContext *plotContext, uint32_t imguiID) const;
    };

    //
    // General stats
    //

    struct FrameSamplers
    {
        std::string name = "Frame";

        Sampler<float, Profiler::BENCH_FRAME_COUNT> cpuFrameTime = { .name = "CPU/frame (ms)" };

        GPUTimer& gpuFrameTime = Profiler::InitTimer<GPUTimer>("GPU/frame (ms)");
        GPUTimer& gpuRenderTime = Profiler::InitTimer<GPUTimer>("GPU/trace (ms)");
        GPUTimer& gpuDenoiserTime = Profiler::InitTimer<GPUTimer>("GPU/denoiser (ms)");
        
        GPUTimer& hiZRenderTime = Profiler::InitTimer<GPUTimer>("GPU/hi-z (ms)");
        GPUTimer& zReprojectionTime = Profiler::InitTimer<GPUTimer>("GPU/zReprojection (ms)");
        GPUTimer& zRenderPassTime = Profiler::InitTimer<GPUTimer>("GPU/zRenderPass (ms)");
        GPUTimer& computeMotionVectorsTimer = Profiler::InitTimer<GPUTimer>("GPU/motion vectors (ms)");
        void BuildUI(ImFont *iconicFont, ImPlotContext *plotContext) const;
    };
    extern FrameSamplers frameSamplers;


    struct ClusterAccelSamplers
    {
        std::string name = "AccelBuilder";

        GPUTimer& clusterTilingTime = Profiler::InitTimer<GPUTimer>("GPU/Cluster Tiling (ms)");
        GPUTimer& fillClustersTime = Profiler::InitTimer<GPUTimer>("GPU/Fill Clusters (ms)");
        GPUTimer& buildClasTime = Profiler::InitTimer<GPUTimer>("GPU/CLAS build (ms)");
        GPUTimer& buildBlasTime = Profiler::InitTimer<GPUTimer>("GPU/BLAS build (ms)");

        Sampler<uint32_t> numClusters = { .name = "Clusters count", };
        Sampler<uint32_t> numTriangles = { .name = "Triangles count", };
        
        donut::math::int2 renderSize = {};

        void BuildUI(ImFont* iconicFont, ImPlotContext* plotContext) const;
    };
    extern ClusterAccelSamplers clusterAccelSamplers;

    struct EvaluatorSamplers
    {
        std::string name = "Subdivision Evaluator";

        TopologyMapStats topologyMapStats;

        bool hasBadTopology = false;
        bool m_topologyQualityButtonPressed = false;
        size_t surfaceTablesByteSizeTotal = 0;

        std::vector<SurfaceTableStats> surfaceTableStats;


        // run-time evaluation

        Sampler<uint32_t> numLimitSamples = { .name = "Limit evaluations", };

        void BuildUI(ImFont* iconicFont, ImPlotContext* plotContext);
    };
    extern EvaluatorSamplers evaluatorSamplers;

    struct MemUsageSamplers
    {
        std::string name = "Memory";

        Sampler<size_t> blasSize = { .name = "BLAS size", };
        Sampler<size_t> blasScratchSize = { .name = "BLAS Scratch", };
        Sampler<size_t> clasSize = { .name = "CLAS size", };
        Sampler<size_t> vertexBufferSize = { .name = "Vertex Buffer", };
        Sampler<size_t> vertexNormalsBufferSize = { .name = "Vertex Normals Buffer", };
        Sampler<size_t> clusterShadingDataSize = { .name = "Cluster Data Buffer", };

        void BuildUI(ImFont* iconicFont, ImPlotContext* plotContext) const;
    };
    extern MemUsageSamplers memUsageSamplers;

}  // end namespace stats
