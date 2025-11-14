//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <opensubdiv/version.h>

#include <cstdint>
#include <span>

#include <nvrhi/utils.h>
#include <donut/engine/DescriptorTableManager.h>
// clang-format on

namespace OpenSubdiv::OPENSUBDIV_VERSION::Tmr
{
    class TopologyMap;
}
namespace stats
{
    struct TopologyMapStats;
}

struct TopologyMap
{
    nvrhi::BufferHandle subpatchTreesArraysBuffer;
    nvrhi::BufferHandle patchPointIndicesArraysBuffer;
    nvrhi::BufferHandle stencilMatrixArraysBuffer;
    nvrhi::BufferHandle plansBuffer;

    donut::engine::DescriptorHandle subpatchTreesDescriptor;
    donut::engine::DescriptorHandle patchPointIndicesDescriptor;
    donut::engine::DescriptorHandle stencilMatrixDescriptor;
    donut::engine::DescriptorHandle plansDescriptor;

    std::unique_ptr<OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap>
        aTopologyMap;

    TopologyMap(const TopologyMap& other) = delete;
    TopologyMap(TopologyMap&& other) = delete;

    TopologyMap(std::unique_ptr<OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap>
        atopologyMap);

    void InitDeviceData(std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable, nvrhi::ICommandList* commandList, bool keepHostData = true);

    // statistics

    struct SubdivisionPlanStats
    {
        uint32_t plansCount = 0;
        size_t plansByteSize = 0;

        uint32_t regularFacePlansCount = 0; // plans created without quadrangulation

        uint32_t maxFaceSize = 0;
        uint32_t sharpnessCount = 0;
        float sharpnessMax = 0.f;

        uint32_t stencilCountMin = ~uint32_t(0);
        uint32_t stencilCountMax = 0;
        float stencilCountAvg = 0.f;
        std::vector<uint32_t> stencilCountHistogram;
    };

    static stats::TopologyMapStats ComputeStatistics(
        const OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap& topologyMap,
        int histogramSize = 50);
};
