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

#pragma once

// clang-format off
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>

using namespace donut::math;

#include "rtxmg/utils/buffer.h"
#include "rtxmg/cluster_builder/tessellator_config.h"
// clang-format on

#include <memory>
#include <span>

struct ClusterAccels
{
    RTXMGBuffer<uint8_t> blasBuffer;
    RTXMGBuffer<uint8_t> clasBuffer;

    RTXMGBuffer<nvrhi::GpuVirtualAddress> clasPtrsBuffer;  // address of each CLAS header in clasBuffer
    RTXMGBuffer<nvrhi::GpuVirtualAddress> blasPtrsBuffer;  // handles in device memory
    RTXMGBuffer<uint32_t> blasSizesBuffer;

    // -------------------------------------------------------------------------
    // Cluster data buffer for shading information
    //
    RTXMGBuffer<ClusterShadingData> clusterShadingDataBuffer;

    // -------------------------------------------------------------------------
    // Vertex Position buffer that we stage into before creating CLASes
    //
    RTXMGBuffer<float3> clusterVertexPositionsBuffer;

    // -------------------------------------------------------------------------
    // Vertex Normal buffer (optional - only allocated when vertex normals are enabled)
    //
    RTXMGBuffer<float3> clusterVertexNormalsBuffer;
};

struct ClusterStatistics
{    
    struct BufferStatistics
    {
        uint32_t m_numClusters = 0;
        uint32_t m_numTriangles = 0;
        size_t m_blasScratchSize = 0;
        size_t m_blasSize = 0;
        size_t m_vertexBufferSize = 0;
        size_t m_vertexNormalsBufferSize = 0;
        size_t m_clasSize = 0;
        size_t m_clusterDataSize = 0;
    };

    BufferStatistics desired;
    BufferStatistics allocated;
};
