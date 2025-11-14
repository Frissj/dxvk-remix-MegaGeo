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

#include <memory>

#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>

#include "rtxmg/utils/buffer.h"

#include "hiz_buffer_constants.h"

using namespace donut::math;

class HiZBuffer
{
public:
    ~HiZBuffer() = default;

    static std::unique_ptr<HiZBuffer> Create(uint2 size,
            std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
            std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
            nvrhi::ICommandList* commandList);

    nvrhi::ITexture* GetTextureObject(uint32_t lod) const { return textureObjects[lod]; }

    // resets the depth values across the hi-z mip levels to +inf
    // note: there should be no need to call this on a per-frame basis
    void Clear(nvrhi::ICommandList* commandList);

    // applies max reduction to the input zbuffer data to populate
    // the hi-z mip levels
    void Reduce(nvrhi::ITexture* zbuffer, nvrhi::ICommandList* commandList);

    // composites the hi-z mip levels over an arbitrary rgba texture
    // (starting from a small offset at the bottom left corner)
    void Display(nvrhi::ITexture* output, nvrhi::ICommandList* commandList);
    
    void GetDesc(nvrhi::BindingLayoutDesc* outBindingLayout, nvrhi::BindingSetDesc* outBindingSet, bool writeable = false) const;

    uint32_t GetNumLevels() const { return m_numLODs; }
    float2 GetInvSize() const { return m_invSize; }
private:
    uint2 m_size = { 0, 0 };
    float2 m_invSize = { 0.f, 0.f };
    uint32_t m_numLODs = 0;

    nvrhi::TextureHandle textureObjects[HIZ_MAX_LODS] = { 0 };

    nvrhi::ShaderHandle m_pass1Shader;
    nvrhi::ShaderHandle m_pass2Shader;
    nvrhi::ShaderHandle m_displayShader;

    nvrhi::BindingLayoutHandle m_passBL;
    nvrhi::ComputePipelineHandle m_pass1PSO;
    nvrhi::ComputePipelineHandle m_pass2PSO;

    nvrhi::BindingLayoutHandle m_displayBL;
    nvrhi::ComputePipelineHandle m_displayPSO;

    nvrhi::SamplerHandle m_sampler;

    nvrhi::BufferHandle m_reduceParamsBuffer;
    nvrhi::BufferHandle m_displayParamsBuffer;
};
