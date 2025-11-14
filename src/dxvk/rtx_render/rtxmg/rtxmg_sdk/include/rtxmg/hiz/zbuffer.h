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

#include "rtxmg/hiz/hiz_buffer.h"

using namespace donut::math;

class ZBuffer
{
    nvrhi::TextureHandle m_currentTexture;
    RTXMGBuffer<float> m_minmaxBuffer;

    nvrhi::ShaderHandle m_minmaxShader;
    nvrhi::ShaderHandle m_displayShader;

    nvrhi::BindingLayoutHandle m_minmaxBL;
    nvrhi::ComputePipelineHandle m_minmaxPSO;

    nvrhi::BindingLayoutHandle m_displayBL;
    nvrhi::ComputePipelineHandle m_displayPSO;

    std::unique_ptr<HiZBuffer> m_hierarchy;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;

public:
    static std::unique_ptr<ZBuffer> Create(uint2 size, 
        std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
        std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
        nvrhi::ICommandList* commandList);

    nvrhi::TextureHandle GetCurrent() { return m_currentTexture; }
    const nvrhi::TextureHandle GetCurrent() const { return m_currentTexture; }

    void Display(nvrhi::ITexture *output, nvrhi::ICommandList* commandList);
    void ReduceHierarchy(nvrhi::ICommandList* commandList);
    void Clear(nvrhi::ICommandList* commandList);

    int GetNumHiZLODs() const
    {
        if (m_hierarchy) return m_hierarchy->GetNumLevels();
        return 0;
    }

    float2 GetInvHiZSize() const
    {
        if (m_hierarchy) return m_hierarchy->GetInvSize();
        return float2(0.f, 0.f);
    }
    nvrhi::ITexture* GetHierarchyTexture(uint32_t level) const { return m_hierarchy->GetTextureObject(level); }
    void GetHiZDesc(nvrhi::BindingLayoutDesc* outBindingLayout, nvrhi::BindingSetDesc* outBindingSet) const
    { 
        m_hierarchy->GetDesc(outBindingLayout, outBindingSet, false);
    }
};
