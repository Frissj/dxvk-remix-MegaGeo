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


#include "rtxmg/hiz/hiz_buffer.h"
#include "rtxmg/hiz/hiz_buffer_reduce_params.h"
#include "rtxmg/hiz/hiz_buffer_display_params.h"
#include "rtxmg/hiz/hiz_buffer_constants.h"

#include "rtxmg/utils/buffer.h"
#include "rtxmg/utils/debug.h"

#include "rtxmg/profiler/statistics.h"

#include <donut/engine/CommonRenderPasses.h>
#include <donut/core/log.h>

using namespace donut;

std::unique_ptr<HiZBuffer> HiZBuffer::Create(uint2 size,
    std::shared_ptr<engine::CommonRenderPasses> commonPasses,
    std::shared_ptr<engine::ShaderFactory> shaderFactory,
    nvrhi::ICommandList* commandList)
{
    auto device = commandList->getDevice();
    auto hiz = std::make_unique<HiZBuffer>();

    hiz->m_size = size;
    hiz->m_invSize = { 1.f / float(size.x), 1.f / float(size.y) };
    hiz->m_sampler = commonPasses->m_LinearClampSampler;

    nvrhi::TextureDesc desc;
    desc.isUAV = true;
    desc.keepInitialState = true;
    desc.format = nvrhi::Format::R32_FLOAT;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;

    uint2 mipSize = hiz->m_size;

    for (uint8_t level = 0; level < HIZ_MAX_LODS; ++level)
    {
        desc.width = mipSize.x;
        desc.height = mipSize.y;

        std::stringstream ss;
        ss << "HiZ Buffer Level " << (int)level;

        desc.debugName = ss.str();
        hiz->textureObjects[level] = device->createTexture(desc);

        hiz->m_numLODs = level + 1;

        mipSize = { mipSize.x >> 1, mipSize.y >> 1 };
        if (mipSize.x == 0 || mipSize.y == 0)
            break;
    }

    for (uint8_t level = hiz->m_numLODs; level < HIZ_MAX_LODS; ++level)
    {
        desc.width = 1;
        desc.height = 1;

        std::stringstream ss;
        ss << "UNUSED HiZ Buffer Level " << (int)level;

        desc.debugName = ss.str();
        hiz->textureObjects[level] = device->createTexture(desc);
    }

    nvrhi::ShaderDesc clearDesc(nvrhi::ShaderType::Compute);

    hiz->m_pass1Shader = shaderFactory->CreateShader("hiz/hiz_pass1.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);
    if (!hiz->m_pass1Shader)
    {
        log::fatal("Failed to create hiz pass 1 shader");
    }

    hiz->m_pass2Shader = shaderFactory->CreateShader("hiz/hiz_pass2.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);
    if (!hiz->m_pass2Shader)
    {
        log::fatal("Failed to create hiz pass 2 shader");
    }
    hiz->m_displayShader = shaderFactory->CreateShader("hiz/hiz_display.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);
    if (!hiz->m_displayShader)
    {
        log::fatal("Failed to create hiz display shader");
    }

    hiz->m_reduceParamsBuffer = CreateBuffer(1, sizeof(HiZReducePass1Params), "HiZReducePass1Params", device);

    hiz->m_displayParamsBuffer = device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(HiZDisplayParams), "HiZDisplayParams", engine::c_MaxRenderPassConstantBufferVersions));

    return hiz;
}

void HiZBuffer::Display(nvrhi::ITexture* output, nvrhi::ICommandList* commandList)
{
    nvrhi::utils::ScopedMarker marker(commandList, "HiZBuffer::display");
    static constexpr uint32_t spacing = 10u;

    uint2 offset{ spacing, spacing };

    auto device = commandList->getDevice();

    nvrhi::BindingLayoutDesc bindingLayoutDesc;
    nvrhi::BindingSetDesc bindingSetDesc;
    GetDesc(&bindingLayoutDesc, &bindingSetDesc, true);
    bindingLayoutDesc
        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
        .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0))
        .setVisibility(nvrhi::ShaderType::Compute);
    bindingSetDesc
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_displayParamsBuffer))
        .addItem(nvrhi::BindingSetItem::Texture_UAV(0, output));

    // need to write *something* to the constant buffer before we set up the compute state
    HiZDisplayParams params;
    params.level = 0;
    params.offsetX = offset.x;
    params.offsetY = offset.y;
    commandList->writeBuffer(m_displayParamsBuffer, &params, sizeof(params));

    if (!m_displayBL)
    {
        m_displayBL = device->createBindingLayout(bindingLayoutDesc);
        if (!m_displayBL)
        {
            log::fatal("Failed to create binding layout for hiz display");
        }
    }

    nvrhi::BindingSetHandle bindingSet = device->createBindingSet(bindingSetDesc, m_displayBL);
    if (!bindingSet)
    {
        log::fatal("Failed to create binding set for hiz display");
    }

    if (!m_displayPSO)
    {
        nvrhi::ComputePipelineDesc computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(m_displayShader)
            .addBindingLayout(m_displayBL);

        m_displayPSO = device->createComputePipeline(computePipelineDesc);
    }
    
    auto state = nvrhi::ComputeState()
        .setPipeline(m_displayPSO)
        .addBindingSet(bindingSet);

    commandList->setComputeState(state);

    for (uint8_t level = 0; level < HIZ_MAX_LODS; level++)
    {
        if (!textureObjects[level])
            continue;

        constexpr int const blocksize = 32;

        uint2 extent{ textureObjects[level]->getDesc().width,
            textureObjects[level]->getDesc().height };
        uint2 numBlocks{ extent.x / blocksize + 1,
            extent.y / blocksize + 1 };

        HiZDisplayParams params;
        params.level = level;
        params.offsetX = offset.x;
        params.offsetY = offset.y;
        commandList->writeBuffer(m_displayParamsBuffer, &params, sizeof(params));

        commandList->dispatch(numBlocks.x, numBlocks.y);

        offset.x += extent.x + spacing;
    }
}

void HiZBuffer::GetDesc(nvrhi::BindingLayoutDesc* outBindingLayout, nvrhi::BindingSetDesc* outBindingSet, bool writeable) const
{
    *outBindingLayout = nvrhi::BindingLayoutDesc();
    *outBindingSet = nvrhi::BindingSetDesc();

    if (writeable)
    {
        outBindingLayout->addItem(nvrhi::BindingLayoutItem::Texture_UAV(0).
            setSize(HIZ_MAX_LODS));
        for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
        {
            outBindingSet->addItem(nvrhi::BindingSetItem::Texture_UAV(0, textureObjects[i]).setArrayElement(i));
        }
    }
    else
    {
        outBindingLayout->addItem(nvrhi::BindingLayoutItem::Texture_SRV(0).
            setSize(HIZ_MAX_LODS));
        for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
        {
            outBindingSet->addItem(nvrhi::BindingSetItem::Texture_SRV(0, textureObjects[i]).setArrayElement(i));
        }
    }
}

void HiZBuffer::Reduce(nvrhi::ITexture* zbuffer, nvrhi::ICommandList* commandList)
{
    constexpr uint32_t const kGroupSizePass1 = HIZ_GROUP_SIZE * (1 << 3); // 128

    uint32_t zwidth, zheight;

    zwidth = zbuffer->getDesc().width;
    zheight = zbuffer->getDesc().height;

    if (zwidth < kGroupSizePass1 || zheight < kGroupSizePass1)
        return;

    //uint2 dispatchSize = { (uint32_t(zwidth) + kGroupSizePass1 - 1) / kGroupSizePass1,
    //                   (uint32_t(zheight) + kGroupSizePass1 - 1) / kGroupSizePass1, };

    uint2 dispatchSize = m_size;

    auto device = commandList->getDevice();

    HiZReducePass1Params params;
    params.zBufferInvSize = float2(1.f / zwidth, 1.f / zheight);
    commandList->writeBuffer(m_reduceParamsBuffer, &params, sizeof(params));

    nvrhi::BindingLayoutDesc bindingLayoutDesc;
    nvrhi::BindingSetDesc bindingSetDesc;
    GetDesc(&bindingLayoutDesc, &bindingSetDesc, true);
    bindingLayoutDesc
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
        .addItem(nvrhi::BindingLayoutItem::Sampler(0))
        .setVisibility(nvrhi::ShaderType::Compute);
    bindingSetDesc
        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, zbuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_reduceParamsBuffer))
        .addItem(nvrhi::BindingSetItem::Sampler(0, m_sampler));

    if (!m_passBL)
    {
        m_passBL = device->createBindingLayout(bindingLayoutDesc);
        if (!m_passBL)
        {
            log::fatal("Failed to create binding layout for hiz reduce");
        }
    }

    nvrhi::BindingSetHandle bindingSet = device->createBindingSet(bindingSetDesc, m_passBL);
    if (!bindingSet)
    {
        log::fatal("Failed to create binding set for hiz reduce");
    }
  
    if (!m_pass1PSO)
    {
        nvrhi::ComputePipelineDesc computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(m_pass1Shader)
            .addBindingLayout(m_passBL);

        m_pass1PSO = device->createComputePipeline(computePipelineDesc);

        computePipelineDesc.setComputeShader(m_pass2Shader);
        m_pass2PSO = device->createComputePipeline(computePipelineDesc);
    }


    auto state = nvrhi::ComputeState()
        .setPipeline(m_pass1PSO)
        .addBindingSet(bindingSet);

    commandList->setComputeState(state);

    stats::frameSamplers.hiZRenderTime.Start(commandList);
    
    {
        nvrhi::utils::ScopedMarker marker(commandList, "HiZBuffer::reduce pass 1");
        commandList->dispatch(dispatchSize.x, dispatchSize.y);
    }

    if (m_numLODs > 5)
    {
        nvrhi::utils::TextureUavBarrier(commandList, textureObjects[4]);

        // apply in-place 1x reduction in second pass.
        constexpr uint32_t const kGroupSizePass2 = HIZ_GROUP_SIZE;
        uint2 dispatchSizePass2 = {
            (uint32_t(m_size.x >> 4) + kGroupSizePass2 - 1) / kGroupSizePass2,
            (uint32_t(m_size.y >> 4) + kGroupSizePass2 - 1) / kGroupSizePass2,
        };

        // ok to leave all bindings the same, just change shader
        state.setPipeline(m_pass2PSO);
        commandList->setComputeState(state);
        nvrhi::utils::ScopedMarker marker(commandList, "HiZBuffer::reduce pass 2");
        commandList->dispatch(dispatchSizePass2.x, dispatchSizePass2.y);
    }

    stats::frameSamplers.hiZRenderTime.Stop();
}

void HiZBuffer::Clear(nvrhi::ICommandList* commandList)
{
    nvrhi::utils::ScopedMarker marker(commandList, "HiZBuffer::clear");
    for (uint8_t level = 0; level < HIZ_MAX_LODS; ++level)
    {
        if (textureObjects[level])
        {
            commandList->clearTextureFloat(textureObjects[level], nvrhi::AllSubresources, nvrhi::Color(std::numeric_limits<float>::max()));
        }
    }
}
