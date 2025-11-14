/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef SHADER_DEBUG_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define SHADER_DEBUG_H

#define ENABLE_SHADER_DEBUG 0

#ifdef __cplusplus
#include <ostream>
#else
#endif

 // Debug pixel
struct ShaderDebugElement
{
    enum PayloadType : uint
    {
        PayloadType_None,
        PayloadType_Uint,
        PayloadType_Uint2,
        PayloadType_Uint3,
        PayloadType_Uint4,
        PayloadType_Int,
        PayloadType_Int2,
        PayloadType_Int3,
        PayloadType_Int4,
        PayloadType_Float,
        PayloadType_Float2,
        PayloadType_Float3,
        PayloadType_Float4
    };

    uint4 uintData;
    float4 floatData;
    uint payloadType;
    uint lineNumber;
    uint2 pad0;


#ifdef __cplusplus
    static bool OutputLambda(std::ostream& ss, const ShaderDebugElement& e)
    {
        if (e.payloadType == ShaderDebugElement::PayloadType_None)
            return false;

        ss << "[Line:" << std::dec << e.lineNumber << "] ";

        if (e.payloadType >= ShaderDebugElement::PayloadType_Float &&
            e.payloadType <= ShaderDebugElement::PayloadType_Float4)
        {
            uint32_t numVectorElements = (e.payloadType - uint32_t(ShaderDebugElement::PayloadType_Float)) + 1;

            ss << std::setprecision(12) << e.floatData.data()[0];
            for (uint32_t i = 1; i < numVectorElements; i++)
                ss << ", " << e.floatData.data()[i];
        }
        else if (e.payloadType >= ShaderDebugElement::PayloadType_Int &&
            e.payloadType <= ShaderDebugElement::PayloadType_Int4)
        {
            uint32_t numVectorElements = (e.payloadType - uint32_t(ShaderDebugElement::PayloadType_Int)) + 1;
            ss << std::dec << static_cast<int>(e.uintData.data()[0]) << std::hex << "(0x" << e.uintData.data()[0] << ")";
            for (uint32_t i = 1; i < numVectorElements; i++)
                ss << ", " << std::dec << static_cast<int>(e.uintData.data()[i]) << std::hex << "(0x" << e.uintData.data()[i] << ")";
        }
        else if (e.payloadType >= ShaderDebugElement::PayloadType_Uint &&
            e.payloadType <= ShaderDebugElement::PayloadType_Uint4)
        {
            uint32_t numVectorElements = (e.payloadType - uint32_t(ShaderDebugElement::PayloadType_Uint)) + 1;
            ss << std::dec << e.uintData.data()[0] << std::hex << "(0x" << e.uintData.data()[0] << ")";
            for (uint32_t i = 1; i < numVectorElements; i++)
                ss << ", " << std::dec << e.uintData.data()[i] << std::hex << "(0x" << e.uintData.data()[i] << ")";
        }

        return true;
    }
#endif
};

#ifndef __cplusplus
#if ENABLE_SHADER_DEBUG
struct ShaderDebugger
{
    RWStructuredBuffer<ShaderDebugElement> output;
    uint3 predicateID;
    uint3 currentID;

    uint AllocateSlot()
    {
        uint bufferSize, bufferStride;
        output.GetDimensions(bufferSize, bufferStride);
        uint maxSize = bufferSize - 1;

        uint result;
        InterlockedAdd(output[0].payloadType, 1, result);
        return (result % maxSize) + 1;
    }

    void _ShaderDebug(float4 value, uint lineNumber, uint payloadType, bool checkPredicate)
    {
        if (!checkPredicate || all(predicateID == currentID))
        {
            ShaderDebugElement element = (ShaderDebugElement)0;
            element.payloadType = payloadType;
            element.lineNumber = lineNumber;
            element.floatData = value;
            element.uintData = 0;
            output[AllocateSlot()] = element;
        }
    }
    void _ShaderDebug(uint4 value, uint lineNumber, uint payloadType, bool checkPredicate)
    {
        if (!checkPredicate || all(predicateID == currentID))
        {
            ShaderDebugElement element = (ShaderDebugElement)0;
            element.payloadType = payloadType;
            element.lineNumber = lineNumber;
            element.floatData = 0.f;
            element.uintData = value;
            output[AllocateSlot()] = element;
        }
    }

    void ShaderDebug(uint4 value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(value, lineNumber, ShaderDebugElement::PayloadType_Uint4, checkPredicate);
    }
    void ShaderDebug(uint3 value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(uint4(value, 0), lineNumber, ShaderDebugElement::PayloadType_Uint3, checkPredicate);
    }
    void ShaderDebug(uint2 value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(uint4(value, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Uint2, checkPredicate);
    }
    void ShaderDebug(uint value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(uint4(value, 0, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Uint, checkPredicate);
    }

    void ShaderDebug(int4 value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(value, lineNumber, ShaderDebugElement::PayloadType_Int4, checkPredicate);
    }
    void ShaderDebug(int3 value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(uint4(value, 0), lineNumber, ShaderDebugElement::PayloadType_Int3, checkPredicate);
    }
    void ShaderDebug(int2 value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(uint4(value, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Int2, checkPredicate);
    }
    void ShaderDebug(int value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(uint4(value, 0, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Int, checkPredicate);
    }

    void ShaderDebug(float4 value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(value, lineNumber, ShaderDebugElement::PayloadType_Float4, checkPredicate);
    }
    void ShaderDebug(float3 value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(float4(value, 0), lineNumber, ShaderDebugElement::PayloadType_Float3, checkPredicate);
    }
    void ShaderDebug(float2 value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(float4(value, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Float2, checkPredicate);
    }
    void ShaderDebug(float value, uint lineNumber, bool checkPredicate = true)
    {
        _ShaderDebug(float4(value, 0, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Float, checkPredicate);
    }
};

static ShaderDebugger g_ShaderDebugger;

static void InitShaderDebugger(RWStructuredBuffer<ShaderDebugElement> output, uint3 predicateID, uint3 currentID)
{
    g_ShaderDebugger.output = output;
    g_ShaderDebugger.predicateID = predicateID;
    g_ShaderDebugger.currentID = currentID;
}

static void InitShaderDebugger(RWStructuredBuffer<ShaderDebugElement> output, uint2 predicateID, uint2 currentID)
{
    InitShaderDebugger(output, uint3(predicateID, 0), uint3(currentID, 0));
}

static void InitShaderDebugger(RWStructuredBuffer<ShaderDebugElement> output, uint predicateID, uint currentID)
{
    InitShaderDebugger(output, uint3(predicateID, 0, 0), uint3(currentID, 0, 0));
}

#define SHADER_DEBUG(value) g_ShaderDebugger.ShaderDebug(value, __LINE__)
#define SHADER_DEBUG_FORCE(value) g_ShaderDebugger.ShaderDebug(value, __LINE__, false)
#define SHADER_DEBUG_INIT(outputBuffer, predicateID, currentID) InitShaderDebugger(outputBuffer, predicateID, currentID)

#else
#define SHADER_DEBUG(value) 
#define SHADER_DEBUG_FORCE(value)
#define SHADER_DEBUG_INIT(outputBuffer, predicateID, currentID)
#endif

#endif // __cplusplus

#endif /* SHADER_DEBUG_H */