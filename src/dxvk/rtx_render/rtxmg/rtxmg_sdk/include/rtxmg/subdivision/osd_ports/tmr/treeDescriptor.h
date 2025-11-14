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

#ifndef OSD_PORTS_TMR_TREE_DESCRIPTOR_H
#define OSD_PORTS_TMR_TREE_DESCRIPTOR_H

#include "rtxmg/subdivision/osd_ports/tmr/types.h"

struct TreeDescriptorHLSL
{
    StructuredBuffer<uint32_t> m_subpatchTrees;
    uint32_t m_treeOffset;

    static uint32_t const NumPatchPointsOffset = 2;

    bool IsRegularFace()
    {
        return unpack(m_subpatchTrees[m_treeOffset], 1, 0) != 0;
    }

    uint32_t GetFaceSize()
    {
        return unpack(m_subpatchTrees[m_treeOffset], 16, 16);
    }

    uint32_t GetSubfaceIndex()
    {
        return unpack(m_subpatchTrees[m_treeOffset], 16, 0);
    }

    uint32_t GetNumControlPoints()
    {
        return unpack(m_subpatchTrees[m_treeOffset], 16, 16);
    }

    uint32_t GetNumPatchPoints(uint16_t level)
    {
        return m_subpatchTrees[m_treeOffset + NumPatchPointsOffset + level];
    }
};

#endif // OSD_PORTS_TMR_TREE_DESCRIPTOR_H