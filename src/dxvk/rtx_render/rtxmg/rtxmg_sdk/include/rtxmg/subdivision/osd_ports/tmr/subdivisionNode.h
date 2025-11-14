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

#ifndef OSD_PORTS_TMR_SUBDIVISION_NODE_H
#define OSD_PORTS_TMR_SUBDIVISION_NODE_H

#include "rtxmg/subdivision/osd_ports/tmr/types.h"
#include "rtxmg/subdivision/osd_ports/tmr/nodeDescriptor.h"

struct SubdivisionNode
{
    StructuredBuffer<uint32_t> m_subpatchTrees;
    StructuredBuffer<Index> m_patchPoints;
    int m_nodeOffset;
    int m_treeOffset;
    int m_patchPointsOffset; // global offset m_patchPoints

    static int maxIsolationLevel() { return 10; }

    // patch points
    static int catmarkRegularPatchSize() { return 16; };
    static int catmarkTerminalPatchSize() { return 25; };
    static int loopRegularPatchSize() { return 12; };

    // node sizes (in 'ints', not bytes)
    static int regularNodeSize(bool singleCrease) { return singleCrease ? 3 : 2; }
    static int endCapNodeSize() { return 2; }
    static int terminalNodeSize() { return 3; }
    static int recursiveNodeSize() { return 6; }

    static int getNumChildren(NodeType type)
    {
        switch (type)
        {
        case NodeType::NODE_TERMINAL: return 1;
        case NodeType::NODE_RECURSIVE: return 4;
        default: return 0;
        }
    }

    static int rootNodeOffset() { return 14; }

    // internal node offsets in tree array
    int descriptorOffset() { return m_nodeOffset; }
    int sharpnessOffset() { return m_nodeOffset + 2; }
    int patchPointsOffset() { return m_nodeOffset + 1; }
    int childOffset(int childIndex) { return m_nodeOffset + 2 + childIndex; }

    float GetSharpness()
    {
        return asfloat(m_subpatchTrees[m_treeOffset + sharpnessOffset()]);
    }

    SubdivisionNode GetChild(int childIndex)
    {
        SubdivisionNode child;
        child.m_subpatchTrees = m_subpatchTrees;
        child.m_patchPoints = m_patchPoints;
        child.m_nodeOffset = m_subpatchTrees[m_treeOffset + childOffset(childIndex)];
        child.m_treeOffset = m_treeOffset;
        child.m_patchPointsOffset = m_patchPointsOffset;
        return child;
    }

    NodeDescriptor GetDesc()
    {
        return MakeNodeDescriptor(m_subpatchTrees[m_treeOffset + descriptorOffset()]);
    }

    int GetPatchPointBase()
    {
        return m_subpatchTrees[m_treeOffset + patchPointsOffset()];
    }

    Index GetPatchPoint(
        int pointIndex,
        int quadrant,
        uint16_t maxLevel)
    {
        int offset = GetPatchPointBase();
        if (offset == INDEX_INVALID)
        {
            return INDEX_INVALID;
        }

        NodeDescriptor desc = GetDesc();
        switch (desc.GetType())
        {
        case NODE_REGULAR:
        case NODE_END:
            offset += pointIndex;
            break;
        case NODE_RECURSIVE:
            offset = (desc.GetDepth() >= maxLevel) && desc.HasEndcap() ? offset + pointIndex : INDEX_INVALID;
            break;
        case NODE_TERMINAL:
            // Unsupported, uses quadrant
            break;
        default:
            break;
        }
        return m_patchPoints[m_patchPointsOffset + offset];
    }
};

#endif // OSD_PORTS_TMR_SUBDIVISION_NODE_H