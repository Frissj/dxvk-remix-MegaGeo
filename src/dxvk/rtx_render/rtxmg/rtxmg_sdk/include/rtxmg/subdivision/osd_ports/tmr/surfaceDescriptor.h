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

#ifndef OSD_PORTS_TMR_SURFACE_DESCRIPTOR_H
#define OSD_PORTS_TMR_SURFACE_DESCRIPTOR_H

#include "types.h"

enum class Domain : uint16_t
{
    Tri = 0,
    Quad,
    Quad_Subface,
};

///
///  \brief Linear Surface descriptor
///
/// Specialized descriptor for linearly interpolated surfaces. Linear surfaces
/// do not require subdivision plans or other external data and can be evaluated
/// directly from the SurfaceTable.
/// 
/// (held by SurfaceTable)
/// 
/// Encoding:
/// 
///  field0        | Bits | Content
///  --------------|:----:|---------------------------------------------------
///  face m_size     | 16   | number of control points in the face
///  subface index | 16   | index of the quad sub-face (or invalid index = 0xFF)
/// 

struct LinearSurfaceDescriptor
{
    void Set(unsigned int firstPoint, uint16_t faceSize, uint16_t quadSubface = ~uint16_t(0));

    void SetNoLimit() { field0 = 0; firstControlPoint = ~uint32_t(0); };
    bool HasLimit() { return GetFaceSize() != 0; }

    uint16_t GetFaceSize() { return uint16_t(unpack(field0, 16, 0)); }
    LocalIndex GetQuadSubfaceIndex() { return LocalIndex(unpack(field0, 16, 16)); }

    Index GetPatchPoint(int pointIndex);
    static Index GetPatchPoint(int pointInex, uint16_t faceSize, LocalIndex subfaceIndex);

    static Domain getDomain(uint16_t faceSize, LocalIndex subfaceIndex);
    Domain GetDomain() { return getDomain(GetFaceSize(), GetQuadSubfaceIndex()); }

    uint32_t field0;
    uint32_t firstControlPoint;
};

inline void LinearSurfaceDescriptor::Set(
    unsigned int firstPoint, uint16_t faceSize, LocalIndex quadSubface)
{
    field0 = pack(faceSize, 16, 0) |
        pack(quadSubface, 16, 16);
    firstControlPoint = firstPoint;
}

inline Index LinearSurfaceDescriptor::GetPatchPoint(int pointIndex, uint16_t faceSize, LocalIndex subfaceIndex)
{
    if (subfaceIndex == LOCAL_INDEX_INVALID)
    {
        assert(pointIndex < faceSize);
        return pointIndex;
    }
    else
    {
        assert(pointIndex < 4);
        // patch point indices layout (N = faceSize) :
        // [ N control points ] 
        // [ 1 face-point ] 
        // [ N edge-points ]
        int N = faceSize;
        switch (pointIndex)
        {
        case 0: return subfaceIndex;
        case 1: return N + 1 + subfaceIndex; // edge-point after
        case 2: return N; // central face-point
        case 3: return N + (subfaceIndex > 0 ? subfaceIndex : N);
        }
    }
    return INDEX_INVALID;
}

inline Index LinearSurfaceDescriptor::GetPatchPoint(int pointIndex)
{
    return GetPatchPoint(pointIndex, GetFaceSize(), GetQuadSubfaceIndex());
}

inline Domain LinearSurfaceDescriptor::getDomain(uint16_t faceSize, LocalIndex subfaceIndex)
{
    if (subfaceIndex == LOCAL_INDEX_INVALID)
    {
        if (faceSize == 4)
        {
            return Domain::Quad;
        }
        else
        {
            return Domain::Tri;
        }
    }
    return Domain::Quad_Subface;
}

///
///  \brief Surface descriptor
///
/// Aggregates pointers into multiple sets of data that need to be assembled in
/// order to evaluate the limit surface for the face of a mesh:
/// 
///   - the indices of the 1-ring set of control points around the face
///   - a pointer to the SubdivisionPlan with all the topological information
///     (composed of an index to a TopologyMap, and the index of the Plan itself
///     within that map)
///   - a subset of flags affecting the evaluation of the surface.
/// 
/// (held by SurfaceTable)
/// 
/// Encoding:
/// 
///  field0              | Bits | Content
///  --------------------|:----:|---------------------------------------------------
///  has limit           | 1    | limit surface cannot be evaluated if false (implies
///                      |      | other fields are expected to be set to 0)
///  param rotation      | 2    | parametric rotation of the subdivision plan
///  edges adjacency     | 4    | per-edge bits set: true if one or more surfaces
///                      |      | adjacent to that edge are irregular (the edge is a
///                      |      | T-junction) ; always false if the surface is irregular
///  topology map        | 5    | index of topology map (optional)
///  plan index          | 20   | index of the plan within the topology map selected
///  

struct SurfaceDescriptor
{
    static const uint32_t kMaxMapIndex = (1 << 5) - 1;
    static const uint32_t kMaxPlanIndex = (1 << 20) - 1;

    void SetNoLimit() { field0 = 0; firstControlPoint = ~uint32_t(0); };
    bool HasLimit() { return unpack(field0, 1, 0); }

    void Set(unsigned int firstPoint, unsigned int planIndex, uint16_t rotation, uint16_t adjacency, unsigned int mapIndex = 0);

    uint16_t GetParametricRotation() { return (uint16_t)unpack(field0, 2, 1); }

    uint16_t GetEdgeAdjacencyBits() { return (uint16_t)unpack(field0, 4, 3); }
    bool GetEdgeAdjacencyBit(uint16_t edgeIndex) { uint16_t edgebits = (uint16_t)unpack(field0, 4, 3); return (edgebits >> edgeIndex) & 0x1; }

    unsigned int GetTopologyMapIndex() { return unpack(field0, 5, 7); }
    unsigned int GetSubdivisionPlanIndex() { return unpack(field0, 20, 12); }

    uint32_t field0;
    uint32_t firstControlPoint;
};

inline void SurfaceDescriptor::Set(
    unsigned int firstPoint, unsigned int planIndex, uint16_t rotation, uint16_t adjacency, unsigned int mapIndex)
{
    assert(planIndex < kMaxPlanIndex && rotation < 4 && mapIndex < kMaxMapIndex);

    field0 = pack(true, 1, 0) |
        pack(rotation, 2, 1) |
        pack(adjacency, 4, 3) |
        pack(mapIndex, 5, 7) |
        pack(planIndex, 20, 12);

    firstControlPoint = firstPoint;
}

#endif /* OSD_PORTS_TMR_SURFACE_DESCRIPTOR_H */