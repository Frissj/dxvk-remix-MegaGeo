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

/// \brief Patch parameterization
///
/// Topological refinement splits coarse mesh faces into refined faces.
///
/// This patch parameterzation describes the relationship between one
/// of these refined faces and its corresponding coarse face. It is used
/// both for refined faces that are represented as full limit surface
/// parametric patches as well as for refined faces represented as simple
/// triangles or quads. This parameterization is needed to interpolate
/// primvar data across a refined face.
///
/// The U,V and refinement level parameters describe the scale and offset
/// needed to map a location on the patch between levels of refinement.
/// The encoding of these values exploits the quad-tree organization of
/// the faces produced by subdivision. We encode the U,V origin of the
/// patch using two 10-bit integer values and the refinement level as
/// a 4-bit integer. This is sufficient to represent up through 10 levels
/// of refinement.
///
/// Special consideration must be given to the refined faces resulting from
/// irregular coarse faces. We adopt a convention similar to Ptex texture
/// mapping and define the parameterization for these faces in terms of the
/// regular faces resulting from the first topological splitting of the
/// irregular coarse face.
///
/// When computing the basis functions needed to evaluate the limit surface
/// parametric patch representing a refined face, we also need to know which
/// edges of the patch are interpolated boundaries. These edges are encoded
/// as a boundary bitmask identifying the boundary edges of the patch in
/// sequential order starting from the first vertex of the refined face.
///
/// A sparse topological refinement (like feature adaptive refinement) can
/// produce refined faces that are adjacent to faces at the next level of
/// subdivision. We identify these transitional edges with a transition
/// bitmask using the same encoding as the boundary bitmask.
///
/// For triangular subdivision schemes we specify the parameterization using
/// a similar method. Alternate triangles at a given level of refinement
/// are parameterized from their opposite corners and encoded as occupying
/// the opposite diagonal of the quad-tree hierarchy. The third barycentric
/// coordinate is dependent on and can be derived from the other two
/// coordinates. This encoding also takes inspiration from the Ptex
/// texture mapping specification.
///
/// Bitfield layout :
///
///  Field0     | Bits | Content
///  -----------|:----:|------------------------------------------------------
///  faceId     | 28   | the faceId of the patch
///  transition | 4    | transition edge mask encoding
///
///  Field1     | Bits | Content
///  -----------|:----:|------------------------------------------------------
///  level      | 4    | the subdivision level of the patch
///  nonquad    | 1    | whether patch is refined from a non-quad face
///  regular    | 1    | whether patch is regular
///  unused     | 1    | unused
///  boundary   | 5    | boundary edge mask encoding
///  v          | 10   | log2 value of u parameter at first patch corner
///  u          | 10   | log2 value of v parameter at first patch corner
///
/// Note : the bitfield is not expanded in the struct due to differences in how
///        GPU & CPU compilers pack bit-fields and endian-ness.
///
/*!
    \verbatim
    Quad Patch Parameterization

    (0,1)                           (1,1)
    +-------+-------+---------------+
    |       |       |               |
    |   L2  |   L2  |               |
    |0,3    |1,3    |               |
    +-------+-------+       L1      |
    |       |       |               |
    |   L2  |   L2  |               |
    |0,2    |1,2    |1,1            |
    +-------+-------+---------------+
    |               |               |
    |               |               |
    |               |               |
    |       L1      |       L1      |
    |               |               |
    |               |               |
    |0,0            |1,0            |
    +---------------+---------------+
    (0,0)                           (1,0)
    \endverbatim
*/
/*!
    \verbatim
    Triangle Patch Parameterization

    (0,1)                           (1,1)  (0,1,0)
    +-------+-------+---------------+       +
    | \     | \     | \             |       | \
    |L2 \   |L2 \   |   \           |       |   \
    |0,3  \ |1,3  \ |     \         |       | L2  \
    +-------+-------+       \       |       +-------+
    | \     | \     |   L1    \     |       | \  L2 | \
    |L2 \   |L2 \   |           \   |       |   \   |   \
    |0,2  \ |1,2  \ |1,1          \ |       | L2  \ | L2  \
    +-------+-------+---------------+       +-------+-------+
    | \             | \             |       | \             | \
    |   \           |   \           |       |   \           |   \
    |     \         |     \         |       |     \    L1   |     \
    |       \       |       \       |       |       \       |       \
    |   L1    \     |   L1    \     |       |   L1    \     |   L1    \
    |           \   |           \   |       |           \   |           \
    |0,0          \ |1,0          \ |       |             \ |             \
    +---------------+---------------+       +---------------+---------------+
    (0,0)                           (1,0)  (0,0,1)                         (1,0,0)
    \endverbatim
*/

#include "rtxmg/subdivision/osd_ports/tmr/types.h"

struct PatchParam
{
    /// \brief Sets the values of the bit fields
    ///
    /// @param faceid face index
    ///
    /// @param u value of the u parameter for the first corner of the face
    /// @param v value of the v parameter for the first corner of the face
    ///
    /// @param depth subdivision level of the patch
    /// @param nonquad true if the root face is not a quad
    ///
    /// @param boundary 5-bits identifying boundary edges (and verts for tris)
    /// @param transition 4-bits identifying transition edges
    ///
    /// @param regular whether the patch is regular
    ///
    void Set(Index faceid, uint16_t u, uint16_t v,
        uint16_t depth, bool nonquad,
        uint16_t boundary, uint16_t transition,
        bool regular = false);

    /// \brief Resets everything to 0
    void Clear() { field0 = field1 = 0; }

    /// \brief Returns the faceid
    Index GetFaceId() { return Index(unpack(field0, 28, 0)); }

    /// \brief Returns the log2 value of the u parameter at
    /// the first corner of the patch
    uint16_t GetU() { return (uint16_t)unpack(field1, 10, 22); }

    /// \brief Returns the log2 value of the v parameter at
    /// the first corner of the patch
    uint16_t GetV() { return (uint16_t)unpack(field1, 10, 12); }

    /// \brief Returns the transition edge encoding for the patch.
    uint16_t GetTransition() { return (uint16_t)unpack(field0, 4, 28); }

    /// \brief Returns the boundary edge encoding for the patch.
    uint16_t GetBoundary() { return (uint16_t)unpack(field1, 5, 7); }

    /// \brief True if the parent base face is a non-quad
    bool NonQuadRoot() { return (unpack(field1, 1, 4) != 0); }

    /// \brief Returns the level of subdivision of the patch
    uint16_t GetDepth() { return (uint16_t)unpack(field1, 4, 0); }

    /// \brief Returns the fraction of unit parametric space covered by this face.
    float GetParamFraction();

    /// \brief A (u,v) pair in the fraction of parametric space covered by this
    /// face is mapped into a normalized parametric space.
    ///
    /// @param u  u parameter
    /// @param v  v parameter
    ///
    void Normalize(inout float u, inout float v);
    void NormalizeTriangle(inout float u, inout float v);

    /// \brief A (u,v) pair in a normalized parametric space is mapped back into the
    /// fraction of parametric space covered by this face.
    ///
    /// @param u  u parameter
    /// @param v  v parameter
    ///
    void Unnormalize(inout float u, inout float v);
    void UnnormalizeTriangle(inout float u, inout float v);

    /// \brief Returns if a triangular patch is parametrically rotated 180 degrees
    bool IsTriangleRotated();

    /// \brief Returns whether the patch is regular
    bool IsRegular() { return (unpack(field1, 1, 5) != 0); }

    unsigned int field0 : 32;
    unsigned int field1 : 32;

    unsigned int pack(unsigned int value, int width, int offset)
    {
        return (unsigned int)((value & ((1U << width) - 1)) << offset);
    }

    unsigned int unpack(unsigned int value, int width, int offset)
    {
        return (unsigned int)((value >> offset) & ((1U << width) - 1));
    }
};

void PatchParam::Set(Index faceid, uint16_t u, uint16_t v,
    uint16_t depth, bool nonquad,
    uint16_t boundary, uint16_t transition,
    bool regular)
{
    field0 = pack(faceid, 28, 0) |
        pack(transition, 4, 28);

    field1 = pack(u, 10, 22) |
        pack(v, 10, 12) |
        pack(boundary, 5, 7) |
        pack(regular, 1, 5) |
        pack(nonquad, 1, 4) |
        pack(depth, 4, 0);
}

float PatchParam::GetParamFraction()
{
    return 1.0f / (float)(1U << (GetDepth() - NonQuadRoot()));
}

void PatchParam::Normalize(inout float u, inout float v)
{

    float fracInv = (float)(1.0f / GetParamFraction());

    u = u * fracInv - (float)GetU();
    v = v * fracInv - (float)GetV();
}

void PatchParam::Unnormalize(inout float u, inout float v)
{
    float frac = (float)GetParamFraction();

    u = (u + (float)GetU()) * frac;
    v = (v + (float)GetV()) * frac;
}

bool PatchParam::IsTriangleRotated()
{
    return (GetU() + GetV()) >= (1U << GetDepth());
}

void PatchParam::NormalizeTriangle(inout float u, inout float v)
{
    if (IsTriangleRotated())
    {
        float fracInv = (float)(1.0f / GetParamFraction());

        int depthFactor = 1U << GetDepth();
        u = (float)(depthFactor - GetU()) - (u * fracInv);
        v = (float)(depthFactor - GetV()) - (v * fracInv);
    }
    else
    {
        Normalize(u, v);
    }
}

void PatchParam::UnnormalizeTriangle(inout float u, inout float v)
{

    if (IsTriangleRotated())
    {
        float frac = GetParamFraction();

        int depthFactor = 1U << GetDepth();
        u = ((float)(depthFactor - GetU()) - u) * frac;
        v = ((float)(depthFactor - GetV()) - v) * frac;
    }
    else
    {
        Unnormalize(u, v);
    }
}
