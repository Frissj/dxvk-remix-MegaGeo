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

#ifndef OSD_PORTS_TMR_TYPES_H
#define OSD_PORTS_TMR_TYPES_H

#ifdef __cplusplus
// Calm visual studio down about "missing types" when trying to highlight HLSL files
#include <cstdint>
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
#else


inline float quantize(float a, uint32_t nbits)
{
    nbits = 32 - nbits;
    int mask = (1u << (32 - nbits)) - 1;
    return asfloat(asint(a) & ~mask);
}

inline float3 quantize(float3 v, uint32_t nbits)
{
    return float3(quantize(v.x, nbits), quantize(v.y, nbits), quantize(v.z, nbits));
}

#define assert(x)

#endif

typedef int       Index;
typedef uint16_t  LocalIndex;

static const LocalIndex LOCAL_INDEX_INVALID = ~LocalIndex(0);
static const Index INDEX_INVALID = -1;

static const uint32_t kMaxIsolationLevel = 10;

inline uint32_t pack(uint32_t value, uint32_t width, uint32_t offset)
{
    return (uint32_t)((value & ((1U << width) - 1)) << offset);
}

inline uint32_t unpack(uint32_t value, uint32_t width, uint32_t offset)
{
    return (uint32_t)((value >> offset) & ((1U << width) - 1));
}


enum SchemeType
{
    SCHEME_BILINEAR = 0,
    SCHEME_CATMARK,
    SCHEME_LOOP
};

enum EndCapType
{
    ENDCAP_NONE = 0,             ///< no endcap
    ENDCAP_BILINEAR_BASIS,       ///< use bilinear quads (4 cp) as end-caps
    ENDCAP_BSPLINE_BASIS,        ///< use BSpline basis patches (16 cp) as end-caps
    ENDCAP_GREGORY_BASIS,        ///< use Gregory basis patches (20 cp) as end-caps
};

enum NodeType
{
    NODE_REGULAR = 0,
    NODE_RECURSIVE = 1,
    NODE_TERMINAL = 2,
    NODE_END = 3,
};

enum PatchDescriptorType
{
    NON_PATCH = 0,     ///< undefined

    POINTS,            ///< points (useful for cage drawing)
    LINES,             ///< lines  (useful for cage drawing)

    QUADS,             ///< 4-sided quadrilateral (bilinear)
    TRIANGLES,         ///< 3-sided triangle

    LOOP,              ///< regular triangular patch for the Loop scheme

    REGULAR,           ///< regular B-Spline patch for the Catmark scheme
    GREGORY,
    GREGORY_BOUNDARY,
    GREGORY_BASIS,
    GREGORY_TRIANGLE
};


#endif // OSD_PORTS_TMR_TYPES_H