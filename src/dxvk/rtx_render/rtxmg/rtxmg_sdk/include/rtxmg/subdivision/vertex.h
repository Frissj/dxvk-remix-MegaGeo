//
//   Copyright 2024 Nvidia
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#pragma once

#ifdef __cplusplus
#include <donut/core/math/math.h>
#include <cstdint>

using namespace donut::math;
#endif

static const uint32_t kPatchSize = 16;

struct LimitFrame
{
    float3 p;
    float3 deriv1;
    float3 deriv2;

    void Clear()
    {
        p = float3(0, 0, 0);
        deriv1 = float3(0, 0, 0);
        deriv2 = float3(0, 0, 0);
    }

    void AddWithWeight(float3 src,
        float weight, float d1Weight, float d2Weight)
    {
        p += weight * src;
        deriv1 += d1Weight * src;
        deriv2 += d2Weight * src;
    }
};

// Texture coordinate with partial derivs w.r.t the parametric U and V directions of the surface
struct TexCoordLimitFrame
{
    float2 uv;
    float2 deriv1;  // (dST/du)
    float2 deriv2;  // (dST/du)

    void Clear()
    {
        uv = float2(0,0);
        deriv1 = float2(0,0);
        deriv2 = float2(0,0);
    }

    void AddWithWeight(float2 src, float weight, float du_weight, float dv_weight)
    {
        uv += weight * src;
        deriv1 += du_weight * src;
        deriv2 += dv_weight * src;
    }
};