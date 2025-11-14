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

#include <donut/core/math/math.h>

using namespace donut::math;

#include <array>
#include <string>

class Camera
{

public:
    bool HasChanged() const { return m_changed; }

    float3 GetDirection() const { return normalize(m_lookat - m_eye); }
    void SetDirection(const float3& dir)
    {
        m_lookat = m_eye + length(m_lookat - m_eye) * dir;
    }

    void Translate(float3 const& v);
    void Rotate(float yaw, float pitch, float roll);
    void Roll(float speed);

    void Dolly(float factor);
    void Pan(float2 speed);
    void Zoom(const float factor);

    void Frame(box3 const& aabb);

    // UVW forms an orthogonal, but not orthonormal basis!
    std::array<float3, 3> const& GetBasis();

    void Print() const;

    float3 GetEye() const { return m_eye; }
    float3 GetLookat() const { return m_lookat; }
    float3 GetUp() const { return m_up; }

    float GetFovY() const { return m_fovY; }
    float GetAspectRatio() const { return m_aspectRatio; }
    float GetZNear() const { return m_zNear; }
    float GetZFar() const { return m_zFar; }

    // These return column vectors but are stored in row_major memory wise
    // Translation is in m[3][j]  
    float4x4 GetViewMatrix() const;
    float4x4 GetProjectionMatrix() const;
    float4x4 GetViewProjectionMatrix() const;

    void SetEye(float3 eye);
    void SetLookat(float3 lookat);
    void SetUp(float3 up);

    void SetFovY(float fovy);
    void SetAspectRatio(float ar);
    void SetNear(float near);
    void SetFar(float far);

    void Set(std::string const& camc_string);

private:
    void ComputeBasis(float3& u, float3& v, float3& w) const;

    std::array<float3, 3> m_basis = {};

    float3 m_eye = float3(1.f);
    float3 m_lookat = float3(0.f);
    float3 m_up = float3(0.f, 1.f, 0.f);

    float m_fovY = 35.f;
    float m_aspectRatio = 1.f;
    float m_zNear = 0.1f;
    float m_zFar = 100.f;

    bool m_changed = true;
};