#pragma once
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

class SubdivisionSurface;

#include <donut/core/math/math.h>
#include <donut/engine/SceneGraph.h>

using namespace donut::math;

enum class GeometryType : uint8_t
{
    GEOMETRY_TYPE_UNKNOWN = 0,
    GEOMETRY_TYPE_STRUCTURED = (1 << 0),
    GEOMETRY_TYPE_UNSTRUCTURED = (1 << 1),
};

struct Instance
{
    std::shared_ptr<donut::engine::MeshInstance> meshInstance;
    affine3 localToWorld = affine3::identity();

    box3 aabb;

    float3 translation = { 0.f, 0.f, 0.f };
    quat rotation = { 1.f, 0.f, 0.f, 0.f };
    float3 scaling = { 1.f, 1.f, 1.f };

    float scale = 1.0f; // For LOD traversal, computed from localToWorld
    float radius = 0.f;
    float edgelength = 0.f;
    uint32_t meshID = ~uint32_t(0);

    GeometryType geometryType = GeometryType::GEOMETRY_TYPE_UNKNOWN;

    // Pointer to all clusters for this instance's resource
    //  const lod::Cluster *const *d_unstructuredClusters = nullptr;

    void Animate(float animTime, float animRate);

    void UpdateLocalTransform();

    void Lerp(Instance const& a, Instance const& b, float t);
};

struct Model
{
    int2 frameRange = { std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::min() };
    std::unique_ptr<SubdivisionSurface> subd;
    std::vector<Instance> instances;
};