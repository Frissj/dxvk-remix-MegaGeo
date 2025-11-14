/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
*
* OpenSubdiv integration for RTX Mega Geometry.
* Builds and manages GPU subdivision surface data.
*/

#pragma once

#include "../../dxvk_device.h"
#include "../../util/util_math.h"
#include "rtxmg_math_types.h"
#include <vector>
#include <memory>

namespace dxvk {

class DxvkContext;

// GPU-side data structures for subdivision surface evaluation
struct SubdivisionSurfaceGPUData {
    // Control points (original game mesh vertices)
    std::vector<float3> controlPoints;
    uint32_t controlPointCount = 0;

    // Topology: control point indices for each triangle
    std::vector<uint32_t> controlPointIndices;

    // Stencil matrix for GPU interpolation
    // Format per triangle: [idx0, idx1, idx2, w0, w1, w2]
    // GPU uses: p = w0*controlPoints[idx0] + w1*controlPoints[idx1] + w2*controlPoints[idx2]
    std::vector<float> stencilMatrix;

    // Surface descriptors (one per patch/triangle)
    std::vector<uint32_t> surfaceDescriptors;

    // Subdivision plan (evaluation instructions for GPU)
    std::vector<uint32_t> subdivisionPlans;

    // Metadata
    uint32_t triangleCount = 0;
    uint32_t isolationLevel = 0;  // 0 = linear, 1-4 = subdivision levels

    bool isValid() const {
        return controlPointCount > 0 && triangleCount > 0;
    }
};

// GPU buffer handles for uploaded subdivision data
struct GPUBufferHandles {
    Rc<DxvkBuffer> controlPointsBuffer;
    Rc<DxvkBuffer> controlPointIndicesBuffer;
    Rc<DxvkBuffer> stencilMatrixBuffer;
    Rc<DxvkBuffer> surfaceDescriptorsBuffer;
    Rc<DxvkBuffer> subdivisionPlansBuffer;

    bool isValid() const {
        return controlPointsBuffer != nullptr &&
               stencilMatrixBuffer != nullptr;
    }
};

class RtxmgSubdivisionBuilder {
public:
    RtxmgSubdivisionBuilder();
    ~RtxmgSubdivisionBuilder();

    // Initialize subdivision builder
    bool initialize();

    // Cleanup
    void shutdown();

    // Build subdivision surface topology from game mesh
    // Uses linear (kBilinear) scheme to preserve exact game geometry
    // Precomputes all GPU-side data structures
    bool buildSubdivisionSurface(
        const std::vector<float3>& positions,
        const std::vector<uint32_t>& indices,
        const std::vector<float3>& normals,
        const std::vector<float2>& texcoords,
        uint32_t isolationLevel,
        SubdivisionSurfaceGPUData& outData);

    // Upload pre-computed surface data to GPU
    // Prepares GPU buffers for cluster filling shader
    bool uploadGPUBuffers(
        Rc<DxvkContext> ctx,
        SubdivisionSurfaceGPUData& surfaceData,
        GPUBufferHandles& outHandles);

private:
    bool m_initialized;
    std::vector<SubdivisionSurfaceGPUData> m_subdivisionSurfaces;

    // Internal helper methods

    // Pre-compute stencil matrix for linear interpolation
    // Stores weights and indices for fast GPU evaluation
    void populateStencilMatrix(
        const std::vector<float3>& positions,
        const std::vector<uint32_t>& indices,
        SubdivisionSurfaceGPUData& outData);

    // Create surface descriptors (one per patch)
    void createSurfaceDescriptors(
        SubdivisionSurfaceGPUData& outData);

    // Create subdivision plan for GPU evaluation
    // Encodes which vertices and weights to use per surface
    void createSubdivisionPlan(
        SubdivisionSurfaceGPUData& outData);
};

} // namespace dxvk
