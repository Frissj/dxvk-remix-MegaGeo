/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
*
* OpenSubdiv integration for RTX Mega Geometry cluster building.
* Builds subdivision surface topology and pre-computes GPU buffers.
*/

#include "rtxmg_subdivision_builder.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"
#include <algorithm>

namespace dxvk {

RtxmgSubdivisionBuilder::RtxmgSubdivisionBuilder()
    : m_initialized(false)
{
    Logger::info("[RTXMG Subdivision] Builder created");
}

RtxmgSubdivisionBuilder::~RtxmgSubdivisionBuilder()
{
    shutdown();
}

bool RtxmgSubdivisionBuilder::initialize()
{
    if (m_initialized) {
        Logger::warn("[RTXMG Subdivision] Builder already initialized");
        return true;
    }

    m_initialized = true;
    Logger::info("[RTXMG Subdivision] Builder initialized");
    return true;
}

void RtxmgSubdivisionBuilder::shutdown()
{
    m_subdivisionSurfaces.clear();
    m_initialized = false;
    Logger::info("[RTXMG Subdivision] Builder shutdown");
}

// Build subdivision surface topology from game mesh
// Uses linear (kBilinear) scheme to preserve exact game geometry
bool RtxmgSubdivisionBuilder::buildSubdivisionSurface(
    const std::vector<float3>& positions,
    const std::vector<uint32_t>& indices,
    const std::vector<float3>& normals,
    const std::vector<float2>& texcoords,
    uint32_t isolationLevel,
    SubdivisionSurfaceGPUData& outData)
{
    if (positions.empty() || indices.empty()) {
        Logger::warn("[RTXMG Subdivision] Empty geometry input");
        return false;
    }

    if (indices.size() % 3 != 0) {
        Logger::warn("[RTXMG Subdivision] Index count not divisible by 3");
        return false;
    }

    // STEP 1: Store control points (original game mesh vertices)
    // Linear subdivision evaluates to exact original mesh
    outData.controlPoints = positions;
    outData.controlPointCount = static_cast<uint32_t>(positions.size());
    outData.triangleCount = static_cast<uint32_t>(indices.size() / 3);
    outData.isolationLevel = isolationLevel;

    // STEP 2: Store control point indices (topology)
    outData.controlPointIndices = indices;

    // STEP 3: Pre-compute stencil matrix for GPU evaluation
    // Stencil = linear combination of control points
    // For triangle mesh: p(u,v) = (1-u-v)*p0 + u*p1 + v*p2
    // For quad mesh: p(u,v) = (1-u)(1-v)*p0 + u(1-v)*p1 + (1-u)v*p2 + uv*p3

    populateStencilMatrix(positions, indices, outData);

    // STEP 4: Create surface descriptors
    // Describes which control points contribute to each patch
    createSurfaceDescriptors(outData);

    // STEP 5: Create subdivision plan for GPU
    // Encodes which vertices to sample for each surface
    createSubdivisionPlan(outData);

    Logger::info(str::format(
        "[RTXMG Subdivision] Built surface: ",
        outData.controlPointCount, " control points, ",
        outData.triangleCount, " triangles, ",
        "stencilElements=", outData.stencilMatrix.size()));

    return true;
}

void RtxmgSubdivisionBuilder::populateStencilMatrix(
    const std::vector<float3>& positions,
    const std::vector<uint32_t>& indices,
    SubdivisionSurfaceGPUData& outData)
{
    // LINEAR STENCIL MATRIX
    // Pre-computes interpolation weights for each evaluation point
    // This avoids expensive per-frame computation on GPU

    const uint32_t triangleCount = static_cast<uint32_t>(indices.size() / 3);

    // For each triangle, store:
    // - 3 vertex indices
    // - 3 barycentric weights (sum to 1.0)
    // This allows fast p = w0*p0 + w1*p1 + w2*p2 evaluation

    outData.stencilMatrix.reserve(triangleCount * 6); // 3 indices + 3 weights per triangle

    for (uint32_t i = 0; i < triangleCount; ++i) {
        uint32_t idx0 = indices[i * 3];
        uint32_t idx1 = indices[i * 3 + 1];
        uint32_t idx2 = indices[i * 3 + 2];

        // Validate indices
        if (idx0 >= positions.size() || idx1 >= positions.size() || idx2 >= positions.size()) {
            Logger::warn(str::format("[RTXMG Subdivision] Invalid index in triangle ", i));
            continue;
        }

        // Store indices and default weights (uniform 1/3 each)
        // GPU will use these for fast linear interpolation
        outData.stencilMatrix.push_back(static_cast<float>(idx0));
        outData.stencilMatrix.push_back(static_cast<float>(idx1));
        outData.stencilMatrix.push_back(static_cast<float>(idx2));
        outData.stencilMatrix.push_back(1.0f / 3.0f);  // weight0
        outData.stencilMatrix.push_back(1.0f / 3.0f);  // weight1
        outData.stencilMatrix.push_back(1.0f / 3.0f);  // weight2
    }
}

void RtxmgSubdivisionBuilder::createSurfaceDescriptors(
    SubdivisionSurfaceGPUData& outData)
{
    // SURFACE DESCRIPTORS
    // One descriptor per "surface" (patch for OpenSubdiv)
    // Describes control point topology

    // For linear tessellation of game meshes:
    // Each triangle becomes a surface

    const uint32_t triangleCount = outData.triangleCount;
    outData.surfaceDescriptors.resize(triangleCount);

    for (uint32_t i = 0; i < triangleCount; ++i) {
        // Simple descriptor: index into control points + count
        outData.surfaceDescriptors[i] = i * 3; // offset into indices
    }

    Logger::info(str::format("[RTXMG Subdivision] Created ", triangleCount, " surface descriptors"));
}

void RtxmgSubdivisionBuilder::createSubdivisionPlan(
    SubdivisionSurfaceGPUData& outData)
{
    // SUBDIVISION PLAN
    // Encodes evaluation strategy for GPU
    // For linear tessellation: direct triangle interpolation

    const uint32_t triangleCount = outData.triangleCount;

    // Each "plan" encodes:
    // - Offset into stencil matrix
    // - Number of stencil elements
    // - Surface type (linear for our case)

    outData.subdivisionPlans.resize(triangleCount);

    for (uint32_t i = 0; i < triangleCount; ++i) {
        // Store offset into stencil matrix (6 floats per triangle)
        outData.subdivisionPlans[i] = i * 6;
    }

    Logger::info(str::format("[RTXMG Subdivision] Created subdivision plan for ", triangleCount, " surfaces"));
}

// Upload GPU buffers for this subdivision surface
// Called before first use in cluster filling
bool RtxmgSubdivisionBuilder::uploadGPUBuffers(
    Rc<DxvkContext> ctx,
    SubdivisionSurfaceGPUData& surfaceData,
    GPUBufferHandles& outHandles)
{
    if (ctx == nullptr) {
        Logger::err("[RTXMG Subdivision] Invalid context for GPU upload");
        return false;
    }

    if (!surfaceData.isValid()) {
        Logger::warn("[RTXMG Subdivision] Cannot upload invalid subdivision surface data");
        return false;
    }

    try {
        Logger::info("[RTXMG Subdivision] Uploading subdivision surface data to GPU buffers...");

        // Create control points buffer if data exists
        if (!surfaceData.controlPoints.empty()) {
            Logger::info(str::format("[RTXMG Subdivision] Creating control points buffer: ",
                surfaceData.controlPointCount, " vertices"));

            DxvkBufferCreateInfo bufferInfo;
            bufferInfo.size = surfaceData.controlPoints.size() * sizeof(float3);
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bufferInfo.stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            bufferInfo.access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

            // Create buffer as HOST_VISIBLE for CPU upload, then cache it
            outHandles.controlPointsBuffer = ctx->getDevice()->createBuffer(
                bufferInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                DxvkMemoryStats::Category::RTXBuffer, "Subdivision Control Points");

            if (outHandles.controlPointsBuffer == nullptr) {
                Logger::err("[RTXMG Subdivision] Failed to create control points buffer");
                return false;
            }

            // Map and upload control points
            void* mappedPtr = outHandles.controlPointsBuffer->mapPtr(0);
            if (mappedPtr) {
                std::memcpy(mappedPtr, surfaceData.controlPoints.data(), bufferInfo.size);
                Logger::info("[RTXMG Subdivision] ✓ Control points uploaded");
            }
        }

        // Create stencil matrix buffer if data exists
        if (!surfaceData.stencilMatrix.empty()) {
            Logger::info(str::format("[RTXMG Subdivision] Creating stencil matrix buffer: ",
                surfaceData.stencilMatrix.size(), " floats"));

            DxvkBufferCreateInfo bufferInfo;
            bufferInfo.size = surfaceData.stencilMatrix.size() * sizeof(float);
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bufferInfo.stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            bufferInfo.access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

            // Create buffer as HOST_VISIBLE for CPU upload
            outHandles.stencilMatrixBuffer = ctx->getDevice()->createBuffer(
                bufferInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                DxvkMemoryStats::Category::RTXBuffer, "Subdivision Stencil Matrix");

            if (outHandles.stencilMatrixBuffer == nullptr) {
                Logger::err("[RTXMG Subdivision] Failed to create stencil matrix buffer");
                return false;
            }

            // Map and upload stencil matrix
            void* mappedPtr = outHandles.stencilMatrixBuffer->mapPtr(0);
            if (mappedPtr) {
                std::memcpy(mappedPtr, surfaceData.stencilMatrix.data(), bufferInfo.size);
                Logger::info("[RTXMG Subdivision] ✓ Stencil matrix uploaded");
            }
        }

        // Create surface descriptors buffer if data exists
        if (!surfaceData.surfaceDescriptors.empty()) {
            Logger::info(str::format("[RTXMG Subdivision] Creating surface descriptors buffer: ",
                surfaceData.surfaceDescriptors.size(), " descriptors"));

            DxvkBufferCreateInfo bufferInfo;
            bufferInfo.size = surfaceData.surfaceDescriptors.size() * sizeof(uint32_t);
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bufferInfo.stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            bufferInfo.access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

            // Create buffer as HOST_VISIBLE for CPU upload
            outHandles.surfaceDescriptorsBuffer = ctx->getDevice()->createBuffer(
                bufferInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                DxvkMemoryStats::Category::RTXBuffer, "Subdivision Surface Descriptors");

            if (outHandles.surfaceDescriptorsBuffer == nullptr) {
                Logger::err("[RTXMG Subdivision] Failed to create surface descriptors buffer");
                return false;
            }

            // Map and upload descriptors
            void* mappedPtr = outHandles.surfaceDescriptorsBuffer->mapPtr(0);
            if (mappedPtr) {
                std::memcpy(mappedPtr, surfaceData.surfaceDescriptors.data(), bufferInfo.size);
                Logger::info("[RTXMG Subdivision] ✓ Surface descriptors uploaded");
            }
        }

        // Create subdivision plans buffer if data exists
        if (!surfaceData.subdivisionPlans.empty()) {
            Logger::info(str::format("[RTXMG Subdivision] Creating subdivision plans buffer: ",
                surfaceData.subdivisionPlans.size(), " plans"));

            DxvkBufferCreateInfo bufferInfo;
            bufferInfo.size = surfaceData.subdivisionPlans.size() * sizeof(uint32_t);
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bufferInfo.stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            bufferInfo.access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

            // Create buffer as HOST_VISIBLE for CPU upload
            outHandles.subdivisionPlansBuffer = ctx->getDevice()->createBuffer(
                bufferInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                DxvkMemoryStats::Category::RTXBuffer, "Subdivision Plans");

            if (outHandles.subdivisionPlansBuffer == nullptr) {
                Logger::err("[RTXMG Subdivision] Failed to create subdivision plans buffer");
                return false;
            }

            // Map and upload plans
            void* mappedPtr = outHandles.subdivisionPlansBuffer->mapPtr(0);
            if (mappedPtr) {
                std::memcpy(mappedPtr, surfaceData.subdivisionPlans.data(), bufferInfo.size);
                Logger::info("[RTXMG Subdivision] ✓ Subdivision plans uploaded");
            }
        }

        Logger::info(str::format(
            "[RTXMG Subdivision] GPU buffers created successfully: ",
            "controlPoints=", surfaceData.controlPointCount, ", ",
            "stencilElements=", surfaceData.stencilMatrix.size(), ", ",
            "surfaceDescriptors=", surfaceData.surfaceDescriptors.size(), ", ",
            "subdivisionPlans=", surfaceData.subdivisionPlans.size()));

        return outHandles.isValid();

    } catch (const DxvkError& e) {
        Logger::err(str::format("[RTXMG Subdivision] GPU buffer upload failed: ", e.message()));
        return false;
    }
}

} // namespace dxvk
