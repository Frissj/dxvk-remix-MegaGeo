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

#include "rtxmg/subdivision/subdivision_surface.h"

#include "rtxmg/utils/buffer.h"
#include "rtxmg/subdivision/topology_cache.h"
#include "rtxmg/subdivision/topology_map.h"
#include "rtxmg/subdivision/far.h"
#include "rtxmg/subdivision/segmented_vector.h"
#include "rtxmg/profiler/statistics.h"

#include <opensubdiv/tmr/surfaceTableFactory.h>
#include <opensubdiv/tmr/subdivisionPlanBuilder.h>
#include <opensubdiv/tmr/subdivisionPlan.h>

#include <donut/engine/CommonRenderPasses.h>
#include <donut/core/log.h>

#include <algorithm>
#include <numeric>
#include <ranges>

// clang-format on

using namespace OpenSubdiv;
using namespace donut;

using TexcoordDeviceData = SubdivisionSurface::SurfaceTableDeviceData;

void initSubdLinearDeviceData(const Tmr::LinearSurfaceTable& surfaceTable,
    TexcoordDeviceData& deviceData, nvrhi::ICommandList* commandList)
{
    deviceData.surfaceDescriptors =
        CreateAndUploadBuffer<Tmr::LinearSurfaceDescriptor>(
            surfaceTable.descriptors, "texture coordinate surface descriptors", commandList);

    deviceData.controlPointIndices = CreateAndUploadBuffer<Vtr::Index>(
        surfaceTable.controlPointIndices, "texture coordinate control point indices", commandList);

    // Support (patch) points
    const uint32_t numSurfaces = surfaceTable.GetNumSurfaces();
    std::vector<uint32_t> patchPointsOffsets(numSurfaces + 1, 0);
    for (uint32_t i = 0; i < numSurfaces; i++)
    {
        Tmr::LinearSurfaceDescriptor desc = surfaceTable.GetDescriptor(i);
        if (!desc.HasLimit())
        {
            patchPointsOffsets[i + 1] = patchPointsOffsets[i];
            continue;
        }

        uint32_t numPatchPoints =
            (desc.GetQuadSubfaceIndex() == Tmr::LOCAL_INDEX_INVALID)
            ? 0
            : desc.GetFaceSize() + 1;
        patchPointsOffsets[i + 1] = patchPointsOffsets[i] + numPatchPoints;
    }

    deviceData.patchPointsOffsets = CreateAndUploadBuffer<uint32_t>(
        patchPointsOffsets, "texture coordinate patch points offsets", commandList);

    deviceData.patchPoints = CreateBuffer(patchPointsOffsets.back(), sizeof(float2),
        "texture coordinate patch points", commandList->getDevice());
}

static void gatherStatistics(Shape const& shape,
    Far::TopologyRefiner const& refiner,
    Tmr::TopologyMap const& topologyMap,
    Tmr::SurfaceTable const& surfTable,
    std::vector<uint16_t> &topologyQuality)
{
    int nsurfaces = surfTable.GetNumSurfaces();

    auto& evalStats = stats::evaluatorSamplers;

    evalStats.topologyMapStats = TopologyMap::ComputeStatistics(topologyMap);

    static constexpr int const histogramSize = 50;

    stats::SurfaceTableStats surfStats;

    surfStats.name = shape.filepath.filename().generic_string();

    surfStats.maxValence = refiner.getLevel(0).getMaxValence();

    surfStats.byteSize = surfTable.GetByteSize();

    topologyQuality.resize(nsurfaces, 0);
    
    size_t stencilSum = 0;

    for (int surfIndex = 0; surfIndex < nsurfaces; ++surfIndex)
    {
        Tmr::SurfaceDescriptor const& desc = surfTable.GetDescriptor(surfIndex);

        if (!desc.HasLimit())
        {
            ++surfStats.holesCount;
            continue;
        }

        auto const plan =
            topologyMap.GetSubdivisionPlan(desc.GetSubdivisionPlanIndex());

        uint16_t& quality = topologyQuality[surfIndex];

        // check face m_size (regular / non-quad)
        if (!plan->IsRegularFace())
            ++surfStats.irregularFaceCount;

        uint32_t faceSize = plan->GetFaceSize();

        if (faceSize > 5)
            quality = std::max(quality, (uint16_t)0xff);

        surfStats.maxFaceSize = std::max(faceSize, surfStats.maxFaceSize);

        // check vertex valences
        const Tmr::Index* controlPoints =
            surfTable.GetControlPointIndices(surfIndex);

        uint32_t maxVertexValence = 0;
        for (uint8_t i = 0; i < faceSize; ++i)
        {
            auto edges = refiner.GetLevel(0).GetVertexEdges(controlPoints[i]);
            maxVertexValence = std::max(maxVertexValence, uint32_t(edges.size()));
        }
        if (maxVertexValence > 8)
            quality = std::max(quality, (uint16_t)0xff);

        // check sharpness
        bool hasSharpness = false;

        if (plan->GetNumNeighborhoods())
        {
            Tmr::Neighborhood const& n = plan->GetNeighborhood(0);

            Tmr::ConstFloatArray corners = n.GetCornerSharpness();
            Tmr::ConstFloatArray creases = n.GetCreaseSharpness();

            if (hasSharpness = !(corners.empty() && creases.empty()))
            {

                auto processSharpness = [&surfStats,
                    &quality](Tmr::ConstFloatArray values)
                    {
                        for (int i = 0; i < values.size(); ++i)
                        {
                            if (values[i] >= 10.f)
                                ++surfStats.infSharpCreases;
                            else
                            {
                                surfStats.sharpnessMax =
                                    std::max(surfStats.sharpnessMax, values[i]);

                                if (values[i] > 8.f)
                                    quality = std::max(quality, (uint16_t)0xff);
                                else if (values[i] > 4.f)
                                    quality = std::max(
                                        quality,
                                        uint16_t((values[i] / Sdc::Crease::SHARPNESS_INFINITE) *
                                            255.f));
                            }
                        }
                    };
                processSharpness(creases);
                processSharpness(corners);
            }
        }

        // check stencil matrix
        size_t nstencils = plan->GetNumStencils();

        if (nstencils == 0)
        {
            if (plan->GetNumControlPoints() == 16)
                ++surfStats.bsplineSurfaceCount;
            else
                ++surfStats.regularSurfaceCount;
        }
        else
        {
            if (hasSharpness)
                ++surfStats.sharpSurfaceCount;
            else
                ++surfStats.isolationSurfaceCount;
        }

        stencilSum += nstencils;

        surfStats.stencilCountMin =
            std::min(surfStats.stencilCountMin, (uint32_t)nstencils);
        surfStats.stencilCountMax =
            std::max(surfStats.stencilCountMax, (uint32_t)nstencils);
    }

    assert((surfStats.holesCount + surfStats.bsplineSurfaceCount +
        surfStats.regularSurfaceCount + surfStats.isolationSurfaceCount +
        surfStats.sharpSurfaceCount) == nsurfaces);

    surfStats.stencilCountAvg = float(stencilSum) / float(nsurfaces);

    surfStats.stencilCountHistogram.resize(histogramSize);

    surfStats.surfaceCount = nsurfaces;

    if (!surfStats.IsCatmarkTopology())
    {
        // if we suspect this was not a sub-d model (likely a triangular mesh), run
        // a second pass of the surfaces to tag all the irregular faces (non-quads)
        // as poor quality
        int const regularFaceSize =
            Sdc::SchemeTypeTraits::GetRegularFaceSize(refiner.GetSchemeType());

        const Vtr::internal::Level& level = refiner.getLevel(0);
        for (int faceIndex = 0, surfaceIndex = 0; faceIndex < level.getNumFaces();
            ++faceIndex)
        {
            if (level.isFaceHole(faceIndex))
                continue;
            if (int nverts = level.getFaceVertices(faceIndex).size();
                nverts == regularFaceSize)
                ++surfaceIndex;
            else
            {
                for (int vert = 0; vert < nverts; ++vert, ++surfaceIndex)
                    topologyQuality[surfaceIndex] = 0xff;
            }
        }
    }

    // fill stencil counts histogram
    if (surfStats.stencilCountMin == surfStats.stencilCountMax)
    {
        // all the surfaces have the same number of stencils
        surfStats.stencilCountHistogram.push_back(nsurfaces);
    }
    else
    {
        surfStats.stencilCountHistogram.resize(histogramSize);

        float delta = float(surfStats.stencilCountMax - surfStats.stencilCountMin) /
            histogramSize;

        for (int surfIndex = 0; surfIndex < nsurfaces; ++surfIndex)
        {

            Tmr::SurfaceDescriptor const& desc = surfTable.GetDescriptor(surfIndex);

            if (!desc.HasLimit())
                continue;

            auto const plan =
                topologyMap.GetSubdivisionPlan(desc.GetSubdivisionPlanIndex());

            uint32_t nstencils = (uint32_t)plan->GetNumStencils();

            uint32_t i = (uint32_t)std::floor(
                float(nstencils - surfStats.stencilCountMin) / delta);

            ++surfStats
                .stencilCountHistogram[std::min(uint32_t(histogramSize - 1), i)];
        }
    }

    surfStats.BuildTopologyRecommendations();

    evalStats.surfaceTablesByteSizeTotal += surfStats.byteSize;
    evalStats.hasBadTopology |= (!surfStats.topologyRecommendations.empty());

    evalStats.surfaceTableStats.emplace_back(std::move(surfStats));
}

static std::vector<uint16_t> quadrangulateFaceToSubshape(
    Shape const& shape, uint32_t nsurfaces)
{
    assert(shape.scheme == Scheme::kCatmark);

    if (shape.nvertsPerFace.empty() || shape.faceToSubshapeIndex.empty() || !nsurfaces)
        return {};

    std::vector<uint16_t> result(nsurfaces);

    // Strong assumption here that this matches the quadrangulation to Create the surface descriptors
    for (uint32_t face = 0, vcount = 0; face < (uint32_t)shape.nvertsPerFace.size(); ++face)
    {
        int nverts = shape.nvertsPerFace[face];

        uint32_t subShapeIndex = shape.faceToSubshapeIndex[face];

        if (nverts == 4)
        {
            assert(vcount < result.size());
            result[vcount++] = static_cast<uint16_t>(subShapeIndex);
        }
        else
        {
            assert(vcount + nverts <= result.size());
            for (int vert = 0; vert < nverts; ++vert)
            {
                result[vcount + vert] = static_cast<uint16_t>(subShapeIndex);
            }
            vcount += nverts;
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// SubdivisionSurface
// -----------------------------------------------------------------------------
SubdivisionSurface::SubdivisionSurface(TopologyCache& topologyCache,
    std::unique_ptr<Shape> shape,
    const std::vector<std::unique_ptr<Shape>> &keyFrameShapes,
    std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable,
    nvrhi::ICommandList* commandList)
{
    m_shape = std::move(shape);

    // Create Far mesh (control cage topology)

    Sdc::SchemeType schemeType = GetSdcType(*m_shape);
    Sdc::Options schemeOptions = GetSdcOptions(*m_shape);
    Tmr::EndCapType endCaps = Tmr::EndCapType::ENDCAP_BSPLINE_BASIS;

    {
        // note: for now the topology cache only supports a single map
        // for a given set of traits ; eventually Tmr::SurfaceTableFactory
        // may support directly topology caches, allowing a given
        // Tmr::SurfaceTable to reference multiple topology maps at run-time.
        Tmr::TopologyMap::Traits traits;
        traits.SetCompatible(schemeType, schemeOptions, endCaps);

        m_topology_map = &topologyCache.get(traits.value);
    }

    Tmr::TopologyMap& topologyMap = *m_topology_map->aTopologyMap;

    std::unique_ptr<Far::TopologyRefiner> refiner;

    refiner.reset(Far::TopologyRefinerFactory<Shape>::Create(
        *m_shape,
        Far::TopologyRefinerFactory<Shape>::Options(schemeType, schemeOptions)));

    Tmr::SurfaceTableFactory tableFactory;

    Tmr::SurfaceTableFactory::Options options;
    options.planBuilderOptions.endCapType = endCaps;
    options.planBuilderOptions.isolationLevel = topologyCache.options.isoLevelSharp;
    options.planBuilderOptions.isolationLevelSecondary = topologyCache.options.isoLevelSmooth;
    options.planBuilderOptions.useSingleCreasePatch = true;
    options.planBuilderOptions.useInfSharpPatch = true;
    options.planBuilderOptions.useTerminalNode = topologyCache.options.useTerminalNodes;
    options.planBuilderOptions.useDynamicIsolation = true;
    options.planBuilderOptions.orderStencilMatrixByLevel = true;
    options.planBuilderOptions.generateLegacySharpCornerPatches = false;

    m_surface_table =
        tableFactory.Create(*refiner, topologyMap, options);

    std::vector<uint16_t> topologyQuality;
    gatherStatistics(*m_shape, *refiner, topologyMap, *m_surface_table, topologyQuality);

    m_topologyQualityBuffer = CreateAndUploadBuffer<uint16_t>(
        topologyQuality, "topology quality", commandList);

    m_topologyQualityDescriptor = descriptorTable->CreateDescriptorHandle(
        nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_topologyQualityBuffer));

    // setup for texcoords
    Tmr::LinearSurfaceTableFactory tableFactoryFvar;
    constexpr int const fvarChannel = 0;
    m_texcoord_surface_table =
        tableFactoryFvar.Create(*refiner, fvarChannel, m_surface_table.get());

    InitDeviceData(commandList);

    m_texcoordsBuffer =
        CreateAndUploadBuffer<float2>(m_shape->uvs, "base texcoords", commandList);

    m_positionsBuffer = CreateAndUploadBuffer<float3>(m_shape->verts, "SubdPosedPositions", commandList);
    m_aabb = m_shape->aabb;

    if (keyFrameShapes.size() > 0)
    {
        // Includes the 0th frame
        size_t nframes = keyFrameShapes.size() + 1;

        m_positionsPrevBuffer = CreateAndUploadBuffer<float3>(m_shape->verts, "SubdPosedPositions", commandList);

        m_positionKeyframeBuffers.resize(nframes);
        m_aabbKeyframes.resize(nframes);

        m_positionKeyframeBuffers[0] = CreateAndUploadBuffer<float3>(
            m_shape->verts, "SubdKeyFramePosition0", commandList);
        m_aabbKeyframes[0] = m_shape->aabb;

        // starts 1 indexed
        uint32_t frameIndex = 1;
        for (auto& keyFrameShape : keyFrameShapes)
        {
            char debugName[50];
            std::snprintf(debugName, std::size(debugName), "SubdKeyFramePosition%d",
                frameIndex);
            m_positionKeyframeBuffers[frameIndex] =
                CreateAndUploadBuffer<float3>(keyFrameShape->verts, debugName, commandList);
            m_aabbKeyframes[frameIndex] = keyFrameShape->aabb;

            frameIndex++;
        }
    }

    m_vertexSurfaceDescriptorDescriptor = descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_vertexDeviceData.surfaceDescriptors));
    m_vertexControlPointIndicesDescriptor = descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_vertexDeviceData.controlPointIndices));
    
    m_positionsDescriptor = descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_positionsBuffer));
    if (m_positionsPrevBuffer)
    {
        m_positionsPrevDescriptor = descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_positionsPrevBuffer));
    }

    m_surfaceToGeometryIndexDescriptor = descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_surfaceToGeometryIndexBuffer));
}

uint32_t SubdivisionSurface::NumVertices() const
{
    return static_cast<uint32_t>(m_positionsBuffer->getDesc().byteSize / sizeof(float3));
}

uint32_t SubdivisionSurface::SurfaceCount() const
{
    return m_surfaceCount;
}

void SubdivisionSurface::InitDeviceData(nvrhi::ICommandList* commandList)
{
    m_surfaceCount = uint32_t(m_surface_table->descriptors.size());

    // Sort surfaces by PureBspline, Bspline, Complex types for shader optimization
    std::vector<Tmr::SurfaceDescriptor> sortedDescriptors = m_surface_table->descriptors;
    std::vector<Tmr::LinearSurfaceDescriptor> sortedTexcoordDescriptors = m_texcoord_surface_table->descriptors;
    auto surfaceToGeometryIndex = quadrangulateFaceToSubshape(*m_shape, m_surfaceCount);

    assert(m_surfaceCount == sortedDescriptors.size());
    assert(m_surfaceCount == sortedTexcoordDescriptors.size());
    assert(m_surfaceCount == surfaceToGeometryIndex.size());

    auto zippedDescriptorsGeometryIndex = std::ranges::views::zip(sortedDescriptors, sortedTexcoordDescriptors, surfaceToGeometryIndex);

    std::ranges::sort(zippedDescriptorsGeometryIndex.begin(), zippedDescriptorsGeometryIndex.end(), [this](const auto& lhs, const auto& rhs)
        {
            // Extract surface descriptors 
            const auto& a = std::get<0>(lhs);
            const auto& b = std::get<0>(rhs);
            bool tieBreakerAB = a.firstControlPoint < b.firstControlPoint;

            // All holes last
            bool aHasLimit = a.HasLimit();
            bool bHasLimit = b.HasLimit();
            if (aHasLimit != bHasLimit)
                return aHasLimit;
            else if (!aHasLimit && !bHasLimit)
                return tieBreakerAB;
            
            // PureBspline
            bool aIsPureBSplinePatch = a.GetSubdivisionPlanIndex() == 0;
            bool bIsPureBSplinePatch = b.GetSubdivisionPlanIndex() == 0;
            if (aIsPureBSplinePatch != bIsPureBSplinePatch)
                return aIsPureBSplinePatch;
            else if (aIsPureBSplinePatch && bIsPureBSplinePatch)
                return tieBreakerAB;

            // BSpline
            const auto* aPlan = m_surface_table->topologyMap.GetSubdivisionPlan(a.GetSubdivisionPlanIndex());
            bool aIsBSplinePatch = aPlan->GetTreeDescriptor().GetNumPatchPoints(Tmr::kMaxIsolationLevel) == 0;

            const auto* bPlan = m_surface_table->topologyMap.GetSubdivisionPlan(b.GetSubdivisionPlanIndex());
            bool bIsBSplinePatch = bPlan->GetTreeDescriptor().GetNumPatchPoints(Tmr::kMaxIsolationLevel) == 0;

            if (aIsBSplinePatch != bIsBSplinePatch)
                return aIsBSplinePatch;
            
            // Complex, Limit Surface
            return tieBreakerAB;
        });

    // Array is sorted lets find the starting index for each stype
    int lastSurfaceType = -1;
    auto UpdateSurfaceOffset = [&lastSurfaceType, this](SurfaceType surfaceType, uint32_t startIndex)
        {
            for (int i = lastSurfaceType + 1; i <= int(surfaceType); i++)
            {
                m_surfaceOffsets[i] = startIndex;
            }
            lastSurfaceType = int(surfaceType);
        };

    for (uint32_t i = 0; i < m_surfaceCount; i++)
    {
        const auto& descriptor = sortedDescriptors[i];
        bool isPureBSplinePatch = descriptor.GetSubdivisionPlanIndex() == 0;
        if (isPureBSplinePatch)
        {
            UpdateSurfaceOffset(SurfaceType::PureBSpline, i);
        }
        else
        {
            const auto* plan = m_surface_table->topologyMap.GetSubdivisionPlan(descriptor.GetSubdivisionPlanIndex());
            bool isBSplinePatch = plan->GetTreeDescriptor().GetNumPatchPoints(Tmr::kMaxIsolationLevel) == 0;
            if (isBSplinePatch)
            {
                UpdateSurfaceOffset(SurfaceType::RegularBSpline, i);
            }
            else
            {
                if (descriptor.HasLimit())
                {
                    UpdateSurfaceOffset(SurfaceType::Limit, i);
                }
                else
                {
                    UpdateSurfaceOffset(SurfaceType::NoLimit, i);
                }
            }
        }
    }
    UpdateSurfaceOffset(SurfaceType::NoLimit, m_surfaceCount);

    std::vector<uint32_t> patchPointsOffsets(m_surfaceCount + 1, 0);
    for (uint32_t iSurface = 0; iSurface < m_surfaceCount; ++iSurface)
    {
        const Tmr::SurfaceDescriptor surface = sortedDescriptors[iSurface];
        if (!surface.HasLimit())
        {
            patchPointsOffsets[iSurface + 1] = patchPointsOffsets[iSurface];
            continue;
        }

        // plan is never going to be null here
        const auto* plan = m_surface_table->topologyMap.GetSubdivisionPlan(
            surface.GetSubdivisionPlanIndex());

        patchPointsOffsets[iSurface + 1] =
            patchPointsOffsets[iSurface] + static_cast<uint32_t>(plan->GetNumPatchPoints());
    }

    // Texcoord Patch Points
    std::vector<uint32_t> texcoordPatchPointsOffsets(m_surfaceCount + 1, 0);
    for (uint32_t i = 0; i < m_surfaceCount; i++)
    {
        Tmr::LinearSurfaceDescriptor desc = sortedTexcoordDescriptors[i];
        if (!desc.HasLimit())
        {
            texcoordPatchPointsOffsets[i + 1] = texcoordPatchPointsOffsets[i];
            continue;
        }

        uint32_t numPatchPoints =
            (desc.GetQuadSubfaceIndex() == Tmr::LOCAL_INDEX_INVALID)
            ? 0
            : desc.GetFaceSize() + 1;
        texcoordPatchPointsOffsets[i + 1] = texcoordPatchPointsOffsets[i] + numPatchPoints;
    }

    m_vertexDeviceData.surfaceDescriptors =
        CreateAndUploadBuffer<Tmr::SurfaceDescriptor>(
            sortedDescriptors, "surface descriptors", commandList);

    m_vertexDeviceData.controlPointIndices = CreateAndUploadBuffer<Vtr::Index>(
        m_surface_table->controlPointIndices, "control point indices", commandList);

    m_vertexDeviceData.patchPoints = CreateBuffer(patchPointsOffsets.back(), sizeof(float3), "patch points", commandList->getDevice());

    m_vertexDeviceData.patchPointsOffsets = CreateAndUploadBuffer<uint32_t>(
        patchPointsOffsets, "patch points offsets", commandList);

    m_surfaceToGeometryIndexBuffer = CreateAndUploadBuffer<uint16_t>(surfaceToGeometryIndex, "surfaceToGeometryIndex", commandList);

    m_texcoordDeviceData.surfaceDescriptors =
        CreateAndUploadBuffer<Tmr::LinearSurfaceDescriptor>(
            sortedTexcoordDescriptors, "texture coordinate surface descriptors", commandList);

    m_texcoordDeviceData.controlPointIndices = CreateAndUploadBuffer<Vtr::Index>(
        m_texcoord_surface_table->controlPointIndices, "texture coordinate control point indices", commandList);

    m_texcoordDeviceData.patchPointsOffsets = CreateAndUploadBuffer<uint32_t>(
        texcoordPatchPointsOffsets, "texture coordinate patch points offsets", commandList);

    m_texcoordDeviceData.patchPoints = CreateBuffer(texcoordPatchPointsOffsets.back(), sizeof(float2),
        "texture coordinate patch points", commandList->getDevice());
}

bool SubdivisionSurface::HasAnimation() const
{
    return !m_positionKeyframeBuffers.empty();
}

uint32_t SubdivisionSurface::NumKeyframes() const
{
    return (uint32_t)m_positionKeyframeBuffers.size();
}

static inline box3 lerpAabb(const box3& a, const box3& b, float t)
{
    box3 result;
    result.m_mins = lerp(a.m_mins, b.m_mins, t);
    result.m_maxs = lerp(a.m_maxs, b.m_maxs, t);
    return result;
}

void SubdivisionSurface::Animate(float animTime, float frameRate)
{
    if (!HasAnimation())
        return;

    uint32_t nframes = static_cast<uint32_t>(m_positionKeyframeBuffers.size());

    float frameTime = m_frameOffset + animTime * frameRate;
    float frame = std::truncf(frameTime);

    // animation implicitly loops if frameTime >= NumKeyframes
    m_f0 = static_cast<int>(frame) % nframes;
    m_f1 = (m_f0 + 1) % nframes;

    m_dt = frameTime - frame;

    m_aabb = lerpAabb(m_aabbKeyframes[m_f0], m_aabbKeyframes[m_f1], animTime);
}
