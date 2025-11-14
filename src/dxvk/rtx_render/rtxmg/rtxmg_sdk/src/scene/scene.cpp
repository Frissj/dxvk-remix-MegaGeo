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

 //#include "args.h"

#include <donut/core/log.h>
#include <donut/engine/Scene.h>
#include <donut/engine/TextureCache.h>
#include <json/json.h>
#include <nvrhi/common/misc.h>
#include <execution>
#include <filesystem>
#include <numeric>
#include <numbers>
#include <ranges>
#include <set>

#include "rtxmg/cluster_builder/cluster_accels.h"
#include "rtxmg/profiler/statistics.h"
#include "rtxmg/scene/box_extent.h"
#include "rtxmg/scene/scene.h"
#include "rtxmg/scene/json.h"
#include "rtxmg/subdivision/shape.h"
#include "rtxmg/subdivision/subdivision_surface.h"
#include "rtxmg/subdivision/topology_cache.h"
#include "rtxmg/utils/buffer.h"

using namespace donut;

namespace fs = std::filesystem;

RTXMGScene::RTXMGScene(nvrhi::IDevice* device,
    const fs::path& mediapath,
    std::shared_ptr<CommonRenderPasses> commonPasses,
    ShaderFactory& shaderFactory,
    std::shared_ptr<vfs::IFileSystem> fs,
    std::shared_ptr<TextureCache> textureCache,
    std::shared_ptr<DescriptorTableManager> descriptorTable,
    std::shared_ptr<SceneTypeFactory> sceneTypeFactory,
    int2 initialFrameRange, int isoLevelSharp, int isoLevelSmooth)
    : Scene(device, shaderFactory, fs, textureCache,
        descriptorTable, sceneTypeFactory),
    m_commonPasses(commonPasses),
    m_mediaPath(mediapath),
    m_isoLevelSharp(isoLevelSharp),
    m_isoLevelSmooth(isoLevelSmooth)
{
    m_attributes.frameRange = initialFrameRange;
    m_UseResourceDescriptorHeapBindless = true;
}

void RTXMGScene::InsertModel(Model&& model)
{
    if (model.subd)
    {
        m_attributes.frameRange.x =
            std::min(m_attributes.frameRange.x, model.frameRange.x);
        m_attributes.frameRange.y =
            std::max(m_attributes.frameRange.y, model.frameRange.y);

        for (auto& instance : model.instances)
            instance.meshID = uint32_t(m_subdMeshes.size());

        m_subdMeshes.emplace_back(std::move(model.subd));

        std::ranges::move(model.instances, std::back_inserter(m_instances));
    }
}

std::span<Instance const> RTXMGScene::GetSubdMeshInstances() const
{
    return const_cast<RTXMGScene*>(this)->GetSubdMeshInstances();
}
std::span<Instance> RTXMGScene::GetSubdMeshInstances()
{
    if (!m_subdMeshes.empty())
        return std::span<Instance>(m_instances).subspan(0);
    return {};
}

uint32_t RTXMGScene::TotalSubdPatchCount() const
{
    const auto& instances = GetSubdMeshInstances();
    const auto& subds = GetSubdMeshes();

    uint32_t sum{ 0 };
    for (auto i = instances.begin(); i != instances.end(); ++i)
        sum += subds[i->meshID]->SurfaceCount();
    return sum;
}

void RTXMGScene::Animate(float animTime, float animRate)
{
    for (auto& subd : m_subdMeshes)
    {
        subd->Animate(animTime, animRate);
    }

    for (auto& instance : m_instances)
    {
        instance.Animate(animTime, animRate);
    }
}

static Instance& operator << (Instance& instance, const Json::Value& node)
{
    if (const auto& value = node["translation"]; !value.isNull())
        value >> instance.translation;

    if (const auto& value = node["rotation"]; !value.isNull())
    {
        if (node.isArray() && node.size() == 4)
            throw std::runtime_error("expecting 4-component quaternion for node's 'rotation' (use 'euler' otherwise)");
        value >> instance.rotation;
    }
    else if (const auto& value = node["euler"]; !value.isNull())
    {
        float3 euler = { 0.0, 0.0, 0.0 };
        value >> euler;
        euler *= float(std::numbers::pi) / 180.f;
        instance.rotation = donut::math::rotationQuat<float>(euler);
    }

    if (const auto& value = node["scaling"]; !value.isNull())
        value >> instance.scaling;

    instance.UpdateLocalTransform();

    return instance;
}


//
// View
//

static View& operator << (View& view, const Json::Value& node)
{
    if (const auto& value = node["position"]; !value.isNull())
        value >> view.position;
    if (const auto& value = node["lookat"]; !value.isNull())
        value >> view.lookat;
    if (const auto& value = node["up"]; !value.isNull())
        value >> view.up;
    if (const auto& value = node["fov"]; !value.isNull())
        value >> view.fov;

    return view;
}


static RTXMGScene::Attributes& operator << (RTXMGScene::Attributes& attrs, const Json::Value& node)
{
    if (const auto& value = node["audio"]; value.isString())
        value >> attrs.audio;

    if (const auto& value = node["audio start time"]; value.isDouble())
        value >> attrs.audioStartTime;

    if (const auto& value = node["frame range"]; value.isArray())
        value >> attrs.frameRange;

    if (const auto& value = node["frame rate"]; value.isDouble())
        value >> attrs.frameRate;

    return attrs;
}

fs::path RTXMGScene::ResolveMediapath(const fs::path& m_filepath, const fs::path& mediapath)
{
    if (m_filepath.empty())
        return {};

    if (fs::is_regular_file(m_filepath))
        return m_filepath;

    if (!mediapath.empty() && fs::is_regular_file(mediapath / m_filepath))
        return mediapath / m_filepath;

    return {};
}

void RTXMGScene::LoadSceneFile(const std::filesystem::path& m_filepath, std::unique_ptr<ObjImporter>& objImporter, nvrhi::ICommandList* commandList)
{
    fs::path fp = m_filepath;

    Json::Value jsonRoot;

    try
    {
        jsonRoot = readFile(fp);
    }
    catch (const std::exception& e)
    {
        log::fatal("failed to Parse JSON file '%s': %s", fp.generic_string().c_str(), e.what());
    }
    if (jsonRoot.isObject())
    {
        objImporter->SetModelPath(fp.parent_path());

        const Json::Value& models = jsonRoot["models"];
        const Json::Value& graph = jsonRoot["graph"];

        if (!models.isArray() || !graph.isArray())
            throw std::runtime_error("need valid 'models' and 'graph' arrays in '" + fp.generic_string() + "'");

        uint32_t nmodels = models.size();

        for (uint32_t i = 0; i < graph.size(); ++i)
        {
            const Json::Value& node = graph[i];

            Instance instance;

            instance << node;

            std::string nodeName;
            if (const auto& name = node["name"]; name.isString())
                nodeName = name.asString();

            if (const auto& modelNode = node["model"]; !modelNode.isNull())
            {
                if (!modelNode.isIntegral())
                    throw std::runtime_error("'model' value for graph node '" + nodeName + "' must be an index");

                int modelIndex = modelNode.asInt();
                if (modelIndex < 0 || modelIndex >= (int)nmodels)
                    throw std::runtime_error("out of bounds 'model' index for graph node '" + nodeName + "'");

                const Json::Value& modelName = models[modelIndex];

                if (!modelName.isString())
                    throw std::runtime_error("invalid model path in 'models' section");

                float frameOffset = 0.0f;
                if (const auto& offset = node["frameoffset"]; offset.isDouble())
                {
                    frameOffset = offset.asFloat();
                }

                auto model =
                    objImporter->Load(modelName.asString(), *m_TextureCache, { 0, 0 }, instance, commandList);

                if (model.has_value())
                    InsertModel(std::move(*model));
            }

            if (const auto& type = node["type"]; type.isString())
                throw std::runtime_error("'type' token for graph node '" + nodeName + "' not supported");

            if (const auto& parent = node["parent"]; !parent.isNull())
                throw std::runtime_error("'parent' token for graph node '" + nodeName + "' not supported");

            if (const auto& children = node["children"]; !children.isNull())
                throw std::runtime_error("'children' token for graph node '" + nodeName + "' not supported");
        }

        if (Json::Value& view = jsonRoot["view"]; view.isObject())
        {
            m_view = std::make_unique<View>();
            *m_view << view;
        }

        if (Json::Value& settings = jsonRoot["settings"]; settings.isObject())
        {
            m_sceneSettings = settings; // so the app can use these settings to override some of its own behavior
            m_attributes << settings;
        }
    }
    else
    {
        log::fatal("failed to Parse JSON file '%s'", fp.generic_string().c_str());
    }
}

bool RTXMGScene::LoadWithExecutor(const std::filesystem::path& filename,
    tf::Executor* executor)
{
    if (executor != nullptr)
    {
        log::warning("RTXMGScene::LoadWithExecutor: executor based loading is not "
            "supported, ignoring");
    }
    log::info("RTXMGScene::LoadWithExecutor: %s", filename.string().c_str());

    TopologyCache topologyCache(TopologyCache::Options{
        .isoLevelSharp = (uint8_t)m_isoLevelSharp,
        .isoLevelSmooth = (uint8_t)m_isoLevelSmooth,
        });

    fs::path sanitizedFilePath = filename;
    std::string sceneName = sanitizedFilePath.empty() ? "default_scene" : sanitizedFilePath.filename().generic_string();

    auto commandList = m_Device->createCommandList();
    commandList->open();
    {
        std::unique_ptr<ObjImporter> objImporter =
            std::make_unique<ObjImporter>(m_fs, m_mediaPath, m_SceneTypeFactory, m_DescriptorTable, topologyCache);

        if (sanitizedFilePath.empty() || sanitizedFilePath.extension() == ".obj")
        {
            // obj importer will default to a cube without a filename
            log::info("RTXMGScene::LoadWithExecutor: Loading an OBJ file");

            auto model =
                objImporter->Load(sanitizedFilePath, *m_TextureCache, m_attributes.frameRange,
                    Instance{}, commandList);

            if (model.has_value())
            {
                InsertModel(std::move(*model));
            }
            else
            {
                log::fatal("RTXMGScene::LoadWithExecutor: Failed to load the OBJ file");
            }
        }
        else if (sanitizedFilePath.extension() == ".json")
        {
            log::info("RTXMGScene::LoadWithExecutor: Loading a JSON file");
            LoadSceneFile(sanitizedFilePath, objImporter, commandList);
        }
        else
        {
            log::fatal("RTXMGScene::LoadWithExecutor: Unsupported file format");
            return false;
        }

        m_attributes.averageInstanceScale = 0.0f;
        for (const auto& instance : m_instances)
        {
            m_attributes.averageInstanceScale +=
                maxComponent(instance.aabb.m_maxs - instance.aabb.m_mins);
            m_attributes.aabb |= instance.aabb;
        }

        m_attributes.averageInstanceScale /= float(m_instances.size());
        if ((m_attributes.frameRange.y - m_attributes.frameRange.x) > 1 &&
            m_attributes.frameRate == 0.f)
        {
            m_attributes.frameRate = 24.0f;
        }

        m_topologyMaps = topologyCache.InitDeviceData(m_DescriptorTable, commandList);

        m_inputPath = sanitizedFilePath.lexically_normal().generic_string();
    }
    commandList->close();
    m_Device->executeCommandList(commandList);

    /// Convert control cage to donut scenegraph for easy viz
    std::shared_ptr<SceneGraphNode> root = std::make_shared<SceneGraphNode>();
    root->SetName(sceneName);
    m_Models.emplace_back(SceneImportResult{ root });

    m_SceneGraph = std::make_shared<SceneGraph>();
    m_SceneGraph->SetRootNode(m_Models.front().rootNode);

    
    std::vector<std::shared_ptr<MeshInfo>> sceneMeshInfos;
    for (auto& subdMesh : m_subdMeshes)
    {
        auto shape = subdMesh->GetShape();
    
        // Add scene mesh info with dummy vertices
        // This is to set up GeometryData and Material data in a donut compatible way
        auto meshInfo = m_SceneTypeFactory->CreateMesh();
        sceneMeshInfos.push_back(meshInfo);
        meshInfo->name = shape->filepath.empty() ? "default_shape" : shape->filepath.generic_string();
        meshInfo->buffers = std::make_shared<BufferGroup>(); // empty dummy buffer group
        meshInfo->buffers->positionData.resize(4, float3::zero()); // add dummy vertex buffer. Must be 16 byte aligned
        meshInfo->objectSpaceBounds = shape->aabb;
        meshInfo->totalIndices = 0;
        meshInfo->totalVertices = 0;

        auto addTexture = [&sanitizedFilePath, this](const fs::path& shapePath, const std::string& mtlLib, const std::string& texPath, bool* enable) -> std::shared_ptr<LoadedTexture>
            {
                *enable = false;
                if (texPath.empty())
                {
                    return nullptr;
                }
                fs::path fp = (((shapePath.parent_path() / mtlLib)).parent_path() / texPath).lexically_normal();
                if (!is_regular_file(fp))
                {
                    fp = ResolveMediapath(fp, m_mediaPath);
                }
                if (is_regular_file(fp))
                {
                    *enable = true;
                    return m_TextureCache->LoadTextureFromFileDeferred(fp, false);
                }
                else
                {
                    log::warning("Texture %s not found...", fp.generic_string().c_str());

                }
                return nullptr;
            };

        std::vector<std::shared_ptr<Material>> materials(shape->mtls.size());

        meshInfo->geometries.reserve(shape->subshapes.size());
        size_t numFaces = shape->nvertsPerFace.size();
        size_t vertIndex = 0;
        for (uint32_t subshapeIndex = 0; subshapeIndex < shape->subshapes.size(); subshapeIndex++)
        {
            const auto& subshape = shape->subshapes[subshapeIndex];

            // lazy initialize of materials
            if (!materials[subshape.mtlBind].get())
            {
                const auto& mtl = shape->mtls[subshape.mtlBind];
                auto material = m_SceneTypeFactory->CreateMaterial();
                material->baseOrDiffuseColor = mtl->kd;
                material->specularColor = mtl->ks;
                // material->specularExponent = mtl->ns;
                material->emissiveColor = mtl->ke;
                material->roughness = mtl->Pr;
                material->baseOrDiffuseTexture = addTexture(shape->filepath, shape->mtllib, mtl->map_kd, &material->enableBaseOrDiffuseTexture);
                material->metalRoughOrSpecularTexture = addTexture(shape->filepath, shape->mtllib, mtl->map_pr, &material->enableMetalRoughOrSpecularTexture);

                // hijack emissive texture for metalness for now (so we can use the existing Donut material structs)
                material->emissiveTexture = addTexture(shape->filepath, shape->mtllib, mtl->map_pm, &material->enableEmissiveTexture);

                // hijack occlusion texture for specular for now (so we can use the existing Donut material structs)
                material->occlusionTexture = addTexture(shape->filepath, shape->mtllib, mtl->map_ks, &material->enableOcclusionTexture);

                material->normalTexture = addTexture(shape->filepath, shape->mtllib, mtl->map_bump, &material->enableNormalTexture);
                material->normalTextureScale = mtl->bm;
                material->metalness = mtl->Pm;

                materials[subshape.mtlBind] = material;
            }

            std::shared_ptr<MeshGeometry> geometry = std::make_shared<MeshGeometry>();
            geometry->material = materials[subshape.mtlBind];
            geometry->numIndices = 0;
            geometry->numVertices = 0;
            geometry->objectSpaceBounds = box3::empty();
            geometry->indexOffsetInMesh = 0;
            geometry->vertexOffsetInMesh = 0;
            meshInfo->geometries.push_back(geometry);

            size_t endFaceIndex = (subshapeIndex + 1) < shape->subshapes.size() ?
                shape->subshapes[subshapeIndex + 1].startFaceIndex :
                numFaces;

            for (size_t faceIndex = subshape.startFaceIndex; faceIndex < endFaceIndex; faceIndex++)
            {
                for (size_t v_offset = 0; v_offset < shape->nvertsPerFace[faceIndex];
                    ++v_offset)
                {
                    geometry->objectSpaceBounds |=
                        shape->verts[shape->faceverts[vertIndex + v_offset]];
                }
                vertIndex += shape->nvertsPerFace[faceIndex];
            }

            meshInfo->objectSpaceBounds |= geometry->objectSpaceBounds;

            if (geometry->material->normalTexture.get() && geometry->material->enableNormalTexture)
            {
                subdMesh->m_hasDisplacementMaterial = true;
            }
        }
    }

    auto subdMeshInstances = GetSubdMeshInstances();
    uint32_t instanceIndex = 0;
    for (auto& instance : subdMeshInstances)
    {
        auto meshInfo = sceneMeshInfos[instance.meshID];
        auto meshInstance = m_SceneTypeFactory->CreateMeshInstance(meshInfo);

        // connect our mesh instances to donut's
        instance.meshInstance = meshInstance;

        auto node = std::make_shared<SceneGraphNode>();

        node->SetScaling(double3(instance.scaling));
        node->SetRotation(dquat(instance.rotation));
        node->SetTranslation(double3(instance.translation));

        m_SceneGraph->Attach(root, node);

        node->SetLeaf(meshInstance);

        std::string meshInstanceName = meshInfo->name + "_" + std::to_string(instanceIndex);
        node->SetName(meshInstanceName);

        instanceIndex++;
    }

    PrintSceneGraph(root);

    SceneGraphWalker walker(m_SceneGraph->GetRootNode().get());
    int boxCount = 0;
    while (walker)
    {
        auto leaf = walker->GetLeaf();
        if (leaf)
        {
            auto box = leaf->GetLocalBoundingBox();
            m_attributes.averageInstanceScale += MaxBoxExtent(box);
            boxCount++;
        }
        walker.Next(true);
    }
    m_attributes.averageInstanceScale /= boxCount;

    return true;
}
