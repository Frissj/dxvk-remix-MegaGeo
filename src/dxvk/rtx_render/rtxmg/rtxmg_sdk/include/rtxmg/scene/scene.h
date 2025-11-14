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

#pragma once

#include <json/json.h>

#include "rtxmg/scene/obj_importer.h"

#include "rtxmg/subdivision/topology_map.h"
#include "rtxmg/cluster_builder/cluster_accel_builder.h"

#include <donut/core/math/math.h>
#include <donut/engine/Scene.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/shaders/bindless.h>

class MaterialCache;

using namespace donut::engine;
using namespace donut::math;

struct View
{
    float3 position = { 0.f, 0.f, -1.f };
    float3 lookat = { 0.f, 0.f, -1.f };
    float3 up = { 1.f, 1.f, 1.f };
    float fov = 35.f;
};

enum TextureType
{
    ALBEDO = 0,
    ROUGHNESS,
    SPECULAR,
    DISPLACEMENT,
    ENVMAP,
    TEXTURE_TYPE_COUNT
};

class RTXMGScene : public Scene
{
public:
    struct Attributes
    {
        std::string audio;
        float audioStartTime = 0.f;

        int2 frameRange = { std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::min() };
        float frameRate = 0.f;

        float averageInstanceScale = 0.f;
        box3 aabb;
    };

    RTXMGScene(nvrhi::IDevice* device,
        const fs::path& mediapath,
        std::shared_ptr<CommonRenderPasses> commonPasses,
        ShaderFactory& shaderFactory,
        std::shared_ptr<donut::vfs::IFileSystem> fs,
        std::shared_ptr<TextureCache> textureCache,
        std::shared_ptr<DescriptorTableManager> descriptorTable,
        std::shared_ptr<SceneTypeFactory> sceneTypeFactory,
        int2 initialFrameRange, int isoLevelSharp, int isoLevelSmooth);

    bool LoadWithExecutor(const std::filesystem::path& filename,
        tf::Executor* executor) override;

    const Attributes& GetAttributes() const { return m_attributes; }
    void InsertModel(Model&& model);

    const std::vector<std::unique_ptr<SubdivisionSurface>>&
        GetSubdMeshes() const
    {
        return m_subdMeshes;
    }

    std::vector<Instance>& GetInstances()
    {
        return m_instances;
    }

    const std::vector<std::unique_ptr<TopologyMap const>>&
        GetTopologyMaps() const
    {
        return m_topologyMaps;
    }
    std::span<Instance>       GetSubdMeshInstances();
    std::span<Instance const> GetSubdMeshInstances() const;
    uint32_t TotalSubdPatchCount() const;

    const View* GetView() const { return m_view.get(); }

    void Animate(float animTime, float animRate);
    
    nvrhi::SamplerHandle GetDisplacementSampler() const { return m_commonPasses->m_LinearWrapSampler; }

    const Json::Value& GetSceneSettings() const { return m_sceneSettings; }
    std::string& GetInputPath() { return m_inputPath; }

    static fs::path ResolveMediapath(const fs::path& m_filepath, const fs::path& mediapath);
protected:
    void LoadSceneFile(const std::filesystem::path& filename, std::unique_ptr<ObjImporter>& objImporter, nvrhi::ICommandList* commandList);


private:
    Attributes m_attributes;
    Json::Value m_sceneSettings;
    const fs::path& m_mediaPath;

    int m_isoLevelSharp;
    int m_isoLevelSmooth;

    // starting viewpoint
    std::unique_ptr<View> m_view;

    std::vector<std::unique_ptr<TopologyMap const>> m_topologyMaps;
    std::vector<std::unique_ptr<SubdivisionSurface>> m_subdMeshes;
    std::vector<Instance> m_instances;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    
    std::string m_inputPath;
};