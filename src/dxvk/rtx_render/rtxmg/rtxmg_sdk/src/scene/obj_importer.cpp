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

#include "rtxmg/scene/obj_importer.h"
#include "rtxmg/utils/buffer.h"

#include "rtxmg/subdivision/shape.h"
#include "rtxmg/subdivision/subdivision_surface.h"

#include <donut/core/log.h>
#include <donut/engine/Scene.h>
#include <execution>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <set>

using namespace donut;
using namespace donut::engine;
using namespace donut::math;
namespace fs = std::filesystem;

// parses filenames of type 'filename[start-stop].ext'
static std::optional<int2> getSequenceRange(const std::string& str)
{
    size_t open = str.find('[');
    size_t close = str.find(']');
    if (open == std::string::npos || close == std::string::npos)
        return {};

    std::string_view range = { str.data() + open + 1, str.data() + close };

    size_t delim = range.find('-');
    if (delim == std::string::npos)
        return {};

    int2 framerange = { 0, 0 };
    std::from_chars(range.data(), range.data() + delim, framerange.x);
    std::from_chars(range.data() + delim + 1, range.data() + range.size(),
        framerange.y);
    return framerange;
}

static std::string getSequenceFormat(const std::string& str, int2 frameRange)
{
    size_t open = str.find('[');
    size_t close = str.find(']');
    if (open == std::string::npos || close == std::string::npos)
        return str;

    std::string prefix = str.substr(0, open);
    std::string suffix = str.substr(close + 1, std::string::npos);

    // test some common padding formats
    std::array<const char*, 3> formats = { "%d", "%03d", "%04d" };

    for (const char* format : formats)
    {
        char buf[16];
        std::snprintf(buf, std::size(buf), format, frameRange.x);

        if (fs::is_regular_file(prefix + buf + suffix))
            return prefix + format + suffix;
    }
    return str;
}

ObjImporter::ObjImporter(std::shared_ptr<vfs::IFileSystem> fs,
    const fs::path& mediapath,
    std::shared_ptr<SceneTypeFactory> sceneTypeFactory,
    std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTableManager,
    TopologyCache& topologyCache)
    : m_fs(std::move(fs)), m_sceneTypeFactory(std::move(sceneTypeFactory)),
    m_topologyCache(topologyCache),
    m_descriptorTableManager(std::move(descriptorTableManager)),
    m_mediaPath(mediapath)
{}

std::optional<Model> ObjImporter::Load(const fs::path& fileName, TextureCache& textureCache,
    int2 frameRange, const Instance& parent,
    nvrhi::ICommandList* commandList) const
{
    fs::path fp = m_modelPath.empty() ? fileName : m_modelPath / fileName;

    if (!fs::is_regular_file(fp) && fs::is_regular_file(m_mediaPath / fp))
        fp = m_mediaPath / fp;

    if (auto range = getSequenceRange(fp.string()))
    {
        frameRange = *range;
        fp = getSequenceFormat(fp.string(), frameRange);
    }

    Model model;
    int nframes = 1;
    if (frameRange.y > frameRange.x)
        nframes = frameRange.y - frameRange.x + 1;


    std::unique_ptr<Shape> shape;
    if (fp.empty())
    {
        shape = Shape::DefaultShape();
    }
    else
    {
        if (fp.extension() == ".eddbin")
        {
            log::warning("EDDBin files are not supported by the ObjImporter");
            return {};
        }
        if (nframes == 1)
        {
            shape = Shape::LoadObjFile(fp.string().c_str());
        }
        else
        {
            char buf[1024];
            std::snprintf(buf, std::size(buf), fp.generic_string().c_str(),
                frameRange.x);
            shape = Shape::LoadObjFile(buf);
        }
    }
    if (shape->mtlbind.empty())
    {
        shape->mtlbind.resize(shape->nvertsPerFace.size(), 0);
        shape->mtls.push_back(std::make_unique<Shape::material>());
    }

    int32_t fallbackErrorMaterial = -1;
    size_t originalMaterialRange = shape->mtls.size();

    shape->subshapes.reserve(originalMaterialRange);
    shape->faceToSubshapeIndex.reserve(shape->mtlbind.size());
    int currMtl = -1;
    for (size_t m_idx = 0; m_idx < shape->mtlbind.size(); ++m_idx)
    {
        int nextMtl = shape->mtlbind[m_idx];
        if (nextMtl < 0 || nextMtl >= originalMaterialRange)
        {
            // missing mtl for face
            log::warning("Missing material bind for face: %d\n", m_idx);
            if (fallbackErrorMaterial == -1)
            {
                fallbackErrorMaterial = (int32_t)shape->mtls.size();
                shape->mtls.push_back(std::make_unique<Shape::material>());
            }
            nextMtl = fallbackErrorMaterial;
        }

        if (nextMtl != currMtl)
        {
            currMtl = nextMtl;
            shape->subshapes.push_back({ m_idx, currMtl });
        }

        assert(shape->subshapes.size());
        assert(shape->subshapes.size() < std::numeric_limits<uint16_t>::max());
        shape->faceToSubshapeIndex.push_back(uint16_t(shape->subshapes.size()) - 1);
    }

    std::vector<std::unique_ptr<Shape>> keyFrameShapes;
    if (nframes > 1)
    {
        std::vector<int> frames(nframes - 1);
        std::iota(frames.begin(), frames.end(), 1);

        keyFrameShapes.resize(frames.size());

        std::for_each(std::execution::par_unseq, frames.begin(), frames.end(), [&](int frame)
            {
                char buf[1024];
                std::snprintf(buf, std::size(buf), fp.generic_string().c_str(), frame + frameRange.x);

                keyFrameShapes[frame - 1] = (Shape::LoadObjFile(buf, false, false));
            });
    }

    std::unique_ptr<SubdivisionSurface> subd =
        std::make_unique<SubdivisionSurface>(m_topologyCache, std::move(shape), keyFrameShapes,
            m_descriptorTableManager, commandList);

    Instance instance = parent;
    instance.aabb = subd->m_aabb * instance.localToWorld;

    model.frameRange = frameRange;
    model.subd = std::move(subd);
    model.instances.emplace_back(instance);

    return model;
}

void Instance::Animate(float animTime, float animRate) {}

void Instance::UpdateLocalTransform()
{
    localToWorld = donut::math::scaling(scaling);

    localToWorld *= rotation.toAffine();

    localToWorld *= donut::math::translation(translation);
}

void Instance::Lerp(Instance const& a, Instance const& b, float t)
{
    auto lerp = [](auto a, auto b, float t) { return (1.f - t) * a + t * b; };

    localToWorld.m_linear =
        lerp(a.localToWorld.m_linear, b.localToWorld.m_linear, t);
    localToWorld.m_translation =
        lerp(a.localToWorld.m_translation, b.localToWorld.m_translation, t);

    // TODO: AABB lerp

    radius = lerp(a.radius, b.radius, t);
    edgelength = lerp(a.edgelength, b.edgelength, t);
}
