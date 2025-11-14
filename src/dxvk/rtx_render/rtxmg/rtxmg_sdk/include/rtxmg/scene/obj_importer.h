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

#include <filesystem>
#include <memory>
#include <optional>

#include <nvrhi/utils.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/DescriptorTableManager.h>

#include "rtxmg/scene/model.h"

namespace donut::vfs
{
    class IFileSystem;
}

namespace donut::engine
{
    struct SceneImportResult;
    class TextureCache;
    class SceneTypeFactory;
} // namespace donut::engine

namespace tf
{
    class Executor;
}

namespace fs = std::filesystem;

class TopologyCache;

class ObjImporter
{
protected:
    std::shared_ptr<donut::vfs::IFileSystem> m_fs;
    std::shared_ptr<donut::engine::SceneTypeFactory> m_sceneTypeFactory;
    std::shared_ptr<donut::engine::DescriptorTableManager> m_descriptorTableManager;
    TopologyCache& m_topologyCache;

    fs::path m_modelPath;
    const fs::path& m_mediaPath;

public:
    explicit ObjImporter(
        std::shared_ptr<donut::vfs::IFileSystem> fs,
        const fs::path& mediapath,
        std::shared_ptr<donut::engine::SceneTypeFactory> sceneTypeFactory,
        std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTableManager,
        TopologyCache& topologyCache);

    std::optional<Model> Load(const std::filesystem::path& fileName,
        donut::engine::TextureCache& textureCache,
        int2 frameRange, const Instance& parent,
        nvrhi::ICommandList* commandList) const;

    void SetModelPath(fs::path&& path) { m_modelPath = std::move(path); }
};
