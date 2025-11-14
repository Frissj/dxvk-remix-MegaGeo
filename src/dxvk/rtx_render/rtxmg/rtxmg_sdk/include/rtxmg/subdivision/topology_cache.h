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
// clang-format off
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "rtxmg/subdivision/topology_map.h"
// clang-format on

// Thread-safe cache for Tmr::TopologyMap
//
// The topology cache collects a set of topology maps used to hash the topology
// of subD surfaces in a scene based on 'traits' (subdivision rules). Once all
// the meshes have been parsed, the topology maps can be serialized and hoisted
// in device memory.
//
// note: ownership of the host-side transient Tmr::TopologyyMaps is passed on to
// the device container (::TopologyMap) so that we can still support a CPU code
// path.
//

class TopologyCache
{
public:
    struct Options
    {
        // see Tmr::SubdivisionPlanBuilder::Options for details
        uint8_t const isoLevelSharp = 6;
        uint8_t const isoLevelSmooth = 3;
        bool const useTerminalNodes = false;
    } const options;

    TopologyCache(Options const& options);

    TopologyMap& get(uint8_t traits);

    TopologyMap const& Get(uint8_t traits) const { return this->Get(traits); }

    bool Empty() const;

    size_t Size() const;

    void Clear();

    // note: all hashing must be completed before hoisting maps into device memory
    // !
    std::vector<std::unique_ptr<TopologyMap const>>
        InitDeviceData(std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable, 
            nvrhi::ICommandList* commandList, bool keepHostData = true);

private:
    mutable std::mutex m_mtx;

    std::map<uint8_t, std::unique_ptr<TopologyMap>> m_topologyMaps;
};
