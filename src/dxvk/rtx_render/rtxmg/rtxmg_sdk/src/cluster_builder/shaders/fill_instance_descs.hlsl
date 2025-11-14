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
#pragma pack_matrix(row_major)

#include <nvrhi/nvrhiHLSL.h>
#include "rtxmg/cluster_builder/fill_instance_descs_params.h"

ConstantBuffer<FillInstanceDescsParams> g_Params : register(b0);

StructuredBuffer<nvrhi::GpuVirtualAddress> t_BlasAddresses : register(t0);
RWStructuredBuffer<nvrhi::rt::IndirectInstanceDesc> u_InstanceDescs : register(u0);

[numthreads(kFillInstanceDescsThreads, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint32_t instanceIndex = threadIdx.x;
    if (instanceIndex > g_Params.numInstances)
        return;

    nvrhi::GpuVirtualAddress blasAddress = t_BlasAddresses[instanceIndex];
    
    nvrhi::rt::IndirectInstanceDesc instanceDesc = u_InstanceDescs[instanceIndex];
    instanceDesc.blasDeviceAddress = blasAddress;
    u_InstanceDescs[instanceIndex] = instanceDesc;
}