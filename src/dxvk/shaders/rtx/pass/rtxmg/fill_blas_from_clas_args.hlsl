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
#include "rtxmg/cluster_builder/fill_blas_from_clas_args_params.h"
#include "rtxmg/cluster_builder/copy_cluster_offset_params.h"

ConstantBuffer<FillBlasFromClasArgsParams> g_Params : register(b0);

StructuredBuffer<uint2> t_ClusterOffsetCounts : register(t0);
RWStructuredBuffer<nvrhi::rt::cluster::IndirectArgs> u_BlasFromClasArgs : register(u0);

[numthreads(kFillBlasFromClasArgsThreads, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint32_t instanceIndex = threadIdx.x;
    if (instanceIndex > g_Params.numInstances)
        return;

    uint2 offsetCount = t_ClusterOffsetCounts[instanceIndex * ClusterDispatchType::NumTypes + ClusterDispatchType::All];

    nvrhi::rt::cluster::IndirectArgs args = (nvrhi::rt::cluster::IndirectArgs)0;
    args.clusterCount = offsetCount.y;
    args.clusterAddresses = g_Params.clasAddressesBaseAddress + sizeof(nvrhi::GpuVirtualAddress) * offsetCount.x;
    u_BlasFromClasArgs[instanceIndex] = args;
}