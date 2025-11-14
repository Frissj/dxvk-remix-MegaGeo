/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Copy Cluster Offset - Compute shader to populate cluster offset/count buffer
*
* This shader reads GPU-written tessellation counters and writes per-instance
* cluster offset/count pairs for use by fill_blas_from_clas_args.
* Ported from NVIDIA RTX Mega Geometry SDK for RTX Remix integration.
*/

#include "rtx/pass/rtxmg/rtxmg_bindings.slangh"

// Constants
static const uint kThreadsPerGroup = 1;

// Parameters
struct CopyClusterOffsetParams {
  uint instanceIndex;
  uint totalInstances;
  uint _pad0;
  uint _pad1;
};

// Constant buffer (SDK MATCH: binding 0, NOT push constants!)
[[vk::binding(0, 0)]] ConstantBuffer<CopyClusterOffsetParams> cb;

// Input: tessellation counters (written by compute_cluster_tiling) - shifted +1
[[vk::binding(1, 0)]] StructuredBuffer<TessellationCounters> tessCountersIn;

// Output: cluster offset/count pairs - shifted +1
[[vk::binding(2, 0)]] RWStructuredBuffer<uint2> clusterOffsetCountsOut;

// SDK Constants: ClusterDispatchType
static const uint kClusterDispatchTypeAll = 3;
static const uint kClusterDispatchTypeNumTypes = 4;

[shader("compute")]
[numthreads(kThreadsPerGroup, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
  uint instanceIndex = cb.instanceIndex;
  uint totalClusterCount = tessCountersIn[0].clusters;

  // CRITICAL: Buffer writes must use instanceIndex * NumTypes + dispatchTypeIndex (SDK MATCH)
  // This shader is called multiple times per instance, once per ClusterDispatchType

  // Note: cb.dispatchTypeIndex is available but RTX Remix doesn't use it
  // For now, write to "All" type (index 3) which contains complete cluster counts
  // This matches the expected indexing pattern in fill_blas_from_clas_args.comp.slang

  uint dispatchTypeIndex = kClusterDispatchTypeAll;
  uint writeIndex = instanceIndex * kClusterDispatchTypeNumTypes + dispatchTypeIndex;

  uint dispatchClusterCount = 0;
  uint clusterOffset = 0;

  if (instanceIndex == 0) {
    // First instance gets all clusters starting at offset 0
    dispatchClusterCount = totalClusterCount;
    clusterOffset = 0;
  } else {
    // Read previous instance's "All" type data to calculate cumulative offset
    uint2 prevOffsetCount = clusterOffsetCountsOut[(instanceIndex - 1) * kClusterDispatchTypeNumTypes + kClusterDispatchTypeAll];
    clusterOffset = prevOffsetCount.x + prevOffsetCount.y;
    dispatchClusterCount = totalClusterCount - clusterOffset;
  }

  // Write offset and count for this instance (to the correct multiplied index)
  clusterOffsetCountsOut[writeIndex] = uint2(clusterOffset, dispatchClusterCount);
}
