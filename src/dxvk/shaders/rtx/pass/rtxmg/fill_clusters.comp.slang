/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
*
* Fill Clusters - Vertex generation shader
*
* This compute shader fills cluster vertices by sampling the input
* triangle mesh at UV positions determined by the cluster layout.
*
* Uses simple barycentric interpolation to avoid geometry deformation.
*/

#include "rtx/pass/rtxmg/rtxmg_bindings.slangh"
#include "rtx/utility/common.slangh"

// Constants
static const uint kThreadsPerWave = 32;
static const uint kWavesPerGroup = 4;
static const uint kThreadsPerGroup = kThreadsPerWave * kWavesPerGroup;

// Fill clusters parameters (not in bindings header)
struct FillClustersParams {
  uint surfaceStart;
  uint surfaceEnd;
  uint enableVertexNormals;  // Generate vertex normals
  uint clusterPattern;       // REGULAR or SLANTED

  float4x4 localToWorld;     // Local to world transform (for normal transform)

  // Phase 4: GPU batching support
  uint enableBatching;       // Enable GPU batching mode (0 = sequential, 1 = batched)
  uint instanceCount;        // Number of instances in batch
  uint instanceIndex;
  uint maxClusterIndex;

  int debugSurfaceIndex;
  int debugClusterIndex;
  int debugLaneIndex;
  uint _pad2;
};

// Register-based buffer declarations (match sample layout)
ConstantBuffer<FillClustersParams> cb : register(b0);

// Input buffers (t0-t4)
StructuredBuffer<GridSampler> gridSamplers : register(t0);
StructuredBuffer<uint2> clusterOffsetCounts : register(t1);
StructuredBuffer<Cluster> clusters : register(t2);
StructuredBuffer<float3> inputPositions : register(t3);
StructuredBuffer<float3> inputNormals : register(t4);

// Additional surface data (t5-t6)
ByteAddressBuffer inputTexcoords : register(t5);
ByteAddressBuffer inputIndices : register(t6);
StructuredBuffer<SurfaceInfo> surfaceInfo : register(t7);

// Padding for subdivision buffers (t8-t19) - not used in this simplified version
// These are declared but empty to match descriptor set layout with sample

// Output buffers (u0-u2)
RWStructuredBuffer<float3> clusterVertexPositionsOut : register(u0);
RWStructuredBuffer<ClusterShadingData> clusterShadingDataOut : register(u1);
RWStructuredBuffer<float3> clusterVertexNormalsOut : register(u2);

// Calculate UV position within surface for a cluster vertex
float2 calculateVertexUV(
  Cluster cluster,
  GridSampler sampler,
  uint2 vertexIdx2D)
{
  // Total grid size
  uint gridSizeX = max(sampler.edgeSegments[0], sampler.edgeSegments[2]);
  uint gridSizeY = max(sampler.edgeSegments[1], sampler.edgeSegments[3]);

  // Vertex position in global grid
  uint globalX = cluster.offsetX + vertexIdx2D.x;
  uint globalY = cluster.offsetY + vertexIdx2D.y;

  // Normalize to [0,1]
  float u = float(globalX) / float(gridSizeX);
  float v = float(globalY) / float(gridSizeY);

  return float2(u, v);
}

// Sample triangle mesh at UV position using barycentric interpolation
void sampleTriangleMesh(
  StructuredBuffer<float3> positions,
  StructuredBuffer<float3> normals,
  ByteAddressBuffer texcoords,
  ByteAddressBuffer indices,
  SurfaceInfo surface,
  float2 uv,
  out float3 outPosition,
  out float3 outNormal,
  out float2 outTexcoord)
{
  // CRITICAL FIX: Validate surface data before any access
  const uint maxIndexCount = 0x10000000;  // 256M indices max
  if (surface.firstIndex >= maxIndexCount || surface.indexCount < 3) {
    outPosition = float3(0, 0, 0);
    outNormal = float3(0, 0, 1);
    outTexcoord = float2(0, 0);
    return;
  }

  // For simplicity, assume the surface is a quad made of 2 triangles
  // UV [0,1] maps to the quad

  // Find which triangle we're in
  uint triIndex;
  float3 bary;

  if (uv.x + uv.y <= 1.0) {
    // Lower-left triangle (0,1,2)
    triIndex = 0;
    bary = float3(1.0 - uv.x - uv.y, uv.x, uv.y);
  } else {
    // Upper-right triangle (2,1,3)
    triIndex = 1;
    bary = float3(uv.x + uv.y - 1.0, 1.0 - uv.y, 1.0 - uv.x);
  }

  // Get triangle indices (ByteAddressBuffer requires byte offset)
  uint i0 = indices.Load((surface.firstIndex + triIndex * 3 + 0) * 4);
  uint i1 = indices.Load((surface.firstIndex + triIndex * 3 + 1) * 4);
  uint i2 = indices.Load((surface.firstIndex + triIndex * 3 + 2) * 4);

  // Clamp indices to valid range
  i0 = min(i0, surface.vertexCount - 1);
  i1 = min(i1, surface.vertexCount - 1);
  i2 = min(i2, surface.vertexCount - 1);

  // Interpolate position
  float3 p0 = positions[surface.firstVertex + i0];
  float3 p1 = positions[surface.firstVertex + i1];
  float3 p2 = positions[surface.firstVertex + i2];
  outPosition = p0 * bary.x + p1 * bary.y + p2 * bary.z;

  // Interpolate normal
  float3 n0 = normals[surface.firstVertex + i0];
  float3 n1 = normals[surface.firstVertex + i1];
  float3 n2 = normals[surface.firstVertex + i2];
  outNormal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);

  // Interpolate texcoord (ByteAddressBuffer - load 2 floats = 8 bytes)
  float2 t0 = asfloat(texcoords.Load2((surface.firstVertex + i0) * 8));
  float2 t1 = asfloat(texcoords.Load2((surface.firstVertex + i1) * 8));
  float2 t2 = asfloat(texcoords.Load2((surface.firstVertex + i2) * 8));
  outTexcoord = t0 * bary.x + t1 * bary.y + t2 * bary.z;
}

[shader("compute")]
[numthreads(kThreadsPerWave * kWavesPerGroup, 1, 1)]
void main(uint3 threadIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID)
{
  uint laneIndex = threadIdx.x % kThreadsPerWave;
  uint waveIndex = threadIdx.x / kThreadsPerWave;

  // Cluster index from wave-based batching
  uint groupClusterIndex = groupIdx.x * kWavesPerGroup + waveIndex;

  // Get offset and count for this instance's clusters
  uint2 offsetCount = clusterOffsetCounts[cb.instanceIndex * 4 + 3];  // All types dispatch

  if (groupClusterIndex >= offsetCount.y)
    return; // Early exit for waves beyond cluster count

  uint clusterIndex = groupClusterIndex + offsetCount.x;

  // Validate cluster index
  if (clusterIndex >= cb.maxClusterIndex)
    return;

  Cluster cluster = clusters[clusterIndex];
  uint surfaceIndex = cluster.iSurface;
  GridSampler sampler = gridSamplers[surfaceIndex];
  SurfaceInfo surface = surfaceInfo[surfaceIndex];

  uint vertexCountPerCluster = cluster.sizeX * cluster.sizeY;

  // Process cluster vertices in wave-parallel loop
  for (uint vertexIdx = laneIndex; vertexIdx < vertexCountPerCluster; vertexIdx += kThreadsPerWave)
  {
    // Calculate 2D grid position
    uint2 vertexIdx2D = uint2(vertexIdx % cluster.sizeX, vertexIdx / cluster.sizeX);

    // Calculate UV position
    float2 uv = calculateVertexUV(cluster, sampler, vertexIdx2D);

    // Evaluate vertex position and normal
    float3 position;
    float3 normal;
    float2 texcoord;

    // Use simple barycentric mesh sampling (no deformation)
    sampleTriangleMesh(
      inputPositions,
      inputNormals,
      inputTexcoords,
      inputIndices,
      surface,
      uv,
      position,
      normal,
      texcoord
    );

    // Write results to output buffers
    uint outputIndex = cluster.nVertexOffset + vertexIdx;
    clusterVertexPositionsOut[outputIndex] = position;

    if (cb.enableVertexNormals)
    {
      clusterVertexNormalsOut[outputIndex] = normal;
    }
  }
}
