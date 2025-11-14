#pragma once

struct ClusterTiling
{
    uint16_t2 tilingSize;   // number of tiles in x and y direction
    uint16_t2  clusterSize;  // number of quads in x and y direction inside tile

    inline uint32_t ClusterCount() { return uint32_t(tilingSize.x) * uint32_t(tilingSize.y); }
    inline uint32_t ClusterVertexCount() { return (clusterSize.x + 1) * (clusterSize.y + 1); }
    inline uint32_t VertexCount() { return ClusterVertexCount() * ClusterCount(); }

    inline uint16_t2 ClusterIndex2D(uint32_t rowMajorIndex)
    {
        return uint16_t2((uint16_t)(rowMajorIndex % tilingSize.x), (uint16_t)(rowMajorIndex / tilingSize.x));
    }

    inline uint16_t2 QuadOffset2D(uint32_t rowMajorIndex)
    {
        return ClusterIndex2D(rowMajorIndex) * uint16_t2(clusterSize.x, clusterSize.y);
    }

    inline uint2 VertexIndex2D(uint32_t rowMajorIndex)
    {
        uint32_t verticesU = clusterSize.x + 1;
        return uint2(rowMajorIndex % verticesU, rowMajorIndex / verticesU);
    }
};

struct SurfaceTiling
{
    enum
    {
        REGULAR = 0,
        RIGHT,
        TOP,
        CORNER,
        N_SUB_TILINGS
    };
    ClusterTiling subTilings[N_SUB_TILINGS];
    uint16_t2       quadOffsets[N_SUB_TILINGS];  // quad offset of the tiling in x and y direction

    uint32_t inline ClusterCount()
    {
        uint32_t sum = 0;
        for (int iTiling = 0; iTiling < N_SUB_TILINGS; ++iTiling)
            sum += subTilings[iTiling].ClusterCount();
        return sum;
    }

    uint32_t inline VertexCount()
    {
        uint32_t sum = 0;
        for (int iTiling = 0; iTiling < N_SUB_TILINGS; ++iTiling)
            sum += subTilings[iTiling].VertexCount();
        return sum;
    }

    uint16_t2 inline ClusterOffset(uint16_t iTiling, uint32_t iCluster)
    {
        return quadOffsets[iTiling] + subTilings[iTiling].QuadOffset2D(iCluster);
    }
};

inline SurfaceTiling MakeSurfaceTiling(uint16_t2 surfaceSize)
{
    SurfaceTiling ret;
    uint16_t targetEdgeSegments = 8;

    uint16_t2 regularGridSize;
    uint16_t2  modCluster;
    {
        uint16_t2 divClusters = uint16_t2((uint16_t)(surfaceSize.x / targetEdgeSegments),
            (uint16_t)(surfaceSize.y / targetEdgeSegments));
        modCluster = uint16_t2((uint16_t)(surfaceSize.x % targetEdgeSegments),
            (uint16_t)(surfaceSize.y % targetEdgeSegments));

        uint32_t maxEdgeSegments = kMaxClusterEdgeSegments;
        if (divClusters.x > 0 && modCluster.x + targetEdgeSegments <= maxEdgeSegments)
        {
            divClusters.x -= 1;
            modCluster.x += targetEdgeSegments;
        }
        if (divClusters.y > 0 && modCluster.y + targetEdgeSegments <= maxEdgeSegments)
        {
            divClusters.y -= 1;
            modCluster.y += targetEdgeSegments;
        }
        regularGridSize = divClusters;
    }

    ret.subTilings[SurfaceTiling::REGULAR].tilingSize = regularGridSize;
    ret.subTilings[SurfaceTiling::REGULAR].clusterSize = uint16_t2(targetEdgeSegments, targetEdgeSegments);
    ret.quadOffsets[SurfaceTiling::REGULAR] = uint16_t2(0u, 0u);

    ret.subTilings[SurfaceTiling::RIGHT].tilingSize = uint16_t2(1u, regularGridSize.y);
    ret.subTilings[SurfaceTiling::RIGHT].clusterSize = uint16_t2(modCluster.x, targetEdgeSegments);
    ret.quadOffsets[SurfaceTiling::RIGHT] = uint16_t2((uint16_t)(regularGridSize.x * targetEdgeSegments), 0u);

    ret.subTilings[SurfaceTiling::TOP].tilingSize = uint16_t2(regularGridSize.x, 1u);
    ret.subTilings[SurfaceTiling::TOP].clusterSize = uint16_t2(targetEdgeSegments, modCluster.y);
    ret.quadOffsets[SurfaceTiling::TOP] = uint16_t2(0u, (uint16_t)(regularGridSize.y * targetEdgeSegments));

    ret.subTilings[SurfaceTiling::CORNER].tilingSize = uint16_t2(1u, 1u);
    ret.subTilings[SurfaceTiling::CORNER].clusterSize = modCluster;
    ret.quadOffsets[SurfaceTiling::CORNER] = uint16_t2((uint16_t)(regularGridSize.x * targetEdgeSegments),
        (uint16_t)(regularGridSize.y * targetEdgeSegments));

    return ret;
}

