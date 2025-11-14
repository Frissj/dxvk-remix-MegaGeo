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

Texture2D<float> zbuffer: register(t0);
RWStructuredBuffer<uint> minmax: register(u0); // floats-as-uints because depth values are non-negative.

[numthreads(16, 16, 1)]
void main(uint2 threadIdx : SV_GroupThreadID, uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint32_t width, height;
    zbuffer.GetDimensions(width, height);

    uint32_t x = dispatchThreadId.x;
    uint32_t y = dispatchThreadId.y;

    if ((x >= width) || (y >= height))
    {
        return;
    }

    float depth = zbuffer[dispatchThreadId];
    if (depth < 0) depth = 0;

    uint wmin = asuint(depth), wmax = asuint(depth);
    if (isinf(depth))
    {
        wmin = 0xffffffff;
        wmax = 0;
    }

    for (int i = 16; i >= 1; i /= 2)
    {
        uint targetLane = WaveGetLaneIndex() ^ i;

        wmin = min(wmin, WaveReadLaneAt(wmin, targetLane));
        wmax = max(wmax, WaveReadLaneAt(wmax, targetLane));
    }
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        uint orig;
        InterlockedMin(minmax[0], asuint(wmin), orig);
        InterlockedMax(minmax[1], asuint(wmax), orig);
    }
}