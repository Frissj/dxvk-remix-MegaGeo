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

#include <vector>

// clang-format off

template <typename T>
struct segmented_vector
{
    std::vector<T>        elements;      // element vector
    std::vector<uint32_t> offsets{ 0 };  // offset vector first element 0
    std::vector<uint32_t> sizes;         // segment sizes

    template <typename U>  // a container
    void Append(const U& segment)
    {
        sizes.push_back(static_cast<uint32_t>(segment.size()));
        offsets.push_back(offsets.back() + sizes.back());
        elements.insert(elements.end(), segment.begin(), segment.end());
    }

    void Append(const T* a_elements, uint32_t n_elements)
    {
        sizes.push_back(n_elements);
        offsets.push_back(offsets.back() + sizes.back());
        elements.insert(elements.end(), &a_elements[0], &a_elements[n_elements]);
    }

    void Reserve(size_t n)
    {
        offsets.reserve(n);
        sizes.reserve(n);
    }

    T* Data() { return elements.data(); }
    size_t Size() { return elements.size(); }
};

