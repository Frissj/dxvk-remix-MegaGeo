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

#include <donut/core/log.h>

#include <iomanip>
#include <sstream>
#include <typeinfo>

template<int N>
inline std::ostream& operator<<(std::ostream& ss, const donut::math::vector<float, N>& e)
{
    ss << "{";
    ss << std::setprecision(12) << e.data()[0];
    for (uint32_t i = 1; i < donut::math::vector<float, N>::DIM; i++)
        ss << ", " << e.data()[i];
    ss << "}";
    return ss;
}

template<int N>
inline std::ostream& operator<<(std::ostream& ss, const donut::math::vector<uint32_t, N>& e)
{
    ss << "{";
    ss << std::dec << e.data()[0] << std::hex << "(0x" << e.data()[0] << ")";
    for (uint32_t i = 1; i < donut::math::vector<uint32_t, N>::DIM; i++)
        ss << ", " << std::dec << e.data()[i] << std::hex << "(0x" << e.data()[i] << ")";
    ss << "}";
    return ss;
}

namespace vectorlog
{
    template<typename T>
    struct OutputLambda
    {
        typedef const std::function<bool(std::ostream& ss, const T& e)>& Type;
    };

    struct FormatOptions
    {
        bool wrap = true;
        bool header = true; // automatic header of elements
        bool elementIndex = true; // output index of the elemnt

        // specify a range
        size_t startIndex = 0;
        size_t count = 64;
    };

    template<typename T>
    static bool OutputElement(std::ostream& ss, const T& e)
    {
        if (std::is_same<T, float>::value)
            ss << std::setprecision(4) << e;
        else if (std::is_same<T, uint64_t>::value)
            ss << std::hex << "0x" << e;
        else
            ss << e;

        return true;
    }


    static void EndLine(std::stringstream& ss, std::ostream* optOutputStream)
    {
        if (optOutputStream)
        {
            (*optOutputStream) << ss.str() << std::endl;
        }
        else
        {
            donut::log::info(ss.str().c_str());
            ss.str("");
            ss.clear();
        }
    }

    template<typename T>
    static void OutputStream(const std::vector<T>& data, typename OutputLambda<T>::Type outputElementLambda, std::ostream* optOutputStream, const FormatOptions &options)
    {
        if (options.startIndex >= data.size())
            return;

        std::stringstream ss;

        auto begin = data.begin();
        auto iter = begin + options.startIndex;
        auto end = begin + std::min(size_t(options.startIndex + options.count), data.size());
        uint32_t numDigits = (end - iter) > 0 ? uint32_t(ceilf(log10f(float(end - iter)))) : 1;
        auto outputIndex = [&ss, &options, &numDigits](size_t index)
            {
                if (options.elementIndex)
                {
                    ss << "[" << std::dec << std::setw(numDigits) << index << "] ";
                }
            };

        if (options.wrap)
        {
            std::ostream::pos_type startLength = ss.tellp();

            bool continueOutput = true;
            if (iter != end)
            {
                outputIndex(iter - begin);
                continueOutput = outputElementLambda(ss, *iter++);
            }
            while (iter != end && continueOutput)
            {
                std::ostream::pos_type currentLength = ss.tellp() - startLength;
                if (currentLength > 80u)
                {
                    EndLine(ss, optOutputStream);
                    startLength = ss.tellp();
                }
                else
                {
                    ss << ", ";
                }
                outputIndex(iter - begin);
                continueOutput = outputElementLambda(ss, *iter++);
            }

            std::ostream::pos_type currentLength = ss.tellp() - startLength;
            if (currentLength > 0u)
            {
                EndLine(ss, optOutputStream);
                startLength = ss.tellp();
            }
        }
        else
        {
            bool continueOutput = true;
            while (iter != end && continueOutput)
            {
                outputIndex(iter - begin);
                continueOutput = outputElementLambda(ss, *iter++);
                EndLine(ss, optOutputStream);
            }
        }
    }

    template<typename T>
    static void Log(const std::vector<T>& data, typename OutputLambda<T>::Type outputElementLambda, FormatOptions options = {})
    {
        OutputStream(data, outputElementLambda, nullptr, options);
    }

    template<typename T>
    static void Log(const std::vector<T>& data, FormatOptions options = {})
    {
        OutputStream(data, vectorlog::OutputElement<T>, nullptr, options);
    }
}

