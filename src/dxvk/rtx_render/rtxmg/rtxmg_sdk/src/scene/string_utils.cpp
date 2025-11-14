//
// Copyright (c) 2012-2016, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include "rtxmg/scene/string_utils.h"

#include <donut/core/log.h>

#include <fstream>

namespace fs = std::filesystem;

char const* ParseInt(char const* ptr, int* value)
{
    ptr = SkipWhiteSpace(ptr);

    int sign = 1;
    if (*ptr == '-')
    {
        sign = -1;
        ++ptr;
    }

    int num = 0;
    while (IsDigit(*ptr))
        num = 10 * num + (*ptr++ - '0');

    *value = sign * num;
    return ptr;
}

char const* ParseDouble(char const* ptr, double* value)
{
    static double const kPowersPos[] = {
        1.0e0,  1.0e1,  1.0e2,  1.0e3,  1.0e4,  1.0e5,  1.0e6,
        1.0e7,  1.0e8,  1.0e9,  1.0e10, 1.0e11, 1.0e12, 1.0e13,
        1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19,
    };

    static double const kPowersNeg[] = {
        1.0e0,   1.0e-1,  1.0e-2,  1.0e-3,  1.0e-4,  1.0e-5,  1.0e-6,
        1.0e-7,  1.0e-8,  1.0e-9,  1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13,
        1.0e-14, 1.0e-15, 1.0e-16, 1.0e-17, 1.0e-18, 1.0e-19,
    };
    static constexpr uint8_t npowers = sizeof(kPowersPos) / sizeof(double);

    double sign = 1.0;

    ptr = SkipWhiteSpace(ptr);

    if (*ptr == '-')
    {
        sign = -1;
        ++ptr;
    }
    else if (*ptr == '+')
    {
        ++ptr;
    }

    double num = 0.0;
    while (IsDigit(*ptr))
        num = 10.0 * num + (double)(*ptr++ - '0');

    if (*ptr == '.')
        ++ptr;

    double frac = 0.0, div = 1.0;
    while (IsDigit(*ptr))
    {
        frac = 10.0 * frac + (double)(*ptr++ - '0');
        div *= 10.0;
    }
    num += frac / div;

    if (IsExponent(*ptr))
    {
        ptr++;
        double const* powers = nullptr;
        if (*ptr == '+')
        {
            powers = kPowersPos;
            ++ptr;
        }
        else if (*ptr == '-')
        {
            powers = kPowersNeg;
            ++ptr;
        }
        else
        {
            powers = kPowersPos;
        }

        int e = 0;
        while (IsDigit(*ptr))
            e = 10 * e + (*ptr++ - '0');

        num *= (e >= npowers) ? 0.0 : powers[e];
    }

    *value = sign * num;

    return ptr;
}

std::unique_ptr<uint8_t[]> ReadBigFile(fs::path const& m_filepath,
    uint64_t* size)
{
    if (std::ifstream file(m_filepath, std::ios::binary); file.is_open())
    {
        auto tstart = std::chrono::steady_clock::now();

        std::streampos start = file.tellg();
        file.seekg(0, std::ios::end);
        std::streampos end = file.tellg();
        file.seekg(0, std::ios::beg);
        uint64_t length = end - start;

        if (length > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
            return nullptr;

        std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>(length + 1);
        file.read((char*)data.get(), length);

        data[length] = '\0';

        if (size)
            *size = length;

        {
            auto tstop = std::chrono::steady_clock::now();
            std::chrono::duration<float> elapsed = tstop - tstart;
            donut::log::info("read (%.2f seconds) '%s'", elapsed.count(),
                m_filepath.generic_string().c_str());
        }

        if (file.good())
            return std::move(data);
    }
    donut::log::error("reading '%s'\n", m_filepath.generic_string().c_str());
    return nullptr;
}

char const* ParseString(char const* ptr, std::string* value)
{
    char const* wordEnd = SkipWord(ptr);
    *value = std::string(ptr, wordEnd);
    return wordEnd;
}

std::string ReadASCIIFile(char const* m_filepath)
{
    std::ifstream ifs(m_filepath);

    if (!ifs)
        throw std::runtime_error(std::string("Cannot find: ") + m_filepath);

    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();

    std::string s = ss.str();
    if (s.empty())
        throw std::runtime_error(std::string("Read error: ") + m_filepath);

    return std::move(s);
}

char const* sgets(char* s, int size, char** stream)
{
    for (int i = 0; i < size; ++i)
    {
        if ((*stream)[i] == '\n' || (*stream)[i] == '\0')
        {

            memcpy(s, *stream, i);
            s[i] = '\0';

            if ((*stream)[i] == '\0')
                return 0;
            else
            {
                (*stream) += i + 1;
                return s;
            }
        }
    }
    return 0;
}
std::istream& operator>>(std::istream& is, float3& v)
{
    // parse [x,y,z]
    char st;
    is >> st >> v.x >> st >> v.y >> st >> v.z >> st;
    return is;
}
std::ostream& operator<<(std::ostream& os, float3& v)
{
    os << "[" << v.x << "," << v.y << "," << v.z << "]";
    return os;
}
std::ostream& operator<<(std::ostream& os, box3& b)
{
    os << b.m_mins << " --> " << b.m_maxs;
    return os;
}
