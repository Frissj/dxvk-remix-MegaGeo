
#pragma once

// clang-format off

#include <cassert>
#include <filesystem>
#include <string>

#include <donut/core/math/math.h>

using namespace donut::math;

namespace Json
{
    class Value;
}

Json::Value readFile(const std::filesystem::path& m_filepath);

template <typename T> T read(const Json::Value& node, const T& defaultValue) { assert(false); return T{}; }

template <> std::string read<std::string>(const Json::Value& node, const std::string& defaultValue);

template <> bool read<bool>(const Json::Value& node, const bool& defaultValue);

template <> int8_t read<int8_t>(const Json::Value& node, const int8_t& defaultValue);
template <> int16_t read<int16_t>(const Json::Value& node, const int16_t& defaultValue);
template <> int32_t read<int32_t>(const Json::Value& node, const int32_t& defaultValue);
template <> int2 read<int2>(const Json::Value& node, const int2& defaultValue);
template <> int3 read<int3>(const Json::Value& node, const int3& defaultValue);
template <> int4 read<int4>(const Json::Value& node, const int4& defaultValue);

template <> uint8_t read<uint8_t>(const Json::Value& node, const uint8_t& defaultValue);
template <> uint16_t read<uint16_t>(const Json::Value& node, const uint16_t& defaultValue);
template <> uint32_t read<uint32_t>(const Json::Value& node, const uint32_t& defaultValue);
template <> uint2 read<uint2>(const Json::Value& node, const uint2& defaultValue);
template <> uint3 read<uint3>(const Json::Value& node, const uint3& defaultValue);
template <> uint4 read<uint4>(const Json::Value& node, const uint4& defaultValue);

template <> float read<float>(const Json::Value& node, const float& defaultValue);
template <> float2 read<float2>(const Json::Value& node, const float2& defaultValue);
template <> float3 read<float3>(const Json::Value& node, const float3& defaultValue);
template <> float4 read<float4>(const Json::Value& node, const float4& defaultValue);

template <> double read<double>(const Json::Value& node, const double& defaultValue);
template <> double2 read<double2>(const Json::Value& node, const double2& defaultValue);
template <> double3 read<double3>(const Json::Value& node, const double3& defaultValue);
template <> double4 read<double4>(const Json::Value& node, const double4& defaultValue);

template<typename T> void operator >> (const Json::Value& node, T& dest)
{
    dest = read<T>(node, dest);
}
// clang-format on
#pragma once
