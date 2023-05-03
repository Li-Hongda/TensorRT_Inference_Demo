// #pragma once
#ifndef COMMEN_H
#define COMMEN_H

#ifndef CHECK
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)
#endif

// tensorrt
#include<logger.h>
#include<parserOnnxConfig.h>
#include<NvInfer.h>
// #include "NvInferPlugin.h"

// cuda
#include<cuda_runtime.h>
// #include<stdio.h>
// #include <thrust/sort.h>
// #include<math.h>
// #include<cuda_device_runtime_api.h>
// #include<cuda_runtime_api.h>
// #include<device_launch_parameters.h>
// #include<device_atomic_functions.h>

// opencv
#include<opencv2/opencv.hpp>

// cpp std
#include<algorithm>
#include<cstdlib>
#include <cstring>
// #include <sys/types.h>
// #include <sys/stat.h>
// #include<dirent.h>
#include<numeric>
#include<fstream>
#include<iostream>
#include<sstream>
#include<vector>
#include<map>
// #include <regex>
// #include <glob.h>
// #include <unistd.h>
// #include <unordered_map>
#include<chrono>

#include "yaml-cpp/yaml.h"

constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
{
    return val * (1 << 10);
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

void setReportableSeverity(sample::Severity severity);
std::vector<std::string> ReadFolder(const std::string &image_path);
std::string replace(std::string str, const std::string& from, const std::string& to);
std::map<int, std::string> ReadImageNetLabel(const std::string &fileName);
int CheckDir(const std::string &path);

#endif