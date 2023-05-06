// #pragma once
#ifndef COMMEN_H
#define COMMEN_H


// tensorrt
#include<logger.h>
#include<sampleUtils.h>
#include<parserOnnxConfig.h>
#include<NvInfer.h>
// #include "NvInferPlugin.h"

// cuda
#include<cuda_runtime.h>
// #include<stdio.h>
// #include <thrust/sort.h>
// #include<math.h>
// #include<cuda_device_runtime_api.h>
#include<cuda_runtime_api.h>
// #include<device_launch_parameters.h>
// #include<device_atomic_functions.h>

// opencv
#include<opencv2/opencv.hpp>

// cpp std
#include<algorithm>
#include<cstdlib>
#include <cstring>
#include<math.h>
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

#ifndef CUDA_CHECK
#define CUDA_CHECK(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);
#endif


std::vector<std::string> get_names(const std::string &image_path);
std::string replace(std::string str, const std::string& from, const std::string& to);
int check_dir(const std::string &path, const bool is_mkdir);

#endif