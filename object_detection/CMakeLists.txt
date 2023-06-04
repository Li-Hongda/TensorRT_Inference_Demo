cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")

project(object_detection)

set(CMAKE_CXX_STANDARD 14)

#CUDA
set(CMAKE_CUDA_COMPILER /home/lhd/cuda-11.1/bin/nvcc)
enable_language(CUDA)
find_package(CUDA REQUIRED)
link_directories(/home/lhd/cuda-11.1/lib64)
list(APPEND ALL_INCLUDE ${CUDA_INCLUDE_DIRS})
message(STATUS "Find CUDA headers at ${CUDA_INCLUDE_DIRS}")

#TensorRT
set(TENSORRT_ROOT /home/lhd/downloads/TensorRT-8.6.0.12)
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD}
        PATH_SUFFIXES lib lib64 lib/x64)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h 
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include)     
list(APPEND ALL_LIBS ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER} ${TENSORRT_LIBRARY_PLUGIN} )
list(APPEND ALL_INCLUDE ${TENSORRT_INCLUDE_DIR})

file(GLOB SAMPLES_LIBRARY
        ${TENSORRT_ROOT}/samples/common/logger.cpp 
        ${TENSORRT_ROOT}/samples/common/sampleOptions.cpp
        ${TENSORRT_ROOT}/samples/common/sampleUtils.cpp)
set(SAMPLES_INCLUDE_DIR ${TENSORRT_ROOT}/samples/common)
list(APPEND ALL_INCLUDE ${SAMPLES_INCLUDE_DIR})
message(STATUS "Find TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Find TensorRT sample headers at ${SAMPLES_INCLUDE_DIR}")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV headers at ${OpenCV_INCLUDE_DIRS}")
list(APPEND ALL_LIBS ${OpenCV_LIBRARIES})
list(APPEND ALL_INCLUDE ${OpenCV_INCLUDE_DIRS})

# YAML
set(YAML_LIBRARY ../yaml-cpp/build)
set(YAML_INCLUDE ../yaml-cpp/include)
link_directories(${YAML_LIBRARY})
list(APPEND ALL_INCLUDE ${YAML_INCLUDE})


# Project
set(PROJECT_INCLUDE  ../include)

message(STATUS "Find project include at ${PROJECT_INCLUDE}")
list(APPEND ALL_INCLUDE ${PROJECT_INCLUDE})

include_directories(${ALL_INCLUDE})

add_subdirectory(yolov5)
add_subdirectory(yolov6)
add_subdirectory(yolov7)
add_subdirectory(yolov8)
add_subdirectory(yolox)
add_subdirectory(rtdetr)
set(LIBRARY_OUTPUT_PATH ../libs)
add_library(build SHARED ../src/build.cpp)

set(EXECUTABLE_OUTPUT_PATH ../../bin)
add_executable(object_detection main.cpp)
target_link_libraries(object_detection yaml-cpp build yolov5 yolov6 yolov7 yolov8 yolox rtdetr cudart)

