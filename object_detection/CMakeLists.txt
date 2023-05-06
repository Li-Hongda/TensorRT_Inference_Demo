cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")

project(object_detection)

set(CMAKE_CXX_STANDARD 14)

#CUDA
find_package(CUDA REQUIRED)
list(APPEND ALL_INCLUDE ${CUDA_INCLUDE_DIRS})
message(STATUS "Find CUDA headers at ${CUDA_INCLUDE_DIRS}")

#TensorRT
set(TENSORRT_ROOT /home/lhd/downloads/TensorRT-8.6.0.12)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h 
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include)     
list(APPEND ALL_INCLUDE ${TENSORRT_INCLUDE_DIR})

set(SAMPLES_INCLUDE_DIR ${TENSORRT_ROOT}/samples/common)
list(APPEND ALL_INCLUDE ${SAMPLES_INCLUDE_DIR})
message(STATUS "Find TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Find TensorRT sample headers at ${SAMPLES_INCLUDE_DIR}")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV headers at ${OpenCV_INCLUDE_DIRS}")
list(APPEND ALL_INCLUDE ${OpenCV_INCLUDE_DIRS})

# YAML
set(YAML_LIBRARY ../yaml-cpp-yaml-cpp-0.6.3/build)
set(YAML_INCLUDE ../yaml-cpp-yaml-cpp-0.6.3/include)
link_directories(${YAML_LIBRARY})
list(APPEND ALL_INCLUDE ${YAML_INCLUDE})


# Project
set(PROJECT_INCLUDE  ../include)

message(STATUS "Find project include at ${PROJECT_INCLUDE}")
list(APPEND ALL_INCLUDE ${PROJECT_INCLUDE})

include_directories(${ALL_INCLUDE})


# add_subdirectory(nanodet)
# add_subdirectory(Swin-Transformer)
add_subdirectory(yolov5)
# add_subdirectory(YOLOv6)
add_subdirectory(yolov7)
add_subdirectory(yolov8)
set(LIBRARY_OUTPUT_PATH ../libs)
add_library(build SHARED ../src/build.cpp)

set(EXECUTABLE_OUTPUT_PATH ../../bin)
add_executable(object_detection main.cpp)
target_link_libraries(object_detection yaml-cpp build yolov5 yolov7 yolov8)

