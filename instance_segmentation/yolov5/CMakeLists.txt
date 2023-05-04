cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")

project(yolov5)

set(CMAKE_CXX_STANDARD 14)

#CUDA
find_package(CUDA REQUIRED)
list(APPEND ALL_LIBS 
  ${CUDA_LIBRARIES} 
  ${CUDA_cublas_LIBRARY} 
  ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} ${CUDA_nppidei_LIBRARY} ${CUDA_nppial_LIBRARY})
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
# message(STATUS "Find OpenCV librays at ${OpenCV_LIBRARIES}")
list(APPEND ALL_LIBS ${OpenCV_LIBRARIES})
list(APPEND ALL_INCLUDE ${OpenCV_INCLUDE_DIRS})


# YAML
set(YAML_LIBRARY ../../yaml-cpp-yaml-cpp-0.6.3/build)
set(YAML_INCLUDE ../../yaml-cpp-yaml-cpp-0.6.3/include)
link_directories(${YAML_LIBRARY})
list(APPEND ALL_INCLUDE ${YAML_INCLUDE})


# Project
set(PROJECT_INCLUDE  ../../include)
message(STATUS "Find project include at ${PROJECT_INCLUDE}")
list(APPEND ALL_INCLUDE ${PROJECT_INCLUDE})
set(PROJECT_LIBRARY ../../src/common.cpp  
                    ../../src/basemodel.cpp 
                    ../../src/detection.cpp 
                    ../../src/yolo.cpp 
                    ../../src/yolov5.cpp)

include_directories(${ALL_INCLUDE})

set(LIBRARY_OUTPUT_PATH ../../libs)
add_library(yolov5 SHARED ${PROJECT_LIBRARY} ${SAMPLES_LIBRARY})
target_link_libraries(yolov5 ${ALL_LIBS} yaml-cpp)

# set(EXECUTABLE_OUTPUT_PATH ../../bin)
# add_executable(yolov5 main.cpp)
# target_link_libraries(yolov5 det ${ALL_LIBS} yaml-cpp)





