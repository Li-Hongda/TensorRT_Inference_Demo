cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")

project(yolov5)

set(CMAKE_CXX_STANDARD 14)

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
                    ../../src/instance_segmentation.cpp
                    ../../src/yolo.cpp 
                    ../../src/yolov5.cpp
                    ../../src/cuda_function.cu)

include_directories(${ALL_INCLUDE})

set(LIBRARY_OUTPUT_PATH ../../libs)
add_library(yolov5 SHARED ${PROJECT_LIBRARY} ${SAMPLES_LIBRARY})
target_link_libraries(yolov5 ${ALL_LIBS} yaml-cpp)
