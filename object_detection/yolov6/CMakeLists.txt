cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "Release")

project(yolov6)

set(CMAKE_CXX_STANDARD 14)

# YAML
set(YAML_LIBRARY ../../yaml-cpp/build)
set(YAML_INCLUDE ../../yaml-cpp/include)
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
                    ../../src/yolov6.cpp
                    ../../src/cuda_function.cu)

include_directories(${ALL_INCLUDE})

set(LIBRARY_OUTPUT_PATH ../../libs)
add_library(yolov6 SHARED ${PROJECT_LIBRARY} ${SAMPLES_LIBRARY})
target_link_libraries(yolov6 ${ALL_LIBS} yaml-cpp)
