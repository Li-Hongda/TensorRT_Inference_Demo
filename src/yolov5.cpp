#include "yolov5.h"

YOLOv5::YOLOv5(const YAML::Node &config) : YOLO(config) {}
// void YOLOv5::allocateBuffers(std::shared_ptr<nvinfer1::ICudaEngine> engine){
//     int a = 1;
// }

YOLOv5_seg::YOLOv5_seg(const YAML::Node &config) : YOLO_seg(config) {}

// void YOLOv5_seg::allocateBuffers(std::shared_ptr<nvinfer1::ICudaEngine> engine){
//     int nbBindings = engine->getNbBindings();
//     bufferSize.resize(nbBindings);
//     for (int i = 0; i < nbBindings; ++i) {
//         nvinfer1::Dims dims = engine->getBindingDimensions(i);
//         nvinfer1::DataType dtype = engine->getBindingDataType(i);
//         names[i] = engine->getBindingName(i);
//         int64_t totalSize = sample::volume(dims) * sample::dataTypeSize(dtype);
//         bufferSize[i] = totalSize;
//         CUDA_CHECK(cudaMalloc(&buffers[i], totalSize));
//     }
//     for(int i = 1; i < nbBindings; i++){
//         outSize += int(bufferSize[i] / sizeof(float) / batchSize);
//     } 
//     std::cout << outSize << std::endl;
// }