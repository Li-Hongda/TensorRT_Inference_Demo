#ifndef BUILD_H
#define BUILD_H


// #include "Swin-Transformer.h"
#include "yolov5.h"
// #include "YOLOv6.h"
#include "yolov7.h"
#include "yolov8.h"
#include "rtdetr.h"

std::shared_ptr<Model> build_model(std::string model_arch, std::string cfg);
// char **argv
#endif
