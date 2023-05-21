#include "yolov6.h"

YOLOv6::YOLOv6(const YAML::Node &config) : YOLO(config) {
    int index = 0;
    num_bboxes = 0;
    for (const int &stride : strides)
    {
        num_bboxes += int(imageHeight / stride) * int(imageWidth / stride);
        index+=1;
    }
}
