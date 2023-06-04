#include "yolox.h"

YOLOX::YOLOX(const YAML::Node &config) : YOLO(config) {
    num_bboxes = 0;
    for (const int &stride : strides) {
        num_bboxes += int(imageHeight / stride) * int(imageWidth / stride);
    }
}
