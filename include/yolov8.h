#ifndef YOLOV8_H
#define YOLOV8_H

#include "yolo.h"

class YOLOv8 : public YOLO {
public:
    explicit YOLOv8(const YAML::Node &config);
};

class YOLOv8_seg : public YOLO_seg {
public:
    explicit YOLOv8_seg(const YAML::Node &config);
};

#endif