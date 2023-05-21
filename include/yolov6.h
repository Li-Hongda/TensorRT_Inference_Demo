#ifndef YOLOV6_H
#define YOLOV6_H

#include "yolov8.h"

class YOLOv6 : public YOLOv8 {
public:
    explicit YOLOv6(const YAML::Node &config);
};

#endif