#ifndef YOLOV7_H
#define YOLOV7_H

#include "yolo.h"

class YOLOv7 : public YOLO {
public:
    explicit YOLOv7(const YAML::Node &config);
};

#endif