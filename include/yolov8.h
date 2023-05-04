#ifndef YOLOV8_H
#define YOLOV8_H

#include "baseyolo.h"

class YOLOv8 : public YOLO {
public:
    explicit YOLOv8(const YAML::Node &config);
};

#endif