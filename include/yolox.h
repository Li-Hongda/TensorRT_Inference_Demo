#ifndef YOLOX_H
#define YOLOX_H

#include "yolov8.h"

class YOLOX : public YOLO {
public:
    explicit YOLOX(const YAML::Node &config);
};

#endif