#ifndef YOLOV5_H
#define YOLOV5_H

#include "yolo.h"

class YOLOv5 : public YOLO {
public:
    explicit YOLOv5(const YAML::Node &config);
};

class YOLOv5_seg :public YOLO_seg {
public:
    explicit YOLOv5_seg(const YAML::Node &config);
};
#endif
