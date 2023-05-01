#ifndef YOLOV5_H
#define YOLOV5_H

#include "baseyolo.h"

class YOLOv5 : public YOLO {
public:
    explicit YOLOv5(const YAML::Node &config);
};

// class YOLOv5_cls :public Classification {
// public:
//     explicit YOLOv5_cls(const YAML::Node &config);
// };
#endif
