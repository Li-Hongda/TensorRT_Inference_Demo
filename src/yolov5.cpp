#include "yolov5.h"

YOLOv5::YOLOv5(const YAML::Node &config) : YOLO(config) {}

YOLOv5_seg::YOLOv5_seg(const YAML::Node &config) : YOLO_seg(config) {}
