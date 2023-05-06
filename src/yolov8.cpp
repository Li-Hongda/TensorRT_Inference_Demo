#include "yolov8.h"

YOLOv8::YOLOv8(const YAML::Node &config) : YOLO(config) {}

YOLOv8_seg::YOLOv8_seg(const YAML::Node &config) : YOLO_seg(config) {}
