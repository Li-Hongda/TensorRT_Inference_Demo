#ifndef YOLOV8_H
#define YOLOV8_H

#include "yolo.h"

class YOLOv8 : public YOLO {
public:
    explicit YOLOv8(const YAML::Node &config);
    std::vector<Detections> PostProcess(const std::vector<cv::Mat> &imgBatch, float *output);
};

class YOLOv8_seg : public YOLO_seg {
public:
    explicit YOLOv8_seg(const YAML::Node &config);
protected:    
    std::vector<Segmentations> PostProcess(const std::vector<cv::Mat> &imgBatch, float *output1, float *output2);
};

#endif