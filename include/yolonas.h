#ifndef YOLONAS_H
#define YOLONAS_H

#include "yolov8.h"

class YOLONAS : public YOLOv8 {
public:
    explicit YOLONAS(const YAML::Node &config);
    std::vector<Detections> PostProcess(const std::vector<cv::Mat> &imgBatch, float* output);
};

#endif