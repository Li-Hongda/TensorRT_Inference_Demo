#ifndef RTDETR_H
#define RTDETR_H

#include "detection.h"
#include "instance_segmentation.h"

class RTDETR : public Detection {
public:
    explicit RTDETR(const YAML::Node &config);
protected:
    std::vector<Detections> PostProcess(const std::vector<cv::Mat> &vec_Mat, float* output) override;
};

#endif
