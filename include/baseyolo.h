#ifndef YOLO_H
#define YOLO_H

#include "detection.h"

class YOLO : public Detection {
public:
    explicit YOLO(const YAML::Node &config);

protected:
    std::vector<Detections> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;
};

#endif
