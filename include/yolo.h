#ifndef YOLO_H
#define YOLO_H

#include "detection.h"
#include "instance_segmentation.h"

class YOLO : public Detection {
public:
    explicit YOLO(const YAML::Node &config);
    std::vector<Detections> InferenceImages(std::vector<cv::Mat> &imgBatch) noexcept;
protected:
    std::vector<Detections> PostProcess(const std::vector<cv::Mat> &vec_Mat, float* output) override;
    float nms_thr;
    std::vector<int> strides;
};

class YOLO_seg : public InstanceSegmentation {
public:
    explicit YOLO_seg(const YAML::Node &config);
    std::vector<Segmentations> InferenceImages(std::vector<cv::Mat> &imgBatch) noexcept;
protected:
    std::vector<Segmentations> PostProcess(const std::vector<cv::Mat> &vec_Mat, float* output1, float* output2) override;
    float nms_thr;
    std::vector<int> strides;
};

#endif
