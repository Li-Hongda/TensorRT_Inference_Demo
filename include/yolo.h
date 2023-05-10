#ifndef YOLO_H
#define YOLO_H

#include "detection.h"
#include "instance_segmentation.h"

class YOLO : public Detection {
public:
    explicit YOLO(const YAML::Node &config);
protected:
    std::vector<Detections> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;
};

class YOLO_seg : public InstanceSegmentation {
public:
    explicit YOLO_seg(const YAML::Node &config);
    cv::Rect get_downscale_rect(float bbox[4], float scale);
    std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets);
protected:
    std::vector<Segmentations> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output1, float *output2) override;
};

#endif
