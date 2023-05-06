#ifndef YOLO_H
#define YOLO_H

#include "detection.h"
#include "instance_segmentation.h"

class YOLO : public Detection {
public:
    explicit YOLO(const YAML::Node &config);
    // virtual void allocateBuffers(std::shared_ptr<nvinfer1::ICudaEngine> engine);
protected:
    std::vector<Detections> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;
};

class YOLO_seg : public InstanceSegmentation {
public:
    explicit YOLO_seg(const YAML::Node &config);

protected:
    std::vector<Segmentations> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;
};

#endif
