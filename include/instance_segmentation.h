#ifndef INSTANCE_SEGMENTATION_H
#define INSTANCE_SEGMENTATION_H

#include "detection.h"

struct Instance:Box {
    cv::Mat mask;
    // float mask[32];
};

struct Segmentations {
    std::vector<Instance> segs;
};

class InstanceSegmentation : public Model
{
public:
    explicit InstanceSegmentation(const YAML::Node &config);
    std::vector<Segmentations> InferenceImages(std::vector<cv::Mat> &imgBatch);
    void Inference(const std::string &input_path, const std::string &save_path, const bool video) override;
    virtual void Inference(const std::string &input_path, const std::string &save_path) override;
    void Visualize(const std::vector<Segmentations> &segmentations, std::vector<cv::Mat> &imgBatch,
                     std::vector<std::string> image_names);
    void Visualize(const std::vector<Segmentations> &segmentations, std::vector<cv::Mat> &imgBatch,
                     cv::String save_name, int fps, cv::Size size); 
    static float DIoU(const Instance &det_a, const Instance &det_b);

protected:
    virtual std::vector<Segmentations> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output1, float *output2)=0;
    cv::Mat scale_mask(cv::Mat mask, cv::Mat img);
    void NMS(std::vector<Instance> &segmentations);
    int num_classes;
    float obj_threshold;
    float nms_threshold;
    std::string type;
    std::vector<std::string> class_labels;
    std::vector<cv::Scalar> class_colors;
    std::vector<int> strides;
    int num_rows = 0;
};

#endif