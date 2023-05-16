#ifndef INSTANCE_SEGMENTATION_H
#define INSTANCE_SEGMENTATION_H

#include "detection.h"

struct Instance {
    float x;
    float y;
    float w;
    float h;
    int label;
    float score;    
    cv::Mat mask;
    float pred_mask[32];
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
    cv::Mat scale_mask(cv::Mat mask, cv::Mat img);

protected:
    virtual std::vector<Segmentations> PostProcess(const std::vector<cv::Mat> &vec_Mat, float* output1, float* output2)=0;
    void NMS(std::vector<Instance> &segmentations);
    uint8_t* cpu_mask_buffer = nullptr;
    int num_classes;
    float conf_thr;
    float nms_thr;
    std::string type;
    std::vector<std::string> class_labels;
    std::vector<cv::Scalar> class_colors;
    std::vector<int> strides;
    int num_rows = 0;
};

#endif