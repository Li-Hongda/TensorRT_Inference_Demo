#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include "basemodel.h"

struct ClassRes{
    int classes;
    float prob;
};

class Classification : public Model
{
public:
    explicit Classification(const YAML::Node &config);
    std::vector<ClassRes> InferenceImages(std::vector<cv::Mat> &imgBatch);
    void InferenceFolder(const std::string &input_path) override;
    void Visualize(const std::vector<ClassRes> &results, std::vector<cv::Mat> &imgBatch,
                     std::vector<std::string> image_names);

protected:
    std::vector<ClassRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output);
    std::map<int, std::string> class_labels;
    int num_classes;
};

#endif //TENSORRT_INFERENCE_CLASSIFICATION_H
