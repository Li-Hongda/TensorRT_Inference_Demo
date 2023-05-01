#ifndef BASEMODEL_H
#define BASEMODEL_H

#include "common.h"

class Model
{
public:
    explicit Model(const YAML::Node &config);
    ~Model();
    void LoadEngine();
//    virtual float *InferenceImage(std::vector<float> image_data) = 0;
    virtual void InferenceFolder(const std::string &folder_name) = 0;

protected:
    bool ReadTrtFile();
    void OnnxToTRTModel();
    std::vector<float> PreProcess(std::vector<cv::Mat> &image);
    void ModelInference(std::vector<float> image_data, float *output);
    std::string onnx_file;
    std::string engine_file;
    std::string labels_file;
    bool video;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    void *buffers[2];
    std::vector<int64_t> bufferSize;
    cudaStream_t stream;
    int outSize;
    std::vector<float> img_mean;
    std::vector<float> img_std;
    std::string resize;
};
#endif