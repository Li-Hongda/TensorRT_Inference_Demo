#ifndef BASEMODEL_H
#define BASEMODEL_H

#include "common.h"

class Model
{
public:
    explicit Model(const YAML::Node &config);
    ~Model();
    void LoadEngine();
    virtual void Inference(const std::string &input_path, const std::string &save_path, const bool video) = 0;
    virtual void Inference(const std::string &input_path, const std::string &save_path) = 0;

protected:
    bool ReadTrtFile();
    void OnnxToTRTModel();
    std::vector<float> PreProcess(std::vector<cv::Mat> &image);
    std::string onnx_file;
    std::string engine_file;
    std::string mode;
    int batchSize;
    int inputChannel;
    int imageWidth;
    int imageHeight;
    std::string names[10];
    float **cpu_buffers = new float* [10];
    void *gpu_buffers[10]{};
    std::vector<int64_t> bufferSize;    
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;    
    // nvinfer1::ICudaEngine *engine = nullptr;
    // nvinfer1::IExecutionContext *context = nullptr;
    float kSoloImageMean[3]={123.675, 116.28, 103.53};
    float kSoloImageStd[3]={58.395, 57.12, 57.375};

    cudaStream_t stream;
    std::vector<float> imgMean;
    std::vector<float> imgStd;   
};
#endif