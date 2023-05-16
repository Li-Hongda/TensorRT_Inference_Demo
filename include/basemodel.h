#ifndef BASEMODEL_H
#define BASEMODEL_H

#include "common.h"
#include "cuda_function.h"

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
    void PreProcess(std::vector<cv::Mat> &image);
    std::string onnx_file;
    std::string engine_file;
    std::string mode;
    AffineMatrix dst2src;
    int batchSize;
    int inputChannel;
    int imageWidth;
    int imageHeight;
    std::string names[10];
    float** cpu_buffers = new float* [10];
    float* gpu_buffers[10]{};
    // float* cpu_mask_buffer = nullptr;
    std::vector<int64_t> bufferSize;    
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream;
    std::vector<float> imgMean;
    std::vector<float> imgStd;   
};
#endif