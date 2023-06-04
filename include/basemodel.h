#ifndef BASEMODEL_H
#define BASEMODEL_H

#include "common.h"
#include "cuda_function.h"

class Model {
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
    int dynamic;
    int batchSize;
    int imageWidth;
    int imageHeight;
    float* cpu_buffer;
    float* gpu_buffers[10]{};
    std::vector<int64_t> bufferSize;    
    std::vector<AffineMatrix> dst2src;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream;
    Norm norm;
    // float imgScale;
    // std::vector<float> imgMean;
    // std::vector<float> imgStd;   
};
#endif