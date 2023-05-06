#ifndef BASEMODEL_H
#define BASEMODEL_H

#include "common.h"

class Model
{
public:
    explicit Model(const YAML::Node &config);
    ~Model();
    virtual void LoadEngine();
    // void LoadEngine();
    // virtual inline void allocateBuffers(std::shared_ptr<nvinfer1::ICudaEngine> engine);
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
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;    
    // nvinfer1::ICudaEngine *engine = nullptr;
    // nvinfer1::IExecutionContext *context = nullptr;
    void *buffers[2];
    std::vector<int64_t> bufferSize;
    cudaStream_t stream;
    int outSize;
    std::vector<float> imgMean;
    std::vector<float> imgStd;   
};
#endif