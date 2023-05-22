#include "basemodel.h"

Model::Model(const YAML::Node &config) {
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    mode = config["mode"].as<std::string>();
    batchSize = config["batchSize"].as<int>();
    imageWidth = config["imageWidth"].as<int>();
    imageHeight = config["imageHeight"].as<int>();
    imgMean = config["imgMean"].as<std::vector<float>>();
    imgStd = config["imgStd"].as<std::vector<float>>();
}

Model::~Model() {
    cudaStreamDestroy(stream);
    for (int i = 0; i < engine->getNbBindings(); i++){
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    }
};

void Model::OnnxToTRTModel() {
    // create the builder
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
    assert(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    auto config = builder->createBuilderConfig();

    auto parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
    if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()))) {
        sample::gLogError << "Failure while parsing ONNX file" << std::endl;
    }

    // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
    if (mode == "fp16")
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    else if  (mode == "int8")
        config->setFlag(nvinfer1::BuilderFlag::kINT8);

    nvinfer1::IHostMemory* data = builder->buildSerializedNetwork(*network, *config);
    assert(data);
    
    std::ofstream file;
    file.open(engine_file, std::ios::binary | std::ios::out);
    file.write((const char *) data->data(), data->size());
    file.close();

    delete parser;
    delete config;
    delete network;
    delete builder;
}

bool Model::ReadTrtFile() {
    std::string cached_engine;
    std::fstream file;
    // sample::gLogInfo << "loading filename from:" << engine_file << std::endl;
    nvinfer1::IRuntime *trtRuntime;
    file.open(engine_file, std::ios::binary | std::ios::in);

    if (!file.is_open()) {
        sample::gLogInfo << "read file error: " << engine_file << std::endl;
        cached_engine = "";
    }

    while (file.peek() != EOF) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        cached_engine.append(buffer.str());
    }
    file.close();

    trtRuntime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
    this->engine = std::unique_ptr<nvinfer1::ICudaEngine>(trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr));
    // sample::gLogInfo << "deserialize done" << std::endl;
}

void Model::LoadEngine(){
    // create and load engine
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        ReadTrtFile();
    } else {
        OnnxToTRTModel();
        ReadTrtFile();
        assert(this->engine != nullptr);
    }
    this->context = std::unique_ptr<nvinfer1::IExecutionContext>(this->engine->createExecutionContext());
    assert(this->context != nullptr);

    //get gpu_buffers
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);
    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        names[i] = engine->getBindingName(i);
        int64_t totalSize = sample::volume(dims) * sample::dataTypeSize(dtype);
        cpu_buffers[i] = (float* )malloc(totalSize);
        bufferSize[i] = totalSize;
        CUDA_CHECK(cudaMalloc(&gpu_buffers[i], totalSize));
    }
    //get stream  
    cudaStreamCreate(&stream);
}

void Model::PreProcess(std::vector<cv::Mat>& img_batch) {
    dst2src.reserve(batchSize);
    for (size_t i = 0; i < img_batch.size(); i++) {
        int height = img_batch[i].rows; 
        int width = img_batch[i].cols;
        float scale = std::min(imageHeight / height, imageWidth / width);
        cv::Mat s2d = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * width + imageWidth + scale - 1) * 0.5,
        0.f, scale, (-scale * height + imageHeight + scale - 1) * 0.5);
        cv::Mat d2s = cv::Mat::zeros(2, 3, CV_32FC1);
        cv::invertAffineTransform(s2d, d2s);
        AffineMatrix mat;
        memcpy(&mat, d2s.ptr(), sizeof(mat));
        dst2src.emplace_back(mat);
        preprocess(img_batch[i].ptr(), mat, width, height, &gpu_buffers[0][bufferSize[0] * i], imageWidth, imageHeight, stream); 
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}