#include "basemodel.h"

Model::Model(const YAML::Node &config) {
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    mode = config["mode"].as<std::string>();
    dynamic = config["dynamic"].as<int>();
    batchSize = config["batchSize"].as<int>();
    imageWidth = config["imageWidth"].as<int>();
    imageHeight = config["imageHeight"].as<int>();
    auto imgMean = config["imgMean"].as<std::vector<float>>();
    auto imgStd = config["imgStd"].as<std::vector<float>>();
    auto imgScale = config["imgScale"].as<float>();
    memcpy(norm.mean, imgMean.data(), sizeof(float) * 3);
    memcpy(norm.std, imgStd.data(), sizeof(float) * 3);
    norm.scale = imgScale;
}

Model::~Model() {
    cudaStreamDestroy(stream);
    for (int i = 0; i < engine->getNbBindings(); i++) {
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    }
};

void Model::OnnxToTRTModel() {
    // create the builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
    assert(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    auto config = builder->createBuilderConfig();
    if (dynamic) {
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1,3,imageWidth,imageHeight));
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(8,3,imageWidth,imageHeight));
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(32,3,imageWidth,imageHeight));

        config->addOptimizationProfile(profile);
    }


    auto parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
    if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()))) {
        sample::gLogError << "Failure while parsing ONNX file" << std::endl;
    }

    // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
    if (mode == "fp16")
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    else if  (mode == "int8")
        // TODO: support int8 calibrate
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

void Model::LoadEngine() {
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
        if (dims.d[0] == -1)
            dims.d[0] = batchSize;
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = sample::volume(dims) * sample::dataTypeSize(dtype);
        bufferSize[i] = totalSize;
        CUDA_CHECK(cudaMalloc(&gpu_buffers[i], totalSize));
    }
    cpu_buffer = (float* )malloc(1000 * 100 * sizeof(float));
    if (dynamic) {
        this->context->setOptimizationProfile(0);
        this->context->setBindingDimensions(0, nvinfer1::Dims4(batchSize, 3, imageHeight, imageWidth));        
    }
    //get stream  
    cudaStreamCreate(&stream);
}

void Model::PreProcess(std::vector<cv::Mat>& img_batch) {
    int size = imageWidth * imageHeight * 3;
    dst2src.reserve(batchSize);
    for (size_t i = 0; i < img_batch.size(); i++) {
        int height = img_batch[i].rows; 
        int width = img_batch[i].cols;
        auto h_scale = static_cast<float>(imageHeight) / static_cast<float>(height);
        float w_scale = static_cast<float>(imageWidth) / static_cast<float>(width);
        float scale = std::min(h_scale, w_scale);
        cv::Mat s2d = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * width + imageWidth + scale - 1) * 0.5,
        0.f, scale, (-scale * height + imageHeight + scale - 1) * 0.5);
        cv::Mat d2s = cv::Mat::zeros(2, 3, CV_32FC1);
        cv::invertAffineTransform(s2d, d2s);
        AffineMatrix mat;
        memcpy(&mat, d2s.ptr(), sizeof(mat));
        dst2src.emplace_back(mat);
        preprocess(img_batch[i].ptr(), mat, width, height, &gpu_buffers[0][size * i], imageWidth, imageHeight, norm, stream); 
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}
