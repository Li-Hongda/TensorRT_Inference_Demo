#include "basemodel.h"
// static float_t* img_buffer_host = nullptr;


Model::Model(const YAML::Node &config) {
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    mode = config["mode"].as<std::string>();
    batchSize = config["batchSize"].as<int>();
    inputChannel = config["inputChannel"].as<int>();
    imageWidth = config["imageWidth"].as<int>();
    imageHeight = config["imageHeight"].as<int>();
    imgMean = config["imgMean"].as<std::vector<float>>();
    imgStd = config["imgStd"].as<std::vector<float>>();
}

Model::~Model() = default;

void Model::OnnxToTRTModel() {
    // create the builder
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
    assert(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    // auto network = builder->createNetworkV2(0U);
    auto config = builder->createBuilderConfig();

    auto parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
    if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()))) {
        sample::gLogError << "Failure while parsing ONNX file" << std::endl;
    }
    // Build the engine
    // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
    if (mode == "fp16")
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    else if  (mode == "int8")
        config->setFlag(nvinfer1::BuilderFlag::kINT8);

    sample::gLogInfo << "start building engine" << std::endl;
    nvinfer1::IHostMemory* data = builder->buildSerializedNetwork(*network, *config);
    sample::gLogInfo << "build engine done" << std::endl;
    assert(data);
    

    std::ofstream file;
    file.open(engine_file, std::ios::binary | std::ios::out);
    sample::gLogInfo << "writing engine file..." << std::endl;
    file.write((const char *) data->data(), data->size());
    sample::gLogInfo << "save engine file done" << std::endl;
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

// inline void Model::allocateBuffers(std::shared_ptr<nvinfer1::ICudaEngine> engine){
//     int nbBindings = engine->getNbBindings();
//     bufferSize.resize(nbBindings);
//     for (int i = 0; i < nbBindings; ++i) {
//         nvinfer1::Dims dims = engine->getBindingDimensions(i);
//         nvinfer1::DataType dtype = engine->getBindingDataType(i);
//         names[i] = engine->getBindingName(i);
//         int64_t totalSize = sample::volume(dims) * sample::dataTypeSize(dtype);
//         bufferSize[i] = totalSize;
//         CUDA_CHECK(cudaMalloc(&buffers[i], totalSize));
//     }
//     outSize = int(bufferSize[1] / sizeof(float) / batchSize);
// }


void Model::LoadEngine(){
    // create and load engine
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        ReadTrtFile();
        // assert(this->engine != nullptr);
    } else {
        OnnxToTRTModel();
        ReadTrtFile();
        assert(this->engine != nullptr);
    }
    this->context = std::unique_ptr<nvinfer1::IExecutionContext>(this->engine->createExecutionContext());
    assert(this->context != nullptr);

    //get buffers
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);
    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        names[i] = engine->getBindingName(i);
        int64_t totalSize = sample::volume(dims) * sample::dataTypeSize(dtype);
        bufferSize[i] = totalSize;
        CUDA_CHECK(cudaMalloc(&buffers[i], totalSize));
    }
    outSize = int(bufferSize[1] / sizeof(float) / batchSize);    
    // allocateBuffers(engine);

    //get stream  
    cudaStreamCreate(&stream);
    // outSize = int(bufferSize[1] / sizeof(float) / batchSize);
}

std::vector<float> Model::PreProcess(std::vector<cv::Mat> &imgBatch) {
    std::vector<float> result(batchSize * imageWidth * imageHeight * inputChannel);
    float *data = result.data();
    for (const cv::Mat &img : imgBatch) {
        if (!img.data)
            continue;
        cv::Mat dst_img;
        if (inputChannel == 1)
            cv::cvtColor(img, dst_img, cv::COLOR_RGB2GRAY);
        else if (inputChannel == 3)
            cv::cvtColor(img, dst_img, cv::COLOR_BGR2RGB);
        float ratio = std::min(float(imageWidth) / float(img.cols), float(imageHeight) / float(img.rows));
        dst_img = cv::Mat::zeros(cv::Size(imageWidth, imageHeight), CV_8UC3);
        cv::Mat rsz_img;
        cv::resize(img, rsz_img, cv::Size(), ratio, ratio);
        rsz_img.copyTo(dst_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
        dst_img.convertTo(dst_img, CV_32F, 1 / 255.0);
        std::vector<cv::Mat> split_img(inputChannel);
        cv::split(dst_img, split_img);

        int channelLength = imageWidth * imageHeight;
        for (int i = 0; i < inputChannel; ++i) {
            split_img[i] = (split_img[i] - imgMean[i]) / imgStd[i];
            memcpy(data, split_img[i].data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
    return result;
}

