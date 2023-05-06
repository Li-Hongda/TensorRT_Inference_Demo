#include "yolo.h"


YOLO::YOLO(const YAML::Node &config) : Detection(config) {}

// void YOLO::allocateBuffers(std::shared_ptr<nvinfer1::ICudaEngine> engine){
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
// }

std::vector<Detections> YOLO::PostProcess(const std::vector<cv::Mat> &imgBatch, float *output) {
    std::vector<Detections> vec_result;
    int index = 0;
    for (const cv::Mat &img : imgBatch)
    {
        Detections result;
        float ratio = std::max(float(img.cols) / float(imageWidth), float(img.rows) / float(imageHeight));

        float *pred = output + index * outSize;
        for (int position = 0; position < num_rows; position++) {
            float *row = pred + position * (num_classes + 5);
            Box box;
            if (row[4] < obj_threshold)
                continue;
            auto max_pos = std::max_element(row + 5, row + num_classes + 5);
            box.score = row[4] * row[max_pos - row];
            box.label = max_pos - row - 5;
            box.x = row[0] * ratio;
            box.y = row[1] * ratio;
            box.w = row[2] * ratio;
            box.h = row[3] * ratio;
            result.dets.emplace_back(box);
        }
        NMS(result.dets);
        vec_result.emplace_back(result);
        index++;
    }
    return vec_result;
}

YOLO_seg::YOLO_seg(const YAML::Node &config) : InstanceSegmentation(config) {}
std::vector<Segmentations> YOLO_seg::PostProcess(const std::vector<cv::Mat> &imgBatch, float *output) {
    std::vector<Segmentations> vec_result;
    int index = 0;
    for (const cv::Mat &img : imgBatch)
    {
        Segmentations result;
        float ratio = std::max(float(img.cols) / float(imageWidth), float(img.rows) / float(imageHeight));

        float *pred = output + index * outSize;
        // float *proto = output + index * 
        for (int position = 0; position < num_rows; position++) {
            float *row = pred + position * (num_classes + 5);
            // Box box;
            // if (row[4] < obj_threshold)
            //     continue;
            // auto max_pos = std::max_element(row + 5, row + num_classes + 5);
            // box.score = row[4] * row[max_pos - row];
            // box.label = max_pos - row - 5;
            // box.x = row[0] * ratio;
            // box.y = row[1] * ratio;
            // box.w = row[2] * ratio;
            // box.h = row[3] * ratio;
            // result.segs.emplace_back(box);
        }
        NMS(result.segs);
        vec_result.emplace_back(result);
        index++;
    }
    return vec_result;
}

