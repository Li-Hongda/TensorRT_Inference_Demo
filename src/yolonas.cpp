#include "yolonas.h"

YOLONAS::YOLONAS(const YAML::Node &config) : YOLOv8(config) {
    num_bboxes = 0;
    for (const int &stride : strides) {
        num_bboxes += int(imageHeight / stride) * int(imageWidth / stride);
    }
}

std::vector<Detections> YOLONAS::PostProcess(const std::vector<cv::Mat> &imgBatch, float* output) {
    std::vector<Detections> vec_result;
    int index = 0;
    auto predSize = bufferSize[1] / batchSize / sizeof(float);
    for (const cv::Mat &img : imgBatch) {
        Detections result;
        float* pred_per_img = output + index * predSize;
        cuda_postprocess_init(7, imageWidth, imageHeight);
        yolonas_postprocess_box(pred_per_img, num_bboxes, num_classes, 7, conf_thr, nms_thr, dst2src[index], stream, cpu_buffer);
        int num_boxes = std::min((int)cpu_buffer[0], 1000);
        for (int i = 0; i < num_boxes; i++) {
            Box box;
            float* ptr = cpu_buffer + 1 + 7 * i;
            if (!ptr[6]) continue;
            box.x = ptr[0];
            box.y = ptr[1];
            box.w = ptr[2];
            box.h = ptr[3];
            box.score = ptr[4];
            box.label = ptr[5];            
            result.dets.emplace_back(box);
        }
        vec_result.emplace_back(result);
        index++;
    }
    dst2src.clear();        
    return vec_result;
}