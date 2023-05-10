#include "yolov8.h"

YOLOv8::YOLOv8(const YAML::Node &config) : YOLO(config) {
    int index = 0;
    num_rows = 0;
    for (const int &stride : strides)
    {
        num_rows += int(imageHeight / stride) * int(imageWidth / stride);
        index+=1;
    }     
}

std::vector<Detections> YOLOv8::PostProcess(const std::vector<cv::Mat> &imgBatch, float *output) {
    std::vector<Detections> vec_result;
    int index = 0;
    auto predSize = bufferSize[1] / sizeof(float);
    for (const cv::Mat &img : imgBatch)
    {
        Detections result;
        float ratio = std::max(float(img.cols) / float(imageWidth), float(img.rows) / float(imageHeight));
        float *pred = output + index * predSize;
        for (int position = 0; position < num_rows; position++) {
            float *row = pred + position * (num_classes + 4);
            Box box;
            auto max_pos = std::max_element(row + 4, row + num_classes + 4);
            if (*max_pos < obj_threshold) continue;
            box.score = row[max_pos - row];
            box.label = max_pos - row - 4;
            box.x = row[0] * ratio;
            box.y = row[1] * ratio;
            box.w = row[2] * ratio;
            box.h = row[3] * ratio;
            // box = regularization(box, img.cols, img.rows);
            result.dets.emplace_back(box);
        }
        NMS(result.dets);
        vec_result.emplace_back(result);
        index++;
    }
    return vec_result;
}

YOLOv8_seg::YOLOv8_seg(const YAML::Node &config) : YOLO_seg(config) {}
