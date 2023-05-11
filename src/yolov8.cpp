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
    for (const cv::Mat &img : imgBatch){
        Detections result;
        float ratio = std::max(float(img.cols) / float(imageWidth), float(img.rows) / float(imageHeight));
        float *pred_per_img = output + index * predSize;
        for (int position = 0; position < num_rows; position ++){
            float *pred_per_obj = new float[84];
            for (int pos = 0; pos < 84; pos++){
                memcpy((pred_per_obj + pos), pred_per_img + position + pos * 8400, sizeof(float));
            }
            Box box;
            auto max_pos = std::max_element(pred_per_obj + 4, pred_per_obj + num_classes + 4); 
            if (*max_pos < obj_threshold) continue;      
            box.score = pred_per_obj[max_pos - pred_per_obj];
            box.label = max_pos - pred_per_obj - 4;
            box.x = pred_per_obj[0] * ratio;
            box.y = pred_per_obj[1] * ratio;
            box.w = pred_per_obj[2] * ratio;
            box.h = pred_per_obj[3] * ratio;
            result.dets.emplace_back(box);
            delete[] pred_per_obj;
        }
        NMS(result.dets);
        vec_result.emplace_back(result);
        index++;
    }       
    return vec_result;
}

YOLOv8_seg::YOLOv8_seg(const YAML::Node &config) : YOLO_seg(config) {}
