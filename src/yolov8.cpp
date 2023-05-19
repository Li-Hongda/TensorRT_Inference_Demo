#include "yolov8.h"

YOLOv8::YOLOv8(const YAML::Node &config) : YOLO(config) {
    int index = 0;
    num_bboxes = 0;
    for (const int &stride : strides)
    {
        num_bboxes += int(imageHeight / stride) * int(imageWidth / stride);
        index+=1;
    }     
}

std::vector<Detections> YOLOv8::PostProcess(const std::vector<cv::Mat> &imgBatch, float* output) {
    std::vector<Detections> vec_result;
    int index = 0;
    auto predSize = bufferSize[1] / sizeof(float);
    for (const cv::Mat &img : imgBatch) {
        Detections result;
        float* pred_per_img = output + index * predSize;
        cuda_postprocess_init(7, imageWidth, imageHeight);
        yolov8_postprocess_box(pred_per_img, num_bboxes, num_classes, 7, conf_thr, nms_thr, stream, cpu_buffers[1]);
        int num_boxes = std::min((int)cpu_buffers[1][0], 1000);
        for (int i = 0; i < num_boxes; i++) {
            Box box;
            float* ptr = cpu_buffers[1] + 1 + 7 * i;
            if (!ptr[6]) continue;
            auto l = ptr[0];
            auto t = ptr[1];
            auto r = ptr[0] + ptr[2];
            auto b = ptr[1] + ptr[3];
            auto new_l = dst2src.v0 * l + dst2src.v1 * t + dst2src.v2;
            auto new_r = dst2src.v0 * r + dst2src.v1 * b + dst2src.v2;
            auto new_t = dst2src.v3 * l + dst2src.v4 * t + dst2src.v5;
            auto new_b = dst2src.v3 * r + dst2src.v4 * b + dst2src.v5;
            box.x = new_l;
            box.y = new_t;
            box.w = new_r - new_l;
            box.h = new_b - new_t;                                
            box.score = ptr[4];
            box.label = ptr[5];
            result.dets.emplace_back(box);
        }
        vec_result.emplace_back(result);
        index++;
    }    
    return vec_result;
}


YOLOv8_seg::YOLOv8_seg(const YAML::Node &config) : YOLO_seg(config) {
    int index = 0;
    num_bboxes = 0;
    for (const int &stride : strides)
    {
        num_bboxes += int(imageHeight / stride) * int(imageWidth / stride);
        index+=1;
    }   
}

std::vector<Segmentations> YOLOv8_seg::PostProcess(const std::vector<cv::Mat> &imgBatch, float* output1, float* output2) {
    std::vector<Segmentations> vec_result;
    int index = 0;
    auto protoSize = bufferSize[1] / sizeof(float);
    auto predSize = bufferSize[2] / sizeof(float);
    cuda_postprocess_init(39, imageWidth, imageHeight);
    for (const cv::Mat &img : imgBatch) {
        Segmentations result;
        float* proto = output1 + index * protoSize;
        float* pred_per_img = output2 + index * predSize;
        yolov8_postprocess_box_mask(pred_per_img, num_bboxes, num_classes, 39, conf_thr, nms_thr, stream, cpu_buffers[1]);
        int num_boxes = std::min((int)cpu_buffers[1][0], 1000);
        for (int i = 0; i < num_boxes; i++) {
            Instance ins;
            float* ptr = cpu_buffers[1] + 1 + 39 * i;
            if (!ptr[6]) continue;
            auto l = ptr[0];
            auto t = ptr[1];
            auto r = ptr[0] + ptr[2];
            auto b = ptr[1] + ptr[3];
            auto new_l = dst2src.v0 * l + dst2src.v1 * t + dst2src.v2;
            auto new_r = dst2src.v0 * r + dst2src.v1 * b + dst2src.v2;
            auto new_t = dst2src.v3 * l + dst2src.v4 * t + dst2src.v5;
            auto new_b = dst2src.v3 * r + dst2src.v4 * b + dst2src.v5;
            ins.x = new_l;
            ins.y = new_t;
            ins.w = new_r - new_l;
            ins.h = new_b - new_t;                                            
            ins.score = ptr[4];
            ins.label = ptr[5];
            process_mask(ptr, proto, cpu_mask_buffer, 39, imageWidth / 4, imageHeight / 4, 32, imageWidth * imageHeight / 16, stream);
            cv::Mat mask(imageWidth / 4, imageHeight / 4, CV_8UC1);
            memcpy(mask.ptr(), cpu_mask_buffer, imageWidth * imageHeight * sizeof(uint8_t) / 16);
            cv::resize(mask, mask, cv::Size(imageWidth, imageHeight));
            ins.mask = mask;
            result.segs.emplace_back(ins);
        }      
        vec_result.emplace_back(result);
        index++;
    }
    return vec_result;
}