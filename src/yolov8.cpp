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
        yolov8_postprocess_box(pred_per_img, num_bboxes, num_classes, 7, conf_thr, nms_thr, dst2src, stream, cpu_buffers[1]);
        int num_boxes = std::min((int)cpu_buffers[1][0], 1000);
        for (int i = 0; i < num_boxes; i++) {
            Box box;
            float* ptr = cpu_buffers[1] + 1 + 7 * i;
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
        yolov8_postprocess_box_mask(pred_per_img, num_bboxes, num_classes, 39, conf_thr, nms_thr, dst2src, stream, cpu_buffers[1]);
        int num_boxes = std::min((int)cpu_buffers[1][0], 1000);
        for (int i = 0; i < num_boxes; i++) {
            Instance ins;
            float* ptr = cpu_buffers[1] + 1 + 39 * i;
            if (!ptr[6]) continue;
            ins.x = ptr[0];
            ins.y = ptr[1];
            ins.w = ptr[2];
            ins.h = ptr[3];  
            ins.score = ptr[4];
            ins.label = ptr[5]; 
            // process_mask_init(39, ptr[2], ptr[3]);
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