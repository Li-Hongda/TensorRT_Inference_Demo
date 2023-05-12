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

// std::vector<Detections> YOLOv8::PostProcess(const std::vector<cv::Mat> &imgBatch, float *output) {
//     std::vector<Detections> vec_result;
//     int index = 0;
//     auto predSize = bufferSize[1] / sizeof(float);
//     for (const cv::Mat &img : imgBatch){
//         Detections result;
//         float ratio = std::max(float(img.cols) / float(imageWidth), float(img.rows) / float(imageHeight));
//         float *pred_per_img = output + index * predSize;
//         for (int position = 0; position < num_rows; position ++){
//             float *pred_per_obj = new float[84];
//             for (int pos = 0; pos < 84; pos++){
//                 memcpy((pred_per_obj + pos), pred_per_img + position + pos * 8400, sizeof(float));
//             }
//             Box box;
//             auto max_pos = std::max_element(pred_per_obj + 4, pred_per_obj + num_classes + 4); 
//             if (*max_pos < obj_threshold) continue;      
//             box.score = pred_per_obj[max_pos - pred_per_obj];
//             box.label = max_pos - pred_per_obj - 4;

//             auto l = pred_per_obj[0] - pred_per_obj[2] / 2;
//             auto t = pred_per_obj[1] - pred_per_obj[3] / 2;
//             auto r = pred_per_obj[0] + pred_per_obj[2] / 2;
//             auto b = pred_per_obj[1] + pred_per_obj[3] / 2;
//             auto new_l = dst2src.v0 * l + dst2src.v1 * t + dst2src.v2;
//             auto new_r = dst2src.v0 * r + dst2src.v1 * b + dst2src.v2;
//             auto new_t = dst2src.v3 * l + dst2src.v4 * t + dst2src.v5;
//             auto new_b = dst2src.v3 * r + dst2src.v4 * b + dst2src.v5;
//             box.x = new_l;
//             box.y = new_t;
//             box.w = new_r - new_l;
//             box.h = new_b - new_t;

//             result.dets.emplace_back(box);
//             delete[] pred_per_obj;
//         }
//         NMS(result.dets);
//         vec_result.emplace_back(result);
//         index++;
//     }       
//     return vec_result;
// }


std::vector<Detections> YOLOv8::PostProcess(const std::vector<cv::Mat> &imgBatch, float *output) {
    std::vector<Detections> vec_result;
    int index = 0;
    auto predSize = bufferSize[1] / sizeof(float);
    for (const cv::Mat &img : imgBatch){
        Detections result;
        float ratio = std::max(float(img.cols) / float(imageWidth), float(img.rows) / float(imageHeight));
        float *pred_per_img = output + index * predSize;
        for (int position = 0; position < num_rows; position++) {
            float *pred_per_obj = pred_per_img + position * (num_classes + 4);
            Box box;
            auto max_pos = std::max_element(pred_per_obj + 4, pred_per_obj + num_classes + 4);
            if (*max_pos < obj_threshold) continue;  
            box.score = pred_per_obj[max_pos - pred_per_obj];
            box.label = max_pos - pred_per_obj - 4;

            // 将得到的box坐标映射回原图。
            auto l = pred_per_obj[0] - pred_per_obj[2] / 2;
            auto t = pred_per_obj[1] - pred_per_obj[3] / 2;
            auto r = pred_per_obj[0] + pred_per_obj[2] / 2;
            auto b = pred_per_obj[1] + pred_per_obj[3] / 2;
            auto new_l = dst2src.v0 * l + dst2src.v1 * t + dst2src.v2;
            auto new_r = dst2src.v0 * r + dst2src.v1 * b + dst2src.v2;
            auto new_t = dst2src.v3 * l + dst2src.v4 * t + dst2src.v5;
            auto new_b = dst2src.v3 * r + dst2src.v4 * b + dst2src.v5;
            box.x = new_l;
            box.y = new_t;
            box.w = new_r - new_l;
            box.h = new_b - new_t;
        
            result.dets.emplace_back(box);
        }
        NMS(result.dets);
        vec_result.emplace_back(result);
        index++;
    }       
    return vec_result;
}

YOLOv8_seg::YOLOv8_seg(const YAML::Node &config) : YOLO_seg(config) {
    int index = 0;
    num_rows = 0;
    for (const int &stride : strides)
    {
        num_rows += int(imageHeight / stride) * int(imageWidth / stride);
        index+=1;
    }   
}

std::vector<Segmentations> YOLOv8_seg::PostProcess(const std::vector<cv::Mat> &imgBatch, float *output1, float *output2) {
    std::vector<Segmentations> vec_result;
    int index = 0;
    auto protoSize = bufferSize[1] / sizeof(float);
    auto predSize = bufferSize[2] / sizeof(float);
    for (const cv::Mat &img : imgBatch)
    {
        Segmentations result;
        float *proto = output1 + index * protoSize;
        float *pred_per_img = output2 + index * predSize;
        for (int position = 0; position < num_rows; position++) {
            float *pred_per_obj = pred_per_img + position * (num_classes + 4 + 32);
            if (pred_per_obj[4] < obj_threshold) continue;
            Instance ins;
            cv::Mat mask_mat = cv::Mat::zeros(imageHeight / 4, imageWidth / 4, CV_32FC1);

            auto max_pos = std::max_element(pred_per_obj + 4, pred_per_obj + num_classes + 4);
            float temp[32];
            memcpy(&temp, pred_per_obj + num_classes + 4, 32 * 4);
            ins.score = pred_per_obj[4] * pred_per_obj[max_pos - pred_per_obj];
            ins.label = max_pos - pred_per_obj - 4;
            auto l = pred_per_obj[0] - pred_per_obj[2] / 2;
            auto t = pred_per_obj[1] - pred_per_obj[3] / 2;
            auto r = pred_per_obj[0] + pred_per_obj[2] / 2;
            auto b = pred_per_obj[1] + pred_per_obj[3] / 2;
            auto new_l = dst2src.v0 * l + dst2src.v1 * t + dst2src.v2;
            auto new_r = dst2src.v0 * r + dst2src.v1 * b + dst2src.v2;
            auto new_t = dst2src.v3 * l + dst2src.v4 * t + dst2src.v5;
            auto new_b = dst2src.v3 * r + dst2src.v4 * b + dst2src.v5;
            ins.x = new_l;
            ins.y = new_t;
            ins.w = new_r - new_l;
            ins.h = new_b - new_t;
            float box[4] = {pred_per_obj[0], pred_per_obj[1], pred_per_obj[2], pred_per_obj[3]};
            auto scale_box = get_downscale_rect(box, 4);
            for (int x = scale_box.x; x < scale_box.x + scale_box.width; x++) {
                for (int y = scale_box.y; y < scale_box.y + scale_box.height; y++) {
                    float e = 0.0f;
                    for (int j = 0; j < 32; j++) {
                        int index = j * protoSize / 32 + y * mask_mat.cols + x;
                        if (index >= 0 && index < protoSize) { 
                            e += temp[j] * proto[index];
                        }
                    }
                    e = 1.0f / (1.0f + expf(-e));
                    if (e > 0.5)
                        mask_mat.at<float>(y, x) = 1;
                }
            }
            cv::resize(mask_mat, mask_mat, cv::Size(imageWidth, imageHeight));
            ins.mask = mask_mat;          
            result.segs.emplace_back(ins);
        }
        NMS(result.segs);
        vec_result.emplace_back(result);
        index++;
    }

    return vec_result;
}
