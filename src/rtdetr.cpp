#include "rtdetr.h"


RTDETR::RTDETR(const YAML::Node &config) : Detection(config) {}

std::vector<Detections> RTDETR::PostProcess(const std::vector<cv::Mat> &imgBatch, float *output) {
    std::vector<Detections> vec_result;
    int index = 0;
    auto predSize = bufferSize[1] / sizeof(float);
    for (const cv::Mat &img : imgBatch)
    {
        Detections result;
        float *pred_per_img = output + index * predSize;
        for (int position = 0; position < num_rows; position++) {
            float *pred_per_obj = pred_per_img + position * (num_classes + 5);
            if (pred_per_obj[4] < obj_threshold) continue;
            Box box;
            auto max_pos = std::max_element(pred_per_obj + 5, pred_per_obj + num_classes + 5);
            box.score = pred_per_obj[4] * pred_per_obj[max_pos - pred_per_obj];
            box.label = max_pos - pred_per_obj - 5;

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