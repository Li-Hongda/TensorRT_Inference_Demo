#include "yolo.h"


YOLO::YOLO(const YAML::Node &config) : Detection(config) {}

std::vector<Detections> YOLO::PostProcess(const std::vector<cv::Mat> &imgBatch, float *output) {
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

YOLO_seg::YOLO_seg(const YAML::Node &config) : InstanceSegmentation(config) {}

std::vector<Segmentations> YOLO_seg::PostProcess(const std::vector<cv::Mat> &imgBatch, float *output1, float *output2) {
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
            float *pred_per_obj = pred_per_img + position * (num_classes + 5 + 32);
            if (pred_per_obj[4] < obj_threshold) continue;
            Instance ins;
            cv::Mat mask_mat = cv::Mat::zeros(imageHeight / 4, imageWidth / 4, CV_32FC1);

            auto max_pos = std::max_element(pred_per_obj + 5, pred_per_obj + num_classes + 5);
            float temp[32];
            memcpy(&temp, pred_per_obj + num_classes + 5, 32 * 4);
            ins.score = pred_per_obj[4] * pred_per_obj[max_pos - pred_per_obj];
            ins.label = max_pos - pred_per_obj - 5;
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

cv::Rect YOLO_seg::get_downscale_rect(float bbox[4], float scale) {
    float left = bbox[0] - bbox[2] / 2;
    float top = bbox[1] - bbox[3] / 2;
    left /= scale;
    top /= scale;
    auto width  = bbox[2] / scale;
    auto height = bbox[3] / scale;
    if (left < 0) left = 0.0;
    if (top < 0) top = 0.0;
    if (left + width > imageWidth / scale) width = imageWidth / scale - left;
    if (top + height > imageHeight / scale) height = imageHeight / scale -top;

    return cv::Rect(left, top, width, height);
}