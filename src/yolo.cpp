#include "yolo.h"


YOLO::YOLO(const YAML::Node &config) : Detection(config) {}

std::vector<Detections> YOLO::PostProcess(const std::vector<cv::Mat> &imgBatch, float *output) {
    std::vector<Detections> vec_result;
    int index = 0;
    auto predSize = bufferSize[1] / sizeof(float);
    for (const cv::Mat &img : imgBatch)
    {
        Detections result;
        float ratio = std::max(float(img.cols) / float(imageWidth), float(img.rows) / float(imageHeight));
        float *pred_per_img = output + index * predSize;
        for (int position = 0; position < num_rows; position++) {
            float *pred_per_obj = pred_per_img + position * (num_classes + 5);
            if (pred_per_obj[4] < obj_threshold) continue;
            Box box;
            auto max_pos = std::max_element(pred_per_obj + 5, pred_per_obj + num_classes + 5);
            box.score = pred_per_obj[4] * pred_per_obj[max_pos - pred_per_obj];
            box.label = max_pos - pred_per_obj - 5;
            box.x = pred_per_obj[0] * ratio;
            box.y = pred_per_obj[1] * ratio;
            box.w = pred_per_obj[2] * ratio;
            box.h = pred_per_obj[3] * ratio;
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
        float ratio = std::max(float(img.cols) / float(imageWidth), float(img.rows) / float(imageHeight));
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
            ins.x = pred_per_obj[0] * ratio;
            ins.y = pred_per_obj[1] * ratio;
            ins.w = pred_per_obj[2] * ratio;
            ins.h = pred_per_obj[3] * ratio;
            float box[4] = {ins.x, ins.y, ins.w, ins.h};
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