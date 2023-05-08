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
        float *pred = output + index * predSize;
        for (int position = 0; position < num_rows; position++) {
            float *row = pred + position * (num_classes + 5);
            if (row[4] < obj_threshold) continue;
            Box box;
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
        float *pred = output2 + index * predSize;
        // float *proto = output + index * 
        for (int position = 0; position < num_rows; position++) {
            float *row = pred + position * (num_classes + 5 + 32);
            if (row[4] < obj_threshold) continue;
            Instance ins;
            cv::Mat mask_mat = cv::Mat::zeros(imageHeight / 4, imageWidth / 4, CV_32FC1);

            auto max_pos = std::max_element(row + 5, row + num_classes + 5);
            float temp[32];
            memcpy(&temp, row + num_classes + 5, 32 * 4);
            ins.score = row[4] * row[max_pos - row];
            ins.label = max_pos - row - 5;
            ins.x = row[0] * ratio;
            ins.y = row[1] * ratio;
            ins.w = row[2] * ratio;
            ins.h = row[3] * ratio;
            // // auto scale_box = cv::Rect(round((ins.x - ins.w / 2)/ 4), round((ins.y - ins.h / 2)/ 4), round(ins.w / 4), round(ins.h / 4));
            // float box[4] = {ins.x, ins.y, ins.w, ins.h};
            // auto scale_box = get_downscale_rect(box,4);
            // for (int x = scale_box.x; x < scale_box.x + scale_box.width; x++) {
            //     for (int y = scale_box.y; y < scale_box.y + scale_box.height; y++) {
            //         float e = 0.0f;
            // //         for (int j = 0; j < 32; j++) {
            // //             e += temp[j] * proto[j * protoSize / 32 + y * mask_mat.cols + x];
            // //         }
            //         e = 1.0f / (1.0f + expf(-e));
            //         mask_mat.at<float>(y, x) = e;
            //     }
            // }
            // cv::resize(mask_mat, mask_mat, cv::Size(imageWidth, imageHeight));
            
            // float box[4] = {ins.x, ins.y, ins.w, ins.h};
            // for (int x = 0; x < mask_mat.cols; x++) {
            //     for (int y = 0; y < mask_mat.rows; y++) {
            //         float e = 0.0f;
            //         for (int j = 0; j < 32; j++) {
            //             e += temp[j] * proto[j * protoSize / 32 + y * mask_mat.cols + x];
            //         }
            //         e = 1.0f / (1.0f + expf(-e));
            //         mask_mat.at<float>(y, x) = e;
            //     }
            // }
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
  float right = bbox[0] + bbox[2] / 2;
  float bottom = bbox[1] + bbox[3] / 2;
  left /= scale;
  top /= scale;
  right /= scale;
  bottom /= scale;
  return cv::Rect(round(left), round(top), round(right - left), round(bottom - top));
}