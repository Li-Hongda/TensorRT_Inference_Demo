#include "baseyolo.h"


YOLO::YOLO(const YAML::Node &config) : Detection(config) {}

std::vector<Detections> YOLO::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<Detections> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat)
    {
        Detections result;
        float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT)  ? float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);
        float *out = output + index * outSize;
        for (int position = 0; position < num_rows; position++) {
            float *row = out + position * (num_classes + 5);
            Box box;
            if (row[4] < obj_threshold)
                continue;
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