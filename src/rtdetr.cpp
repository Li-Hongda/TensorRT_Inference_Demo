#include "rtdetr.h"


RTDETR::RTDETR(const YAML::Node &config) : Detection(config) {
    num_bboxes = config["num_queries"].as<int>();
}

std::vector<Detections> RTDETR::InferenceImages(std::vector<cv::Mat> &imgBatch) noexcept{
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    PreProcess(imgBatch);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();

    //gpu inference
    auto t_start = std::chrono::high_resolution_clock::now();
    auto gpu_buf = (void **)gpu_buffers;
    this->context->enqueueV2(gpu_buf, stream, nullptr);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    auto t_start_post = std::chrono::high_resolution_clock::now();
    auto boxes = PostProcess(imgBatch, gpu_buffers[1], gpu_buffers[2]);
    auto t_end_post = std::chrono::high_resolution_clock::now();
    float total_post = std::chrono::duration<float, std::milli>(t_end_post - t_start_post).count();
    // std::cout << "preprocess time: "<< total_pre << "ms " <<
    // "detection inference time: " << total_inf << "ms " 
    // "postprocess time: " << total_post << "ms " << std::endl; 
    return boxes;
}


std::vector<Detections> RTDETR::PostProcess(const std::vector<cv::Mat> &imgBatch, float* output1, float* output2) {
    std::vector<Detections> vec_result;
    int index = 0;
    auto predboxSize = bufferSize[1] / sizeof(float);
    auto predscoreSize = bufferSize[2] / sizeof(float);
    for (const cv::Mat &img : imgBatch) {
        Detections result;
        float* box_per_img = output1 + index * predboxSize;
        float* score_per_img = output2 + index * predscoreSize;
        cuda_postprocess_init(6, imageWidth, imageHeight);
        auto t_start_post = std::chrono::high_resolution_clock::now();
        rtdetr_postprocess_box(box_per_img, score_per_img, num_bboxes, num_classes, 6, 
                               conf_thr, imageWidth, imageHeight, dst2src, stream, cpu_buffers[2]);
        auto t_end_post = std::chrono::high_resolution_clock::now();
        float total_post = std::chrono::duration<float, std::milli>(t_end_post - t_start_post).count(); 
        std::cout << "postprocess time: " << total_post << "ms " << std::endl;
        int num_boxes = std::min((int)cpu_buffers[2][0], 300);
        for (int i = 0; i < num_boxes; i++) {
            Box box;
            float* ptr = cpu_buffers[2] + 1 + 6 * i;
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
        // cuda_postprocess_destroy();
    }
    return vec_result;
}