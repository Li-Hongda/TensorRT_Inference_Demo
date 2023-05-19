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
    // for(int i=1;i<engine->getNbBindings(); ++i){
    //     CUDA_CHECK(cudaMemcpyAsync(cpu_buffers[i], gpu_buffers[i], bufferSize[i], cudaMemcpyDeviceToHost, stream));
    // }    
    auto t_start_post = std::chrono::high_resolution_clock::now();
    auto boxes = PostProcess(imgBatch, gpu_buffers[1], gpu_buffers[2]);
    auto t_end_post = std::chrono::high_resolution_clock::now();
    float total_post = std::chrono::duration<float, std::milli>(t_end_post - t_start_post).count();
    std::cout << "preprocess time: "<< total_pre << "ms." <<
    "detection inference time: " << total_inf << " ms." 
    "postprocess time: " << total_post << " ms." << std::endl; 
    return boxes;
}


std::vector<Detections> RTDETR::PostProcess(const std::vector<cv::Mat> &imgBatch, float* output1, float* output2) {
    std::vector<Detections> vec_result;
    int index = 0;
    auto predboxSize = bufferSize[1] / sizeof(float);
    auto predscoreSize = bufferSize[2] / sizeof(float);
    for (const cv::Mat &img : imgBatch)
    {
        Detections result;
        float* box_per_img = output1 + index * predboxSize;
        float* score_per_img = output2 + index * predscoreSize;
        cuda_postprocess_init(6, imageWidth, imageHeight);
        rtdetr_postprocess_box(box_per_img, score_per_img, num_bboxes, num_classes, 6, 
                               conf_thr, imageWidth, imageHeight, stream, cpu_buffers[2]);
        int num_boxes = std::min((int)cpu_buffers[2][0], 300);
        for (int i = 0; i < num_boxes; i++) {
            Box box;
            float* ptr = cpu_buffers[2] + 1 + 6 * i;
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
            // box.x = ptr[0];
            // box.y = ptr[1];
            // box.w = ptr[2];
            // box.h = ptr[3]; 
            // box.score = ptr[4];
            // box.label = ptr[5];                          
            result.dets.emplace_back(box);
        }
        vec_result.emplace_back(result);
        index++;
        // cuda_postprocess_destroy();
    }
    return vec_result;
}