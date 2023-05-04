#include "classification.h"

Classification::Classification(const YAML::Node &config) : Model(config) {
    labels_file = config["labels_file"].as<std::string>();
    class_labels = ReadImageNetLabel(labels_file);
    num_classes = class_labels.size();
}

std::vector<ClassRes> Classification::InferenceImages(std::vector<cv::Mat> &imgBatch) {
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = PreProcess(imgBatch);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    std::cout << "classification prepare image take: " << total_pre << " ms." << std::endl;
    auto *output = new float[outSize * batchSize];;
    auto t_start = std::chrono::high_resolution_clock::now();
    ModelInference(image_data, output);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "classification inference take: " << total_inf << " ms." << std::endl;
    auto r_start = std::chrono::high_resolution_clock::now();
    auto results = PostProcess(imgBatch, output);
    auto r_end = std::chrono::high_resolution_clock::now();
    float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
    std::cout << "classification postprocess take: " << total_res << " ms." << std::endl;
    delete[] output;
    return results;
}

void Classification::InferenceFolder(const std::string &input_path) {
    std::vector<std::string> image_list = get_names(input_path);
    int index = 0;
    int batch_id = 0;
    std::vector<cv::Mat> vec_Mat(batchSize);
    std::vector<std::string> vec_name(batchSize);
    float total_time = 0;
    for (const std::string &image_name : image_list) {
        index++;
        std::cout << "Processing: " << image_name << std::endl;
        cv::Mat src_img = cv::imread(image_name);
        if (src_img.data) {
            cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
            vec_Mat[batch_id] = src_img.clone();
            vec_name[batch_id] = image_name;
            batch_id++;
        }
        if (batch_id == batchSize or index == image_list.size()) {
            auto start_time = std::chrono::high_resolution_clock::now();
            auto cls_results = InferenceImages(vec_Mat);
            auto end_time = std::chrono::high_resolution_clock::now();
            Visualize(cls_results, vec_Mat, vec_name);
            vec_Mat = std::vector<cv::Mat>(batchSize);
            batch_id = 0;
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
        }
    }
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}

std::vector<ClassRes> Classification::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<ClassRes> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat) {
        ClassRes result;
        float *out = output + index * outSize;
        auto max_pos = std::max_element(out, out + outSize);
        result.classes = max_pos - out;
        result.prob = out[result.classes];
        vec_result.emplace_back(result);
        index++;
    }
    return vec_result;
}

void Classification::Visualize(const std::vector<ClassRes> &results, std::vector<cv::Mat> &imgBatch,
                        std::vector<std::string> image_names=std::vector<std::string>()) {
    for (int i = 0; i < (int)imgBatch.size(); i++) {
        auto org_img = imgBatch[i];
        if (!org_img.data)
            continue;
        auto result = results[i];
        if (!image_names.empty()) {
            std::string rst_name = class_labels[result.classes] + ".jpg";;
            std::cout << rst_name << std::endl;
        }
    }
}
