#include "detection.h"

Detection::Detection(const YAML::Node &config) : Model(config) {
    conf_thr = config["conf_thr"].as<float>();
    type = config["type"].as<std::string>();

    if (type == "coco"){
        class_colors = Color::coco;
        class_labels = Category::coco;
    }
    else {
        class_colors = Color::voc;
        class_labels = Category::voc;
    }
    num_classes = class_labels.size();

}

void Detection::Inference(const std::string &input_path, const cv::String &save_path, const bool video) {

    cv::VideoCapture capture;
    capture.open(input_path);
    cv::Size size = cv::Size((int)capture.get(cv::CAP_PROP_FRAME_WIDTH), (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));        
    int fps = capture.get(cv::CAP_PROP_FPS);        
    auto total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
    std::vector<cv::Mat> imgBatch;
    imgBatch.reserve(batchSize);

    std::vector<Detections> dets;
    dets.reserve(total_frames);
    std::vector<cv::Mat> imgs;
    imgs.reserve(total_frames);

    int index = 0;
    float total_time = 0;
    cv::Mat frame;

    while (capture.isOpened())
    {
        index++;
        if (imgBatch.size() < batchSize) // get input
        {
            capture.read(frame);

            if (frame.empty())
            {
                sample::gLogWarning << "no more video or camera frame" << std::endl;
                auto start_time = std::chrono::high_resolution_clock::now();
                std::vector<Detections> det_results = InferenceImages(imgBatch);
                auto end_time = std::chrono::high_resolution_clock::now();
                dets.insert(dets.end(), det_results.begin(), det_results.end());
                imgs.insert(imgs.end(), imgBatch.begin(), imgBatch.end());                    
                imgBatch.clear(); // clear
                total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
                break;
            }
            else
            {
                imgBatch.emplace_back(frame.clone());
            }
        }
        else // infer
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            auto det_results = InferenceImages(imgBatch);
            auto end_time = std::chrono::high_resolution_clock::now();
            dets.insert(dets.end(), det_results.begin(), det_results.end());
            imgs.insert(imgs.end(), imgBatch.begin(), imgBatch.end());
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count(); 
            imgBatch.clear(); 
        }
    }
    Visualize(dets, imgs, save_path, fps, size);

}

void Detection::Inference(const std::string &input_path, const std::string &save_path) {    
    std::vector<std::string> image_list = get_names(input_path);
    
    int index = 0;
    std::vector<cv::Mat> imgBatch;
    imgBatch.reserve(batchSize);
    std::vector<std::string> imgInfo;
    imgInfo.reserve(batchSize);
    float total_time = 0;
    cuda_preprocess_init(kMaxInputImageSize);
    for (const std::string &image_name : image_list) {
        index++;
        // TODO: figure out why double free.
        auto load_start = std::chrono::high_resolution_clock::now();
        cv::Mat img = cv::imread(image_name);
        auto load_end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<float, std::milli>(load_end - load_start).count();
        imgBatch.emplace_back(img.clone());
        auto save_name = replace(image_name, input_path, save_path);
        imgInfo.emplace_back(save_name);
        
        if (imgBatch.size() == batchSize or index == image_list.size()){
            auto infer_start = std::chrono::high_resolution_clock::now();
            auto det_results = InferenceImages(imgBatch);
            Visualize(det_results, imgBatch, imgInfo);
            auto infer_end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<float, std::milli>(infer_end - infer_start).count();            
            imgBatch.clear();
            imgInfo.clear(); 
        }
    }
    delete [] cpu_buffers;
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms " << std::endl;
    std::cout << "Average FPS is " << 1000 * image_list.size() / total_time << std::endl;
}

std::vector<Detections> Detection::PostProcess(const std::vector<cv::Mat> &vec_Mat, float* output) {}


void Detection::Visualize(const std::vector<Detections> &detections, 
                          std::vector<cv::Mat> &imgBatch,
                          std::vector<std::string> image_names) {
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5f;
    float thickness = 0.5;    
    for (int i = 0; i < (int)imgBatch.size(); i++) {
        auto img = imgBatch[i];
        if (!img.data)
            continue;
        auto bboxes = detections[i].dets;
        for(const auto &bbox : bboxes) {
            auto score = cv::format("%.3f", bbox.score);
            std::string text = class_labels[bbox.label] + "|" + score;
            cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
            cv::Point org;
            org.x = bbox.x;
            org.y = bbox.y + text_size.height + 2;            
            cv::Rect text_back = cv::Rect(org.x, org.y - text_size.height, text_size.width, text_size.height + 5); 
            cv::rectangle(img, text_back, class_colors[bbox.label], -1);
            cv::putText(img, text, org, font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
            cv::Rect rect(bbox.x, bbox.y, bbox.w, bbox.h);
            cv::rectangle(img, rect, class_colors[bbox.label], 2, cv::LINE_8, 0);
        }
        std::string img_name = image_names[i];
        cv::imwrite(img_name, img);
    }
}

void Detection::Visualize(const std::vector<Detections> &detections, 
                          std::vector<cv::Mat> &frames,
                          const cv::String save_name, 
                          int fps, cv::Size size) {
    auto fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    cv::VideoWriter writer(save_name, fourcc, fps, size, true);
    for (int i = 0; i < (int)frames.size(); i++){
        auto frame = frames[i];
        if (!frame.data)
            continue;
        auto bboxes = detections[i].dets;
        for(const auto &bbox : bboxes) {
            auto score = cv::format("%.3f", bbox.score);
            std::string text = class_labels[bbox.label] + "|" + score;
            cv::putText(frame, text, cv::Point(bbox.x - bbox.w / 2, bbox.y - bbox.h / 2 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, class_colors[bbox.label], 2);
            cv::Rect rect(bbox.x - bbox.w / 2, bbox.y - bbox.h / 2, bbox.w, bbox.h);
            cv::rectangle(frame, rect, class_colors[bbox.label], 2, cv::LINE_8, 0);
        }        
        writer.write(frame);
    }
    writer.release();    
}
