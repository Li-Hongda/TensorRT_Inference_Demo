#include "detection.h"

Detection::Detection(const YAML::Node &config) : Model(config) {
    obj_threshold = config["obj_threshold"].as<float>();
    nms_threshold = config["nms_threshold"].as<float>();
    type = config["type"].as<std::string>();
    strides = config["strides"].as<std::vector<int>>();

    int index = 0;
    for (const int &stride : strides)
    {
        num_rows += int(imageHeight / stride) * int(imageWidth / stride) * 3;
        index+=1;
    }    
    
    if (type == "coco80"){
        class_colors = Color::coco80;
        class_labels = Category::coco80;
    }
    else {
        class_colors = Color::coco91;
        class_labels = Category::coco91;
    }
    num_classes = class_labels.size();

}

std::vector<Detections> Detection::InferenceImages(std::vector<cv::Mat> &imgBatch) noexcept{
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = PreProcess(imgBatch);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();

    
    CUDA_CHECK(cudaMemcpyAsync(gpu_buffers[0], image_data.data(), bufferSize[0], cudaMemcpyHostToDevice, stream));

    //gpu inference
    auto t_start = std::chrono::high_resolution_clock::now();
    // this->context->executeV2(gpu_buffers);
    this->context->enqueueV2(gpu_buffers, stream, nullptr);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    for(int i=1;i<engine->getNbBindings(); ++i){
        CUDA_CHECK(cudaMemcpyAsync(cpu_buffers[i], gpu_buffers[i], bufferSize[i], cudaMemcpyDeviceToHost, stream));
    } 
    cudaStreamSynchronize(stream); 
    
    auto t_start_post = std::chrono::high_resolution_clock::now();
    auto boxes = PostProcess(imgBatch, cpu_buffers[1]);
    auto t_end_post = std::chrono::high_resolution_clock::now();
    float total_post = std::chrono::duration<float, std::milli>(t_end_post - t_start_post).count();
    std::cout << "preprocess time: "<< total_pre << "ms." <<
    "detection inference time: " << total_inf << " ms." 
    "postprocess time: " << total_post << " ms." << std::endl; 
    return boxes;
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

    for (const std::string &image_name : image_list) {
        index++;
        // TODO: figure out why double free.
        // std::cout << "Processing: " << image_name << std::endl;
        cv::Mat img = cv::imread(image_name);
        imgBatch.emplace_back(img.clone());
        auto save_name = replace(image_name, input_path, save_path);
        imgInfo.emplace_back(save_name);
        
        if (imgBatch.size() == batchSize or index == image_list.size()){
            auto start_time = std::chrono::high_resolution_clock::now();
            auto det_results = InferenceImages(imgBatch);
            auto end_time = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
            Visualize(det_results, imgBatch, imgInfo);
            imgBatch.clear();
            imgInfo.clear();
        }
    }
    delete [] cpu_buffers;
    // sample::gLogError << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
    std::cout << "Average FPS is " << 1000 * image_list.size() / total_time << std::endl;
}

void Detection::NMS(std::vector<Box> &detections) {
    sort(detections.begin(), detections.end(), [=](const Box &left, const Box &right) {
        return left.score > right.score;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].label == detections[j].label)
            {
                float iou = DIoU(detections[i], detections[j]);
                if (iou > nms_threshold)
                    detections[j].score = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const Box &det)
    { return det.score == 0; }), detections.end());
}

float Detection::DIoU(const Box &det_a, const Box &det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = pow((center_a - center_b).x , 2) + pow((center_a - center_b).y, 2) ;
    float distance_c = pow((left_up - right_down).x, 2) + pow((left_up - right_down).y, 2);
    float inter_l = std::max(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2);
    float inter_t = std::max(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2);
    float inter_r = std::min(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2);
    float inter_b = std::min(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2);
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        // return inter_area / union_area;
        return inter_area / union_area - distance_d / distance_c;
}

void Detection::Visualize(const std::vector<Detections> &detections, std::vector<cv::Mat> &imgBatch,
                            std::vector<std::string> image_names=std::vector<std::string>()) {
    for (int i = 0; i < (int)imgBatch.size(); i++) {
        auto img = imgBatch[i];
        if (!img.data)
            continue;
        auto bboxes = detections[i].dets;
        for(const auto &bbox : bboxes) {
            auto score = cv::format("%.3f", bbox.score);
            std::string text = class_labels[bbox.label] + "|" + score;
            cv::putText(img, text, cv::Point(bbox.x - bbox.w / 2, bbox.y - bbox.h / 2 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, class_colors[bbox.label], 2);
            cv::Rect rect(bbox.x - bbox.w / 2, bbox.y - bbox.h / 2, bbox.w, bbox.h);
            cv::rectangle(img, rect, class_colors[bbox.label], 2, cv::LINE_8, 0);
        }
        std::string img_name = image_names[i];
        cv::imwrite(img_name, img);
    }
}

void Detection::Visualize(const std::vector<Detections> &detections, std::vector<cv::Mat> &frames,
                            const cv::String save_name, int fps, cv::Size size) {
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