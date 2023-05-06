#include "instance_segmentation.h"

InstanceSegmentation::InstanceSegmentation(const YAML::Node &config) : Model(config) {
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

std::vector<Segmentations> InstanceSegmentation::InferenceImages(std::vector<cv::Mat> &imgBatch) {
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = PreProcess(imgBatch);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    auto *output = new float[outSize * batchSize];

    auto t_start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(buffers[0], image_data.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);

    //gpu inference
    this->context->executeV2(buffers);
    // this->context->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);    
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "detection inference take: " << total_inf << " ms." << std::endl;
    auto t_start_post = std::chrono::high_resolution_clock::now();
    auto boxes = PostProcess(imgBatch, output);
    auto t_end_post = std::chrono::high_resolution_clock::now();
    float total_post = std::chrono::duration<float, std::milli>(t_end_post - t_start_post).count();
    std::cout << "preprocess take: "<< total_pre << "ms." <<
    "detection inference take: " << total_inf << " ms." 
    "postprocess take: " << total_post << " ms." << std::endl;
    delete[] output;
    return boxes;
}

void InstanceSegmentation::Inference(const std::string &input_path, const cv::String &save_path, const bool video) {

    cv::VideoCapture capture;
    capture.open(input_path);
    cv::Size size = cv::Size((int)capture.get(cv::CAP_PROP_FRAME_WIDTH), (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));        
    int fps = capture.get(cv::CAP_PROP_FPS);        
    auto total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
    std::vector<cv::Mat> imgBatch;
    imgBatch.reserve(batchSize);

    std::vector<Segmentations> segs;
    segs.reserve(total_frames);
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
                std::vector<Segmentations> seg_results = InferenceImages(imgBatch);
                auto end_time = std::chrono::high_resolution_clock::now();
                segs.insert(segs.end(), seg_results.begin(), seg_results.end());
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
            auto seg_results = InferenceImages(imgBatch);
            auto end_time = std::chrono::high_resolution_clock::now();
            segs.insert(segs.end(), seg_results.begin(), seg_results.end());
            imgs.insert(imgs.end(), imgBatch.begin(), imgBatch.end());
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count(); 
            imgBatch.clear(); 
        }
    }
    Visualize(segs, imgs, save_path, fps, size);

}

void InstanceSegmentation::Inference(const std::string &input_path, const std::string &save_path) {    
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
        std::cout << "Processing: " << image_name << std::endl;
        cv::Mat img = cv::imread(image_name);
        imgBatch.emplace_back(img.clone());
        auto save_name = replace(image_name, input_path, save_path);
        imgInfo.emplace_back(save_name);
        
        if (imgBatch.size() == batchSize or index == image_list.size()){
            auto start_time = std::chrono::high_resolution_clock::now();
            auto seg_results = InferenceImages(imgBatch);
            auto end_time = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
            Visualize(seg_results, imgBatch, imgInfo);
            imgBatch.clear();
            imgInfo.clear();
        }
        
    }
    // sample::gLogError << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
    std::cout << "Average FPS is " << 1000 * image_list.size() / total_time << std::endl;
}

void InstanceSegmentation::NMS(std::vector<Instances> &segmentations) {
    sort(segmentations.begin(), segmentations.end(), [=](const Instances &left, const Instances &right) {
        return left.score > right.score;
    });

    for (int i = 0; i < (int)segmentations.size(); i++)
        for (int j = i + 1; j < (int)segmentations.size(); j++)
        {
            if (segmentations[i].label == segmentations[j].label)
            {
                float iou = DIoU(segmentations[i], segmentations[j]);
                if (iou > nms_threshold)
                    segmentations[j].score = 0;
            }
        }

    segmentations.erase(std::remove_if(segmentations.begin(), segmentations.end(), [](const Instances &det)
    { return det.score == 0; }), segmentations.end());
}

float InstanceSegmentation::DIoU(const Instances &det_a, const Instances &det_b) {
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
        return inter_area / union_area - distance_d / distance_c;
}

void InstanceSegmentation::Visualize(const std::vector<Segmentations> &segmentations, std::vector<cv::Mat> &imgBatch,
                            std::vector<std::string> image_names=std::vector<std::string>()) {
    for (int i = 0; i < (int)imgBatch.size(); i++) {
        auto img = imgBatch[i];
        if (!img.data)
            continue;
        auto bboxes = segmentations[i].segs;
        for(const auto &bbox : bboxes) {
            auto score = cv::format("%.3f", bbox.score);
            std::string text = class_labels[bbox.label] + "|" + score;
            cv::putText(img, text, cv::Point(bbox.x - bbox.w / 2, bbox.y - bbox.h / 2 - 5),
                    cv::FONT_HERSHEY_COMPLEX, 0.7, class_colors[bbox.label], 2);
            cv::Rect rect(bbox.x - bbox.w / 2, bbox.y - bbox.h / 2, bbox.w, bbox.h);
            cv::rectangle(img, rect, class_colors[bbox.label], 2, cv::LINE_8, 0);
        }
        std::string img_name = image_names[i];
        cv::imwrite(img_name, img);
    }
}

void InstanceSegmentation::Visualize(const std::vector<Segmentations> &segmentations, std::vector<cv::Mat> &frames,
                            const cv::String save_name, int fps, cv::Size size) {
    auto fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    cv::VideoWriter writer(save_name, fourcc, fps, size, true);
    for (int i = 0; i < (int)frames.size(); i++){
        auto frame = frames[i];
        if (!frame.data)
            continue;
        auto bboxes = segmentations[i].segs;
        for(const auto &bbox : bboxes) {
            auto score = cv::format("%.3f", bbox.score);
            std::string text = class_labels[bbox.label] + "|" + score;
            cv::putText(frame, text, cv::Point(bbox.x - bbox.w / 2, bbox.y - bbox.h / 2 - 5),
                    cv::FONT_HERSHEY_COMPLEX, 0.7, class_colors[bbox.label], 2);
            cv::Rect rect(bbox.x - bbox.w / 2, bbox.y - bbox.h / 2, bbox.w, bbox.h);
            cv::rectangle(frame, rect, class_colors[bbox.label], 2, cv::LINE_8, 0);
        }        
        writer.write(frame);
    }
    writer.release();    
}