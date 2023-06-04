#include "instance_segmentation.h"

InstanceSegmentation::InstanceSegmentation(const YAML::Node &config) : Model(config) {
    conf_thr = config["conf_thr"].as<float>();
    type = config["type"].as<std::string>();
    cpu_mask_buffer = (uint8_t*) malloc(imageHeight * imageWidth * sizeof(uint8_t) / 16);  
    
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

void InstanceSegmentation::Inference(const std::string &input_path, const cv::String &save_path, const bool video) {
    // TODO: fix bugs for video inference.    
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
    cuda_preprocess_init(maxImageSize);
    while (capture.isOpened()) {
        index++;
        if (imgBatch.size() < batchSize) {
            capture.read(frame);
            if (frame.empty()) {
                std::cout << "no more video or camera frame" << std::endl;
                auto start_time = std::chrono::high_resolution_clock::now();
                std::vector<Segmentations> seg_results = InferenceImages(imgBatch);
                auto end_time = std::chrono::high_resolution_clock::now();
                segs.insert(segs.end(), seg_results.begin(), seg_results.end());
                imgs.insert(imgs.end(), imgBatch.begin(), imgBatch.end());                    
                imgBatch.clear(); 
                total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
                break;
            } else {
                imgBatch.emplace_back(frame.clone());
            }
        }
        else {
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
    cuda_preprocess_init(maxImageSize);
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
            auto start_time = std::chrono::high_resolution_clock::now();
            auto seg_results = InferenceImages(imgBatch);
            auto end_time = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
            Visualize(seg_results, imgBatch, imgInfo);
            imgBatch.clear();
            imgInfo.clear();
        }
        
    }
    // sample::gLogError << "Average processing time is " << total_time / image_list.size() << "ms " << std::endl;
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms " << std::endl;
    std::cout << "Average FPS is " << 1000 * image_list.size() / total_time << std::endl;
}

void InstanceSegmentation::Visualize(const std::vector<Segmentations> &segmentations, 
                                     std::vector<cv::Mat> &imgBatch, 
                                     std::vector<std::string> image_names) {
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5f;
    float thickness = 0.5;    
    for (int i = 0; i < (int)imgBatch.size(); i++) {
        auto img = imgBatch[i];
        if (!img.data) continue;
        auto instances = segmentations[i].segs;
        for(const auto &ins : instances) {
            cv::Mat mask = ins.mask;
            cv::Mat img_mask = scale_mask(mask, img);
            cv::Mat reg_img = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
            for (int row = ins.y; row < ins.y + ins.h; row++) {	
                if (row < 0 || row >= img.rows) continue;
                cv::Vec<uint8_t, 1> *data_Ptr = reg_img.ptr<cv::Vec<uint8_t, 1>> (row);
                for (int col = ins.x; col < ins.x + ins.w; col++) {
                    if (col < 0 || col >= img.cols) continue;
                    data_Ptr[col][0] = 1;
                }
            } 
            cv::bitwise_and(img_mask, reg_img, img_mask);

            std::vector<cv::Mat> contours;
            cv::Mat hierarchy;
            cv::Mat colored_img = img.clone();
            cv::findContours(img_mask, contours, hierarchy, 
                             cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(colored_img, contours, -1, class_colors[ins.label], -1, cv::LINE_8,
                             hierarchy, 100);                
            img = 0.4 * colored_img + 0.6 * img;

            auto score = cv::format("%.3f", ins.score);
            std::string text = class_labels[ins.label] + "|" + score;
            cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
            cv::Point org;
            org.x = ins.x ;
            org.y = ins.y + text_size.height + 2;
            cv::Rect text_back = cv::Rect(org.x, org.y - text_size.height, text_size.width, text_size.height + 5); 
            cv::rectangle(img, text_back, class_colors[ins.label], -1);
            cv::putText(img, text, org, font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
            cv::Rect rect(ins.x, ins.y, ins.w, ins.h);
            cv::rectangle(img, rect, class_colors[ins.label], 2, cv::LINE_8, 0);
        }
        std::string img_name = image_names[i];
        cv::imwrite(img_name, img);
    }
}

void InstanceSegmentation::Visualize(const std::vector<Segmentations> &segmentations, 
                                     std::vector<cv::Mat> &frames,
                                     const cv::String save_name, 
                                     int fps, cv::Size size) {
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5f;
    float thickness = 0.5;        
    auto fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    cv::VideoWriter writer(save_name, fourcc, fps, size, true);
    for (int i = 0; i < (int)frames.size(); i++){
        auto frame = frames[i];
        if (!frame.data)
            continue;
        auto instances = segmentations[i].segs;
        for(const auto &ins : instances) {
            cv::Mat mask = ins.mask;
            cv::Mat img_mask = scale_mask(mask, frame);
            cv::Mat reg_img = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
            for (int row = ins.y; row < ins.y + ins.h; row++) {	
                if (row < 0 || row >= frame.rows) continue;
                cv::Vec<uint8_t, 1> *data_Ptr = reg_img.ptr<cv::Vec<uint8_t, 1>> (row);
                for (int col = ins.x; col < ins.x + ins.w; col++) {
                    if (col < 0 || col >= frame.cols) continue;
                    data_Ptr[col][0] = 1;
                }
            } 
            cv::bitwise_and(img_mask, reg_img, img_mask);

            std::vector<cv::Mat> contours;
            cv::Mat hierarchy;
            cv::Mat colored_img = frame.clone();
            cv::findContours(img_mask, contours, hierarchy, 
                             cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(colored_img, contours, -1, class_colors[ins.label], -1, cv::LINE_8,
                             hierarchy, 100);                
            frame = 0.4 * colored_img + 0.6 * frame;

            auto score = cv::format("%.3f", ins.score);
            std::string text = class_labels[ins.label] + "|" + score;
            cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
            cv::Point org;
            org.x = ins.x ;
            org.y = ins.y + text_size.height + 2;
            cv::Rect text_back = cv::Rect(org.x, org.y - text_size.height, text_size.width, text_size.height + 5); 
            cv::rectangle(frame, text_back, class_colors[ins.label], -1);
            cv::putText(frame, text, org, font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
            cv::Rect rect(ins.x, ins.y, ins.w, ins.h);
            cv::rectangle(frame, rect, class_colors[ins.label], 2, cv::LINE_8, 0);
        }        
        writer.write(frame);
    }
    writer.release();    
}

cv::Mat InstanceSegmentation::scale_mask(cv::Mat mask, cv::Mat img) {
  int x, y, w, h;
  float r_w = imageWidth / (img.cols * 1.0);
  float r_h = imageHeight / (img.rows * 1.0);
  if (r_h > r_w) {
    w = imageWidth;
    h = r_w * img.rows;
    x = 0;
    y = (imageHeight - h) / 2;
  } else {
    w = r_h * img.cols;
    h = imageHeight;
    x = (imageWidth - w) / 2;
    y = 0;
  }
  cv::Rect r(x, y, w, h);
  cv::Mat res;
  cv::resize(mask(r), res, img.size());
  return res;
}
