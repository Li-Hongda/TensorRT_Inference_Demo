#include "detection.h"

Detection::Detection(const YAML::Node &config) : Model(config) {
    obj_threshold = config["obj_threshold"].as<float>();
    nms_threshold = config["nms_threshold"].as<float>();
    type = config["type"].as<std::string>();
    strides = config["strides"].as<std::vector<int>>();
    num_anchors = config["num_anchors"].as<std::vector<int>>();

    int index = 0;
    for (const int &stride : strides)
    {
        int num_anchor = num_anchors[index] !=0 ? num_anchors[index] : 1;
        num_rows += int(IMAGE_HEIGHT / stride) * int(IMAGE_WIDTH / stride) * num_anchor;
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

std::vector<Detections> Detection::InferenceImages(std::vector<cv::Mat> &vec_img) {
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = PreProcess(vec_img);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    // sample::gLogInfo << "detection prepare image take: " << total_pre << " ms." << std::endl;
    auto *output = new float[outSize * BATCH_SIZE];;
    auto t_start = std::chrono::high_resolution_clock::now();
    ModelInference(image_data, output);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    // sample::gLogInfo << "detection inference take: " << total_inf << " ms." << std::endl;
    auto r_start = std::chrono::high_resolution_clock::now();
    auto boxes = PostProcess(vec_img, output);
    auto r_end = std::chrono::high_resolution_clock::now();
    float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
    // sample::gLogInfo << "detection postprocess take: " << total_res << " ms." << std::endl;
    delete[] output;
    return boxes;
}

// void Detection::Inference(const std::string &input_path, const bool video) {
//     // int ret;
//     // ret = CheckDir(input_path);
//     // if (ret == 0)
//     //     InferenceFolder(input_path);
//     if (video == true){
//         int total_frames = 0;

//         cv::VideoCapture capture;
//         capture.open(input_path);
// 		total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
// 	cv::Mat frame;	
//     std::vector<cv::Mat> imgs_batch;
//     int i = 0; // debug
// 	int batchi = 0;

// 	while (capture.isOpened())
// 	{
// 		// if (batchi >= total_batches && source != utils::InputStream::CAMERA)
// 		// {
// 		// 	break;
// 		// }
// 		if (imgs_batch.size() < BATCH_SIZE) // get input
// 		{
// 			capture.read(frame);

// 			if (frame.empty())
// 			{
// 				// sample::gLogWarning << "no more video or camera frame" << std::endl;
// 				InferenceImages(frame);
// 				imgs_batch.clear(); // clear
// 				//sample::gLogInfo << imgs_batch.capacity() << std::endl;
// 				batchi++;
// 				break;
// 			}	
// 			else
// 			{
// 				imgs_batch.emplace_back(frame.clone()); 
// 			}
			
// 		}
// 		else // infer
// 		{
// 			task(yolo, param, imgs_batch, delay_time, batchi, is_show, is_save);
// 			imgs_batch.clear(); // clear
// 			//sample::gLogInfo << imgs_batch.capacity() << std::endl;
// 			batchi++;
// 		}
// 	}		
//         // totalBatches = (total_frames % param.batch_size == 0) ?
// 		// 	(total_frames / param.batch_size) : (total_frames / param.batch_size + 1);
//     }

// }

void Detection::InferenceFolder(const std::string &input_path) {
    std::vector<std::string> image_list = ReadFolder(input_path);
    int index = 0;
    int batch_id = 0;
    std::vector<cv::Mat> imgBatch(BATCH_SIZE);
    std::vector<std::string> imgInfo(BATCH_SIZE);
    float total_time = 0;
    for (const std::string &image_name : image_list) {
        index++;
        // TODO: figure out why double free.
        // sample::gLogInfo << "Processing: " << image_name << std::endl;
        std::cout << "Processing: " << image_name << std::endl;
        cv::Mat src_img = cv::imread(image_name);
        if (src_img.data) {

            cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
            imgBatch[batch_id] = src_img.clone();
            imgInfo[batch_id] = image_name;
            batch_id++;
        }
        if (batch_id == BATCH_SIZE or index == image_list.size()) {
            auto start_time = std::chrono::high_resolution_clock::now();
            auto det_results = InferenceImages(imgBatch);
            auto end_time = std::chrono::high_resolution_clock::now();
            Visualize(det_results, imgBatch, imgInfo);
            imgBatch = std::vector<cv::Mat>(BATCH_SIZE);
            batch_id = 0;
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
        }
    }
    // sample::gLogInfo << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}

void Detection::NMS(std::vector<Bbox> &detections) {
    sort(detections.begin(), detections.end(), [=](const Bbox &left, const Bbox &right) {
        return left.score > right.score;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].label == detections[j].label)
            {
                float iou = BoxIoU(detections[i], detections[j]);
                if (iou > nms_threshold)
                    detections[j].score = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const Bbox &det)
    { return det.score == 0; }), detections.end());
}

float Detection::BoxIoU(const Bbox &det_a, const Bbox &det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}

void Detection::Visualize(const std::vector<Detections> &detections, std::vector<cv::Mat> &vec_img,
                            std::vector<std::string> image_name=std::vector<std::string>()) {
    for (int i = 0; i < (int)vec_img.size(); i++) {
        auto org_img = vec_img[i];
        if (!org_img.data)
            continue;
        auto rects = detections[i].dets;
        cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
        for(const auto &rect : rects) {
            char t[256];
            sprintf(t, "%.2f", rect.score);
            std::string name = class_labels[rect.label] + "-" + t;
            cv::putText(org_img, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5),
                    cv::FONT_HERSHEY_COMPLEX, 0.7, class_colors[rect.label], 2);
            cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
            cv::rectangle(org_img, rst, class_colors[rect.label], 2, cv::LINE_8, 0);
        }
        if (!image_name.empty()) {
            int pos = image_name[i].find_last_of('.');
            std::string rst_name = image_name[i].insert(pos, "_");
            // sample::gLogInfo << rst_name << std::endl;
            cv::imwrite(rst_name, org_img);
        }
    }
}
