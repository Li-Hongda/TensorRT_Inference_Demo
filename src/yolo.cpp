#include "yolo.h"


YOLO::YOLO(const YAML::Node &config) : Detection(config) {}

#if 0 //CPU postprocess deprecated
std::vector<Detections> YOLO::PostProcess(const std::vector<cv::Mat> &imgBatch, float* output) {
    std::vector<Detections> vec_result;
    int index = 0;
    auto predSize = bufferSize[1] / sizeof(float);
    for (const cv::Mat &img : imgBatch) {
        Detections result;
        float* pred_per_img = output + index * predSize;
        for (int position = 0; position < num_rows; position++) {
            float* pred_per_obj = pred_per_img + position * (num_classes + 5);
            if (pred_per_obj[4] < conf_thr) continue;
            Box box;
            auto max_pos = std::max_element(pred_per_obj + 5, pred_per_obj + num_classes + 5);
            box.score = pred_per_obj[4] * pred_per_obj[max_pos - pred_per_obj];
            box.label = max_pos - pred_per_obj - 5;

            // 将得到的box坐标映射回原图。
            auto l = pred_per_obj[0] - pred_per_obj[2] / 2;
            auto t = pred_per_obj[1] - pred_per_obj[3] / 2;
            auto r = pred_per_obj[0] + pred_per_obj[2] / 2;
            auto b = pred_per_obj[1] + pred_per_obj[3] / 2;
            auto new_l = dst2src.v0 * l + dst2src.v1 * t + dst2src.v2;
            auto new_r = dst2src.v0 * r + dst2src.v1 * b + dst2src.v2;
            auto new_t = dst2src.v3 * l + dst2src.v4 * t + dst2src.v5;
            auto new_b = dst2src.v3 * r + dst2src.v4 * b + dst2src.v5;
            box.x = new_l;
            box.y = new_t;
            box.w = new_r - new_l;
            box.h = new_b - new_t;
        
            result.dets.emplace_back(box);
        }
        NMS(result.dets);
        vec_result.emplace_back(result);
        index++;
    } 
    return vec_result;
}
#endif

std::vector<Detections> YOLO::PostProcess(const std::vector<cv::Mat> &imgBatch, float* output) {
    std::vector<Detections> vec_result;
    int index = 0;
    auto predSize = bufferSize[1] / sizeof(float);
    for (const cv::Mat &img : imgBatch) {
        Detections result;
        float* pred_per_img = output + index * predSize;
        cuda_postprocess_init(7, imageWidth, imageHeight);
        postprocess_box(pred_per_img, num_rows, num_classes, 7, conf_thr, nms_thr, stream, cpu_buffers[1]);
        int num_boxes = std::min((int)cpu_buffers[1][0], 1000);
        for (int i = 0; i < num_boxes; i++) {
            Box box;
            float* ptr = cpu_buffers[1] + 1 + 7 * i;
            if (!ptr[6]) continue;
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
            result.dets.emplace_back(box);
        }
        vec_result.emplace_back(result);
        index++;
    }    
    return vec_result;
}


YOLO_seg::YOLO_seg(const YAML::Node &config) : InstanceSegmentation(config) {}

#if 1
std::vector<Segmentations> YOLO_seg::PostProcess(const std::vector<cv::Mat> &imgBatch, float* output1, float* output2) {
    std::vector<Segmentations> vec_result;
    int index = 0;
    auto protoSize = bufferSize[1] / sizeof(float);
    auto predSize = bufferSize[2] / sizeof(float);
    cuda_postprocess_init(39, imageWidth, imageHeight);
    for (const cv::Mat &img : imgBatch) {
        Segmentations result;
        float* proto = output1 + index * protoSize;
        float* pred_per_img = output2 + index * predSize;
        postprocess_box_mask(pred_per_img, num_rows, num_classes, 39, conf_thr, nms_thr, stream, cpu_buffers[1]);
        int num_boxes = std::min((int)cpu_buffers[1][0], 1000);
        for (int i = 0; i < num_boxes; i++) {
            Instance ins;
            float* ptr = cpu_buffers[1] + 1 + 39 * i;
            memcpy(&ins.pred_mask, ptr + 7, 32 * 4);
            if (!ptr[6]) continue;
            ins.x = ptr[0];
            ins.y = ptr[1];
            ins.w = ptr[2];
            ins.h = ptr[3];
            ins.score = ptr[4];
            ins.label = ptr[5];
            result.segs.emplace_back(ins);
        } 
        for (int i = 0; i < result.segs.size(); i++){
            cv::Mat mask = cv::Mat::zeros(imageHeight / 4, imageWidth / 4, CV_32FC1);
            float box[4] = {result.segs[i].x, result.segs[i].y, result.segs[i].w, result.segs[i].h};
            auto scale_box = get_downscale_rect(box, 4);
            for (int x = scale_box.x; x < scale_box.x + scale_box.width; x++) {
                for (int y = scale_box.y; y < scale_box.y + scale_box.height; y++) {
                    float e = 0.0f;
                    for (int j = 0; j < 32; j++) {
                        int index = j * protoSize / 32 + y * mask.cols + x;
                        e += result.segs[i].pred_mask[j] * proto[index];
                    }
                    e = 1.0f / (1.0f + expf(-e));
                    if (e > 0.5) mask.at<float>(y, x) = 1;
                }
            }
            cv::resize(mask, mask, cv::Size(imageWidth, imageHeight));
            auto l = result.segs[i].x;
            auto t = result.segs[i].y;
            auto r = result.segs[i].x + result.segs[i].w;
            auto b = result.segs[i].y + result.segs[i].h;
            auto new_l = dst2src.v0 * l + dst2src.v1 * t + dst2src.v2;
            auto new_r = dst2src.v0 * r + dst2src.v1 * b + dst2src.v2;
            auto new_t = dst2src.v3 * l + dst2src.v4 * t + dst2src.v5;
            auto new_b = dst2src.v3 * r + dst2src.v4 * b + dst2src.v5;
            result.segs[i].x = new_l;
            result.segs[i].y = new_t;
            result.segs[i].w = new_r - new_l;
            result.segs[i].h = new_b - new_t;  
            result.segs[i].mask = mask;
        }        
        vec_result.emplace_back(result);
        index++;
    }
    return vec_result;
}
#endif



#if 0 //GPU process_mask,  to be finished.
std::vector<Segmentations> YOLO_seg::PostProcess(const std::vector<cv::Mat> &imgBatch, float* output1, float* output2) {
    std::vector<Segmentations> vec_result;
    int index = 0;
    auto protoSize = bufferSize[1] / sizeof(float);
    auto predSize = bufferSize[2] / sizeof(float);
    cuda_postprocess_init(39, imageWidth, imageHeight);
    for (const cv::Mat &img : imgBatch){
        Segmentations result;
        float* proto = output1 + index * protoSize;
        float* pred_per_img = output2 + index * predSize;
        
        postprocess_box_mask(pred_per_img, num_rows, num_classes, 39, conf_thr, nms_thr, stream, cpu_buffers[1]);

        // postprocess_box_mask(pred_per_img, proto, num_rows, num_classes, 39, 160, 160, conf_thr, nms_thr, stream, cpu_buffers[1], cpu_mask_buffer);
        int num_boxes = std::min((int)cpu_buffers[1][0], 1000);
        for (int i = 0; i < num_boxes; i++) {
            Instance ins;
            float* ptr = cpu_buffers[1] + 1 + 39 * i;
            if (!ptr[6]) continue;
            auto l = ptr[0];
            auto t = ptr[1];
            auto r = ptr[0] + ptr[2];
            auto b = ptr[1] + ptr[3];
            auto new_l = dst2src.v0 * l + dst2src.v1 * t + dst2src.v2;
            auto new_r = dst2src.v0 * r + dst2src.v1 * b + dst2src.v2;
            auto new_t = dst2src.v3 * l + dst2src.v4 * t + dst2src.v5;
            auto new_b = dst2src.v3 * r + dst2src.v4 * b + dst2src.v5;
            ins.x = new_l;
            ins.y = new_t;
            ins.w = new_r - new_l;
            ins.h = new_b - new_t;                                            
            ins.score = ptr[4];
            ins.label = ptr[5];
            process_mask(ptr, proto, cpu_mask_buffer, 39, 160, 160, 32, 25600, stream);
            cv::Mat mask(imageWidth / 4, imageHeight / 4, CV_32F);
            memcpy(mask.ptr(), cpu_mask_buffer, imageWidth * imageHeight * sizeof(float) / 16);
            cv::resize(mask, mask, cv::Size(imageWidth, imageHeight));
            ins.mask = mask;
            result.segs.emplace_back(ins);
        }
        vec_result.emplace_back(result);        
        index++;
    }
    // cuda_postprocess_destroy();
    return vec_result;
}
#endif

cv::Rect YOLO_seg::get_downscale_rect(float bbox[4], float scale) {
    float left = bbox[0] / scale;
    float top = bbox[1] / scale;
    float width  = bbox[2] / scale;
    float height = bbox[3] / scale;
    if (left < 0) left = 0.0;
    if (top < 0) top = 0.0;
    if (left + width > imageWidth / scale) width = imageWidth / scale - left;
    if (top + height > imageHeight / scale) height = imageHeight / scale -top;

    return cv::Rect(left, top, width, height);
}