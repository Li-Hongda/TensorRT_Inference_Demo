#ifndef DETECTION_H
#define DETECTION_H

#include "basemodel.h"

namespace Category {
    const std::vector<std::string> coco = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
    const std::vector<std::string> voc = {
        "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable",
        "dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tv/monitor"
    };
}

namespace Color {
    const std::vector<cv::Scalar> coco { 
        cv::Scalar(220, 20, 60), cv::Scalar(119, 11, 32), cv::Scalar(0, 0, 142), cv::Scalar(0, 0, 230), cv::Scalar(106, 0, 228), 
        cv::Scalar(0, 60, 100), cv::Scalar(0, 80, 100), cv::Scalar(0, 0, 70), cv::Scalar(0, 0, 192), cv::Scalar(250, 170, 30), 
        cv::Scalar(100, 170, 30), cv::Scalar(220, 220, 0), cv::Scalar(175, 116, 175), cv::Scalar(250, 0, 30), cv::Scalar(165, 42, 42), 
        cv::Scalar(255, 77, 255), cv::Scalar(0, 226, 252), cv::Scalar(182, 182, 255), cv::Scalar(0, 82, 0), cv::Scalar(120, 166, 157), 
        cv::Scalar(110, 76, 0), cv::Scalar(174, 57, 255), cv::Scalar(199, 100, 0), cv::Scalar(72, 0, 118), cv::Scalar(255, 179, 240), 
        cv::Scalar(0, 125, 92), cv::Scalar(209, 0, 151), cv::Scalar(188, 208, 182), cv::Scalar(0, 220, 176), cv::Scalar(255, 99, 164), 
        cv::Scalar(92, 0, 73), cv::Scalar(133, 129, 255), cv::Scalar(78, 180, 255), cv::Scalar(0, 228, 0), cv::Scalar(174, 255, 243), 
        cv::Scalar(45, 89, 255), cv::Scalar(134, 134, 103), cv::Scalar(145, 148, 174), cv::Scalar(255, 208, 186), cv::Scalar(197, 226, 255), 
        cv::Scalar(171, 134, 1), cv::Scalar(109, 63, 54), cv::Scalar(207, 138, 255), cv::Scalar(151, 0, 95), cv::Scalar(9, 80, 61), 
        cv::Scalar(84, 105, 51), cv::Scalar(74, 65, 105), cv::Scalar(166, 196, 102), cv::Scalar(208, 195, 210), cv::Scalar(255, 109, 65), 
        cv::Scalar(0, 143, 149), cv::Scalar(179, 0, 194), cv::Scalar(209, 99, 106), cv::Scalar(5, 121, 0), cv::Scalar(227, 255, 205), 
        cv::Scalar(147, 186, 208), cv::Scalar(153, 69, 1), cv::Scalar(3, 95, 161), cv::Scalar(163, 255, 0), cv::Scalar(119, 0, 170), 
        cv::Scalar(0, 182, 199), cv::Scalar(0, 165, 120), cv::Scalar(183, 130, 88), cv::Scalar(95, 32, 0), cv::Scalar(130, 114, 135), 
        cv::Scalar(110, 129, 133), cv::Scalar(166, 74, 118), cv::Scalar(219, 142, 185), cv::Scalar(79, 210, 114), cv::Scalar(178, 90, 62), 
        cv::Scalar(65, 70, 15), cv::Scalar(127, 167, 115), cv::Scalar(59, 105, 106), cv::Scalar(142, 108, 45), cv::Scalar(196, 172, 0), 
        cv::Scalar(95, 54, 80), cv::Scalar(128, 76, 255), cv::Scalar(201, 57, 1), cv::Scalar(246, 0, 122), cv::Scalar(191, 162, 208)
    };
    const std::vector<cv::Scalar> voc {
        cv::Scalar(106, 0, 228), cv::Scalar(119, 11, 32), cv::Scalar(165, 42, 42), cv::Scalar(0, 0, 192), cv::Scalar(197, 226, 255), 
        cv::Scalar(0, 60, 100), cv::Scalar(0, 0, 142), cv::Scalar(255, 77, 255), cv::Scalar(153, 69, 1), cv::Scalar(120, 166, 157), 
        cv::Scalar(0, 182, 199), cv::Scalar(0, 226, 252), cv::Scalar(182, 182, 255), cv::Scalar(0, 0, 230), cv::Scalar(220, 20, 60), 
        cv::Scalar(163, 255, 0), cv::Scalar(0, 82, 0), cv::Scalar(3, 95, 161), cv::Scalar(0, 80, 100), cv::Scalar(183, 130, 88)
    };
};

struct Box {
    float x;
    float y;
    float w;
    float h;
    float score;
    int label;
};
const static int maxImageSize = 2048 * 2048;
struct Detections {
    std::vector<Box> dets;
};

class Detection : public Model {
public:

    explicit Detection(const YAML::Node &config);
    virtual std::vector<Detections> InferenceImages(std::vector<cv::Mat> &imgBatch) = 0;
    void Inference(const std::string &input_path, const std::string &save_path, const bool video) override;
    void Inference(const std::string &input_path, const std::string &save_path) override;
    void Visualize(const std::vector<Detections> &detections, std::vector<cv::Mat> &imgBatch,
                     std::vector<std::string> image_names);
    void Visualize(const std::vector<Detections> &detections, std::vector<cv::Mat> &imgBatch,
                     cv::String save_name, int fps, cv::Size size); 

protected:
    virtual std::vector<Detections> PostProcess(const std::vector<cv::Mat> &vec_Mat, float* output);
    int num_classes;
    float conf_thr;
    std::string type;
    std::vector<std::string> class_labels;
    std::vector<cv::Scalar> class_colors;
    int num_bboxes = 0;
};

#endif