#ifndef DETECTION_H
#define DETECTION_H

#include "basemodel.h"

struct Box {
    float x;
    float y;
    float w;
    float h;
    int label;
    float score;
};

namespace Category{
    const std::vector<std::string> coco80 = {
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
    const std::vector<std::string> coco91 = { 
        "person", "bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","street sign","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
        "hat","backpack","umbrella","shoe","eye glasses","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
        "baseball glove","skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","mirror","dining table","window",
        "desk","toilet","door","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","blender",
        "book","clock","vase","scissors","teddy bear","hair drier","toothbrush","hair brush" 
    };    
}


namespace Color{
    const std::vector<cv::Scalar> coco80{ 
        cv::Scalar(128, 77, 207),cv::Scalar(65, 32, 208),cv::Scalar(0, 224, 45),cv::Scalar(3, 141, 219),cv::Scalar(80, 239, 253),cv::Scalar(239, 184, 12),
        cv::Scalar(7, 144, 145),cv::Scalar(161, 88, 57),cv::Scalar(0, 166, 46),cv::Scalar(218, 113, 53),cv::Scalar(193, 33, 128),cv::Scalar(190, 94, 113),
        cv::Scalar(113, 123, 232),cv::Scalar(69, 205, 80),cv::Scalar(18, 170, 49),cv::Scalar(89, 51, 241),cv::Scalar(153, 191, 154),cv::Scalar(27, 26, 69),
        cv::Scalar(20, 186, 194),cv::Scalar(210, 202, 167),cv::Scalar(196, 113, 204),cv::Scalar(9, 81, 88),cv::Scalar(191, 162, 67),cv::Scalar(227, 73, 120),
        cv::Scalar(177, 31, 19),cv::Scalar(133, 102, 137),cv::Scalar(146, 72, 97),cv::Scalar(145, 243, 208),cv::Scalar(2, 184, 176),cv::Scalar(219, 220, 93),
        cv::Scalar(238, 253, 234),cv::Scalar(197, 169, 160),cv::Scalar(204, 201, 106),cv::Scalar(13, 24, 129),cv::Scalar(40, 38, 4),cv::Scalar(5, 41, 34),
        cv::Scalar(46, 94, 129),cv::Scalar(102, 65, 107),cv::Scalar(27, 11, 208),cv::Scalar(191, 240, 183),cv::Scalar(225, 76, 38),cv::Scalar(193, 89, 124),
        cv::Scalar(30, 14, 175),cv::Scalar(144, 96, 90),cv::Scalar(181, 186, 86),cv::Scalar(102, 136, 34),cv::Scalar(158, 71, 15),cv::Scalar(183, 81, 247),
        cv::Scalar(73, 69, 89),cv::Scalar(123, 73, 232),cv::Scalar(4, 175, 57),cv::Scalar(87, 108, 23),cv::Scalar(105, 204, 142),cv::Scalar(63, 115, 53),
        cv::Scalar(105, 153, 126),cv::Scalar(247, 224, 137),cv::Scalar(136, 21, 188),cv::Scalar(122, 129, 78),cv::Scalar(145, 80, 81),cv::Scalar(51, 167, 149),
        cv::Scalar(162, 173, 20),cv::Scalar(252, 202, 17),cv::Scalar(10, 40, 3),cv::Scalar(150, 90, 254),cv::Scalar(169, 21, 68),cv::Scalar(157, 148, 180),
        cv::Scalar(131, 254, 90),cv::Scalar(7, 221, 102),cv::Scalar(19, 191, 184),cv::Scalar(98, 126, 199),cv::Scalar(210, 61, 56),cv::Scalar(252, 86, 59),
        cv::Scalar(102, 195, 55),cv::Scalar(160, 26, 91),cv::Scalar(60, 94, 66),cv::Scalar(204, 169, 193),cv::Scalar(126, 4, 181),cv::Scalar(229, 209, 196),
        cv::Scalar(195, 170, 186),cv::Scalar(155, 207, 148)
    };
    const std::vector<cv::Scalar> coco91{
        cv::Scalar(148, 99, 164),cv::Scalar(65, 172, 90),cv::Scalar(18, 117, 190),cv::Scalar(173, 208, 229),cv::Scalar(37, 162, 147),cv::Scalar(121, 99, 42),
        cv::Scalar(218, 173, 104),cv::Scalar(193, 213, 138),cv::Scalar(142, 168, 45),cv::Scalar(107, 143, 94),cv::Scalar(242, 89, 7),cv::Scalar(87, 218, 248),
        cv::Scalar(126, 168, 9),cv::Scalar(86, 152, 105),cv::Scalar(155, 135, 251),cv::Scalar(73, 234, 44),cv::Scalar(177, 37, 42),cv::Scalar(219, 215, 54),
        cv::Scalar(124, 207, 143),cv::Scalar(7, 81, 209),cv::Scalar(254, 18, 130),cv::Scalar(71, 54, 73),cv::Scalar(172, 198, 63),cv::Scalar(64, 217, 224),
        cv::Scalar(105, 224, 25),cv::Scalar(41, 52, 130),cv::Scalar(220, 27, 193),cv::Scalar(65, 222, 86),cv::Scalar(250, 150, 201),cv::Scalar(201, 150, 105),
        cv::Scalar(104, 96, 142),cv::Scalar(111, 230, 54),cv::Scalar(105, 24, 22),cv::Scalar(42, 226, 101),cv::Scalar(67, 26, 144),cv::Scalar(155, 113, 106),
        cv::Scalar(152, 196, 216),cv::Scalar(58, 68, 152),cv::Scalar(68, 230, 213),cv::Scalar(169, 143, 129),cv::Scalar(191, 102, 41),cv::Scalar(5, 73, 170),
        cv::Scalar(15, 73, 233),cv::Scalar(95, 13, 71),cv::Scalar(25, 92, 218),cv::Scalar(85, 173, 16),cv::Scalar(247, 158, 17),cv::Scalar(36, 28, 8),
        cv::Scalar(31, 100, 134),cv::Scalar(131, 71, 45),cv::Scalar(158, 190, 91),cv::Scalar(90, 207, 220),cv::Scalar(125, 77, 228),cv::Scalar(40, 156, 67),
        cv::Scalar(35, 250, 69),cv::Scalar(229, 61, 245),cv::Scalar(210, 201, 106),cv::Scalar(184, 35, 131),cv::Scalar(47, 124, 120),cv::Scalar(1, 114, 23),
        cv::Scalar(99, 181, 17),cv::Scalar(77, 141, 151),cv::Scalar(79, 33, 95),cv::Scalar(194, 111, 146),cv::Scalar(187, 199, 138),cv::Scalar(129, 215, 40),
        cv::Scalar(160, 209, 144),cv::Scalar(139, 121, 58),cv::Scalar(97, 208, 197),cv::Scalar(185, 105, 171),cv::Scalar(160, 96, 136),cv::Scalar(232, 26, 26),
        cv::Scalar(34, 165, 109),cv::Scalar(19, 86, 215),cv::Scalar(205, 209, 199),cv::Scalar(131, 91, 25),cv::Scalar(51, 201, 16),cv::Scalar(64, 35, 128),
        cv::Scalar(120, 161, 247),cv::Scalar(123, 164, 190),cv::Scalar(15, 191, 40),cv::Scalar(11, 44, 117),cv::Scalar(198, 136, 70),cv::Scalar(14, 224, 240),
        cv::Scalar(60, 186, 193),cv::Scalar(253, 190, 129),cv::Scalar(134, 228, 173),cv::Scalar(219, 156, 214),cv::Scalar(137, 67, 254),cv::Scalar(178, 223, 250),
        cv::Scalar(219, 199, 139)
    };    
};


struct Detections {
    std::vector<Box> dets;
};

class Detection : public Model
{
public:
    explicit Detection(const YAML::Node &config);
    std::vector<Detections> InferenceImages(std::vector<cv::Mat> &imgBatch);
    void Inference(const std::string &input_path, const std::string &save_path, const bool video) override;
    virtual void Inference(const std::string &input_path, const std::string &save_path) override;
    // void Inference(const std::string &input_path, const bool video) override;
    // void Inference(const std::string &input_path) override;
    void Visualize(const std::vector<Detections> &detections, std::vector<cv::Mat> &imgBatch,
                     std::vector<std::string> image_names);
    void Visualize(const std::vector<Detections> &detections, std::vector<cv::Mat> &imgBatch,
                     cv::String save_name, int fps, cv::Size size); 
    static float BoxIoU(const Box &det_a, const Box &det_b);

protected:
    virtual std::vector<Detections> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output)=0;
    void NMS(std::vector<Box> &detections);
    // std::map<int, std::string> class_labels;
    int num_classes;
    float obj_threshold;
    float nms_threshold;
    std::string type;
    std::vector<std::string> class_labels;
    std::vector<cv::Scalar> class_colors;
    std::vector<int> strides;
    int num_rows = 0;
};

#endif