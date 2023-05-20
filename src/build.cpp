#include "build.h"

std::shared_ptr<Model> build_model(std::string model_arch, std::string cfg) {
    YAML::Node root = YAML::LoadFile(cfg);
    auto model = std::shared_ptr<Model>();
    if (model_arch == "yolov5")
        model = std::make_shared<YOLOv5>(root[model_arch]);
    else if (model_arch == "yolov5-seg")
        model = std::make_shared<YOLOv5_seg>(root[model_arch]);
    // else if (model_arch == "yolov6")
    //     model = std::make_shared<YOLOv6>(root[model_arch]);
    else if (model_arch == "yolov7")
        model = std::make_shared<YOLOv7>(root[model_arch]);
    else if (model_arch == "yolov8")
        model = std::make_shared<YOLOv8>(root[model_arch]);
    else if (model_arch == "yolov8-seg")
        model = std::make_shared<YOLOv8_seg>(root[model_arch]); 
    else if (model_arch == "rtdetr")
        model = std::make_shared<RTDETR>(root[model_arch]);                
    else
        std::cout << "No model arch matched!" << std::endl;
    return model;
}

