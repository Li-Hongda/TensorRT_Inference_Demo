#include "../../include/yolov5.h"


int main(int argc, char** argv)
{
    std::string config = argv[1];
    std::string inputpath = argv[2];
    std::string savepath = argv[3];
    YAML::Node root = YAML::LoadFile(config);
    YOLOv5 YOLOv5(root["yolov5"]);
    YOLOv5.LoadEngine();
    YOLOv5.Inference(inputpath, savepath);
    return 0;
}