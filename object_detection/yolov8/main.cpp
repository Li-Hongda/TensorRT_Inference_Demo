#include "../../include/yolov8.h"


int main(int argc, char** argv)
{
    std::string config = argv[1];
    std::string inputpath = argv[2];
    std::string savepath = argv[3];
    YAML::Node root = YAML::LoadFile(config);
    YOLOv8 YOLOv8(root["yolov8"]);
    YOLOv8.LoadEngine();
    YOLOv8.Inference(inputpath, savepath);
    return 0;
}