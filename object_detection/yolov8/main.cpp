#include "../../include/yolov8.h"


int main(int argc, char** argv)
{
    std::string inputpath = argv[1];
    std::string cfg_dir = "../configs";
    std::string cfg_suffix = ".yaml";
    std::string savedir = "../results";
    auto savepath = savedir + "/" + "yolov8" + "/";
    auto cfg = cfg_dir + "/" + "yolov8" + cfg_suffix;    
    YAML::Node root = YAML::LoadFile(cfg);
    YOLOv8 YOLOv8(root["yolov8"]);
    YOLOv8.LoadEngine();
    YOLOv8.Inference(inputpath, savepath);
    return 0;
}