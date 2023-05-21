#include "../../include/yolov5.h"


int main(int argc, char** argv)
{
    std::string inputpath = argv[1];
    std::string cfg_dir = "../configs";
    std::string cfg_suffix = ".yaml";
    std::string savedir = "../results";
    auto savepath = savedir + "/" + "yolov5" + "/";
    auto cfg = cfg_dir + "/" + "yolov5" + cfg_suffix;    
    YAML::Node root = YAML::LoadFile(cfg);
    YOLOv5 YOLOv5(root["yolov5"]);
    YOLOv5.LoadEngine();
    YOLOv5.Inference(inputpath, savepath);
    return 0;
}