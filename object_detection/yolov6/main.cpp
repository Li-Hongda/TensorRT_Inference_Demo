#include "../../include/yolov6.h"


int main(int argc, char** argv)
{
    std::string inputpath = argv[1];
    std::string cfg_dir = "../configs";
    std::string cfg_suffix = ".yaml";
    std::string savedir = "../results";
    auto savepath = savedir + "/" + "yolov6" + "/";
    auto cfg = cfg_dir + "/" + "yolov6" + cfg_suffix;    
    YAML::Node root = YAML::LoadFile(cfg);
    YOLOv6 YOLOv6(root["yolov6"]);
    YOLOv6.LoadEngine();
    YOLOv6.Inference(inputpath, savepath);
    return 0;
}