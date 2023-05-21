#include "../../include/yolov7.h"


int main(int argc, char** argv)
{
    std::string inputpath = argv[1];
    std::string cfg_dir = "../configs";
    std::string cfg_suffix = ".yaml";
    std::string savedir = "../results";
    auto savepath = savedir + "/" + "yolov7" + "/";
    auto cfg = cfg_dir + "/" + "yolov7" + cfg_suffix;    
    YAML::Node root = YAML::LoadFile(cfg);
    YOLOv7 YOLOv7(root["yolov7"]);
    YOLOv7.LoadEngine();
    YOLOv7.Inference(inputpath, savepath);
    return 0;
}