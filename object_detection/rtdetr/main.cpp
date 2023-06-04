#include "../../include/rtdetr.h"

int main(int argc, char** argv) {
    std::string inputpath = argv[1];
    std::string cfg_dir = "../configs";
    std::string cfg_suffix = ".yaml";
    std::string savedir = "../results";
    auto savepath = savedir + "/" + "rtdetr" + "/";
    auto cfg = cfg_dir + "/" + "rtdetr" + cfg_suffix;    
    YAML::Node root = YAML::LoadFile(cfg);
    RTDETR RTDETR(root["rtdetr"]);
    RTDETR.LoadEngine();
    RTDETR.Inference(inputpath, savepath);
    return 0;
}