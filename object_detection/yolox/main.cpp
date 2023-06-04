#include "../../include/yolox.h"


int main(int argc, char** argv) {
    std::string inputpath = argv[1];
    std::string cfg_dir = "../configs";
    std::string cfg_suffix = ".yaml";
    std::string savedir = "../results";
    auto savepath = savedir + "/" + "yolox" + "/";
    auto cfg = cfg_dir + "/" + "yolox" + cfg_suffix;    
    YAML::Node root = YAML::LoadFile(cfg);
    YOLOX YOLOX(root["yolox"]);
    YOLOX.LoadEngine();
    YOLOX.Inference(inputpath, savepath);
    return 0;
}