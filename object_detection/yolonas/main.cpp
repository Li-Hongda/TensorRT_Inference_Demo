#include "../../include/yolonas.h"


int main(int argc, char** argv) {
    std::string inputpath = argv[1];
    std::string cfg_dir = "../configs";
    std::string cfg_suffix = ".yaml";
    std::string savedir = "../results";
    auto savepath = savedir + "/" + "yolonas" + "/";
    auto cfg = cfg_dir + "/" + "yolonas" + cfg_suffix;    
    YAML::Node root = YAML::LoadFile(cfg);
    YOLONAS YOLONAS(root["yolonas"]);
    YOLONAS.LoadEngine();
    YOLONAS.Inference(inputpath, savepath);
    return 0;
}