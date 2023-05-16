#include "../../include/rtdetr.h"

int main(int argc, char** argv)
{
    std::string config = argv[1];
    std::string inputpath = argv[2];
    std::string savepath = argv[3];
    YAML::Node root = YAML::LoadFile(config);
    RTDETR RTDETR(root["rtdetr"]);
    RTDETR.LoadEngine();
    RTDETR.Inference(inputpath, savepath);
    return 0;
}