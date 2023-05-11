#include "../../include/yolov7.h"


int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
    {
        "{inputpath     || path of images or videos   }"
        "{video         || enable video inference     }"         
    });
    if (argc < 3)
    {
        std::cout << "Please design config file and folder name!" << std::endl;
        return -1;
    }
    std::string config = argv[1];
    std::string inputpath = argv[2];
    std::string savepath = argv[3];
    YAML::Node root = YAML::LoadFile(config);
    YOLOv7 YOLOv7(root["yolov7"]);
    YOLOv7.LoadEngine();
    YOLOv7.Inference(inputpath, savepath);
    return 0;
}