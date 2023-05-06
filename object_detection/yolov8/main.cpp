#include "../../include/yolov8.h"


int main(int argc, char** argv)
{
    // cv::CommandLineParser parser(argc, argv,
    // {
    //     "{config 	|| config file of model			  }"
    //     "{inputpath || path of images or videos       }"
    //     "{savePath  || save path of results	          }" 
        
    //     // "{size      || image (h, w), eg: 640		  }"
    //     // "{batch_size|| batch size           		  }"
    //     // "{video     || video's path					  }"
    //     // "{img       || image's path					  }"
    //     // "{cam_id    || camera's device id,eg:0		  }"
    //     // "{show      || if show the result			  }"
        
    // });
    if (argc < 3)
    {
        std::cout << "Please design config file and folder name!" << std::endl;
        return -1;
    }
    std::string config = argv[1];
    std::string inputpath = argv[2];
    std::string savepath = argv[3];
    YAML::Node root = YAML::LoadFile(config);
    YOLOv8 YOLOv8(root["yolov8"]);
    YOLOv8.LoadEngine();
    YOLOv8.Inference(inputpath, savepath);
    return 0;
}