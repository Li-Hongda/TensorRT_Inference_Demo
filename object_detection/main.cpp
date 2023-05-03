#include "build.h"

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv,
    {
        "{config 	|| config file of model			  }"
        "{inputpath || path of images or videos       }"
        "{savePath  || save path of results	          }" 
        
        // "{size      || image (h, w), eg: 640		  }"
        // "{batch_size|| batch size           		  }"
        // "{video     || video's path					  }"
        // "{img       || image's path					  }"
        // "{cam_id    || camera's device id,eg:0		  }"
        // "{show      || if show the result			  }"
        
    });    
    if (argc < 4)
    {
        std::cout << "Please design model arch, config file and folder name!" << std::endl;
        return -1;
    }
    std::string inputpath = argv[3];
    std::string savepath = argv[4];
    std::string video = argv[5];
    auto model = build_model(argv);
    if (model == nullptr)
        return -1;
    model->LoadEngine();
    if (video == "true")
        model->Inference(inputpath, savepath, true);
    else
        model->Inference(inputpath, savepath);
    // model->Inference(inputpath, video);
    return 0;
}
