#include "build.h"

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv,
    {
        "{model_arch 	|| arch of model			  }"
        "{inputpath     || path of images or videos   }"
        "{video         || enable video inference     }"         
    });    
    if (argc < 4)
    {
        std::cout << "Please design model arch, input_path and folder name!" << std::endl;
        return -1;
    }
    std::string model_arch = argv[1];
    std::string inputpath = argv[2];
    std::string video = argv[3];
    std::string cfg_dir = "../configs";
    std::string cfg_suffix = ".yaml";
    std::string savedir = "../results";
    auto savepath = savedir + "/" + model_arch;
    auto cfg = cfg_dir + "/" + model_arch + cfg_suffix;
    auto model = build_model(model_arch, cfg);
    check_dir(savepath, false);
    if (model == nullptr)
        return -1;
    model->LoadEngine();
    if (video == "true")
        model->Inference(inputpath, savepath, true);
    else
        model->Inference(inputpath, savepath);
    return 0;
}
