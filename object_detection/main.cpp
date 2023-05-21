#include "build.h"

int main(int argc, char **argv)
{
    std::string model_arch = argv[1];
    std::string inputpath = argv[2];
    bool video = false;
    if (argc > 3) video = true;
    std::string cfg_dir = "../configs";
    std::string cfg_suffix = ".yaml";
    std::string savedir = "../results";
    auto savepath = savedir + "/" + model_arch + "/";
    auto cfg = cfg_dir + "/" + model_arch + cfg_suffix;
    auto model = build_model(model_arch, cfg);
    check_dir(savepath, false);
    if (model == nullptr)
        return -1;
    model->LoadEngine();
    if (video)
        model->Inference(inputpath, savepath, true);
    else
        model->Inference(inputpath, savepath);
    return 0;
}
