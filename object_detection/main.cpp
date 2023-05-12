#include "build.h"

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv,
    {
        "{config 	|| config file of model			  }"
        "{inputpath || path of images or videos       }"
        "{savePath  || save path of results	          }" 
    });  
    // int dev = 0;
    // cudaDeviceProp devProp;
    // CHECK(cudaGetDeviceProperties(&devProp, dev));
    // std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    // std::cout << "SM的数量:" << devProp.multiProcessorCount << std::endl;
    // std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    // std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    // std::cout << "每个EM的最大线程数:" << devProp.maxThreadsPerMultiProcessor << std::endl;
    // std::cout << "每个SM的最大线程束数:" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;    
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
