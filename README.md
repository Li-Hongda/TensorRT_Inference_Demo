# TensorRT_Inference_Demo
<div align="center">
<img src="assets/000000005001.jpg" height="200px" >
<img src="assets/000000007816.jpg" height="200px" >

</div>

<div align="center">

  [![Cuda](https://img.shields.io/badge/CUDA-11.3-%2376B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit-archive)
  [![](https://img.shields.io/badge/TensorRT-8.6.0.12-%2376B900.svg?style=flat&logo=tensorrt)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
  [![](https://img.shields.io/badge/ubuntu-20.04-orange.svg?style=flat&logo=ubuntu)](https://releases.ubuntu.com/20.04/)
</div>

## 1.Introduction
This repo use TensorRT-8.x to deploy well-trained models, both image preprocessing and postprocessing are performed with CUDA, which realizes high-speed inference.
## 2.Update
<details open>
<summary>update process</summary>

+ 2023.05.01 🚀 Create the repo.
+ 2023.05.03 🚀 Support yolov5 detection.
+ 2023.05.05 🚀 Support yolov7 and yolov5 instance-segmentation.
+ 2023.05.10 🚀 Support yolov8 detection and instance-segmentation.
+ 2023.05.12 🚀 Support cuda preprocess for speed up.
+ 2023.05.16 🚀 Support cuda box postprocess.
+ 2023.05.19 🚀 Support cuda mask postprocess and support rtdetr.
</details>

## 3.Support Models
<details open>
<summary>supported models</summary>

- [x] [YOLOv5](https://github.com/ultralytics/yolov5)<br>
- [x] [YOLOv5-seg](https://github.com/ultralytics/yolov5)<br>
- [x] [YOLOv7](https://github.com/WongKinYiu/yolov7)<br>
- [x] [YOLOv8](https://github.com/ultralytics/ultralytics)<br>
- [x] [YOLOv8-seg](https://github.com/ultralytics/ultralytics)<br>
- [x] [RT-DETR](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr)<br>
- [ ] [YOLOv6](https://github.com/meituan/YOLOv6) (to be continued)<br>
- [ ] [YOLO-NAS](https://github.com/Deci-AI/super-gradients) (to be continued)<br>
</details>

All speed tests were performed on RTX 3090 with COCO Val set.The time calculated here is the sum of the time of image loading, preprocess, inference and postprocess, so it's going to be slower than what's reported in the paper.
<div align='center'>

| Models | BatchSize | Mode | Resolution |  FPS  |
|-|-|:-:|:-:|:-:|
| YOLOv5-s v7.0  | 1 | FP32 | 640x640 | 468 |
| YOLOv5-s v7.0  | 32 | FP32 | 640x640 | - |
| YOLOv5-seg-s v7.0  | 1 | FP32 | 640x640 | - |
| YOLOv7  | 1 | FP32 | 640x640 | 154 |
| YOLOv8-s  | 1 | FP32 | 640x640 | 171 |
| YOLOv8-s  | 1 | FP32 | 640x640 | - |
| RT-DETR  | 1 | FP32 | 640x640 | - |
| RT-DETR  | 1 | FP32 | 640x640 | - |
</div>


## 4.Usage


1. Clone the repo.
```
git clone https://github.com/Li-Hongda/TensorRT_Inference_Demo.git
```
2. Install the dependencies.
### TensorRT
Following [NVIDIA offical docs](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing) to install TensorRT.

### yaml-cpp
```
git clone https://github.com/jbeder/yaml-cpp
mkdir build && cd build
cmake ..
make -j20
cmake -DYAML_BUILD_SHARED_LIBS=on ..
make -j20
cd ..
```


3. Change the path [here](https://github.com/Li-Hongda/TensorRT_Inference_Demo/blob/main/object_detection/CMakeLists.txt#L19) to your TensorRT path, and [here](https://github.com/Li-Hongda/TensorRT_Inference_Demo/blob/main/object_detection/CMakeLists.txt#L11) to your CUDA path. Then,
```
cd TensorRT_Inference_Demo/object_detection
mkdir build && cd build
cmake ..
make -j$(nproc)
```
4. The executable file will be generated in `bin` in the repo directory if compile successfully.Then enjoy yourself with command like this:
```
cd bin
./object_detection yolov5 /path/to/input/dir 
```

> Notes:
> 1. The output of the model is required for post-processing is num_bboxes (imageHeight x image Width) x num_pred(num_cls + coordinates + confidence),while the output of YOLOv8 is num_pred * num_bboxes,which means the predicted values of the same box are not contiguous in memory.For convenience, the corresponding dimensions of the original pytorch output need to be transposed when exporting to ONNX model.



## 5.Reference
[0].https://github.com/NVIDIA/TensorRT<br>
[1].https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#c_topics<br>
[2].https://github.com/linghu8812/tensorrt_inference<br>
[3].https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#<br>
[4].https://blog.csdn.net/bobchen1017?type=blog<br>



