# TensorRT_Inference_Demo

<div align="center">

  [![Cuda](https://img.shields.io/badge/CUDA-11.3-%2376B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit-archive)
  [![](https://img.shields.io/badge/TensorRT-8.6.0.12-%2376B900.svg?style=flat&logo=tensorrt)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
  [![](https://img.shields.io/badge/ubuntu-20.04-orange.svg?style=flat&logo=ubuntu)](https://releases.ubuntu.com/20.04/)
</div>

## 1.Introduction
This repo use TensorRT-8.x to deploy well-trained models.

## 2.Update

- [x] [YOLOv5](https://github.com/ultralytics/yolov5)
- [x] [YOLOv5-seg](https://github.com/ultralytics/yolov5)
- [x] [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [x] [YOLOv8](https://github.com/ultralytics/ultralytics)
- [ ] [YOLOv8-seg](https://github.com/ultralytics/ultralytics)


## 3.Support Models

| Models | Device | BatchSize | Mode | Input Shape(HxW) | FPS |
|-|-|:-:|:-:|:-:|:-:|
| YOLOv5-n v7.0 |RTX3090 | 1 | FP32 | 640x640 | 264 |
| YOLOv5-s v7.0 |RTX3090 | 1 | FP32 | 640x640 | 210 |
| YOLOv5-s v7.0 |RTX3090 | 32 | FP32 | 640x640 | - |
| YOLOv5-m v7.0 |RTX3090 | 1 | FP32 | 640x640 | 140 |
| YOLOv5-l v7.0 |RTX3090 | 1 | FP32 | 640x640 | 105 |
| YOLOv5-x v7.0 |RTX3090 | 1 | FP32 | 640x640 | 75 |
| YOLOv7 |RTX3090 | 1 | FP32 | 640x640 | 115 |
| YOLOv7x |RTX3090 | 1 | FP32 | 640x640 | - |
| YOLOv8-n |RTX3090 | 1 | FP32 | 640x640 | 222 |
| YOLOv8-s |RTX3090 | 1 | FP32 | 640x640 | 171 |
| YOLOv8-m |RTX3090 | 1 | FP32 | 640x640 | 122 |
| YOLOv8-l |RTX3090 | 1 | FP32 | 640x640 | 88 |
| YOLOv8-x |RTX3090 | 1 | FP32 | 640x640 | 68 |
| RT-DETR |RTX3090 | 1 | FP32 | 640x640 | - |
| RT-DETR |RTX3090 | 1 | FP32 | 640x640 | - |
| SOLO(r50) |RTX3090 | 1 | FP32 | 480x640 | - |
| SOLOv2(r50) |RTX3090 | 1 | INT8 | 480x640 | - |

## 4.Install
1. Clone the repo.
```
git clone https://github.com/Li-Hongda/TensorRT_Inference_Demo.git
cd TensorRT_Inference_Demo/object_detection
```
2. Change the path [here]() to your TensorRT path, and [here]() to your CUDA path. Then,
```
mkdir build && cd build
cmake ..
make -j$(nproc)
```
3. The executable file will be generated in `bin` in the repo directory if compile successfully.Then enjoy yourself with command like this:
```
cd bin
./object_detection yolov5 /path/to/input/dir false
```

