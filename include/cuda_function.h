#pragma once

#include "common.h"
#include <cstdint>


struct AffineMatrix {
    float v0, v1, v2;
    float v3, v4, v5;  
};

void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();

void cuda_postprocess_init(int num_out, int width, int height);
void cuda_postprocess_destroy();

void preprocess(uint8_t* src, AffineMatrix d2s, int src_width, int src_height,
                     float* dst, int dst_width, int dst_height,
                     cudaStream_t stream);					 


void postprocess_box(float* predict, int num_bboxes, int num_out, int num_classes, float conf_thr,
				float nms_thr, cudaStream_t stream, float* dst);

void yolov8_postprocess_box(float* predict, int num_bboxes, int num_out, int num_classes, float conf_thr,
				float nms_thr, cudaStream_t stream, float* dst);

void postprocess_box_mask(float* predict, int num_bboxes, int num_classes, int num_out, 
						  float conf_thr, float nms_thr, cudaStream_t stream, float* dst);

void yolov8_postprocess_box_mask(float* predict, int num_bboxes, int num_classes, int num_out, 
						  float conf_thr, float nms_thr, cudaStream_t stream, float* dst);                          

void process_mask(float* out, float* proto, uint8_t* dst , int num_out, int dst_width, 
				  int dst_height, int out_w, int proto_size,
				  cudaStream_t stream);