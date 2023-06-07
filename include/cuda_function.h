#pragma once

#include "common.h"
#include <cstdint>

struct Norm {
    float mean[3];
    float std[3];
	float scale;
};

struct AffineMatrix {
    float v0, v1, v2;
    float v3, v4, v5;  
};

void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();

void cuda_postprocess_init(int num_out, int width, int height);
void cuda_postprocess_destroy();

void process_mask_init(int num_out, int width, int height);

void preprocess(uint8_t* src, AffineMatrix d2s, int src_width, int src_height,
                     float* dst, int dst_width, int dst_height, Norm norm,
                     cudaStream_t stream);					 

void postprocess_box(float* predict, int num_bboxes, int num_classes, int num_out, float conf_thr,
				float nms_thr, AffineMatrix mat, cudaStream_t stream, float* dst);				

void yolov8_postprocess_box(float* predict, int num_bboxes, int num_classes, int num_out, float conf_thr,
				float nms_thr, AffineMatrix mat, cudaStream_t stream, float* dst);

void rtdetr_postprocess_box(float* predict_box, float* predict_cls, int num_bboxes,  int num_classes, int num_out,
							float conf_thr, int imageWidth, int imageHeight, AffineMatrix mat, cudaStream_t stream, float* dst);

void yolonas_postprocess_box(float* predict, int num_bboxes, int num_classes, int num_out, float conf_thr,
				float nms_thr, AffineMatrix mat, cudaStream_t stream, float* dst);

void postprocess_box_mask(float* predict, int num_bboxes, int num_classes, int num_out, 
						  float conf_thr, float nms_thr, AffineMatrix mat, cudaStream_t stream, float* dst);

void yolov8_postprocess_box_mask(float* predict, int num_bboxes, int num_classes, int num_out, 
						  float conf_thr, float nms_thr, AffineMatrix mat, cudaStream_t stream, float* dst);

void process_mask(float* out, float* proto, uint8_t* dst , int num_out, int dst_width, 
				  int dst_height, int out_w, int proto_size,
				  cudaStream_t stream);