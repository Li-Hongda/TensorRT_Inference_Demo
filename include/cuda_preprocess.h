#pragma once

#include "common.h"
#include <cstdint>


struct AffineMatrix {
    float v0, v1, v2;
    float v3, v4, v5;  
};

void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();

void preprocess(uint8_t* src, AffineMatrix d2s, int src_width, int src_height,
                     float* dst, int dst_width, int dst_height,
                     cudaStream_t stream);					 


