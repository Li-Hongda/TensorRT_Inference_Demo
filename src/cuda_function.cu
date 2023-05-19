#include "cuda_function.h"
#include "detection.h"

static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;
// static float* proto_buffer_host = nullptr;
// static float* proto_buffer_device = nullptr;
static float* out_buffer_host = nullptr;
static float* out_buffer_device = nullptr;
static uint8_t* out_mask_buffer_device = nullptr;
static float* single_out_buffer_device = nullptr;

void cuda_postprocess_init(int num_out, int width, int height) {
	CUDA_CHECK(cudaMallocHost((void**)&out_buffer_host, (1 + 1000 * num_out) * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&out_buffer_device, (1 + 1000 * num_out) * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&out_mask_buffer_device, width * height * sizeof(uint8_t) / 16 ));
	CUDA_CHECK(cudaMalloc((void**)&single_out_buffer_device, num_out * sizeof(float)));

}



void cuda_preprocess_init(int max_image_size) {
	CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));
	CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy() {
	CUDA_CHECK(cudaFree(img_buffer_device));
	CUDA_CHECK(cudaFreeHost(img_buffer_host));
}

void cuda_postprocess_destroy() {
	CUDA_CHECK(cudaFreeHost(out_buffer_host));
	CUDA_CHECK(cudaFree(out_buffer_device));
	CUDA_CHECK(cudaFree(out_mask_buffer_device));
	// CUDA_CHECK(cudaFree(single_out_buffer_device));	
}


static __global__ void warpaffine_kernel_bilinear(
    uint8_t* src, int src_line_size, int src_width,
    int src_height, float* dst, int dst_width,
    int dst_height, uint8_t padding_value,
    AffineMatrix d2s, int size) {
	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position >= size) return;

	float m_x1 = d2s.v0; float m_y1 = d2s.v1;
	float m_z1 = d2s.v2; float m_x2 = d2s.v3;
	float m_y2 = d2s.v4; float m_z2 = d2s.v5;

	int dx = position % dst_width;
	int dy = position / dst_width;
	float src_x = m_x1 * dx + m_y1 * dy + m_z1;
	float src_y = m_x2 * dx + m_y2 * dy + m_z2;
	float c0 = padding_value, c1 = padding_value, c2 = padding_value;


	// 双线性插值
	if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height) {
		// out of range
	} else {
		int y_low = floorf(src_y);
		int x_low = floorf(src_x);
		int y_high = y_low + 1;
		int x_high = x_low + 1;

		uint8_t pad_val[] = {padding_value, padding_value, padding_value};
		float ly = src_y - y_low;
		float lx = src_x - x_low;
		float hy = 1 - ly;
		float hx = 1 - lx;
		float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
		uint8_t* v1 = pad_val; uint8_t* v2 = pad_val;
		uint8_t* v3 = pad_val; uint8_t* v4 = pad_val;

		if (y_low >= 0) {
		if (x_low >= 0) v1 = src + y_low * src_line_size + x_low * 3;
		if (x_high < src_width) v2 = src + y_low * src_line_size + x_high * 3;
		}

		if (y_high < src_height) {
		if (x_low >= 0) v3 = src + y_high * src_line_size + x_low * 3;

		if (x_high < src_width) v4 = src + y_high * src_line_size + x_high * 3;
		}

		c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
		c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
		c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
	}

	// bgr to rgb 
	float t = c2;
	c2 = c0;
	c0 = t;

	// normalization
	c0 = c0 / 255.0f;
	c1 = c1 / 255.0f;
	c2 = c2 / 255.0f;

	// rgbrgbrgb to rrrgggbbb
	int area = dst_width * dst_height;
	float* pdst_c0 = dst + dy * dst_width + dx;
	float* pdst_c1 = pdst_c0 + area;
	float* pdst_c2 = pdst_c1 + area;
	*pdst_c0 = c0;
	*pdst_c1 = c1;
	*pdst_c2 = c2;
}

static __device__ float sigmoid(float a) {
	float e = 1.0f / (1.0f + expf(-a));
	return e;
}

static __device__ float box_iou(
    float aleft, float atop, float awidth, float aheight,
    float bleft, float btop, float bwidth, float bheight) {

    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aleft + awidth, bleft + bwidth);
    float cbottom = min(atop + aheight, btop + bheight);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = max(0.0f, awidth) * max(0.0f, aheight);
    float b_area = max(0.0f, bwidth) * max(0.0f, bheight);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float* bboxes, float threshold, int num_out)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    int count = (int)*bboxes;
    if (position > count) return; 

    float* pcurrent = bboxes + 1  + position * num_out; 
    for (int i = 0; i < count; ++i){
        float* pitem = bboxes + 1 + i * num_out;
		
        if (i == position || pcurrent[5]!= pitem[5] ) continue;
        
        if (pitem[4] >= pcurrent[4]){
            if (pitem[4] == pcurrent[4] && i < position) continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold){
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

static __global__ void decode_box_kernel(float* predict, int num_bboxes, int num_out,
										 int num_classes, float confidence_threshold,
    									 float* parray) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pred_per_obj = predict + position * (num_classes + 5);
    if (pred_per_obj[4] < confidence_threshold) return;
	
    float* cls_score = pred_per_obj + 5;

    float score = *cls_score++;

    int label = 0;
    for (int i = 1; i < num_classes; i++, ++cls_score) {
        if (*cls_score > score) {   
            score = *cls_score;
            label = i;
        }
    }
    score *= pred_per_obj[4];
    if (score < confidence_threshold) return;
    float cx = pred_per_obj[0];
    float cy = pred_per_obj[1];
    float width = pred_per_obj[2];
    float height = pred_per_obj[3];	
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;

    int index = atomicAdd(parray, 1);
    
    float* pout_item = parray + 1 + index * num_out;
    pout_item[0] = left;
    pout_item[1] = top;
    pout_item[2] = width;
    pout_item[3] = height;
    pout_item[4] = score;
    pout_item[5] = label;
	pout_item[6] = 1;		
}

static __global__ void yolov8_decode_box_kernel(float* predict, int num_bboxes, int num_out,
										 int num_classes, float confidence_threshold,
    									 float* parray) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pred_per_obj = predict + position * (num_classes + 4);
	
    float* cls_score = pred_per_obj + 4;

    float score = *cls_score++;

    int label = 0;
    for (int i = 1; i < num_classes; i++, ++cls_score) {
        if (*cls_score > score) {   
            score = *cls_score;
            label = i;
        }
    }
    if (score < confidence_threshold) return;
    float cx = pred_per_obj[0];
    float cy = pred_per_obj[1];
    float width = pred_per_obj[2];
    float height = pred_per_obj[3];	
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;

    int index = atomicAdd(parray, 1);
    
    float* pout_item = parray + 1 + index * num_out;
    pout_item[0] = left;
    pout_item[1] = top;
    pout_item[2] = width;
    pout_item[3] = height;
    pout_item[4] = score;
    pout_item[5] = label;
	pout_item[6] = 1;		
}


static __global__ void rtdetr_decode_box_kernel(float* predict_box, float* predict_cls, int num_bboxes, 
                                                int num_out, int num_classes, float confidence_threshold,
    									        int imgWidth, int imgHeight, float* parray) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* box_pre_obj = predict_box + position * 4;
	
    float* cls_score = predict_cls + position * 80;

    float score = *cls_score++;

    int label = 0;
    for (int i = 1; i < num_classes; i++, ++cls_score) {
        if (*cls_score > score) {   
            score = *cls_score;
            label = i;
        }
    }
    score = sigmoid(score);
    if (score < confidence_threshold) return;
    float cx = box_pre_obj[0] * imgWidth;
    float cy = box_pre_obj[1] * imgHeight;
    float width = box_pre_obj[2] * imgWidth;
    float height = box_pre_obj[3] * imgHeight;	
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;

    int index = atomicAdd(parray, 1);
    
    float* pout_item = parray + 1 + index * num_out;
    pout_item[0] = left;
    pout_item[1] = top;
    pout_item[2] = width;
    pout_item[3] = height;
    pout_item[4] = score;
    pout_item[5] = label;	
}


static __global__ void decode_box_mask_kernel(float* predict, int num_bboxes, int num_out,
										 int num_classes, float confidence_threshold,
    									 float* parray) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;    
	if (position >= num_bboxes) return;
    float* pred_per_obj = predict + position * (num_classes + 5 + 32);
    if (pred_per_obj[4] < confidence_threshold) return;
	
    float* cls_score = pred_per_obj + 5;

    float score = *cls_score++;

    int label = 0;
    for (int i = 1; i < num_classes; i++, ++cls_score) {
        if (*cls_score > score) {   
            score = *cls_score;
            label = i;
        }
    }
    score *= pred_per_obj[4];
    if (score < confidence_threshold) return;
    float cx = pred_per_obj[0];
    float cy = pred_per_obj[1];
    float width = pred_per_obj[2];
    float height = pred_per_obj[3];	
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;

    int index = atomicAdd(parray, 1);
    
    float* pout_item = parray + 1 + index * num_out;
    pout_item[0] = left;
    pout_item[1] = top;
    pout_item[2] = width;
    pout_item[3] = height;
    pout_item[4] = score;
    pout_item[5] = label;
	pout_item[6] = 1;	
	for (int idx = 0;idx < 32; idx++) {
		pout_item[7 + idx] = pred_per_obj[85 + idx];
	}
}


static __global__ void yolov8_decode_box_mask_kernel(float* predict, int num_bboxes, int num_out,
										 int num_classes, float confidence_threshold,
    									 float* parray) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;    
	if (position >= num_bboxes) return;
    float* pred_per_obj = predict + position * (num_classes + 4 + 32);
	
    float* cls_score = pred_per_obj + 4;

    float score = *cls_score++;

    int label = 0;
    for (int i = 1; i < num_classes; i++, ++cls_score) {
        if (*cls_score > score) {   
            score = *cls_score;
            label = i;
        }
    }
    if (score < confidence_threshold) return;
    float cx = pred_per_obj[0];
    float cy = pred_per_obj[1];
    float width = pred_per_obj[2];
    float height = pred_per_obj[3];	
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;

    int index = atomicAdd(parray, 1);
    
    float* pout_item = parray + 1 + index * num_out;
    pout_item[0] = left;
    pout_item[1] = top;
    pout_item[2] = width;
    pout_item[3] = height;
    pout_item[4] = score;
    pout_item[5] = label;
	pout_item[6] = 1;	
	for (int idx = 0;idx < 32; idx++) {
		pout_item[7 + idx] = pred_per_obj[84 + idx];
	}
}

static __global__ void process_mask_kernel(float* out, float* proto, uint8_t* dst , 
										   int dst_width, int dst_height, int out_w, int proto_size) {
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    int left = out[0] / 4;
    int top = out[1] / 4;

    int cx = left + dx;
    int cy = top + dy;
    if (cx >= dst_width || cy >= dst_height) return;
    if (cx < 0 || cx >= dst_width || cy < 0 || cy >= dst_height) {
        dst[cy * dst_width + cx] = 0;
        return;
    }
    float c = 0;
	for(int j = 0; j < out_w; ++j){
		c += out[j + 7] * proto[(j * dst_height + cy) * dst_width + cx];
	}
	if (sigmoid(c) > 0.5) {
        dst[cy * dst_width + cx] = 1;
	} else {
        dst[cy * dst_width + cx] = 0;
	}
}

void preprocess(uint8_t* src, AffineMatrix d2s, int src_width, 
				int src_height, float* dst, int dst_width, 
				int dst_height, cudaStream_t stream) {	
	int size = dst_height * dst_width;
	int threads = 256;
	int blocks = ceil(size / (float)threads);
	memcpy(img_buffer_host, src, 3 * src_height * src_width);
	CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, 
		3 * src_height * src_width, cudaMemcpyHostToDevice, stream));
	warpaffine_kernel_bilinear<<<blocks, threads, 0, stream>>>(
		img_buffer_device, src_width * 3, src_width,
		src_height, dst, dst_width,
		dst_height, 0, d2s, size);	
}

void postprocess_box(float* predict, int num_bboxes, int num_classes, int num_out,
	float conf_thr, float nms_thr, cudaStream_t stream, float* dst) {
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;
    decode_box_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_out, num_classes, 
												  conf_thr, out_buffer_device);
    block = 512;
    grid = (1000 + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(out_buffer_device, nms_thr, num_out);
	CUDA_CHECK(cudaMemcpyAsync(dst, out_buffer_device, sizeof(int) + 1000 * num_out * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

void yolov8_postprocess_box(float* predict, int num_bboxes, int num_classes, int num_out,
	float conf_thr, float nms_thr, cudaStream_t stream, float* dst) {
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;
    yolov8_decode_box_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_out, num_classes, 
												  conf_thr, out_buffer_device);
    block = 512;
    grid = (1000 + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(out_buffer_device, nms_thr, num_out);
	CUDA_CHECK(cudaMemcpyAsync(dst, out_buffer_device, sizeof(int) + 1000 * num_out * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

void rtdetr_postprocess_box(float* predict_box, float* predict_cls, int num_bboxes, int num_classes, int num_out,
	                        float conf_thr, int imageWidth, int imageHeight, cudaStream_t stream, float* dst) {
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;
    rtdetr_decode_box_kernel<<<grid, block, 0, stream>>>(predict_box, predict_cls, num_bboxes, num_out, num_classes, 
												  conf_thr,imageWidth, imageHeight, out_buffer_device);
	CUDA_CHECK(cudaMemcpyAsync(dst, out_buffer_device, sizeof(int) + 300 * num_out * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

void postprocess_box_mask(float* predict, int num_bboxes, int num_classes, 
	int num_out, float conf_thr, float nms_thr, cudaStream_t stream, float* dst) {
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;
    decode_box_mask_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_out, num_classes, 
												  conf_thr, out_buffer_device);
    block = 512;
    grid = (1000 + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(out_buffer_device, nms_thr, num_out);
	CUDA_CHECK(cudaMemcpyAsync(dst, out_buffer_device, sizeof(int) + 1000 * num_out * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

void yolov8_postprocess_box_mask(float* predict, int num_bboxes, int num_classes, 
	int num_out, float conf_thr, float nms_thr, cudaStream_t stream, float* dst) {
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;
    yolov8_decode_box_mask_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_out, num_classes, 
												  conf_thr, out_buffer_device);
    block = 512;
    grid = (1000 + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(out_buffer_device, nms_thr, num_out);
	CUDA_CHECK(cudaMemcpyAsync(dst, out_buffer_device, sizeof(int) + 1000 * num_out * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

void process_mask(float* out, float* proto, uint8_t* dst , int num_out, 
                  int dst_width, int dst_height, int out_w, int proto_size, cudaStream_t stream) {
	// int threads = 512;
	// int blocks = ceil(proto_size / threads);
	// CUDA_CHECK(cudamemcpyH)
	// memcpy(proto_buffer_host, proto, proto_size);
    int width = out[2] / 4 + 0.5f;
    int height = out[3] / 4 + 0.5f;
	dim3 blockSize(32, 32);
    dim3 gridSize((width + 31) / 32, (height + 31) / 32);   
	CUDA_CHECK(cudaMemcpyAsync(single_out_buffer_device, out, 
		num_out * sizeof(float), cudaMemcpyHostToDevice, stream));
	process_mask_kernel<<<gridSize, blockSize, 0, stream>>>(single_out_buffer_device, proto, out_mask_buffer_device, 
                        dst_width, dst_height, out_w, proto_size);
	CUDA_CHECK(cudaMemcpyAsync(dst, out_mask_buffer_device, sizeof(uint8_t) * (dst_width * dst_height), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
}
