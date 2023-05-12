#include "cuda_preprocess.h"

static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;

void cuda_preprocess_init(int max_image_size) {
	// prepare input data in pinned memory
	CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));
	// prepare input data in device memory
	CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy() {
  CUDA_CHECK(cudaFree(img_buffer_device));
  CUDA_CHECK(cudaFreeHost(img_buffer_host));
}


static __global__ void warpaffine_kernel_bilinear(
    uint8_t* src, int src_line_size, int src_width,
    int src_height, float* dst, int dst_width,
    int dst_height, uint8_t padding_value,
    AffineMatrix d2s, int edge) {
	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position >= edge) return;

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

void preprocess(uint8_t* src, AffineMatrix d2s, int src_width, 
				int src_height, float* dst, int dst_width, 
				int dst_height, cudaStream_t stream) {
    dim3 block_size(32, 32);
    dim3 grid_size((dst_width + 31) / 32, (dst_height + 31) / 32); 		
	int jobs = dst_height * dst_width;
	int threads = 256;
	int blocks = ceil(jobs / (float)threads);
	memcpy(img_buffer_host, src, 3 * src_height * src_width);
	CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, 
		3 * src_height * src_width, cudaMemcpyHostToDevice, stream));
	warpaffine_kernel_bilinear<<<blocks, threads, 0, stream>>>(
		img_buffer_device, src_width * 3, src_width,
		src_height, dst, dst_width,
		dst_height, 0, d2s, jobs);	
}
