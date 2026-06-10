#ifndef EXTRACT_GPU_SIFT_DESCRIPTOR_CUH
#define EXTRACT_GPU_SIFT_DESCRIPTOR_CUH

#include <cuda_runtime.h>
#include <math.h>

#include "gpu_settings.h"

static constexpr float PI_F = 3.14159265358979323846f;
static constexpr float TWO_PI_F = 2.0f * PI_F;

__device__ __forceinline__
float wrap_angle_0_2pi(float angle)
{
    while (angle < 0.0f)
        angle += TWO_PI_F;
    while (angle >= TWO_PI_F)
        angle -= TWO_PI_F;
    return angle;
}

__device__ __forceinline__
float tex_clamped(cudaTextureObject_t tex, float x, float y, int width, int height)
{
    x = fminf(fmaxf(x, 0.0f), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0f), static_cast<float>(height - 1));
    return tex2D<float>(tex, x, y);
}

__device__ __forceinline__
void extract_gpu_sift_descriptor(
    cudaTextureObject_t tex,
    int width,
    int height,
    float cx,
    float cy,
    float orientation,
    float* desc)
{
    #pragma unroll
    for (int i = 0; i < SIFT_DESCRIPTOR_DIM; ++i)
        desc[i] = 0.0f;

    const int grid = SIFT_DESCRIPTOR_GRID_SIZE;
    const int bins = SIFT_DESCRIPTOR_ORIENTATION_BINS;
    const int sample_width = SIFT_DESCRIPTOR_SAMPLE_WIDTH;
    const int half_width = sample_width / 2;
    const float cell_width = static_cast<float>(sample_width) / static_cast<float>(grid);
    const float cos_t = cosf(orientation);
    const float sin_t = sinf(orientation);
    const float sigma = 0.5f * static_cast<float>(sample_width);
    const float inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma);

    for (int yy = -half_width; yy < half_width; ++yy) {
        for (int xx = -half_width; xx < half_width; ++xx) {
            const float local_x = static_cast<float>(xx) + 0.5f;
            const float local_y = static_cast<float>(yy) + 0.5f;

            const int cell_x = static_cast<int>((local_x + static_cast<float>(half_width)) / cell_width);
            const int cell_y = static_cast<int>((local_y + static_cast<float>(half_width)) / cell_width);
            if (cell_x < 0 || cell_x >= grid || cell_y < 0 || cell_y >= grid)
                continue;

            const float img_x = cx + local_x * cos_t - local_y * sin_t;
            const float img_y = cy + local_x * sin_t + local_y * cos_t;

            const float gx = tex_clamped(tex, img_x + 1.0f, img_y, width, height) -
                             tex_clamped(tex, img_x - 1.0f, img_y, width, height);
            const float gy = tex_clamped(tex, img_x, img_y + 1.0f, width, height) -
                             tex_clamped(tex, img_x, img_y - 1.0f, width, height);
            const float mag = sqrtf(gx * gx + gy * gy);
            if (mag <= 1e-6f)
                continue;

            const float rel_angle = wrap_angle_0_2pi(atan2f(gy, gx) - orientation);
            int bin = static_cast<int>(floorf(rel_angle * static_cast<float>(bins) / TWO_PI_F));
            if (bin >= bins)
                bin = bins - 1;

            const float gaussian = expf(-(local_x * local_x + local_y * local_y) * inv_two_sigma2);
            const int idx = (cell_y * grid + cell_x) * bins + bin;
            desc[idx] += mag * gaussian;
        }
    }

    float norm_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < SIFT_DESCRIPTOR_DIM; ++i)
        norm_sq += desc[i] * desc[i];
    float inv_norm = rsqrtf(fmaxf(norm_sq, 1e-12f));

    norm_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < SIFT_DESCRIPTOR_DIM; ++i) {
        desc[i] = fminf(desc[i] * inv_norm, 0.2f);
        norm_sq += desc[i] * desc[i];
    }
    inv_norm = rsqrtf(fmaxf(norm_sq, 1e-12f));
    #pragma unroll
    for (int i = 0; i < SIFT_DESCRIPTOR_DIM; ++i)
        desc[i] = desc[i] * inv_norm * static_cast<float>(SIFT_DESCRIPTOR_SCALE);
}

__device__ __forceinline__
float sift_l2_distance(const float* a, const float* b)
{
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < SIFT_DESCRIPTOR_DIM; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

#endif // EXTRACT_GPU_SIFT_DESCRIPTOR_CUH
