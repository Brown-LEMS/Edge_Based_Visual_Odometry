#ifndef PATCH_PHOTOMETRY_HELPERS_CUH
#define PATCH_PHOTOMETRY_HELPERS_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
#include <math.h>
#include "gpu_kernels.h"
#include "get_interp_photometry_in_texture.cuh"

__device__ __forceinline__ 
float 
mean_patch(
    const float* __restrict__ p)
{
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < TOTAL_NUM_OF_PATCH_PIXELS; ++k)
        s += p[k];
    return s / static_cast<float>(TOTAL_NUM_OF_PATCH_PIXELS);
}

static __device__ __forceinline__ 
void 
sample_rot_patch_I(
    cudaTextureObject_t tex,
    float cx, 
    float cy, 
    float cos_L, 
    float sin_L, 
    int w, int h, 
    float* __restrict__ pix)
{
    int ix = 0;
    for (int vv = -PATCH_RADIUS; vv <= PATCH_RADIUS; ++vv) {
        for (int uu = -PATCH_RADIUS; uu <= PATCH_RADIUS; ++uu) {
            float sx = cx + static_cast<float>(uu) * cos_L - static_cast<float>(vv) * sin_L;
            float sy = cy + static_cast<float>(uu) * sin_L + static_cast<float>(vv) * cos_L;
            pix[ix++] = sample_I(tex, sx, sy, w, h);
        }
    }
}

static __device__ __forceinline__ 
void 
sample_rot_patch_gproj(
    cudaTextureObject_t tex,
    float cx, float cy,
    float cos_L, float sin_L,
    float dirx, float diry,
    int w, int h,
    float* __restrict__ g_proj)
{
    int ix = 0;
    for (int vv = -PATCH_RADIUS; vv <= PATCH_RADIUS; ++vv) {
        for (int uu = -PATCH_RADIUS; uu <= PATCH_RADIUS; ++uu) {
            float sx = cx + static_cast<float>(uu) * cos_L - static_cast<float>(vv) * sin_L;
            float sy = cy + static_cast<float>(uu) * sin_L + static_cast<float>(vv) * cos_L;
            float gx = 0.0f, gy = 0.0f;
            sample_grad_I(tex, sx, sy, w, h, gx, gy);
            g_proj[ix++] = -(gx * dirx + gy * diry);
        }
    }
}

#endif