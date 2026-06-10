#ifndef GET_INTERP_PHOTOMETRY_IN_TEXTURE_CUH
#define GET_INTERP_PHOTOMETRY_IN_TEXTURE_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
#include <math.h>
#include "gpu_kernels.h"

__device__ __forceinline__ float clamp_tex_xy(float x, int limit_exclusive) {
    return fminf(fmaxf(x, 0.0f), static_cast<float>(limit_exclusive - 1));
}

__device__ __forceinline__ float sample_I(cudaTextureObject_t tex, float x, float y, int w, int h) {
    x = clamp_tex_xy(x, w);
    y = clamp_tex_xy(y, h);
    return tex2D<float>(tex, x, y);
}

//> Sobel-like central differences on the intensity texture
//> TODO: convolve the image with the Gaussian derivatives
__device__ __forceinline__ void sample_grad_I(cudaTextureObject_t tex, float x, float y, int w, int h, float& gx, float& gy)
{
    gx = 0.5f * (sample_I(tex, x + 1.0f, y, w, h) - sample_I(tex, x - 1.0f, y, w, h));
    gy = 0.5f * (sample_I(tex, x, y + 1.0f, w, h) - sample_I(tex, x, y - 1.0f, w, h));
}

#endif