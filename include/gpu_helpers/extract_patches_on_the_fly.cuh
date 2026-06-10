#ifndef EXTRACT_PATCHES_ON_THE_FLY_CUH
#define EXTRACT_PATCHES_ON_THE_FLY_CUH

#include <cuda_runtime.h>
#include <math.h>

#include "gpu_settings.h"

//> Helper to extract a patch directly from texture into GPU thread-local registers
static __device__ __forceinline__ void extract_patch_stats_on_the_fly(
    cudaTextureObject_t imgTex,
    float shifted_cx, float shifted_cy, 
    float cos_t, float sin_t,
    /* outputs */
    float* patch_data, 
    float& mean, float& variance)
{
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;
    
    //> Loop over the patch pixels
    for (int v = -PATCH_RADIUS; v <= PATCH_RADIUS; v++) {
        for (int u = -PATCH_RADIUS; u <= PATCH_RADIUS; u++) {
            float sx = shifted_cx + u * cos_t - v * sin_t;
            float sy = shifted_cy + u * sin_t + v * cos_t;
            
            float val = tex2D<float>(imgTex, sx, sy);
            patch_data[count++] = val;
            sum += val;
            sum_sq += val * val;
        }
    }
    
    int total_pixels = (2 * PATCH_RADIUS + 1) * (2 * PATCH_RADIUS + 1);
    mean = sum / total_pixels;
    variance = (sum_sq / total_pixels) - (mean * mean);
    variance = (variance < 1e-5f) ? (1e-5f) : variance;
}

#endif