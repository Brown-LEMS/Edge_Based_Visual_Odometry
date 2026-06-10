#ifndef COMPUTE_NCC_CUH
#define COMPUTE_NCC_CUH

#include <cuda_runtime.h>
#include <math.h>

#include "gpu_settings.h"

static __device__ __forceinline__ 
float 
compute_NCC(
    const float* p1, float m1, float v1,
    const float* p2, float m2, float v2)
{
    float cross_cov = 0.0f;
    for (int i = 0; i < TOTAL_NUM_OF_PATCH_PIXELS; i++) {
        cross_cov += (p1[i] - m1) * (p2[i] - m2);
    }
    cross_cov /= TOTAL_NUM_OF_PATCH_PIXELS;
    return cross_cov / sqrtf(v1 * v2);
}

#endif