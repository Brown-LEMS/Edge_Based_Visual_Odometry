#ifndef EPIPOLAR_GEOM_HELPERS_CUH
#define EPIPOLAR_GEOM_HELPERS_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
#include <math.h>
#include "gpu_kernels.h"

__device__ __forceinline__ 
void 
project_point_to_normalized_line(float px, float py, float a, float b, float c, float* ox, float* oy)
{
    float rho = a * px + b * py + c;
    *ox = px - a * rho;
    *oy = py - b * rho;
}

//> Matches Stereo_Matches::refine_edge_disparity tangent from epip coeffs (degrees use (-coeff1, coeff0)).
__device__ __forceinline__ 
void 
get_unit_epipolar_tangent(float na, float nb, float* tx, float* ty)
{
    float vx = -nb;
    float vy = na;
    float len = sqrtf(vx * vx + vy * vy);
    if (len < 1e-8f) {
        *tx = 1.0f;
        *ty = 0.0f;
        return;
    }
    *tx = vx / len;
    *ty = vy / len;
}

#endif