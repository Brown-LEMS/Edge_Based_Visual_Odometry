#ifndef GET_EPIPOLAR_LINE_END_POINTS_CUH
#define GET_EPIPOLAR_LINE_END_POINTS_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
#include <math.h>
#include "gpu_kernels.h"

//> Helper function for finding the intersection of the epipolar line and the image boundaries
__device__ bool get_epipolar_line_endpoints(
    float a, float b, float c, 
    int W, int H, 
    float& x0, float& y0, float& x1, float& y1) 
{
    Point2D_GPU intersection_points[2];
    int count = 0;

    //> Check intersection with x = 0 and x = W
    if (abs(b) > 1e-6f) {
        float y_left = -c / b;
        if (y_left >= 0 && y_left <= H) 
            intersection_points[count++] = {0.0f, y_left};
        
        float y_right = -(a * W + c) / b;
        if (y_right >= 0 && y_right <= H && count < 2) 
            intersection_points[count++] = {static_cast<float>(W), y_right};
    }
    
    //> Check intersection with y = 0 and y = H
    if (abs(a) > 1e-6f && count < 2) {
        float x_top = -c / a;
        if (x_top >= 0 && x_top <= W) 
            intersection_points[count++] = {x_top, 0.0f};
        
        if (count < 2) {
            float x_bottom = -(b * H + c) / a;
            if (x_bottom >= 0 && x_bottom <= W) 
                intersection_points[count++] = {x_bottom, static_cast<float>(H)};
        }
    }

    //> If there are two intersection points, return true
    if (count == 2) {
        x0 = intersection_points[0].x; y0 = intersection_points[0].y;
        x1 = intersection_points[1].x; y1 = intersection_points[1].y;
        return true;
    }
    return false; // Line does not intersect the image
}

#endif // GET_EPIPOLAR_LINE_END_POINTS_CUH