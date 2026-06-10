#ifndef EVAL_EDGES_IN_GRID_CELLS_CUH
#define EVAL_EDGES_IN_GRID_CELLS_CUH

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>
#include <iostream>
#include <math.h>
#include "gpu_kernels.h"
#include "definitions.h"

__device__ __forceinline__
bool edge_pair_passes_orientation_prefilter_deg(float ori_left, float ori_right, float tol_deg)
{
    float diff_rad = fabsf(ori_left - ori_right);
    float diff_deg = diff_rad * (180.f / CUDART_PI_F);
    if (diff_deg > 180.f)
        diff_deg = 360.f - diff_deg;
    if (diff_deg < tol_deg)
        return true;
    if (fabsf(diff_deg - 180.f) < tol_deg)
        return true;
    return false;
}

__device__ __forceinline__
void
eval_edges_in_grid_cells(
    Edge_GPU left_edge,
    int cx, int cy, int grid_width, 
    int left_edge_idx, float a, float b, float c,
    float seg_x0, float seg_y0, float seg_x1, float seg_y1,
    const Edge_GPU* d_right_edges, const int* d_edge_indices,
    const int* d_cell_start_idx, const int* d_cell_counts,
    Match_by_Edge_Index* d_matches, int* d_match_count, int max_matches)
{
    int cell_id = cy * grid_width + cx;
    int start_idx = d_cell_start_idx[cell_id];
    int count = d_cell_counts[cell_id];

    for (int i = 0; i < count; i++) {
        //> Memory Lookup: Get the original index, then the actual point
        int right_edge_idx = d_edge_indices[start_idx + i];
        Edge_GPU right_edge = d_right_edges[right_edge_idx];

        //> Standard left <-> right stereo: match must lie on finite epipolar segment in the right image (t in [0,1]),
        //> not only near the infinite line
        // if (left_edge.location_x < right_edge.location_x)
        //     continue;

        float dist_to_epipolar_line = fabsf(a * right_edge.location_x + b * right_edge.location_y + c);

        float vx_seg = seg_x1 - seg_x0;
        float vy_seg = seg_y1 - seg_y0;
        float v2_seg = vx_seg * vx_seg + vy_seg * vy_seg;
        if (v2_seg < 1e-6f)
            continue;
        float t_seg = ((right_edge.location_x - seg_x0) * vx_seg + (right_edge.location_y - seg_y0) * vy_seg) / v2_seg;
        if (t_seg < 0.f || t_seg > 1.f)
            continue;

        //> Squared Euclidean “disparity” (CPU apply_Disparity_Filtering uses cv::norm on 2D offset)
        float disparity_squared = (left_edge.location_x - right_edge.location_x) * (left_edge.location_x - right_edge.location_x) + \
                                  (left_edge.location_y - right_edge.location_y) * (left_edge.location_y - right_edge.location_y);

        //> Orientation prefilter
        bool b_pass_OR = edge_pair_passes_orientation_prefilter_deg( left_edge.orientation, right_edge.orientation, static_cast<float>(EDGE_PAIR_ORIENTATION_PREFILTER_DEG) );

        //> EP + disparity + orientation
        if ( dist_to_epipolar_line < EPIPOLAR_LINE_DIST_THRESH && disparity_squared < MAX_DISPARITY * MAX_DISPARITY && b_pass_OR ) {
            int insert_pos = atomicAdd(d_match_count, 1);
            if (insert_pos < max_matches) {
                d_matches[insert_pos].left_edge_idx = left_edge_idx;
                d_matches[insert_pos].right_edge_idx = right_edge_idx;
            }
        }
    }
}

#endif