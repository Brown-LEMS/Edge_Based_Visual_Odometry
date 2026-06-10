#ifndef EPIPOLAR_RASTERIZATION_CU
#define EPIPOLAR_RASTERIZATION_CU

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>

#include "definitions.h"
#include "gpu_kernels.h"
#include "gpu_settings.h"
#include "gpu_helpers/get_epipolar_line_end_points.cuh"
#include "gpu_helpers/eval_edges_in_grid_cells.cuh"

__global__ void epipolar_rasterization_thick_band_kernel(
    const Edge_GPU* d_left_edges,       //> Left edge array
    const Edge_GPU* d_right_edges,      //> Right edge array 
    const int* d_edge_indices,          //> Pointer from Thrust (sorted right edge indices)
    const int* d_cell_start_idx,        //> Pointer from Thrust (start index of the right edge indices in each grid cell)
    const int* d_num_of_edges_in_cell,  //> Pointer from Thrust (number of right edge indices in each grid cell)
    const float* d_F,                   //> Fundamental matrix (3x3, flattened to 9 floats)
    int num_of_left_edges,              //> Number of left edges
    int img_right_W, int img_right_H,   //> Width and height of the right image
    int num_of_grid_cells_in_width,     //> Number of grid cells in the image width
    int num_of_grid_cells_in_height,    //> Number of grid cells in the image height
    Match_by_Edge_Index* d_matches,     //> Output buffer
    int* d_match_count,                 //> Global atomic counter for outputs
    int max_matches)                    //> To prevent buffer overflow (max number of matches)
{
    int left_edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (left_edge_idx >= num_of_left_edges) return;

    Edge_GPU left_edge = d_left_edges[left_edge_idx];

    //> Compute and normalize epipolar line
    float a = d_F[0] * left_edge.location_x + d_F[1] * left_edge.location_y + d_F[2];
    float b = d_F[3] * left_edge.location_x + d_F[4] * left_edge.location_y + d_F[5];
    float c = d_F[6] * left_edge.location_x + d_F[7] * left_edge.location_y + d_F[8];

    float norm = sqrtf(a * a + b * b);
    if (norm < 1e-6f) return; 
    a /= norm; b /= norm; c /= norm;

    //> Find intersection points with the image boundaries
    float x0, y0, x1, y1;
    if (!get_epipolar_line_endpoints(a, b, c, img_right_W, img_right_H, x0, y0, x1, y1)) return;

    //> Core idea: major-axis thick band traversal
    if (fabsf(b) > fabsf(a)) {
        //> If the epipolar line is more horizontal, perform x-major traversal
        
        //> Find grid column boundaries
        int cx_0 = max(0, min((int)(x0 / GRID_CELL_SIZE), num_of_grid_cells_in_width - 1));
        int cx_1 = max(0, min((int)(x1 / GRID_CELL_SIZE), num_of_grid_cells_in_width - 1));
        int start_cx = min(cx_0, cx_1);
        int end_cx = max(cx_0, cx_1);
        
        //> Compute how much the epipolar line needs to expand vertically to cover the thick band
        float vertical_expansion = EPIPOLAR_LINE_DIST_THRESH / fabsf(b);

        for (int cx = start_cx; cx <= end_cx; cx++) {
            float x_left = cx * GRID_CELL_SIZE;
            float x_right = (cx + 1) * GRID_CELL_SIZE;
            
            //> Get the y coordinates where the line enters and exits this column
            float y_left = (-a * x_left - c) / b;
            float y_right = (-a * x_right - c) / b;
            
            //> Expand the bounding box by the thickness
            float y_min = fminf(y_left, y_right) - vertical_expansion;
            float y_max = fmaxf(y_left, y_right) + vertical_expansion;
            
            //> Convert to cell indices; adopting floorf to handle negative values safely before clamping
            int cy_start = max(0, (int)floorf(y_min / GRID_CELL_SIZE));
            int cy_end = min(num_of_grid_cells_in_height - 1, (int)floorf(y_max / GRID_CELL_SIZE));
            
            //> Evaluate edges in the grid cells
            for (int cy = cy_start; cy <= cy_end; cy++) {
                eval_edges_in_grid_cells(left_edge, cx, cy, num_of_grid_cells_in_width, left_edge_idx, a, b, c,
                                         x0, y0, x1, y1,
                                         d_right_edges, d_edge_indices, d_cell_start_idx, d_num_of_edges_in_cell,
                                         d_matches, d_match_count, max_matches);
            }
        }
    } 
    else {
        //> If the epipolar line is more vertical, perform y-major traversal
        
        //> Find grid row boundaries
        int cy_0 = max(0, min((int)(y0 / GRID_CELL_SIZE), num_of_grid_cells_in_height - 1));
        int cy_1 = max(0, min((int)(y1 / GRID_CELL_SIZE), num_of_grid_cells_in_height - 1));
        int start_cy = min(cy_0, cy_1);
        int end_cy = max(cy_0, cy_1);
        
        //> Compute how much the epipolar line needs to expand horizontally to cover the thick band
        float horizontal_expansion = EPIPOLAR_LINE_DIST_THRESH / fabsf(a);

        for (int cy = start_cy; cy <= end_cy; cy++) {
            float y_top = cy * GRID_CELL_SIZE;
            float y_bottom = (cy + 1) * GRID_CELL_SIZE;
            
            //> Get the x coordinates where the line enters and exits this row
            float x_top = (-b * y_top - c) / a;
            float x_bottom = (-b * y_bottom - c) / a;
            
            //> Expand the bounding box by the thickness
            float x_min = fminf(x_top, x_bottom) - horizontal_expansion;
            float x_max = fmaxf(x_top, x_bottom) + horizontal_expansion;
            
            //> Convert to cell indices; adopting floorf to handle negative values safely before clamping
            int cx_start = max(0, (int)floorf(x_min / GRID_CELL_SIZE));
            int cx_end = min(num_of_grid_cells_in_width - 1, (int)floorf(x_max / GRID_CELL_SIZE));
            
            //> Evaluate edges in the grid cells
            for (int cx = cx_start; cx <= cx_end; cx++) {
                eval_edges_in_grid_cells(left_edge, cx, cy, num_of_grid_cells_in_width, left_edge_idx, a, b, c,
                                         x0, y0, x1, y1,
                                         d_right_edges, d_edge_indices, d_cell_start_idx, d_num_of_edges_in_cell,
                                         d_matches, d_match_count, max_matches);
            }
        }
    }
}

void epipolar_rasterization_thick_band_kernel_launcher(
    int device_id,
    const Edge_GPU* d_left_edges, 
    const Edge_GPU* d_right_edges, 
    const int* d_edge_indices, 
    const int* d_cell_start_idx, 
    const int* d_num_of_edges_in_cell, 
    const float* d_F, 
    int num_of_left_edges, 
    int img_right_W, int img_right_H, 
    int num_of_grid_cells_in_width, 
    int num_of_grid_cells_in_height, 
    Match_by_Edge_Index* d_matches, 
    int* d_match_count, 
    int max_matches)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    //> One thread per edge: block size must stay within device limit (typically less than or equal to 1024)
    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block) ? 
                                  (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING) : (max_threads_per_block);
    const int num_blocks = (num_of_left_edges + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);
    
    void *kernel_args[] = {
        reinterpret_cast<void*>(&d_left_edges),
        reinterpret_cast<void*>(&d_right_edges),
        reinterpret_cast<void*>(&d_edge_indices),
        reinterpret_cast<void*>(&d_cell_start_idx),
        reinterpret_cast<void*>(&d_num_of_edges_in_cell),
        reinterpret_cast<void*>(&d_F),
        reinterpret_cast<void*>(&num_of_left_edges),
        reinterpret_cast<void*>(&img_right_W),
        reinterpret_cast<void*>(&img_right_H),
        reinterpret_cast<void*>(&num_of_grid_cells_in_width),
        reinterpret_cast<void*>(&num_of_grid_cells_in_height),
        reinterpret_cast<void*>(&d_matches),
        reinterpret_cast<void*>(&d_match_count),
        reinterpret_cast<void*>(&max_matches)
    };

    cudacheck(cudaLaunchKernel( reinterpret_cast<const void*>(epipolar_rasterization_thick_band_kernel), grid_dim, block_dim, kernel_args, 0, nullptr) );
}

float epipolar_rasterization_thick_band_pipeline(
    int                     device_id,
    /* Data live in the host memory, used for results retrieval */
    const float*            h_F,
    const Edge_GPU*         h_left_edges,
    int                     &h_match_count_out,
    Match_by_Edge_Index*    &h_matches_out,
    /* Data live in the device memory */
    Edge_GPU*               d_left_edges,
    const Edge_GPU*         d_right_edges, 
    const int*              d_edge_indices,
    const int*              d_cell_start_idx, 
    const int*              d_num_of_edges_in_cell, 
    float*                  d_F,
    Match_by_Edge_Index*    d_matches, 
    int*                    d_match_count, 
    /* Others */
    int                     num_of_left_edges, 
    int                     img_right_W, 
    int                     img_right_H, 
    int                     num_of_grid_cells_in_width, 
    int                     num_of_grid_cells_in_height, 
    int                     max_matches,
    cudaEvent_t             start,
    cudaEvent_t             stop)
{
    cudacheck(cudaSetDevice(device_id));

    //> Start the CUDA event timer
    cudacheck(cudaEventRecord(start));
    
    //> Copy data from host to device
    cudacheck( cudaMemcpy( d_F, h_F, 9 * sizeof(float), cudaMemcpyHostToDevice ) );
    cudacheck( cudaMemcpy( d_left_edges,  h_left_edges, num_of_left_edges  * sizeof(Edge_GPU), cudaMemcpyHostToDevice ) );

    //> Initialize the number of edges per grid cell as 0
    cudacheck( cudaMemset(d_match_count,  0, sizeof(int)) );

    //> Create and Launch the GPU kernel: assign each right edge a grid cell id based on its location
    epipolar_rasterization_thick_band_kernel_launcher( device_id, d_left_edges, d_right_edges, d_edge_indices, d_cell_start_idx, d_num_of_edges_in_cell, d_F, num_of_left_edges, \
                                            img_right_W, img_right_H, num_of_grid_cells_in_width, num_of_grid_cells_in_height, \
                                            d_matches, d_match_count, max_matches );
    cudacheck(cudaDeviceSynchronize());

    //> End the CUDA event timer for spatial grid mapping
    float epipolar_rasterization_time = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&epipolar_rasterization_time, start, stop));

    //> Copy match count to host for downstream stages (must stay in sync with d_match_count)
    h_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_match_count_out, d_match_count, sizeof(int), cudaMemcpyDeviceToHost));

    //> Clamp count just in case it exceeded the buffer size
    if (h_match_count_out > max_matches) {
        LOG_WARNING("The number of matches exceeded the buffer size");
        h_match_count_out = max_matches;
    }

    //> TEST: retrieve results
    #if COPY_DATA_FROM_HOST_TO_DEVICE_FOR_TESTING
    if (h_match_count_out > 0) {
        cudacheck(cudaMemcpy(h_matches_out, d_matches, h_match_count_out * sizeof(Match_by_Edge_Index),
                             cudaMemcpyDeviceToHost));
    }
    #endif

    return epipolar_rasterization_time;
}

#endif