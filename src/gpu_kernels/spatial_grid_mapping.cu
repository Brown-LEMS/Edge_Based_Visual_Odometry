#ifndef SPATIAL_GRID_MAPPING_CU
#define SPATIAL_GRID_MAPPING_CU

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "gpu_kernels.h"
#include "gpu_settings.h"

//> Assign each edge a grid cell ID based on its edge location
__global__ 
void 
spatial_grid_mapping_kernel(
    const Edge_GPU* d_edges, 
    int num_of_edges, 
    int num_of_grids_in_width,
    int num_of_grids_in_height,
    int* d_cell_ids) 
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_of_edges) {
        //> Find 2D grid coordinates
        int cx = (int)(d_edges[tid].location_x / GRID_CELL_SIZE);
        int cy = (int)(d_edges[tid].location_y / GRID_CELL_SIZE);
        cx = max(0, min(cx, num_of_grids_in_width - 1));
        cy = max(0, min(cy, num_of_grids_in_height - 1));
        
        //> Flatten 2D grid coordinates to a 1D cell id
        d_cell_ids[tid] = cx + cy * num_of_grids_in_width;
    }
}

//> Find the start index and count the number of edges for each cell after thrust sort
__global__ 
void 
find_grid_boundaries_kernel(
    const int* sorted_cell_ids, 
    int* cell_start_idx, 
    int* num_of_edges_per_cell, 
    int num_points) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_points) {
        int cell_of_tid = sorted_cell_ids[tid];
        
        //> Handle the very first element
        if (tid == 0) {
            cell_start_idx[cell_of_tid] = 0;
        } 
        else {
            int prev_cell_id = sorted_cell_ids[tid - 1];
            
            //> If the cell id of the current edge (cell_of_tid) is different from the cell id of the previous edge (prev_cell_id),
            //> it means a new cell block starts exactly at 'tid'
            if (cell_of_tid != prev_cell_id) {
                cell_start_idx[cell_of_tid] = tid;
                
                //> Compute the number of edges in the previous cell
                num_of_edges_per_cell[prev_cell_id] = tid - cell_start_idx[prev_cell_id];
            }
        }
        
        //> Handle the very last element to close the final count
        if (tid == num_points - 1) {
            num_of_edges_per_cell[cell_of_tid] = num_points - cell_start_idx[cell_of_tid];
        }
    }
}

void spatial_grid_mapping_kernel_launcher(
    int device_id,
    const Edge_GPU* d_edges,
    int num_of_edges,
    int num_of_grids_in_width,
    int num_of_grids_in_height,
    int* d_cell_ids)
{
    cudacheck(cudaSetDevice(device_id));

    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    //> One thread per edge: block size must stay within device limit (typically less than or equal to 1024)
    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block) ? 
                                  (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING) : (max_threads_per_block);
    const int num_blocks = (num_of_edges + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    if (num_of_edges > 0) {
        cuda_assert_device_pointer(d_edges, "d_edges");
        cuda_assert_device_pointer(d_cell_ids, "d_cell_ids");
    }

    spatial_grid_mapping_kernel<<<grid_dim, block_dim>>>(
        d_edges, num_of_edges, num_of_grids_in_width, num_of_grids_in_height, d_cell_ids);
    cudacheck(cudaGetLastError());
}

void find_grid_boundaries_kernel_launcher(
    int device_id,
    const int* d_cell_ids,
    int* d_cell_start_idx,
    int* d_num_of_edges_per_cell,
    int num_of_edges)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    //> One thread per edge: block size must stay within device limit (typically less than or equal to 1024)
    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block) ? 
                                  (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING) : (max_threads_per_block);
    const int num_blocks = (num_of_edges + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);
    
    void *kernel_args[] = {
        reinterpret_cast<void*>(&d_cell_ids),
        reinterpret_cast<void*>(&d_cell_start_idx),
        reinterpret_cast<void*>(&d_num_of_edges_per_cell),
        reinterpret_cast<void*>(&num_of_edges)
    };

    cudacheck(cudaLaunchKernel( reinterpret_cast<const void*>(find_grid_boundaries_kernel), grid_dim, block_dim, kernel_args, 0, nullptr) );
}

float edge_spatial_map_thrust_pipeline(
    int             device_id,
    /* Data live in the host memory */
    const Edge_GPU* h_right_edges,
    int*            &h_cell_ids_out,
    /* Data live in the device memory */
    Edge_GPU*       d_right_edges,
    int*            d_cell_ids,
    int*            d_edge_indices,
    int*            d_cell_start_idx,
    int*            d_num_of_edges_in_cell,
    /* Others */
    std::size_t     num_right_edges,
    int             total_num_of_grid_cells,
    int             num_of_grid_cells_in_width,
    int             num_of_grid_cells_in_height,
    cudaEvent_t     start,
    cudaEvent_t     stop)
{
    cudacheck( cudaSetDevice(device_id) );

    //> Wrap raw pointers in thrust::device_ptr; this allows Thrust to interact with memory allocated via cudaMalloc
    thrust::device_ptr<int> t_cell_ids(d_cell_ids);
    thrust::device_ptr<int> t_edge_indices(d_edge_indices);
    thrust::device_ptr<int> t_cell_start_idx(d_cell_start_idx);
    thrust::device_ptr<int> t_num_of_edges_in_cell(d_num_of_edges_in_cell);

    //> Start the CUDA event timer
    cudacheck(cudaEventRecord(start));

    //> Copy edges data from host to device
    cudacheck( cudaMemcpy( d_right_edges, h_right_edges, num_right_edges * sizeof(Edge_GPU), cudaMemcpyHostToDevice) );
    if (num_right_edges > 0) {
        cuda_assert_device_pointer(d_right_edges, "d_right_edges");
        cuda_assert_device_pointer(d_cell_ids, "d_cell_ids");
    }

    //> Initialize arrays using Thrust. By default the cells are empty.
    thrust::fill(t_cell_start_idx, t_cell_start_idx + total_num_of_grid_cells, 0);
    thrust::fill(t_num_of_edges_in_cell, t_num_of_edges_in_cell + total_num_of_grid_cells, 0);

    //> Fill edge indices with incremental integers used to track original edges during sort
    thrust::sequence(t_edge_indices, t_edge_indices + num_right_edges);

    //> Create and Launch the GPU kernel: assign each right edge a grid cell id based on its location
    spatial_grid_mapping_kernel_launcher(device_id, d_right_edges, static_cast<int>(num_right_edges),
                                         num_of_grid_cells_in_width, num_of_grid_cells_in_height, d_cell_ids);
    cudacheck(cudaDeviceSynchronize());

    //> Sort the indices using the Cell IDs as the key. This groups all points in Cell 0 together, then Cell 1, etc.
    //> It rearranges 't_edge_indices' to match the sorted 't_cell_ids'.
    thrust::sort_by_key(t_cell_ids, t_cell_ids + num_right_edges, t_edge_indices);

    // //> Launch the kernel of finding the boundaries of the sorted arrays
    // find_grid_boundaries_kernel_launcher( device_id, d_cell_ids, d_cell_start_idx, d_num_of_edges_in_cell, static_cast<int>(num_right_edges) );
    // cudacheck(cudaDeviceSynchronize());

    //> Build boundaries from sorted cell ids on host to guarantee non-overlapping ranges.
    //> This avoids subtle indexing races/corruption that can create duplicate candidate scans.
    std::vector<int> h_sorted_cell_ids(num_right_edges, 0);
    std::vector<int> h_cell_start_idx(total_num_of_grid_cells, 0);
    std::vector<int> h_num_of_edges_in_cell(total_num_of_grid_cells, 0);
    cudacheck(cudaMemcpy(h_sorted_cell_ids.data(), d_cell_ids, num_right_edges * sizeof(int), cudaMemcpyDeviceToHost));

    if (num_right_edges > 0) {
        int first_cell = h_sorted_cell_ids[0];
        h_cell_start_idx[first_cell] = 0;
        for (int tid = 1; tid < static_cast<int>(num_right_edges); ++tid) {
            int cell_of_tid = h_sorted_cell_ids[tid];
            int prev_cell = h_sorted_cell_ids[tid - 1];
            if (cell_of_tid != prev_cell) {
                h_cell_start_idx[cell_of_tid] = tid;
                h_num_of_edges_in_cell[prev_cell] = tid - h_cell_start_idx[prev_cell];
            }
        }
        int last_cell = h_sorted_cell_ids[static_cast<int>(num_right_edges) - 1];
        h_num_of_edges_in_cell[last_cell] = static_cast<int>(num_right_edges) - h_cell_start_idx[last_cell];
    }

    cudacheck(cudaMemcpy(d_cell_start_idx, h_cell_start_idx.data(),
                         total_num_of_grid_cells * sizeof(int), cudaMemcpyHostToDevice));
    cudacheck(cudaMemcpy(d_num_of_edges_in_cell, h_num_of_edges_in_cell.data(),
                         total_num_of_grid_cells * sizeof(int), cudaMemcpyHostToDevice));

    //> End the CUDA event timer for spatial grid mapping
    float build_spatial_grid_time = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&build_spatial_grid_time, start, stop));

    //> TEST: Copy cell ids data from device to host
    #if COPY_DATA_FROM_HOST_TO_DEVICE_FOR_TESTING
    cudacheck( cudaMemcpy(h_cell_ids_out, d_cell_ids, num_right_edges * sizeof(int), cudaMemcpyDeviceToHost) );
    #endif

    return build_spatial_grid_time;
}

#endif // SPATIAL_GRID_MAPPING_CU