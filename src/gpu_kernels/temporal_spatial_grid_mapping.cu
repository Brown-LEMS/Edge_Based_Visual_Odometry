#ifndef TEMPORAL_SPATIAL_GRID_MAPPING_CU
#define TEMPORAL_SPATIAL_GRID_MAPPING_CU

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "gpu_kernels.h"
#include "gpu_settings.h"

//> Grid-cell assignment for final stereo mates (mate index == array index).
__global__
void temporal_stereo_mate_spatial_grid_mapping_kernel(
    const Merged_Refined_Stereo_Match_GPU* d_stereo_matches,
    int num_of_stereo_mates,
    int num_of_grids_in_width,
    int num_of_grids_in_height,
    int* d_cell_ids)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_of_stereo_mates) {
        int cx = static_cast<int>(d_stereo_matches[tid].left_location_x / GRID_CELL_SIZE);
        int cy = static_cast<int>(d_stereo_matches[tid].left_location_y / GRID_CELL_SIZE);
        cx = max(0, min(cx, num_of_grids_in_width - 1));
        cy = max(0, min(cy, num_of_grids_in_height - 1));
        d_cell_ids[tid] = cx + cy * num_of_grids_in_width;
    }
}

static void temporal_stereo_mate_spatial_grid_mapping_kernel_launcher(
    int device_id,
    const Merged_Refined_Stereo_Match_GPU* d_stereo_matches,
    int num_of_stereo_mates,
    int num_of_grids_in_width,
    int num_of_grids_in_height,
    int* d_cell_ids)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block)
                                      ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING
                                      : max_threads_per_block;
    const int num_blocks = (num_of_stereo_mates + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    void* kernel_args[] = {
        reinterpret_cast<void*>(&d_stereo_matches),
        reinterpret_cast<void*>(&num_of_stereo_mates),
        reinterpret_cast<void*>(&num_of_grids_in_width),
        reinterpret_cast<void*>(&num_of_grids_in_height),
        reinterpret_cast<void*>(&d_cell_ids)};

    cudacheck(cudaLaunchKernel(reinterpret_cast<const void*>(temporal_stereo_mate_spatial_grid_mapping_kernel),
                               grid_dim, block_dim, kernel_args, 0, nullptr));
}

float temporal_stereo_mate_spatial_map_device_thrust_pipeline(
    int device_id,
    const Merged_Refined_Stereo_Match_GPU* d_stereo_matches,
    int* d_cell_ids,
    int* d_edge_indices,
    int* d_cell_start_idx,
    int* d_num_of_edges_in_cell,
    std::size_t num_stereo_mates,
    int total_num_of_grid_cells,
    int num_of_grid_cells_in_width,
    int num_of_grid_cells_in_height,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    cudacheck(cudaSetDevice(device_id));

    thrust::device_ptr<int> t_cell_ids(d_cell_ids);
    thrust::device_ptr<int> t_edge_indices(d_edge_indices);
    thrust::device_ptr<int> t_cell_start_idx(d_cell_start_idx);
    thrust::device_ptr<int> t_num_of_edges_in_cell(d_num_of_edges_in_cell);

    cudacheck(cudaEventRecord(start));

    thrust::fill(t_cell_start_idx, t_cell_start_idx + total_num_of_grid_cells, 0);
    thrust::fill(t_num_of_edges_in_cell, t_num_of_edges_in_cell + total_num_of_grid_cells, 0);
    thrust::sequence(t_edge_indices, t_edge_indices + num_stereo_mates);

    temporal_stereo_mate_spatial_grid_mapping_kernel_launcher(
        device_id, d_stereo_matches, static_cast<int>(num_stereo_mates),
        num_of_grid_cells_in_width, num_of_grid_cells_in_height, d_cell_ids);
    cudacheck(cudaDeviceSynchronize());

    thrust::sort_by_key(t_cell_ids, t_cell_ids + num_stereo_mates, t_edge_indices);

    std::vector<int> h_sorted_cell_ids(num_stereo_mates, 0);
    std::vector<int> h_cell_start_idx(total_num_of_grid_cells, 0);
    std::vector<int> h_num_of_edges_in_cell(total_num_of_grid_cells, 0);
    cudacheck(cudaMemcpy(h_sorted_cell_ids.data(), d_cell_ids, num_stereo_mates * sizeof(int), cudaMemcpyDeviceToHost));

    if (num_stereo_mates > 0) {
        int first_cell = h_sorted_cell_ids[0];
        h_cell_start_idx[first_cell] = 0;
        for (int tid = 1; tid < static_cast<int>(num_stereo_mates); ++tid) {
            int cell_of_tid = h_sorted_cell_ids[tid];
            int prev_cell = h_sorted_cell_ids[tid - 1];
            if (cell_of_tid != prev_cell) {
                h_cell_start_idx[cell_of_tid] = tid;
                h_num_of_edges_in_cell[prev_cell] = tid - h_cell_start_idx[prev_cell];
            }
        }
        int last_cell = h_sorted_cell_ids[static_cast<int>(num_stereo_mates) - 1];
        h_num_of_edges_in_cell[last_cell] =
            static_cast<int>(num_stereo_mates) - h_cell_start_idx[last_cell];
    }

    cudacheck(cudaMemcpy(d_cell_start_idx, h_cell_start_idx.data(), total_num_of_grid_cells * sizeof(int), cudaMemcpyHostToDevice));
    cudacheck(cudaMemcpy(d_num_of_edges_in_cell, h_num_of_edges_in_cell.data(), total_num_of_grid_cells * sizeof(int), cudaMemcpyHostToDevice));

    float build_spatial_grid_time = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&build_spatial_grid_time, start, stop));

    return build_spatial_grid_time;
}

#endif // TEMPORAL_SPATIAL_GRID_MAPPING_CU
