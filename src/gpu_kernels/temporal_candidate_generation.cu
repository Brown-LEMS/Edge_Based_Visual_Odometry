#ifndef TEMPORAL_CANDIDATE_GENERATION_CU
#define TEMPORAL_CANDIDATE_GENERATION_CU

#include <cuda_runtime.h>
#include <math.h>
#include <cstddef>
#include <iostream>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"
#include "gpu_helpers/eval_edges_in_grid_cells.cuh"

//> CH Notes: 
//> Iterates the CF-left spatial grid cells within the square search box around the KF-left edge
//> position (mirroring the CPU's getCandidatesWithinRadius which collects all candidates whose
//> grid cell falls inside the bounding box).  For every CF mate index found in those cells, look up 
//> the CF-right edge (same index) and check if it is also within the square box around the KF-right edge.
//> This preserves the "quad" structure.
//> Additionally, OR filter is applied to both the KF-left and CF-right edges.
__global__ void temporal_candidate_generation_kernel(
    const Merged_Refined_Stereo_Match_GPU*  d_kf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU*  d_cf_stereo_matches,
    const int*                              d_cf_left_edge_indices,   //> sorted cf_mate_idx (by cell)
    const int*                              d_cf_left_cell_start_idx,
    const int*                              d_cf_left_num_in_cell,
    int                                     grid_W,
    int                                     grid_H,
    int                                     num_kf_mates,
    Match_by_Edge_Index*                    d_candidates_out,
    int*                                    d_candidate_count,
    int                                     max_candidates)
{
    const int kf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (kf_idx >= num_kf_mates) return;

    const Merged_Refined_Stereo_Match_GPU kf_mate = d_kf_stereo_matches[kf_idx];

    //> get the edge locations on the left and right KF stereo mates
    const float kl_x = kf_mate.left_location_x;
    const float kl_y = kf_mate.left_location_y;
    const float kl_o = kf_mate.left_orientation;
    const float kr_x = kf_mate.merged_right_x;
    const float kr_y = kf_mate.merged_right_y;
    const float kr_o = kf_mate.merged_right_orientation;

    //> Grid cell range covering the square search box for the KF-left edge position.
    //> Matches CPU getCandidatesWithinRadius: collect all candidates whose grid cell
    //> is within the bounding box, without an exact L2 distance check.
    const int cx0 = max(0,          static_cast<int>((kl_x - TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));
    const int cx1 = min(grid_W - 1, static_cast<int>((kl_x + TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));
    const int cy0 = max(0,          static_cast<int>((kl_y - TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));
    const int cy1 = min(grid_H - 1, static_cast<int>((kl_y + TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));

    //> Grid cell range for the KF-right edge (used in the right-side box check below).
    const int rx0 = max(0,          static_cast<int>((kr_x - TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));
    const int rx1 = min(grid_W - 1, static_cast<int>((kr_x + TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));
    const int ry0 = max(0,          static_cast<int>((kr_y - TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));
    const int ry1 = min(grid_H - 1, static_cast<int>((kr_y + TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));

    for (int cy = cy0; cy <= cy1; ++cy) {
        for (int cx = cx0; cx <= cx1; ++cx) {
            const int cell  = cy * grid_W + cx;
            const int start = d_cf_left_cell_start_idx[cell];
            const int count = d_cf_left_num_in_cell[cell];

            for (int j = 0; j < count; ++j) {
                const int cf_idx = d_cf_left_edge_indices[start + j];

                const Merged_Refined_Stereo_Match_GPU cf_mate = d_cf_stereo_matches[cf_idx];

                //> Left orientation filter
                if (!edge_pair_passes_orientation_prefilter_deg( kl_o, cf_mate.left_orientation, static_cast<float>(EDGE_PAIR_ORIENTATION_PREFILTER_DEG)))
                    continue;

                //> Right box check: CF-right must fall inside the KF-right bounding box
                const float cr_x = cf_mate.merged_right_x;
                const float cr_y = cf_mate.merged_right_y;
                const int cr_cx = static_cast<int>(cr_x / GRID_CELL_SIZE);
                const int cr_cy = static_cast<int>(cr_y / GRID_CELL_SIZE);
                if (cr_cx < rx0 || cr_cx > rx1 || cr_cy < ry0 || cr_cy > ry1) continue;

                //> Right orientation filter
                if (!edge_pair_passes_orientation_prefilter_deg( kr_o, cf_mate.merged_right_orientation, static_cast<float>(EDGE_PAIR_ORIENTATION_PREFILTER_DEG)))
                    continue;

                //> Emit quad candidate
                //> left_edge_idx  == kf_mate_idx
                //> right_edge_idx == cf_mate_idx
                const int slot = atomicAdd(d_candidate_count, 1);
                if (slot < max_candidates) {
                    d_candidates_out[slot].left_edge_idx  = kf_idx;
                    d_candidates_out[slot].right_edge_idx = cf_idx;
                }
            }
        }
    }
}

void temporal_candidate_generation_kernel_launcher(
    int                                  device_id,
    const Merged_Refined_Stereo_Match_GPU* d_kf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU* d_cf_stereo_matches,
    const int*                           d_cf_left_edge_indices,
    const int*           d_cf_left_cell_start_idx,
    const int*           d_cf_left_num_in_cell,
    int                  grid_W,
    int                  grid_H,
    int                  num_kf_mates,
    Match_by_Edge_Index* d_candidates_out,
    int*                 d_candidate_count,
    int                  max_candidates)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    //> Assigns one thread per KF stereo mate.
    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block)
                                  ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING : max_threads_per_block;
    const int num_blocks = (num_kf_mates + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    void* args[] = {
        (void*)&d_kf_stereo_matches,
        (void*)&d_cf_stereo_matches,
        (void*)&d_cf_left_edge_indices,
        (void*)&d_cf_left_cell_start_idx,
        (void*)&d_cf_left_num_in_cell,
        (void*)&grid_W,
        (void*)&grid_H,
        (void*)&num_kf_mates,
        (void*)&d_candidates_out,
        (void*)&d_candidate_count,
        (void*)&max_candidates
    };

    cudacheck(cudaLaunchKernel( reinterpret_cast<const void*>(temporal_candidate_generation_kernel), grid_dim, block_dim, args, 0, nullptr) );
}

float temporal_candidate_generation_pipeline(
    int                                  device_id,
    const Merged_Refined_Stereo_Match_GPU* d_kf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU* d_cf_stereo_matches,
    const int*                           d_cf_left_edge_indices,
    const int*           d_cf_left_cell_start_idx,
    const int*           d_cf_left_num_in_cell,
    int                  grid_W,
    int                  grid_H,
    int                  num_kf_mates,
    Match_by_Edge_Index* d_candidates_out,
    int*                 d_candidate_count,
    int&                 h_candidate_count_out,
    int                  max_candidates,
    cudaEvent_t          start,
    cudaEvent_t          stop)
{
    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaMemset(d_candidate_count, 0, sizeof(int)));
    cudacheck(cudaEventRecord(start));

    temporal_candidate_generation_kernel_launcher(
        device_id,
        d_kf_stereo_matches,
        d_cf_stereo_matches,
        d_cf_left_edge_indices, d_cf_left_cell_start_idx, d_cf_left_num_in_cell,
        grid_W, grid_H, num_kf_mates,
        d_candidates_out, d_candidate_count, max_candidates);
    cudacheck(cudaDeviceSynchronize());


    float elapsed = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&elapsed, start, stop));

    int device_count = 0;
    cudacheck(cudaMemcpy(&device_count, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (device_count > max_candidates) {
        printf("[Temporal GPU] WARNING: candidate buffer overflow (%d > %d); truncating to buffer size\n",
               device_count, max_candidates);
        h_candidate_count_out = max_candidates;
    } else {
        h_candidate_count_out = device_count;
    }

    return elapsed;
}

#endif // TEMPORAL_CANDIDATE_GENERATION_CU
