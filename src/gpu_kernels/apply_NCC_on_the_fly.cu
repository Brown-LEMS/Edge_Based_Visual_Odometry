#ifndef APPLY_NCC_ON_THE_FLY_CUH
#define APPLY_NCC_ON_THE_FLY_CUH

#include <cuda_runtime.h>
#include <math.h>
#include <cstddef>
#include <iostream>
#include <vector>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"

#include "gpu_helpers/extract_patches_on_the_fly.cuh"
#include "gpu_helpers/compute_NCC.cuh"

//> CH: One thread is assigned to one candidate match
//> The left edge patches are precomputed. The right edge patches, on the other hand, are extracted on the fly.
//> TODO: try to see to what extent this approach becomes slow
__global__ void apply_NCC_filter_kernel(
    /* Data live in the device memory */
    const Match_by_Edge_Index* d_candidate_matches,
    const Edge_GPU* d_left_edges,
    const Edge_GPU* d_right_edges,
    Match_by_Edge_Index* d_final_matches,
    int* d_final_count,
    float* d_final_scores,
    Precomputed_Edge_Patches_Photometry* d_left_patches,
    /* Data live in the CUDA texture memory */
    cudaTextureObject_t right_tex,
    /* Others */
    int num_candidates,
    float ncc_threshold)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates) return;

    //> Get the candidate edge match
    Match_by_Edge_Index candidate = d_candidate_matches[tid];

    //> Left edge patches have been precomputed and stored in the device global memory
    //> There is an overlapping road since multiple threads might evaluate the same left_edge_idx, but L1/L2 cache handles this efficiently.
    const Precomputed_Edge_Patches_Photometry& pL = d_left_patches[candidate.left_edge_idx];

    //> Right edge patches are extracted on the fly from the texture memory
    Edge_GPU right_edge = d_right_edges[candidate.right_edge_idx];
    
    float pR_plus[TOTAL_NUM_OF_PATCH_PIXELS];
    float pR_minus[TOTAL_NUM_OF_PATCH_PIXELS];
    float mR_plus, vR_plus, mR_minus, vR_minus;

    float cos_R = cosf(right_edge.orientation);
    float sin_R = sinf(right_edge.orientation);
    float nx_R = -sin_R * OFFSET_DIST;
    float ny_R = cos_R * OFFSET_DIST;

    extract_patch_stats_on_the_fly(right_tex, right_edge.location_x + nx_R, right_edge.location_y + ny_R, cos_R, sin_R, pR_plus, mR_plus, vR_plus);
    extract_patch_stats_on_the_fly(right_tex, right_edge.location_x - nx_R, right_edge.location_y - ny_R, cos_R, sin_R, pR_minus, mR_minus, vR_minus);

    //> Compute NCC on the four possible combinations of the left and right edge patches
    float sim_pp = compute_NCC(pL.pL_plus,  pL.mL_plus,  pL.vL_plus,  pR_plus,  mR_plus,  vR_plus );
    float sim_pn = compute_NCC(pL.pL_plus,  pL.mL_plus,  pL.vL_plus,  pR_minus, mR_minus, vR_minus);
    float sim_np = compute_NCC(pL.pL_minus, pL.mL_minus, pL.vL_minus, pR_plus,  mR_plus,  vR_plus );
    float sim_nn = compute_NCC(pL.pL_minus, pL.mL_minus, pL.vL_minus, pR_minus, mR_minus, vR_minus);

    //> Take the maximum of the four possible combinations
    float max_ncc = fmaxf(fmaxf(sim_pp, sim_pn), fmaxf(sim_np, sim_nn));

    if (max_ncc > ncc_threshold) {
        int out_idx = atomicAdd(d_final_count, 1);
        d_final_matches[out_idx] = candidate;
        d_final_scores[out_idx] = max_ncc;
    }
}

void apply_NCC_filter_kernel_launcher(
    int device_id,
    const Match_by_Edge_Index* d_candidate_matches,
    const Edge_GPU* d_left_edges,
    const Edge_GPU* d_right_edges,
    Match_by_Edge_Index* d_final_matches,
    int* d_final_count,
    float* d_final_scores,
    Precomputed_Edge_Patches_Photometry* d_left_patches,
    cudaTextureObject_t right_tex,
    int num_candidates,
    float ncc_threshold)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    //> Assign one thread per left edge for pre-computation
    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block) ? 
                                  (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING) : (max_threads_per_block);
    const int num_blocks = (num_candidates + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);
    
    void *kernel_args[] = {
        reinterpret_cast<void*>(&d_candidate_matches),
        reinterpret_cast<void*>(&d_left_edges),
        reinterpret_cast<void*>(&d_right_edges),
        reinterpret_cast<void*>(&d_final_matches),
        reinterpret_cast<void*>(&d_final_count),
        reinterpret_cast<void*>(&d_final_scores),
        reinterpret_cast<void*>(&d_left_patches),
        reinterpret_cast<void*>(&right_tex),
        reinterpret_cast<void*>(&num_candidates),
        reinterpret_cast<void*>(&ncc_threshold)
    };

    cudacheck(cudaLaunchKernel( reinterpret_cast<const void*>(apply_NCC_filter_kernel), grid_dim, block_dim, kernel_args, 0, nullptr) );
}

float apply_NCC_filter_pipeline(
    int                         device_id,
    int                         &h_NCC_match_count_out,
    /* Data live in the CUDA texture memory */
    cudaTextureObject_t         right_tex,
    /* Data live in the device memory */
    const Match_by_Edge_Index*  d_candidate_matches,
    const Edge_GPU*             d_left_edges,
    const Edge_GPU*             d_right_edges,
    Match_by_Edge_Index*        &d_final_matches_passing_NCC_filter,
    int*                        &d_match_count_passing_NCC_filter,
    float*                      &d_ncc_scores_out,
    Precomputed_Edge_Patches_Photometry* d_left_patches,
    /* Others */
    int                         num_candidates,
    cudaEvent_t                 start,
    cudaEvent_t                 stop,
    float                       ncc_threshold)
{
    cudacheck(cudaSetDevice(device_id));

    //> Start the CUDA event timer
    cudacheck(cudaEventRecord(start));

    //> atomicAdd target must be zero-initialized before the kernel launch
    cudacheck(cudaMemset(d_match_count_passing_NCC_filter, 0, sizeof(int)));

    //> Allocate output NCC scores buffer (sized to worst-case = all candidates survive)
    if (num_candidates > 0) {
        cudacheck(cudaMalloc(&d_ncc_scores_out, static_cast<size_t>(num_candidates) * sizeof(float)));
    } 
    else {
        d_ncc_scores_out = nullptr;
    }

    apply_NCC_filter_kernel_launcher(device_id, d_candidate_matches, d_left_edges, d_right_edges,
                                     d_final_matches_passing_NCC_filter, d_match_count_passing_NCC_filter,
                                     d_ncc_scores_out, d_left_patches, right_tex, num_candidates, ncc_threshold);
    cudacheck(cudaDeviceSynchronize());

    //> End the CUDA event timer for NCC filter
    float apply_NCC_filter_time = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&apply_NCC_filter_time, start, stop));

    //> Copy match count to host for downstream stages (must stay in sync with d_match_count)
    h_NCC_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_NCC_match_count_out, d_match_count_passing_NCC_filter, sizeof(int), cudaMemcpyDeviceToHost));

    return apply_NCC_filter_time;
}

#endif