#ifndef APPLY_SECOND_NCC_FILTER_CUH
#define APPLY_SECOND_NCC_FILTER_CUH

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

__global__ 
void 
apply_second_NCC_filter_on_merged_kernel(
    const Precomputed_Edge_Patches_Photometry* d_left_patches,
    const Merged_Refined_Stereo_Match_GPU* d_merged_matches_in,
    Merged_Refined_Stereo_Match_GPU* d_merged_matches_out,
    int* d_merged_count_out,
    float* d_final_scores,
    cudaTextureObject_t right_tex,
    int num_candidates)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates) return;

    Merged_Refined_Stereo_Match_GPU candidate_match = d_merged_matches_in[tid];

    const Precomputed_Edge_Patches_Photometry& pL = d_left_patches[candidate_match.left_edge_idx];

    float pR_plus[TOTAL_NUM_OF_PATCH_PIXELS];
    float pR_minus[TOTAL_NUM_OF_PATCH_PIXELS];
    float mR_plus, vR_plus, mR_minus, vR_minus;

    float cos_R = cosf(candidate_match.merged_right_orientation);
    float sin_R = sinf(candidate_match.merged_right_orientation);
    float nx_R = -sin_R * OFFSET_DIST;
    float ny_R = cos_R * OFFSET_DIST;

    extract_patch_stats_on_the_fly(right_tex, candidate_match.merged_right_x + nx_R, candidate_match.merged_right_y + ny_R, cos_R, sin_R, pR_plus, mR_plus, vR_plus);
    extract_patch_stats_on_the_fly(right_tex, candidate_match.merged_right_x - nx_R, candidate_match.merged_right_y - ny_R, cos_R, sin_R, pR_minus, mR_minus, vR_minus);

    float sim_pp = compute_NCC(pL.pL_plus,  pL.mL_plus,  pL.vL_plus,  pR_plus,  mR_plus,  vR_plus);
    float sim_pn = compute_NCC(pL.pL_plus,  pL.mL_plus,  pL.vL_plus,  pR_minus, mR_minus, vR_minus);
    float sim_np = compute_NCC(pL.pL_minus, pL.mL_minus, pL.vL_minus, pR_plus,  mR_plus,  vR_plus);
    float sim_nn = compute_NCC(pL.pL_minus, pL.mL_minus, pL.vL_minus, pR_minus, mR_minus, vR_minus);
    float max_ncc = fmaxf(fmaxf(sim_pp, sim_pn), fmaxf(sim_np, sim_nn));

    if (max_ncc > static_cast<float>(NCC_THRESH)) {
        int out_idx = atomicAdd(d_merged_count_out, 1);
        for (int i = 0; i < TOTAL_NUM_OF_PATCH_PIXELS; ++i) {
            candidate_match.right_patch.pL_plus[i] = pR_plus[i];
            candidate_match.right_patch.pL_minus[i] = pR_minus[i];
        }
        candidate_match.right_patch.mL_plus = mR_plus;
        candidate_match.right_patch.vL_plus = vR_plus;
        candidate_match.right_patch.mL_minus = mR_minus;
        candidate_match.right_patch.vL_minus = vR_minus;
        d_merged_matches_out[out_idx] = candidate_match;
        d_final_scores[out_idx] = max_ncc;
    }
}

float apply_second_NCC_filter_on_merged_edges_pipeline(
    int                         device_id,
    int                         &h_NCC_match_count_out,
    cudaTextureObject_t         right_tex,
    const Precomputed_Edge_Patches_Photometry* d_left_patches,
    const Merged_Refined_Stereo_Match_GPU* d_merged_matches_in,
    int                         num_candidates,
    Merged_Refined_Stereo_Match_GPU* &d_merged_matches_out,
    int*                        &d_merged_count_out,
    float*                      &d_ncc_scores_out,
    cudaEvent_t                 start,
    cudaEvent_t                 stop)
{
    cudacheck(cudaSetDevice(device_id));

    //> Start the CUDA event timer
    cudacheck(cudaEventRecord(start));

    //> Allocate memory for the merged refined matches and count
    cudacheck(cudaMalloc(&d_merged_matches_out, static_cast<size_t>(num_candidates) * sizeof(Merged_Refined_Stereo_Match_GPU)));
    cudacheck(cudaMalloc(&d_merged_count_out, sizeof(int)));

    //> Initialize the merged count to 0
    cudacheck(cudaMemset(d_merged_count_out, 0, sizeof(int)));

    //> Allocate memory for the NCC scores. THis will be used by the final filter which picks the best right edge candidatefrom the NCC scores.
    cudacheck(cudaMalloc(&d_ncc_scores_out, static_cast<size_t>(num_candidates) * sizeof(float)));

    //> Assign one thread per candidate match
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));
    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block) ?
                                  NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING : max_threads_per_block;
    const int num_blocks = (num_candidates + threads_per_block - 1) / threads_per_block;

    apply_second_NCC_filter_on_merged_kernel<<<num_blocks, threads_per_block>>>
        ( d_left_patches, d_merged_matches_in, d_merged_matches_out,
          d_merged_count_out, d_ncc_scores_out, right_tex, num_candidates );
    cudacheck(cudaPeekAtLastError());
    cudacheck(cudaDeviceSynchronize());

    //> End the CUDA event timer
    float time_of_second_NCC_filter = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&time_of_second_NCC_filter, start, stop));

    h_NCC_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_NCC_match_count_out, d_merged_count_out, sizeof(int), cudaMemcpyDeviceToHost));

    return time_of_second_NCC_filter;
}

#endif