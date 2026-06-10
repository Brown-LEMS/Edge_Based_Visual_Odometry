#ifndef APPLY_BEST_NEARLY_BEST_FILTER_CU
#define APPLY_BEST_NEARLY_BEST_FILTER_CU

#include <cuda_runtime.h>
#include <cfloat>
#include <cstddef>
#include <iostream>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"

__global__ void compute_best_score_per_left_edge_kernel(
    const Match_by_Edge_Index* d_candidate_matches,
    const float* d_candidate_scores,
    float* d_best_scores_per_left_edge,
    int num_candidates,
    int num_left_edges,
    bool b_higher_is_better)
{
    const int left_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (left_idx >= num_left_edges)
        return;

    float best = b_higher_is_better ? -FLT_MAX : FLT_MAX;
    bool found = false;
    for (int i = 0; i < num_candidates; ++i) {
        const Match_by_Edge_Index m = d_candidate_matches[i];
        if (m.left_edge_idx != left_idx)
            continue;
        const float s = d_candidate_scores[i];
        if (!found) {
            best = s;
            found = true;
            continue;
        }
        best = b_higher_is_better ? fmaxf(best, s) : fminf(best, s);
    }

    d_best_scores_per_left_edge[left_idx] = found ? best : (b_higher_is_better ? -FLT_MAX : FLT_MAX);
}

__global__ void apply_best_nearly_best_filter_kernel(
    const Match_by_Edge_Index* d_candidate_matches,
    const float* d_candidate_scores,
    const float* d_best_scores_per_left_edge,
    Match_by_Edge_Index* d_final_matches,
    float* d_final_scores,
    int* d_final_count,
    int num_candidates,
    float ratio_threshold,
    bool b_higher_is_better)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates)
        return;

    const Match_by_Edge_Index candidate = d_candidate_matches[tid];
    if (candidate.left_edge_idx < 0)
        return;

    const float best = d_best_scores_per_left_edge[candidate.left_edge_idx];
    const float score = d_candidate_scores[tid];
    if (!isfinite(best) || !isfinite(score))
        return;

    bool keep = false;
    if (b_higher_is_better) {
        keep = (score >= ratio_threshold * best);
    } 
    else {
        const float denom = fmaxf(ratio_threshold, 1e-6f);
        keep = (score <= best / denom);
    }

    if (keep) {
        const int out_idx = atomicAdd(d_final_count, 1);
        d_final_matches[out_idx] = candidate;
        d_final_scores[out_idx] = score;
    }
}

float apply_best_nearly_best_filter_pipeline(
    int                         device_id,
    int                         &h_bnb_match_count_out,
    const Match_by_Edge_Index*  d_candidate_matches,
    const float*                d_candidate_scores,
    Match_by_Edge_Index*        &d_final_matches_passing_bnb_filter,
    float*                      &d_final_scores_passing_bnb_filter,
    int*                        &d_final_count_passing_bnb_filter,
    int                         num_candidates,
    int                         num_left_edges,
    float                       ratio_threshold,
    bool                        b_higher_is_better,
    cudaEvent_t                 start,
    cudaEvent_t                 stop)
{
    cudacheck(cudaSetDevice(device_id));

    float* d_best_scores_per_left_edge = nullptr;

    //> Start the CUDA event timer
    cudacheck(cudaEventRecord(start));

    //> Allocate memory for the outputs: final matches, final scores, and final number of matches passing the BNB filter
    cudacheck(cudaMalloc(&d_final_matches_passing_bnb_filter, static_cast<size_t>(num_candidates) * sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc(&d_final_count_passing_bnb_filter, sizeof(int)));
    cudacheck(cudaMalloc(&d_final_scores_passing_bnb_filter, static_cast<size_t>(num_candidates) * sizeof(float)));
    cudacheck(cudaMemset(d_final_count_passing_bnb_filter, 0, sizeof(int)));

    //> Allocate memory for the best scores per left edge
    cudacheck(cudaMalloc(&d_best_scores_per_left_edge, static_cast<size_t>(num_left_edges > 0 ? num_left_edges : 1) * sizeof(float)));

    //> The first kernel is to compute the best score per left edge
    //> This assigns one thread per left edge
    const int threads = 256;
    const int left_blocks = (num_left_edges + threads - 1) / threads;
    if (num_left_edges > 0) {
        compute_best_score_per_left_edge_kernel<<<left_blocks, threads>>>(
            d_candidate_matches,
            d_candidate_scores,
            d_best_scores_per_left_edge,
            num_candidates,
            num_left_edges,
            b_higher_is_better);
        cudacheck(cudaPeekAtLastError());
    }

    //> The second kernel is to apply the BNB filter
    //> This assigns one thread per candidate (left <-> candidate right edges)
    const int cand_blocks = (num_candidates + threads - 1) / threads;
    if (num_candidates > 0) {
        apply_best_nearly_best_filter_kernel<<<cand_blocks, threads>>>(
            d_candidate_matches,
            d_candidate_scores,
            d_best_scores_per_left_edge,
            d_final_matches_passing_bnb_filter,
            d_final_scores_passing_bnb_filter,
            d_final_count_passing_bnb_filter,
            num_candidates,
            ratio_threshold,
            b_higher_is_better);
        cudacheck(cudaPeekAtLastError());
    }
    cudacheck(cudaDeviceSynchronize());

    //> End the CUDA event timer for BNB filter
    float apply_bnb_filter_time = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&apply_bnb_filter_time, start, stop));

    h_bnb_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_bnb_match_count_out, d_final_count_passing_bnb_filter, sizeof(int), cudaMemcpyDeviceToHost));

    cudacheck(cudaFree(d_best_scores_per_left_edge));

    return apply_bnb_filter_time;
}

#endif // APPLY_BEST_NEARLY_BEST_FILTER_CU
