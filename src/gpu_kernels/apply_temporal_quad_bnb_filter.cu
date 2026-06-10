#ifndef APPLY_TEMPORAL_QUAD_BNB_FILTER_CU
#define APPLY_TEMPORAL_QUAD_BNB_FILTER_CU

#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>
#include <cstddef>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"

__global__ void compute_temporal_best_score_per_kf_kernel(
    const Match_by_Edge_Index* d_candidates,
    const float* d_rank_scores,
    float* d_best_scores_per_kf,
    int num_candidates,
    int num_kf_mates,
    bool higher_is_better)
{
    const int kf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (kf_idx >= num_kf_mates)
        return;

    float best = higher_is_better ? -FLT_MAX : FLT_MAX;
    bool found = false;
    for (int i = 0; i < num_candidates; ++i) {
        const Match_by_Edge_Index m = d_candidates[i];
        if (m.left_edge_idx != kf_idx)
            continue;

        const float score = d_rank_scores[i];
        if (!isfinite(score))
            continue;

        if (!found) {
            best = score;
            found = true;
        } else {
            best = higher_is_better ? fmaxf(best, score) : fminf(best, score);
        }
    }

    d_best_scores_per_kf[kf_idx] = found ? best : (higher_is_better ? -FLT_MAX : FLT_MAX);
}

__global__ void apply_temporal_quad_bnb_by_best_kernel(
    const Match_by_Edge_Index* d_candidates,
    const float* d_rank_scores,
    const float* d_carry_scores,
    const float* d_best_scores_per_kf,
    Match_by_Edge_Index* d_out_matches,
    float* d_out_carry_scores,
    int* d_out_count,
    int num_candidates,
    int num_kf_mates,
    float ratio_threshold,
    bool higher_is_better)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates)
        return;

    const Match_by_Edge_Index candidate = d_candidates[tid];
    const int kf_idx = candidate.left_edge_idx;
    if (kf_idx < 0 || kf_idx >= num_kf_mates)
        return;

    const float best = d_best_scores_per_kf[kf_idx];
    const float score = d_rank_scores[tid];
    if (!isfinite(best) || !isfinite(score))
        return;

    bool keep = false;
    if (higher_is_better) {
        keep = (score >= ratio_threshold * best);
    } else {
        const float denom = fmaxf(ratio_threshold, 1e-6f);
        keep = (score <= best / denom);
    }

    if (keep) {
        const int out_idx = atomicAdd(d_out_count, 1);
        if (out_idx >= num_candidates)
            return;
        d_out_matches[out_idx] = candidate;
        if (d_out_carry_scores != nullptr)
            d_out_carry_scores[out_idx] = (d_carry_scores != nullptr) ? d_carry_scores[tid] : score;
    }
}

float apply_temporal_quad_bnb_filter_pipeline(
    int device_id,
    int& h_match_count_out,
    const Match_by_Edge_Index* d_candidates,
    const float* d_rank_scores,
    const float* d_carry_scores,
    Match_by_Edge_Index*& d_out_matches,
    float*& d_out_carry_scores,
    int*& d_out_count,
    int num_candidates,
    int num_kf_mates,
    float ratio_threshold,
    bool higher_is_better,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaEventRecord(start));

    float* d_best_scores_per_kf = nullptr;
    const size_t match_bytes = static_cast<size_t>(num_candidates) * sizeof(Match_by_Edge_Index);
    cudacheck(cudaMalloc(&d_out_matches, match_bytes > 0 ? match_bytes : sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc(&d_out_count, sizeof(int)));
    cudacheck(cudaMemset(d_out_count, 0, sizeof(int)));
    cudacheck(cudaMalloc(&d_best_scores_per_kf, static_cast<size_t>(num_kf_mates > 0 ? num_kf_mates : 1) * sizeof(float)));

    if (num_candidates > 0 && d_carry_scores != nullptr) {
        cudacheck(cudaMalloc(&d_out_carry_scores, static_cast<size_t>(num_candidates) * sizeof(float)));
    } 
    else {
        d_out_carry_scores = nullptr;
    }

    const int threads = 256;
    if (num_candidates > 0 && num_kf_mates > 0) {
        const int kf_blocks = (num_kf_mates + threads - 1) / threads;
        compute_temporal_best_score_per_kf_kernel<<<kf_blocks, threads>>>(
            d_candidates,
            d_rank_scores,
            d_best_scores_per_kf,
            num_candidates,
            num_kf_mates,
            higher_is_better);
        cudacheck(cudaPeekAtLastError());

        const int cand_blocks = (num_candidates + threads - 1) / threads;
        apply_temporal_quad_bnb_by_best_kernel<<<cand_blocks, threads>>>(
            d_candidates,
            d_rank_scores,
            d_carry_scores,
            d_best_scores_per_kf,
            d_out_matches,
            d_out_carry_scores,
            d_out_count,
            num_candidates,
            num_kf_mates,
            ratio_threshold,
            higher_is_better);
        cudacheck(cudaPeekAtLastError());
    }
    cudacheck(cudaDeviceSynchronize());

    float elapsed_ms = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&elapsed_ms, start, stop));

    h_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_match_count_out, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_match_count_out > num_candidates) {
        h_match_count_out = num_candidates;
    }

    cudacheck(cudaFree(d_best_scores_per_kf));

    return elapsed_ms;
}

#endif // APPLY_TEMPORAL_QUAD_BNB_FILTER_CU
