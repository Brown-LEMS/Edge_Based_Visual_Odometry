#ifndef APPLY_TEMPORAL_QUAD_NCC_FILTER_CU
#define APPLY_TEMPORAL_QUAD_NCC_FILTER_CU

#include <cuda_runtime.h>
#include <math.h>
#include <cstddef>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"
#include "gpu_helpers/compute_NCC.cuh"

__device__ __forceinline__
float max_patch_pair_ncc(const Precomputed_Edge_Patches_Photometry& pL,
                         const Precomputed_Edge_Patches_Photometry& pR)
{
    const float sim_pp = compute_NCC(pL.pL_plus,  pL.mL_plus,  pL.vL_plus,  pR.pL_plus,  pR.mL_plus,  pR.vL_plus);
    const float sim_pn = compute_NCC(pL.pL_plus,  pL.mL_plus,  pL.vL_plus,  pR.pL_minus, pR.mL_minus, pR.vL_minus);
    const float sim_np = compute_NCC(pL.pL_minus, pL.mL_minus, pL.vL_minus, pR.pL_plus,  pR.mL_plus,  pR.vL_plus);
    const float sim_nn = compute_NCC(pL.pL_minus, pL.mL_minus, pL.vL_minus, pR.pL_minus, pR.mL_minus, pR.vL_minus);
    return fmaxf(fmaxf(sim_pp, sim_pn), fmaxf(sim_np, sim_nn));
}

//> One thread per temporal quad candidate. Both left-left and right-right NCC must pass
//> (mirrors Temporal_Matches::apply_NCC_filtering_quads on CPU).
//> Mate indices: left_edge_idx = kf_mate_idx, right_edge_idx = cf_mate_idx.
__global__ void apply_temporal_quad_ncc_filter_kernel(
    const Match_by_Edge_Index*              d_candidates,
    const Precomputed_Edge_Patches_Photometry* d_kf_left_patches,
    const Precomputed_Edge_Patches_Photometry* d_kf_right_patches,
    const Precomputed_Edge_Patches_Photometry* d_cf_left_patches,
    const Precomputed_Edge_Patches_Photometry* d_cf_right_patches,
    Match_by_Edge_Index*                    d_final_matches_out,
    int*                                    d_final_count,
    float*                                  d_ncc_left_scores_out,
    float*                                  d_ncc_right_scores_out,
    int                                     num_candidates,
    int                                     num_kf_mates,
    int                                     num_cf_mates)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates)
        return;

    const Match_by_Edge_Index candidate = d_candidates[tid];
    const int kf_idx = candidate.left_edge_idx;
    const int cf_idx = candidate.right_edge_idx;

    if (kf_idx < 0 || cf_idx < 0 || kf_idx >= num_kf_mates || cf_idx >= num_cf_mates)
        return;

    const Precomputed_Edge_Patches_Photometry& kf_left = d_kf_left_patches[kf_idx];
    const Precomputed_Edge_Patches_Photometry& kf_right = d_kf_right_patches[kf_idx];
    const Precomputed_Edge_Patches_Photometry& cf_left = d_cf_left_patches[cf_idx];
    const Precomputed_Edge_Patches_Photometry& cf_right = d_cf_right_patches[cf_idx];

    const float ncc_left = max_patch_pair_ncc(kf_left, cf_left);
    if (ncc_left <= TEMPORAL_NCC_THRESHOLD)
        return;

    const float ncc_right = max_patch_pair_ncc(kf_right, cf_right);
    if (ncc_right <= TEMPORAL_NCC_THRESHOLD)
        return;

    const int out_idx = atomicAdd(d_final_count, 1);
    if (out_idx >= num_candidates)
        return;
    d_final_matches_out[out_idx] = candidate;
    d_ncc_left_scores_out[out_idx] = ncc_left;
    d_ncc_right_scores_out[out_idx] = ncc_right;
}

static void apply_temporal_quad_ncc_filter_kernel_launcher(
    int device_id,
    const Match_by_Edge_Index* d_candidates,
    const Precomputed_Edge_Patches_Photometry* d_kf_left_patches,
    const Precomputed_Edge_Patches_Photometry* d_kf_right_patches,
    const Precomputed_Edge_Patches_Photometry* d_cf_left_patches,
    const Precomputed_Edge_Patches_Photometry* d_cf_right_patches,
    Match_by_Edge_Index* d_final_matches_out,
    int* d_final_count,
    float* d_ncc_left_scores_out,
    float* d_ncc_right_scores_out,
    int num_candidates,
    int num_kf_mates,
    int num_cf_mates)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block)
                                  ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING : max_threads_per_block;
    const int num_blocks = (num_candidates + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    void* kernel_args[] = {
        reinterpret_cast<void*>(&d_candidates),
        reinterpret_cast<void*>(&d_kf_left_patches),
        reinterpret_cast<void*>(&d_kf_right_patches),
        reinterpret_cast<void*>(&d_cf_left_patches),
        reinterpret_cast<void*>(&d_cf_right_patches),
        reinterpret_cast<void*>(&d_final_matches_out),
        reinterpret_cast<void*>(&d_final_count),
        reinterpret_cast<void*>(&d_ncc_left_scores_out),
        reinterpret_cast<void*>(&d_ncc_right_scores_out),
        reinterpret_cast<void*>(&num_candidates),
        reinterpret_cast<void*>(&num_kf_mates),
        reinterpret_cast<void*>(&num_cf_mates)};

    cudacheck(cudaLaunchKernel(reinterpret_cast<const void*>(apply_temporal_quad_ncc_filter_kernel), grid_dim, block_dim, kernel_args, 0, nullptr));
}

float apply_temporal_quad_ncc_filter_pipeline(
    int device_id,
    int& h_match_count_out,
    const Match_by_Edge_Index* d_candidates,
    const Precomputed_Edge_Patches_Photometry* d_kf_left_patches,
    const Precomputed_Edge_Patches_Photometry* d_kf_right_patches,
    const Precomputed_Edge_Patches_Photometry* d_cf_left_patches,
    const Precomputed_Edge_Patches_Photometry* d_cf_right_patches,
    Match_by_Edge_Index*& d_final_matches_out,
    int*& d_final_count_out,
    float*& d_ncc_left_scores_out,
    float*& d_ncc_right_scores_out,
    int num_candidates,
    int num_kf_mates,
    int num_cf_mates,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaEventRecord(start));

    const size_t match_bytes = static_cast<size_t>(num_candidates) * sizeof(Match_by_Edge_Index);
    cudacheck(cudaMalloc(&d_final_matches_out, match_bytes > 0 ? match_bytes : sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc(&d_final_count_out, sizeof(int)));
    cudacheck(cudaMemset(d_final_count_out, 0, sizeof(int)));

    if (num_candidates > 0) {
        cudacheck(cudaMalloc(&d_ncc_left_scores_out, static_cast<size_t>(num_candidates) * sizeof(float)));
        cudacheck(cudaMalloc(&d_ncc_right_scores_out, static_cast<size_t>(num_candidates) * sizeof(float)));
    } else {
        d_ncc_left_scores_out = nullptr;
        d_ncc_right_scores_out = nullptr;
    }

    if (num_candidates > 0
        && d_kf_left_patches != nullptr && d_kf_right_patches != nullptr
        && d_cf_left_patches != nullptr && d_cf_right_patches != nullptr) {
        apply_temporal_quad_ncc_filter_kernel_launcher(
            device_id, d_candidates,
            d_kf_left_patches, d_kf_right_patches,
            d_cf_left_patches, d_cf_right_patches,
            d_final_matches_out, d_final_count_out,
            d_ncc_left_scores_out, d_ncc_right_scores_out,
            num_candidates, num_kf_mates, num_cf_mates);
        cudacheck(cudaDeviceSynchronize());
    }

    float elapsed_ms = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&elapsed_ms, start, stop));

    h_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_match_count_out, d_final_count_out, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_match_count_out > num_candidates) {
        h_match_count_out = num_candidates;
    }

    return elapsed_ms;
}

#endif // APPLY_TEMPORAL_QUAD_NCC_FILTER_CU
