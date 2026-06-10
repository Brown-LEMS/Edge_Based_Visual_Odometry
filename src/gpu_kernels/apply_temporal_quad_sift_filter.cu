#ifndef APPLY_TEMPORAL_QUAD_SIFT_FILTER_CU
#define APPLY_TEMPORAL_QUAD_SIFT_FILTER_CU

#include <cuda_runtime.h>
#include <math.h>
#include <cstddef>
#include <cfloat>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"
#include "gpu_helpers/extract_gpu_sift_descriptor.cuh"

__device__ __forceinline__
float min_patch_pair_sift_distance(const Precomputed_Edge_SIFT_Descriptor_GPU& a, const Precomputed_Edge_SIFT_Descriptor_GPU& b)
{
    const float d_pp = sift_l2_distance(a.pL_plus,  b.pL_plus);
    const float d_pn = sift_l2_distance(a.pL_plus,  b.pL_minus);
    const float d_np = sift_l2_distance(a.pL_minus, b.pL_plus);
    const float d_nn = sift_l2_distance(a.pL_minus, b.pL_minus);
    return fminf(fminf(d_pp, d_pn), fminf(d_np, d_nn));
}

__device__ __forceinline__
void extract_mate_right_sift_one_side(
    const Merged_Refined_Stereo_Match_GPU& mate,
    cudaTextureObject_t right_tex,
    int right_width,
    int right_height,
    bool use_plus_side,
    float* out_desc)
{
    const float orientation = mate.merged_right_orientation;
    const float cos_t = cosf(orientation);
    const float sin_t = sinf(orientation);
    const float nx = sin_t * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);
    const float ny = -cos_t * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);
    const float cx = use_plus_side ? (mate.merged_right_x + nx) : (mate.merged_right_x - nx);
    const float cy = use_plus_side ? (mate.merged_right_y + ny) : (mate.merged_right_y - ny);
    extract_gpu_sift_descriptor(right_tex, right_width, right_height, cx, cy, orientation, out_desc);
}

//> Keep at most two full descriptors on the stack (avoids >1 KB local frame).
__device__ __forceinline__
float min_on_the_fly_right_sift_distance(
    const Merged_Refined_Stereo_Match_GPU& kf_mate,
    const Merged_Refined_Stereo_Match_GPU& cf_mate,
    cudaTextureObject_t kf_right_tex,
    cudaTextureObject_t cf_right_tex,
    int right_width,
    int right_height)
{
    float kf_desc[SIFT_DESCRIPTOR_DIM];
    float cf_desc[SIFT_DESCRIPTOR_DIM];
    float min_d = FLT_MAX;

    extract_mate_right_sift_one_side(kf_mate, kf_right_tex, right_width, right_height, true, kf_desc);
    extract_mate_right_sift_one_side(cf_mate, cf_right_tex, right_width, right_height, true, cf_desc);
    min_d = fminf(min_d, sift_l2_distance(kf_desc, cf_desc));

    extract_mate_right_sift_one_side(cf_mate, cf_right_tex, right_width, right_height, false, cf_desc);
    min_d = fminf(min_d, sift_l2_distance(kf_desc, cf_desc));

    extract_mate_right_sift_one_side(kf_mate, kf_right_tex, right_width, right_height, false, kf_desc);
    extract_mate_right_sift_one_side(cf_mate, cf_right_tex, right_width, right_height, true, cf_desc);
    min_d = fminf(min_d, sift_l2_distance(kf_desc, cf_desc));

    extract_mate_right_sift_one_side(cf_mate, cf_right_tex, right_width, right_height, false, cf_desc);
    min_d = fminf(min_d, sift_l2_distance(kf_desc, cf_desc));

    return min_d;
}

//> One thread per temporal quad candidate. Left-left uses precomputed mate descriptors;
//> right-right extracts SIFT on-the-fly.
//> Mate indices: left_edge_idx = kf_mate_idx, right_edge_idx = cf_mate_idx.
__global__ void apply_temporal_quad_sift_filter_kernel(
    const Match_by_Edge_Index* d_candidates,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_kf_left_sift,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_cf_left_sift,
    const Merged_Refined_Stereo_Match_GPU* d_kf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU* d_cf_stereo_matches,
    cudaTextureObject_t kf_right_tex,
    cudaTextureObject_t cf_right_tex,
    int right_width,
    int right_height,
    Match_by_Edge_Index* d_final_matches_out,
    int* d_final_count,
    float* d_sift_left_scores_out,
    float* d_sift_right_scores_out,
    const float* d_ncc_left_scores_in,
    float* d_ncc_left_scores_out,
    int num_candidates,
    int num_kf_mates,
    int num_cf_mates)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates)
        return;

    const Match_by_Edge_Index candidate = d_candidates[tid];
    const int kf_idx = candidate.left_edge_idx;
    const int cf_idx = candidate.right_edge_idx;

    if (kf_idx < 0 || cf_idx < 0 || kf_idx >= num_kf_mates || cf_idx >= num_cf_mates)
        return;

    const float sift_left = min_patch_pair_sift_distance( d_kf_left_sift[kf_idx], d_cf_left_sift[cf_idx] );
    if (sift_left >= TEMPORAL_SIFT_THRESHOLD)
        return;

    const float sift_right = min_on_the_fly_right_sift_distance( d_kf_stereo_matches[kf_idx], d_cf_stereo_matches[cf_idx], kf_right_tex, cf_right_tex, right_width, right_height);
    if (sift_right >= TEMPORAL_SIFT_THRESHOLD)
        return;

    const int out_idx = atomicAdd(d_final_count, 1);
    if (out_idx >= num_candidates)
        return;
    d_final_matches_out[out_idx] = candidate;
    d_sift_left_scores_out[out_idx] = sift_left;
    d_sift_right_scores_out[out_idx] = sift_right;
    if (d_ncc_left_scores_in != nullptr && d_ncc_left_scores_out != nullptr)
        d_ncc_left_scores_out[out_idx] = d_ncc_left_scores_in[tid];
}

static void apply_temporal_quad_sift_filter_kernel_launcher(
    int device_id,
    const Match_by_Edge_Index* d_candidates,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_kf_left_sift,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_cf_left_sift,
    const Merged_Refined_Stereo_Match_GPU* d_kf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU* d_cf_stereo_matches,
    cudaTextureObject_t kf_right_tex,
    cudaTextureObject_t cf_right_tex,
    int right_width,
    int right_height,
    Match_by_Edge_Index* d_final_matches_out,
    int* d_final_count,
    float* d_sift_left_scores_out,
    float* d_sift_right_scores_out,
    const float* d_ncc_left_scores_in,
    float* d_ncc_left_scores_out,
    int num_candidates,
    int num_kf_mates,
    int num_cf_mates)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block)
                                      ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING
                                      : max_threads_per_block;
    const int num_blocks = (num_candidates + threads_per_block - 1) / threads_per_block;

    void* kernel_args[] = {
        reinterpret_cast<void*>(&d_candidates),
        reinterpret_cast<void*>(&d_kf_left_sift),
        reinterpret_cast<void*>(&d_cf_left_sift),
        reinterpret_cast<void*>(&d_kf_stereo_matches),
        reinterpret_cast<void*>(&d_cf_stereo_matches),
        reinterpret_cast<void*>(&kf_right_tex),
        reinterpret_cast<void*>(&cf_right_tex),
        reinterpret_cast<void*>(&right_width),
        reinterpret_cast<void*>(&right_height),
        reinterpret_cast<void*>(&d_final_matches_out),
        reinterpret_cast<void*>(&d_final_count),
        reinterpret_cast<void*>(&d_sift_left_scores_out),
        reinterpret_cast<void*>(&d_sift_right_scores_out),
        reinterpret_cast<void*>(&d_ncc_left_scores_in),
        reinterpret_cast<void*>(&d_ncc_left_scores_out),
        reinterpret_cast<void*>(&num_candidates),
        reinterpret_cast<void*>(&num_kf_mates),
        reinterpret_cast<void*>(&num_cf_mates)};

    cudacheck(cudaLaunchKernel(reinterpret_cast<const void*>(apply_temporal_quad_sift_filter_kernel),
                               dim3(num_blocks), dim3(threads_per_block), kernel_args, 0, nullptr));
}

float apply_temporal_quad_sift_filter_pipeline(
    int device_id,
    int& h_match_count_out,
    const Match_by_Edge_Index* d_candidates,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_kf_left_sift,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_cf_left_sift,
    const Merged_Refined_Stereo_Match_GPU* d_kf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU* d_cf_stereo_matches,
    cudaTextureObject_t kf_right_tex,
    cudaTextureObject_t cf_right_tex,
    int right_width,
    int right_height,
    Match_by_Edge_Index*& d_final_matches_out,
    int*& d_final_count_out,
    float*& d_sift_left_scores_out,
    float*& d_sift_right_scores_out,
    const float* d_ncc_left_scores_in,
    float*& d_ncc_left_scores_out,
    int num_candidates,
    int num_kf_mates,
    int num_cf_mates,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    cudacheck(cudaSetDevice(device_id));
    // The right-side on-the-fly SIFT path uses per-thread descriptor scratch space.
    // A larger stack avoids illegal-address failures from local-memory spill frames.
    cudacheck(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
    cudacheck(cudaEventRecord(start));

    const size_t match_bytes = static_cast<size_t>(num_candidates) * sizeof(Match_by_Edge_Index);
    cudacheck(cudaMalloc(&d_final_matches_out, match_bytes > 0 ? match_bytes : sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc(&d_final_count_out, sizeof(int)));
    cudacheck(cudaMemset(d_final_count_out, 0, sizeof(int)));

    if (num_candidates > 0) {
        cudacheck(cudaMalloc(&d_sift_left_scores_out, static_cast<size_t>(num_candidates) * sizeof(float)));
        cudacheck(cudaMalloc(&d_sift_right_scores_out, static_cast<size_t>(num_candidates) * sizeof(float)));
        if (d_ncc_left_scores_in != nullptr) {
            cudacheck(cudaMalloc(&d_ncc_left_scores_out, static_cast<size_t>(num_candidates) * sizeof(float)));
        } else {
            d_ncc_left_scores_out = nullptr;
        }
    } 
    else {
        d_sift_left_scores_out = nullptr;
        d_sift_right_scores_out = nullptr;
        d_ncc_left_scores_out = nullptr;
    }

    apply_temporal_quad_sift_filter_kernel_launcher(
        device_id, d_candidates,
        d_kf_left_sift, d_cf_left_sift,
        d_kf_stereo_matches, d_cf_stereo_matches,
        kf_right_tex, cf_right_tex, right_width, right_height,
        d_final_matches_out, d_final_count_out,
        d_sift_left_scores_out, d_sift_right_scores_out,
        d_ncc_left_scores_in, d_ncc_left_scores_out,
        num_candidates, num_kf_mates, num_cf_mates);
    cudacheck(cudaDeviceSynchronize());

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

#endif
