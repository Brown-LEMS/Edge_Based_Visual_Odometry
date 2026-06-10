#ifndef APPLY_SIFT_DESCRIPTOR_FILTER_CU
#define APPLY_SIFT_DESCRIPTOR_FILTER_CU

#include <cuda_runtime.h>
#include <math.h>
#include <cstddef>
#include <iostream>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"
#include "gpu_helpers/extract_gpu_sift_descriptor.cuh"

__global__
void apply_SIFT_descriptor_filter_kernel(
    const Match_by_Edge_Index* d_ncc_matches,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_left_sift_descriptors,
    const Edge_GPU* d_right_edges,
    Match_by_Edge_Index* d_final_matches,
    int* d_final_count,
    float* d_final_scores,
    const float* d_ncc_scores_in,
    float* d_ncc_scores_out,
    cudaTextureObject_t right_tex,
    int right_width,
    int right_height,
    int num_ncc_matches,
    float sift_threshold)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ncc_matches)
        return;

    const Match_by_Edge_Index candidate = d_ncc_matches[tid];
    if (candidate.left_edge_idx < 0 || candidate.right_edge_idx < 0)
        return;

    //> get the precomputed left edge SIFT descriptor
    const Precomputed_Edge_SIFT_Descriptor_GPU* left_desc = &d_left_sift_descriptors[candidate.left_edge_idx];
    const Edge_GPU right_edge = d_right_edges[candidate.right_edge_idx];

    const float cos_r = cosf(right_edge.orientation);
    const float sin_r = sinf(right_edge.orientation);
    const float rnx = sin_r * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);
    const float rny = -cos_r * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);

    float r_plus[SIFT_DESCRIPTOR_DIM];
    float r_minus[SIFT_DESCRIPTOR_DIM];

    extract_gpu_sift_descriptor(right_tex, right_width, right_height, right_edge.location_x + rnx, right_edge.location_y + rny, right_edge.orientation, r_plus);
    extract_gpu_sift_descriptor(right_tex, right_width, right_height, right_edge.location_x - rnx, right_edge.location_y - rny, right_edge.orientation, r_minus);

    const float d_pp = sift_l2_distance(left_desc->pL_plus,  r_plus);
    const float d_pn = sift_l2_distance(left_desc->pL_plus,  r_minus);
    const float d_np = sift_l2_distance(left_desc->pL_minus, r_plus);
    const float d_nn = sift_l2_distance(left_desc->pL_minus, r_minus);
    const float min_dist = fminf(fminf(d_pp, d_pn), fminf(d_np, d_nn));

    if (min_dist < sift_threshold) {
        const int out_idx = atomicAdd(d_final_count, 1);
        d_final_matches[out_idx] = candidate;
        d_final_scores[out_idx] = min_dist;
        if (d_ncc_scores_in != nullptr && d_ncc_scores_out != nullptr)
            d_ncc_scores_out[out_idx] = d_ncc_scores_in[tid];
    }
}

void apply_SIFT_descriptor_filter_kernel_launcher(
    int device_id,
    const Match_by_Edge_Index* d_ncc_matches,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_left_sift_descriptors,
    const Edge_GPU* d_right_edges,
    Match_by_Edge_Index* d_final_matches,
    int* d_final_count,
    float* d_final_scores,
    const float* d_ncc_scores_in,
    float* d_ncc_scores_out,
    cudaTextureObject_t right_tex,
    int right_width,
    int right_height,
    int num_ncc_matches,
    float sift_threshold)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block)
                                  ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING : max_threads_per_block;
    const int num_blocks = (num_ncc_matches + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    void *kernel_args[] = {
        reinterpret_cast<void*>(&d_ncc_matches),
        reinterpret_cast<void*>(&d_left_sift_descriptors),
        reinterpret_cast<void*>(&d_right_edges),
        reinterpret_cast<void*>(&d_final_matches),
        reinterpret_cast<void*>(&d_final_count),
        reinterpret_cast<void*>(&d_final_scores),
        reinterpret_cast<void*>(&d_ncc_scores_in),
        reinterpret_cast<void*>(&d_ncc_scores_out),
        reinterpret_cast<void*>(&right_tex),
        reinterpret_cast<void*>(&right_width),
        reinterpret_cast<void*>(&right_height),
        reinterpret_cast<void*>(&num_ncc_matches),
        reinterpret_cast<void*>(&sift_threshold)
    };

    cudacheck(cudaLaunchKernel(reinterpret_cast<const void*>(apply_SIFT_descriptor_filter_kernel), grid_dim, block_dim, kernel_args, 0, nullptr));
}

float apply_SIFT_descriptor_filter_pipeline(
    int                         device_id,
    int                         &h_SIFT_match_count_out,
    cudaTextureObject_t         right_tex,
    const Match_by_Edge_Index*  d_ncc_matches,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_left_sift_descriptors,
    const Edge_GPU*             d_right_edges,
    Match_by_Edge_Index*        &d_final_matches_passing_SIFT_filter,
    int*                        &d_final_count_passing_SIFT_filter,
    float*                      &d_sift_scores_out,
    const float*                d_ncc_scores_in,
    float*&                     d_ncc_scores_out,
    int                         right_img_width,
    int                         right_img_height,
    int                         num_ncc_matches,
    cudaEvent_t                 start,
    cudaEvent_t                 stop,
    float                       sift_threshold)
{
    cudacheck(cudaSetDevice(device_id));

    cudacheck(cudaEventRecord(start));

    cudacheck(cudaMalloc(&d_final_matches_passing_SIFT_filter, static_cast<size_t>(num_ncc_matches) * sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc(&d_final_count_passing_SIFT_filter, sizeof(int)));
    cudacheck(cudaMalloc(&d_sift_scores_out, static_cast<size_t>(num_ncc_matches) * sizeof(float)));
    if (num_ncc_matches > 0 && d_ncc_scores_in != nullptr) {
        cudacheck(cudaMalloc(&d_ncc_scores_out, static_cast<size_t>(num_ncc_matches) * sizeof(float)));
    } 
    else {
        d_ncc_scores_out = nullptr;
    }

    cudacheck(cudaMemset(d_final_count_passing_SIFT_filter, 0, sizeof(int)));

    apply_SIFT_descriptor_filter_kernel_launcher(
        device_id, d_ncc_matches, d_left_sift_descriptors, d_right_edges, d_final_matches_passing_SIFT_filter,
        d_final_count_passing_SIFT_filter, d_sift_scores_out, d_ncc_scores_in, d_ncc_scores_out, right_tex,
        right_img_width, right_img_height, num_ncc_matches, sift_threshold);
    cudacheck(cudaDeviceSynchronize());

    float apply_SIFT_filter_time = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&apply_SIFT_filter_time, start, stop));

    h_SIFT_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_SIFT_match_count_out, d_final_count_passing_SIFT_filter, sizeof(int), cudaMemcpyDeviceToHost));

    return apply_SIFT_filter_time;
}

#endif // APPLY_SIFT_DESCRIPTOR_FILTER_CU
