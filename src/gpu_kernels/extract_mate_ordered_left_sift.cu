#ifndef EXTRACT_MATE_ORDERED_LEFT_SIFT_CU
#define EXTRACT_MATE_ORDERED_LEFT_SIFT_CU

#include <cuda_runtime.h>
#include <cstddef>

#include "gpu_settings.h"
#include "gpu_kernels.h"
#include "gpu_helpers/extract_gpu_sift_descriptor.cuh"

//> Sample left SIFT at each stereo mate's stored left geometry (mate-indexed output).
__global__ void precompute_mate_left_sift_from_texture_kernel(
    const Merged_Refined_Stereo_Match_GPU* __restrict__ d_mates,
    Precomputed_Edge_SIFT_Descriptor_GPU* __restrict__ d_out,
    cudaTextureObject_t left_tex,
    int left_width,
    int left_height,
    int num_mates)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_mates)
        return;

    const float cx = d_mates[i].left_location_x;
    const float cy = d_mates[i].left_location_y;
    const float orientation = d_mates[i].left_orientation;

    const float cos_t = cosf(orientation);
    const float sin_t = sinf(orientation);
    const float nx = sin_t * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);
    const float ny = -cos_t * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);

    extract_gpu_sift_descriptor(
        left_tex, left_width, left_height,
        cx + nx, cy + ny, orientation,
        d_out[i].pL_plus);
    extract_gpu_sift_descriptor(
        left_tex, left_width, left_height,
        cx - nx, cy - ny, orientation,
        d_out[i].pL_minus);
}

void precompute_mate_left_sift_from_texture_pipeline(
    int device_id,
    const Merged_Refined_Stereo_Match_GPU* d_mates,
    cudaTextureObject_t left_tex,
    int left_width,
    int left_height,
    int num_mates,
    Precomputed_Edge_SIFT_Descriptor_GPU*& d_out_sift)
{
    if (num_mates <= 0 || d_mates == nullptr || left_tex == 0 || left_width <= 0 || left_height <= 0) {
        d_out_sift = nullptr;
        return;
    }

    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaMalloc(&d_out_sift,
                         static_cast<size_t>(num_mates) * sizeof(Precomputed_Edge_SIFT_Descriptor_GPU)));

    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));
    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block)
                                      ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING
                                      : max_threads_per_block;
    const int num_blocks = (num_mates + threads_per_block - 1) / threads_per_block;

    precompute_mate_left_sift_from_texture_kernel<<<num_blocks, threads_per_block>>>(
        d_mates, d_out_sift, left_tex, left_width, left_height, num_mates);
    cudacheck(cudaPeekAtLastError());
    cudacheck(cudaDeviceSynchronize());
}

//> Build mate-ordered left SIFT[i] for temporal quad matching (kf_mate_idx / cf_mate_idx indexing).
//> Prefer the per-left-edge precompute table when available; otherwise copy from the merged match struct.
__global__ void extract_mate_ordered_left_sift_kernel(
    const Merged_Refined_Stereo_Match_GPU* __restrict__ d_merged,
    const Precomputed_Edge_SIFT_Descriptor_GPU* __restrict__ d_full_left_sift,
    Precomputed_Edge_SIFT_Descriptor_GPU* __restrict__ d_out,
    int num_mates)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_mates)
        return;

    if (d_full_left_sift != nullptr) {
        const int left_idx = d_merged[i].left_edge_idx;
        d_out[i] = d_full_left_sift[left_idx];
    } else {
        d_out[i] = d_merged[i].left_sift_descriptor;
    }
}

void extract_mate_ordered_left_sift_pipeline(
    int device_id,
    const Merged_Refined_Stereo_Match_GPU* d_merged,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_full_left_sift,
    int num_mates,
    Precomputed_Edge_SIFT_Descriptor_GPU*& d_out_sift)
{
    if (num_mates <= 0 || d_merged == nullptr) {
        d_out_sift = nullptr;
        return;
    }

    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaMalloc(&d_out_sift,
                         static_cast<size_t>(num_mates) * sizeof(Precomputed_Edge_SIFT_Descriptor_GPU)));

    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));
    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block)
                                      ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING
                                      : max_threads_per_block;
    const int num_blocks = (num_mates + threads_per_block - 1) / threads_per_block;

    extract_mate_ordered_left_sift_kernel<<<num_blocks, threads_per_block>>>(
        d_merged, d_full_left_sift, d_out_sift, num_mates);
    cudacheck(cudaPeekAtLastError());
    cudacheck(cudaDeviceSynchronize());
}

#endif
