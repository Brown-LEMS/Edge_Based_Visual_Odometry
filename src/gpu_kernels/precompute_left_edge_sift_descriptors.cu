#ifndef PRECOMPUTE_LEFT_EDGE_SIFT_DESCRIPTORS_CU
#define PRECOMPUTE_LEFT_EDGE_SIFT_DESCRIPTORS_CU

#include <cuda_runtime.h>
#include <math.h>
#include <cstddef>

#include "gpu_settings.h"
#include "gpu_kernels.h"
#include "gpu_helpers/extract_gpu_sift_descriptor.cuh"

__global__
void 
precompute_left_edge_sift_descriptors_kernel(
    const Edge_GPU* d_left_edges,
    Precomputed_Edge_SIFT_Descriptor_GPU* d_out_descriptors,
    cudaTextureObject_t left_tex,
    int left_width,
    int left_height,
    int num_of_left_edges)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_of_left_edges)
        return;

    const Edge_GPU left_edge = d_left_edges[tid];

    const float cos_t = cosf(left_edge.orientation);
    const float sin_t = sinf(left_edge.orientation);
    const float nx = sin_t * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);
    const float ny = -cos_t * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);

    extract_gpu_sift_descriptor(
        left_tex, left_width, left_height,
        left_edge.location_x + nx, left_edge.location_y + ny,
        left_edge.orientation,
        d_out_descriptors[tid].pL_plus);

    extract_gpu_sift_descriptor(
        left_tex, left_width, left_height,
        left_edge.location_x - nx, left_edge.location_y - ny,
        left_edge.orientation,
        d_out_descriptors[tid].pL_minus);
}

void precompute_left_edge_sift_descriptors_kernel_launcher(
    int device_id,
    const Edge_GPU* d_left_edges,
    Precomputed_Edge_SIFT_Descriptor_GPU* d_out_descriptors,
    cudaTextureObject_t left_tex,
    int left_width,
    int left_height,
    int num_of_left_edges)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block)
                                  ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING
                                  : max_threads_per_block;
    const int num_blocks = (num_of_left_edges + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    void* kernel_args[] = {
        reinterpret_cast<void*>(&d_left_edges),
        reinterpret_cast<void*>(&d_out_descriptors),
        reinterpret_cast<void*>(&left_tex),
        reinterpret_cast<void*>(&left_width),
        reinterpret_cast<void*>(&left_height),
        reinterpret_cast<void*>(&num_of_left_edges)
    };

    cudacheck(cudaLaunchKernel( reinterpret_cast<const void*>(precompute_left_edge_sift_descriptors_kernel), grid_dim, block_dim, kernel_args, 0, nullptr) );
}

float precompute_left_edge_sift_descriptors_pipeline(
    int device_id,
    cudaTextureObject_t left_tex,
    Precomputed_Edge_SIFT_Descriptor_GPU*& d_left_sift_descriptors,
    Edge_GPU* d_left_edges,
    int left_img_width,
    int left_img_height,
    std::size_t num_of_left_edges,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    cudacheck(cudaSetDevice(device_id));

    cudacheck(cudaEventRecord(start));

    //> Allocate memory for the left edge SIFT descriptors
    cudacheck(cudaMalloc( &d_left_sift_descriptors, num_of_left_edges * sizeof(Precomputed_Edge_SIFT_Descriptor_GPU)));

    //> Launch the kernel to pre-compute the left edge SIFT descriptors
    precompute_left_edge_sift_descriptors_kernel_launcher(
        device_id,
        d_left_edges,
        d_left_sift_descriptors,
        left_tex,
        left_img_width,
        left_img_height,
        static_cast<int>(num_of_left_edges));
    cudacheck(cudaDeviceSynchronize());

    float precompute_sift_descriptors_time = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&precompute_sift_descriptors_time, start, stop));

    return precompute_sift_descriptors_time;
}

#endif // PRECOMPUTE_LEFT_EDGE_SIFT_DESCRIPTORS_CU
