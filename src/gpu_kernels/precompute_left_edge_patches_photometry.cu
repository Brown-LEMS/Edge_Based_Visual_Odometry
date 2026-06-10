#ifndef PRECOMPUTE_LEFT_EDGE_PATCHES_PHOTOMETRY_CU
#define PRECOMPUTE_LEFT_EDGE_PATCHES_PHOTOMETRY_CU

#include <cuda_runtime.h>
#include <math.h>
#include <cstddef>
#include <iostream>
#include <vector>

#include "gpu_settings.h"
#include "gpu_kernels.h"
#include "gpu_helpers/extract_patches_on_the_fly.cuh"

__global__ 
void 
precompute_left_edge_patches_photometry_kernel(
    /* Data live in the device memory */
    const Edge_GPU* d_left_edges,
    Precomputed_Edge_Patches_Photometry* d_out_patches,
    /* Data live in the CUDA texture memory */
    cudaTextureObject_t imgTex,
    /* Others */
    int num_of_left_edges )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_of_left_edges) return;

    Edge_GPU left_edge = d_left_edges[tid];

    float cos_t = cosf(left_edge.orientation);
    float sin_t = sinf(left_edge.orientation);
    float nx = -sin_t * OFFSET_DIST;
    float ny =  cos_t * OFFSET_DIST;

    //> Extract Patch L+
    extract_patch_stats_on_the_fly( imgTex, left_edge.location_x + nx, left_edge.location_y + ny, cos_t, sin_t, 
                                    d_out_patches[tid].pL_plus, d_out_patches[tid].mL_plus, d_out_patches[tid].vL_plus );

    //> Extract Patch L-
    extract_patch_stats_on_the_fly( imgTex, left_edge.location_x - nx, left_edge.location_y - ny, cos_t, sin_t, 
                                    d_out_patches[tid].pL_minus, d_out_patches[tid].mL_minus, d_out_patches[tid].vL_minus );
}

void precompute_left_edge_patches_photometry_kernel_launcher(
    int device_id,
    const Edge_GPU* d_left_edges,
    Precomputed_Edge_Patches_Photometry* d_out_patches,
    cudaTextureObject_t img_texture_object,
    int num_of_left_edges
)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    //> Assign one thread per left edge for pre-computation
    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block) ? 
                                  (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING) : (max_threads_per_block);
    const int num_blocks = (num_of_left_edges + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);
    
    void *kernel_args[] = {
        reinterpret_cast<void*>(&d_left_edges),
        reinterpret_cast<void*>(&d_out_patches),
        reinterpret_cast<void*>(&img_texture_object),
        reinterpret_cast<void*>(&num_of_left_edges)
    };

    cudacheck(cudaLaunchKernel( reinterpret_cast<const void*>(precompute_left_edge_patches_photometry_kernel), grid_dim, block_dim, kernel_args, 0, nullptr) );
}

float precompute_left_edge_patches_photometry_pipeline(
    int             device_id,
    /* Data live in the CUDA texture memory */
    cudaTextureObject_t left_tex,
    /* Data live in the device memory */
    Precomputed_Edge_Patches_Photometry* &d_left_patches,
    Edge_GPU*       d_left_edges,
    /* Others */
    std::size_t     num_of_left_edges,
    cudaEvent_t     start,
    cudaEvent_t     stop) 
{
    cudacheck(cudaSetDevice(device_id));

    //> Start the CUDA event timer
    cudacheck(cudaEventRecord(start));
    
    //> Allocate intermediate device memory for precomputed patches
    cudacheck(cudaMalloc(&d_left_patches, num_of_left_edges * sizeof(Precomputed_Edge_Patches_Photometry)));

    //> Launch precompute kernel
    precompute_left_edge_patches_photometry_kernel_launcher( device_id, d_left_edges, d_left_patches, left_tex, num_of_left_edges );
    cudacheck(cudaDeviceSynchronize());

    //> End the CUDA event timer for precompute left edge patches photometry
    float precompute_left_edge_patches_photometry_time = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&precompute_left_edge_patches_photometry_time, start, stop));

    return precompute_left_edge_patches_photometry_time;
}


#endif