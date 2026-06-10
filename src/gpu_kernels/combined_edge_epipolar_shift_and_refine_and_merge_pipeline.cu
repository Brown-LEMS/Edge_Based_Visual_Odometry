#ifndef GPU_EPIPOLAR_REFINE_MERGE_CU
#define GPU_EPIPOLAR_REFINE_MERGE_CU

#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "definitions.h"
#include "gpu_kernels.h"
#include "gpu_settings.h"
#include "gpu_helpers/epipolar_geom_helpers.cuh"
#include "gpu_helpers/patch_photometry_helpers.cuh"

struct sort_edge_hypotheses {
    __device__ bool operator()(
        const Refined_Edge_Hypothesis_Match_GPU& a, 
        const Refined_Edge_Hypothesis_Match_GPU& b) const
    {
        if (a.left_edge_idx != b.left_edge_idx)
            return a.left_edge_idx < b.left_edge_idx;
        return a.right_edge_idx < b.right_edge_idx;
    }
};

struct NonZeroStencilPred { __device__ bool operator()(int stencil_value) const { return stencil_value != 0; } };

__device__ int uf_find_min(int* __restrict__ par, int i)
{
    int r = i;
    while (par[r] != r)
        r = par[r];
    while (par[i] != i) {
        int next = par[i];
        par[i] = r;
        i = next;
    }
    return r;
}

__device__ __forceinline__ void uf_union_min(int* __restrict__ par, int ia, int ib)
{
    int ra = uf_find_min(par, ia);
    int rb = uf_find_min(par, ib);
    if (ra == rb)
        return;
    if (ra < rb)
        par[rb] = ra;
    else
        par[ra] = rb;
}

__device__ __forceinline__ void emit_merged_row(
    Merged_Refined_Stereo_Match_GPU* __restrict__ merged_out,
    int* __restrict__ out_count,
    int left_idx,
    float mx,
    float my,
    float morient)
{
    const int oid = atomicAdd(out_count, 1);
    merged_out[oid].left_edge_idx = left_idx;
    merged_out[oid].merged_right_x = mx;
    merged_out[oid].merged_right_y = my;
    merged_out[oid].merged_right_orientation = morient;
}

__global__ 
void 
photometric_epipolar_refine_matches_kernel(
    /* Data live in the CUDA texture memory */
    cudaTextureObject_t right_tex,
    /* Data live in the device memory */
    const float* __restrict__ d_F,
    const Edge_GPU* __restrict__ d_left_edges,
    const Edge_GPU* __restrict__ d_right_edges,
    const Precomputed_Edge_Patches_Photometry* __restrict__ d_left_patches,
    const Match_by_Edge_Index* __restrict__ d_matches,
    Refined_Edge_Hypothesis_Match_GPU* __restrict__ d_out,
    /* Others */
    int num_matches,
    int img_right_w,
    int img_right_h)
{
    //> hypothesis edge pair (match) index
    const int mi = blockIdx.x * blockDim.x + threadIdx.x;
    if (mi >= num_matches)
        return;

    Match_by_Edge_Index m = d_matches[mi];

    //> Sanity check
    if (m.left_edge_idx < 0 || m.right_edge_idx < 0) {
        d_out[mi].left_edge_idx = m.left_edge_idx;
        d_out[mi].right_edge_idx = -1;
        d_out[mi].refined_right_x = 0.f;
        d_out[mi].refined_right_y = 0.f;
        d_out[mi].photometric_rms = 1e6f;
        d_out[mi].source_right_orientation = 0.f;
        return;
    }

    const Edge_GPU left_edge = d_left_edges[m.left_edge_idx];
    const Edge_GPU right_edge = d_right_edges[m.right_edge_idx];
    const Precomputed_Edge_Patches_Photometry patches = d_left_patches[m.left_edge_idx];
    const float theta_R = right_edge.orientation;

    // float na = 0.f, nb = 0.f, nc = 0.f;
    // get_normalized_epipolar_line_coeffs(d_F, left_edge.location_x, left_edge.location_y, &na, &nb, &nc);

    //> Compute and normalize epipolar line
    float na = d_F[0] * left_edge.location_x + d_F[1] * left_edge.location_y + d_F[2];
    float nb = d_F[3] * left_edge.location_x + d_F[4] * left_edge.location_y + d_F[5];
    float nc = d_F[6] * left_edge.location_x + d_F[7] * left_edge.location_y + d_F[8];

    float norm = sqrtf(na * na + nb * nb);
    if (norm < 1e-6f) {
        d_out[mi].left_edge_idx = m.left_edge_idx;
        d_out[mi].right_edge_idx = m.right_edge_idx;
        d_out[mi].refined_right_x = right_edge.location_x;
        d_out[mi].refined_right_y = right_edge.location_y;
        d_out[mi].photometric_rms = 1e6f;
        d_out[mi].source_right_orientation = theta_R;
        return;
    }
    na /= norm; nb /= norm; nc /= norm;

    if (fabsf(na) + fabsf(nb) < 1e-12f) {
        d_out[mi].left_edge_idx = m.left_edge_idx;
        d_out[mi].right_edge_idx = m.right_edge_idx;
        d_out[mi].refined_right_x = right_edge.location_x;
        d_out[mi].refined_right_y = right_edge.location_y;
        d_out[mi].photometric_rms = 1e6f;
        d_out[mi].source_right_orientation = theta_R;
        return;
    }

    float base_rx = right_edge.location_x;
    float base_ry = right_edge.location_y;
    project_point_to_normalized_line(base_rx, base_ry, na, nb, nc, &base_rx, &base_ry);

    float dirx = 0.f, diry = 0.f;
    get_unit_epipolar_tangent(na, nb, &dirx, &diry);

    float cos_L = cosf(left_edge.orientation);
    float sin_L = sinf(left_edge.orientation);
    float side_shift_f = static_cast<float>(PATCH_SIZE) * 0.5f + 1.0f;

    float Lc_plus[TOTAL_NUM_OF_PATCH_PIXELS];
    float Lc_minus[TOTAL_NUM_OF_PATCH_PIXELS];
    for (int k = 0; k < TOTAL_NUM_OF_PATCH_PIXELS; ++k) {
        Lc_plus[k] = patches.pL_plus[k] - patches.mL_plus;
        Lc_minus[k] = patches.pL_minus[k] - patches.mL_minus;
    }

    float alpha = 0.0f;
    float last_rms = 1e6f;

    //> Iterative GN right candidate edge locationrefinement along the epipolar line
    for (int iter = 0; iter < GPU_EP_REFINE_MAX_ITER; ++iter) {
        float sx = alpha * dirx;
        float sy = alpha * diry;

        float cRpus_x = (base_rx - sin_L * side_shift_f) + sx;
        float cRpus_y = (base_ry + cos_L * side_shift_f) + sy;
        float cRmin_x = (base_rx + sin_L * side_shift_f) + sx;
        float cRmin_y = (base_ry - cos_L * side_shift_f) + sy;

        float Iplus[TOTAL_NUM_OF_PATCH_PIXELS];
        float Iminus[TOTAL_NUM_OF_PATCH_PIXELS];
        float Gplus[TOTAL_NUM_OF_PATCH_PIXELS];
        float Gminus[TOTAL_NUM_OF_PATCH_PIXELS];

        sample_rot_patch_I(right_tex, cRpus_x, cRpus_y, cos_L, sin_L, img_right_w, img_right_h, Iplus);
        sample_rot_patch_I(right_tex, cRmin_x, cRmin_y, cos_L, sin_L, img_right_w, img_right_h, Iminus);

        sample_rot_patch_gproj(right_tex, cRpus_x, cRpus_y, cos_L, sin_L, dirx, diry, img_right_w, img_right_h, Gplus);
        sample_rot_patch_gproj(right_tex, cRmin_x, cRmin_y, cos_L, sin_L, dirx, diry, img_right_w, img_right_h, Gminus);

        float mean_Rp = mean_patch(Iplus);
        float mean_Rm = mean_patch(Iminus);

        double H_acc = 0.0;
        double b_acc = 0.0;
        double cost_acc = 0.0;

        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const float* Lcv = (pass == 0) ? Lc_plus : Lc_minus;
            const float* Ir = (pass == 0) ? Iplus : Iminus;
            const float* Gpr = (pass == 0) ? Gplus : Gminus;
            const float meanR_val = (pass == 0) ? mean_Rp : mean_Rm;
            for (int kk = 0; kk < TOTAL_NUM_OF_PATCH_PIXELS; ++kk) {
                float r = Lcv[kk] - (Ir[kk] - meanR_val);
                float gv = Gpr[kk];
                double absr = static_cast<double>(fabsf(r));
                double w = 1.0;
                if (absr > GPU_EP_REFINE_HUBER_DELTA)
                    w = GPU_EP_REFINE_HUBER_DELTA / absr;
                H_acc += w * static_cast<double>(gv) * static_cast<double>(gv);
                b_acc += w * static_cast<double>(gv) * static_cast<double>(r);
                cost_acc += w * static_cast<double>(r) * static_cast<double>(r);
            }
        }

        float Hess = static_cast<float>(H_acc);
        if (Hess < 1e-8f)
            break;

        //> Update the edge displacement along the epipolar line
        float delta = -static_cast<float>(b_acc) / Hess;
        alpha += delta;

        last_rms = sqrtf(static_cast<float>(cost_acc) / static_cast<float>(TOTAL_NUM_OF_PATCH_PIXELS * 2));

        //> Early termination
        bool stop_tol = fabsf(delta) < GPU_EP_REFINE_TOL;
        bool last_it = iter == GPU_EP_REFINE_MAX_ITER - 1;
        if ( stop_tol || last_it )
            break;
    }

    //> Write to the GPU global memory
    d_out[mi].left_edge_idx = m.left_edge_idx;
    d_out[mi].right_edge_idx = m.right_edge_idx;
    d_out[mi].refined_right_x = base_rx + alpha * dirx;
    d_out[mi].refined_right_y = base_ry + alpha * diry;
    d_out[mi].photometric_rms = last_rms;
    d_out[mi].source_right_orientation = theta_R;
}

__global__ 
void 
mark_segment_head_kernel(
    const Refined_Edge_Hypothesis_Match_GPU* sorted, 
    int n, 
    int* flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    flags[i] = (i == 0 || sorted[i].left_edge_idx != sorted[i - 1].left_edge_idx) ? 1 : 0;
}

/*
CH Note: CPU clustering uses Gaussian weights from spatial spread around a centroid.
Here, however, weights are based on photometric RMS instead, so the merge stays cheap.
TODO: Make sure that this works.
*/
__global__ 
void 
merge_distance_only_segments_kernel(
    const Refined_Edge_Hypothesis_Match_GPU* __restrict__ sorted,
    const int* __restrict__ seg_starts_ext,
    int n_segments,
    int n_total_hypotheses,
    Merged_Refined_Stereo_Match_GPU* __restrict__ merged_out,
    int* __restrict__ out_count)
{
    const int seg = blockIdx.x * blockDim.x + threadIdx.x;

    //> Sanity check
    if (seg >= n_segments)
        return;

    (void)n_total_hypotheses;

    const int seg_begin = seg_starts_ext[seg];
    const int seg_end_plus = seg_starts_ext[seg + 1];

    //> get the segment boundary range
    int k = seg_end_plus - seg_begin;
    if (k <= 0)
        return;

    float cluster_thresh_sq = static_cast<float>(CLUSTER_DIST_THRESH) * static_cast<float>(CLUSTER_DIST_THRESH);

    if (k > GPU_MERGE_MAX_SEGMENT_SIZE) {
        //> Avoid large stack / long serial UF: emit hypotheses without merging
        //> This is a fallback mechanism to handle large segments that would otherwise cause stack overflows or long serial UF operations
        for (int t = 0; t < k; ++t) {
            const Refined_Edge_Hypothesis_Match_GPU& h = sorted[seg_begin + t];
            emit_merged_row(merged_out, out_count, h.left_edge_idx, h.refined_right_x, h.refined_right_y,
                            h.source_right_orientation);
        }
        return;
    }

    int par[GPU_MERGE_MAX_SEGMENT_SIZE];
    for (int i = 0; i < k; ++i)
        par[i] = i;

    //> Union-find to group the hypotheses into segments/clusters
    for (int i = 0; i < k; ++i) {
        Refined_Edge_Hypothesis_Match_GPU hi = sorted[seg_begin + i];
        for (int j = i + 1; j < k; ++j) {
            Refined_Edge_Hypothesis_Match_GPU hj = sorted[seg_begin + j];
            float dx = hi.refined_right_x - hj.refined_right_x;
            float dy = hi.refined_right_y - hj.refined_right_y;
            if ((dx * dx + dy * dy) <= cluster_thresh_sq)
                uf_union_min(par, i, j);
        }
    }

    //> Find the root of each segment/cluster and compute the centroid of the segment/cluster
    for (int root_scan = 0; root_scan < k; ++root_scan) {
        if (uf_find_min(par, root_scan) != root_scan)
            continue;

        float wsum = 0.f;
        float sx = 0.f;
        float sy = 0.f;
        float sc = 0.f;
        float cc = 0.f;
        int left_idx = sorted[seg_begin + root_scan].left_edge_idx;

        for (int j = 0; j < k; ++j) {
            if (uf_find_min(par, j) != root_scan)
                continue;
            const Refined_Edge_Hypothesis_Match_GPU& hj = sorted[seg_begin + j];
            float w = expf(-hj.photometric_rms / GPU_EP_REFINE_HUBER_DELTA);
            if (w < 1e-8f)
                w = 1e-8f;
            wsum += w;
            sx += w * hj.refined_right_x;
            sy += w * hj.refined_right_y;
            sc += w * sinf(hj.source_right_orientation);
            cc += w * cosf(hj.source_right_orientation);
        }

        float inv = 1.0f / fmaxf(wsum, 1e-8f);
        float mx = sx * inv;
        float my = sy * inv;
        float morient = atan2f(sc * inv, cc * inv);
        emit_merged_row(merged_out, out_count, left_idx, mx, my, morient);
    }
}

float combined_edge_epipolar_shift_and_refine_and_merge_pipeline(
    int device_id,
    /* Data live in the CUDA texture memory */
    cudaTextureObject_t right_tex,
    /* Data live in the device memory */
    const float* d_F,
    const Edge_GPU* d_left_edges,
    const Edge_GPU* d_right_edges,
    const Precomputed_Edge_Patches_Photometry* d_left_patches,
    const Match_by_Edge_Index* d_matches_in,
    Merged_Refined_Stereo_Match_GPU*& d_merged_matches_out,
    int*& d_merged_count_out,
    /* Others */
    int num_matches_in,
    int img_right_width,
    int img_right_height,    
    cudaEvent_t start,
    cudaEvent_t stop)
{
    cudacheck(cudaSetDevice(device_id));

    //> Record the start time
    cudacheck(cudaEventRecord(start));

    //> Allocate the output buffer
    Refined_Edge_Hypothesis_Match_GPU* d_refined = nullptr;
    cudacheck(cudaMalloc(&d_refined, static_cast<size_t>(num_matches_in) * sizeof(Refined_Edge_Hypothesis_Match_GPU)));

    //> Define number of threads per block and number of blocks
    int threads_per_block = NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING;
    cudacheck(cudaDeviceGetAttribute(&threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));
    threads_per_block = threads_per_block >= 256 ? 256 : threads_per_block;
    int num_of_blocks = (num_matches_in + threads_per_block - 1) / threads_per_block;

    dim3 grid_dim(num_of_blocks, 1, 1);
    dim3 block_dim(threads_per_block, 1, 1);

    //> Luanch the first kernel: assign one thread per hypothesis edge pair
    //> Task: shift right candidate edges onto the epipolar line, 
    //  then refine their locations along the epipolar line through minimizing the photometric residuals
    photometric_epipolar_refine_matches_kernel<<<grid_dim, block_dim>>>(
        right_tex, d_F, d_left_edges, d_right_edges, d_left_patches, d_matches_in, d_refined, 
        num_matches_in, img_right_width, img_right_height);
    cudacheck(cudaPeekAtLastError());
    cudacheck(cudaDeviceSynchronize());

    //> Once the epipolar shift and edge location refinement are done, what we care about is the pairings of right candidate edges per left edge
    //> But these pairings are not in order with respect to the left edges, which thus requires sorting
    //> Since the consolidation step processes all hypotheses for the same left edge together, i.e., contiguous runs in an array
    thrust::device_ptr<Refined_Edge_Hypothesis_Match_GPU> dptr(d_refined);
    thrust::sort( thrust::device, dptr, dptr + num_matches_in, sort_edge_hypotheses() );

    //> Launch the second kernel: just segment the edge hypotheses based on the left edge index
    thrust::device_vector<int> d_flags(num_matches_in);
    {
        dim3 grids((num_matches_in + threads_per_block - 1) / threads_per_block);
        dim3 blocks(threads_per_block);
        mark_segment_head_kernel<<<grids, blocks>>>(d_refined, num_matches_in, thrust::raw_pointer_cast(d_flags.data()));
        cudacheck(cudaPeekAtLastError());
        cudacheck(cudaDeviceSynchronize());
    }

    //> So num_of_segments here is the number of left edges that currently have at least one right hypothesis 
    const int num_of_segments = thrust::reduce(thrust::device, d_flags.begin(), d_flags.end(), 0, thrust::plus<int>());

    //> Sanity check
    if (num_of_segments <= 0) {
        LOG_ERROR("Something's wrong: No valid edge hypotheses found after epipolar shift and refinement");
        cudacheck(cudaFree(d_refined));
        d_merged_matches_out = nullptr;
        d_merged_count_out = nullptr;

        float ms_early = 0.0f;
        cudacheck(cudaEventRecord(stop));
        cudacheck(cudaEventSynchronize(stop));
        cudacheck(cudaEventElapsedTime(&ms_early, start, stop));
        (void)ms_early;
        return 0.0f;
    }

    //> Copy the start index of each segment to the d_seg_start vector
    thrust::device_vector<int> d_seg_start(num_of_segments);
    thrust::copy_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_matches_in), d_flags.begin(), d_seg_start.begin(), NonZeroStencilPred());

    //> d_seg_start_ext is a "segment boundary array", i.e., the start index of each segment and the end index of the last segment is num_matches_in
    //> The whole thing is meant to access the GPU memory in a contiguous manner, i.e., coalescing
    thrust::device_vector<int> d_seg_start_ext(static_cast<size_t>(num_of_segments) + 1u);
    thrust::copy(d_seg_start.begin(), d_seg_start.end(), d_seg_start_ext.begin());
    thrust::fill(d_seg_start_ext.begin() + num_of_segments, d_seg_start_ext.begin() + num_of_segments + 1, num_matches_in);

    //> Allocate the output buffers
    cudacheck(cudaMalloc(&d_merged_matches_out, static_cast<size_t>(num_matches_in) * sizeof(Merged_Refined_Stereo_Match_GPU)));
    cudacheck(cudaMalloc(&d_merged_count_out, sizeof(int)));
    cudacheck(cudaMemset(d_merged_count_out, 0, sizeof(int)));

    int* raw_ext = thrust::raw_pointer_cast(d_seg_start_ext.data());
    const int segs_launch = num_of_segments;

    //> Launch the third kernel: assign one thread per segment (also per block)
    //> Task: each segment is handled by one thread that consolidates the corresponding right candidate edges
    merge_distance_only_segments_kernel<<<segs_launch, 1>>>( d_refined, raw_ext, num_of_segments, num_matches_in, d_merged_matches_out, d_merged_count_out );
    cudacheck(cudaPeekAtLastError());
    cudacheck(cudaDeviceSynchronize());

    //> Clean up refinement scratch buffer
    cudacheck(cudaFree(d_refined));

    float time_of_epipolar_shift_refine_and_merge = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&time_of_epipolar_shift_refine_and_merge, start, stop));

    return time_of_epipolar_shift_refine_and_merge;
}

#endif // GPU_EPIPOLAR_REFINE_MERGE_CU
