#ifndef TEMPORAL_PHOTOMETRIC_REFINE_AND_CLUSTER_CU
#define TEMPORAL_PHOTOMETRIC_REFINE_AND_CLUSTER_CU

#include <cuda_runtime.h>
#include <cuda/std/functional>
#include <cstdio>
#include <cstddef>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "definitions.h"
#include "gpu_kernels.h"
#include "gpu_settings.h"
#include "gpu_helpers/get_interp_photometry_in_texture.cuh"
#include "gpu_helpers/patch_photometry_helpers.cuh"
#include "gpu_helpers/GN_refinement_2D_free.cuh"

//> Assign one thread per BNB-SIFT candidate. Runs GN_refinement_2D_free for both the
// left-left and right-right sides. Emits a Temporal_Refined_Match_GPU entry
// only when both sides converge.
__global__ void temporal_photometric_refine_kernel(
    const Match_by_Edge_Index*                             d_candidates,
    const Precomputed_Edge_Patches_Photometry* __restrict__ d_kf_left_patches,
    const Precomputed_Edge_Patches_Photometry* __restrict__ d_kf_right_patches,
    const Merged_Refined_Stereo_Match_GPU* __restrict__     d_kf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU* __restrict__     d_cf_stereo_matches,
    cudaTextureObject_t   cf_left_tex,
    cudaTextureObject_t   cf_right_tex,
    Temporal_Refined_Match_GPU* __restrict__ d_out,
    int* __restrict__            d_out_count,
    int   num_candidates,
    int   cf_left_img_w,
    int   cf_left_img_h,
    int   cf_right_img_w,
    int   cf_right_img_h)
{
    const int mi = blockIdx.x * blockDim.x + threadIdx.x;
    if (mi >= num_candidates) return;

    const Match_by_Edge_Index m = d_candidates[mi];
    const int kf_idx = m.left_edge_idx;
    const int cf_idx = m.right_edge_idx;

    const Merged_Refined_Stereo_Match_GPU kf_mate = d_kf_stereo_matches[kf_idx];
    const Merged_Refined_Stereo_Match_GPU cf_mate = d_cf_stereo_matches[cf_idx];

    //> -------- Left side (KF-left → CF-left) --------
    const Precomputed_Edge_Patches_Photometry kfpL = d_kf_left_patches[kf_idx];

    float Lc_plus_L[TOTAL_NUM_OF_PATCH_PIXELS];
    float Lc_minus_L[TOTAL_NUM_OF_PATCH_PIXELS];
    for (int k = 0; k < TOTAL_NUM_OF_PATCH_PIXELS; ++k) {
        Lc_plus_L[k]  = kfpL.pL_plus[k]  - kfpL.mL_plus;
        Lc_minus_L[k] = kfpL.pL_minus[k] - kfpL.mL_minus;
    }

    float dx_L = 0.0f, dy_L = 0.0f, rms_L = 1e6f;
    bool  valid_L = false;
    GN_refinement_2D_free(Lc_plus_L, Lc_minus_L,
                        kf_mate.left_location_x, kf_mate.left_location_y, kf_mate.left_orientation,
                        cf_mate.left_location_x, cf_mate.left_location_y,
                        cf_left_tex, cf_left_img_w, cf_left_img_h,
                        dx_L, dy_L, rms_L, valid_L);

    if (!valid_L) return;  // fast-exit: no need to run right side

    //> -------- Right side (KF-right → CF-right) --------
    const Precomputed_Edge_Patches_Photometry kfpR = d_kf_right_patches[kf_idx];

    float Lc_plus_R[TOTAL_NUM_OF_PATCH_PIXELS];
    float Lc_minus_R[TOTAL_NUM_OF_PATCH_PIXELS];
    for (int k = 0; k < TOTAL_NUM_OF_PATCH_PIXELS; ++k) {
        Lc_plus_R[k]  = kfpR.pL_plus[k]  - kfpR.mL_plus;
        Lc_minus_R[k] = kfpR.pL_minus[k] - kfpR.mL_minus;
    }

    float dx_R = 0.0f, dy_R = 0.0f, rms_R = 1e6f;
    bool  valid_R = false;
    GN_refinement_2D_free(Lc_plus_R, Lc_minus_R,
                        kf_mate.merged_right_x, kf_mate.merged_right_y, kf_mate.merged_right_orientation,
                        cf_mate.merged_right_x, cf_mate.merged_right_y,
                        cf_right_tex, cf_right_img_w, cf_right_img_h,
                        dx_R, dy_R, rms_R, valid_R);

    if (!valid_R) return;

    //> Both sides converged — emit
    const int slot = atomicAdd(d_out_count, 1);
    d_out[slot].kf_mate_idx = kf_idx;
    d_out[slot].cf_mate_idx = cf_idx;
    //> Refined CF-left position: CF_iterated = KF - displacement
    d_out[slot].cf_left_x   = kf_mate.left_location_x - dx_L;
    d_out[slot].cf_left_y   = kf_mate.left_location_y - dy_L;
    d_out[slot].rms         = 0.5f * (rms_L + rms_R);
}

//> Sort comparator for Temporal_Refined_Match_GPU (by kf_mate_idx)
struct sort_temporal_refined {
    __device__ bool operator()(const Temporal_Refined_Match_GPU& a,
                               const Temporal_Refined_Match_GPU& b) const
    {
        if (a.kf_mate_idx != b.kf_mate_idx) return a.kf_mate_idx < b.kf_mate_idx;
        return a.rms < b.rms;  // ties: lower RMS first (best candidate first per group)
    }
};

//> Set flags[i]=1 where the kf_mate_idx changes (or i==0).
__global__ 
void 
mark_temporal_segment_head_kernel(
    const Temporal_Refined_Match_GPU* __restrict__ sorted,
    int n, int* __restrict__ flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = (i == 0 || sorted[i].kf_mate_idx != sorted[i - 1].kf_mate_idx) ? 1 : 0;
}

__device__ __forceinline__
void fill_refined_edge_hypothesis_from_temporal(
    const Temporal_Refined_Match_GPU& refined,
    const Merged_Refined_Stereo_Match_GPU& cf_mate,
    Refined_Edge_Hypothesis_Match_GPU& out)
{
    out.left_edge_idx  = refined.kf_mate_idx;
    out.right_edge_idx = refined.cf_mate_idx;
    //> Temporal refine updates CF-left; stored in refined_right_* for reuse with stereo post-NCC kernels.
    out.refined_right_x          = refined.cf_left_x;
    out.refined_right_y          = refined.cf_left_y;
    out.photometric_rms          = refined.rms;
    out.source_right_orientation = cf_mate.left_orientation;
}

//> Assign one thread per segment (unique kf_mate_idx group)
//> CH Notes: uses union-find to merge refined candidates whose CF-left positions are 
//  within CLUSTER_DIST_THRESH. Emits one Refined_Edge_Hypothesis_Match_GPU per cluster.
__global__ 
void 
temporal_cluster_merge_kernel(
    const Temporal_Refined_Match_GPU* __restrict__ sorted,
    const Merged_Refined_Stereo_Match_GPU* __restrict__ d_cf_stereo_matches,
    const int* __restrict__ seg_starts_ext,   //> length = n_segments+1 (extended)
    int n_segments,
    Refined_Edge_Hypothesis_Match_GPU* __restrict__ d_out,
    int* __restrict__                 d_out_count)
{
    const int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= n_segments) return;

    const int seg_begin = seg_starts_ext[seg];
    const int seg_end   = seg_starts_ext[seg + 1];
    const int k         = seg_end - seg_begin;
    if (k <= 0) return;

    const float thresh_sq = static_cast<float>(CLUSTER_DIST_THRESH)
                          * static_cast<float>(CLUSTER_DIST_THRESH);

    if (k == 1) {
        const int oid = atomicAdd(d_out_count, 1);
        const Temporal_Refined_Match_GPU& rep = sorted[seg_begin];
        fill_refined_edge_hypothesis_from_temporal(
            rep, d_cf_stereo_matches[rep.cf_mate_idx], d_out[oid]);
        return;
    }

    // Clamp segment size to avoid stack overflow (same guard as stereo pipeline)
    if (k > GPU_MERGE_MAX_SEGMENT_SIZE) {
        for (int t = 0; t < GPU_MERGE_MAX_SEGMENT_SIZE; ++t) {
            const int oid = atomicAdd(d_out_count, 1);
            const Temporal_Refined_Match_GPU& rep = sorted[seg_begin + t];
            fill_refined_edge_hypothesis_from_temporal(
                rep, d_cf_stereo_matches[rep.cf_mate_idx], d_out[oid]);
        }
        return;
    }

    // Union-Find (path compression) — stored in local memory
    int par[GPU_MERGE_MAX_SEGMENT_SIZE];
    for (int i = 0; i < k; ++i) par[i] = i;

    // Find root with path compression
    auto uf_find = [&](int i) -> int {
        while (par[i] != i) { par[i] = par[par[i]]; i = par[i]; }
        return i;
    };
    auto uf_union = [&](int a, int b) {
        int ra = uf_find(a), rb = uf_find(b);
        if (ra == rb) return;
        if (ra < rb) par[rb] = ra; else par[ra] = rb;
    };

    // Merge nearby candidates
    for (int i = 0; i < k; ++i) {
        const Temporal_Refined_Match_GPU& ai = sorted[seg_begin + i];
        for (int j = i + 1; j < k; ++j) {
            const Temporal_Refined_Match_GPU& aj = sorted[seg_begin + j];
            const float dx = ai.cf_left_x - aj.cf_left_x;
            const float dy = ai.cf_left_y - aj.cf_left_y;
            if (dx * dx + dy * dy <= thresh_sq)
                uf_union(i, j);
        }
    }

    // Emit the lowest-RMS representative for each cluster.
    // Candidates are already sorted by RMS within the segment (sort_temporal_refined).
    bool emitted[GPU_MERGE_MAX_SEGMENT_SIZE];
    for (int i = 0; i < k; ++i) emitted[i] = false;

    for (int i = 0; i < k; ++i) {
        const int root = uf_find(i);
        if (!emitted[root]) {
            emitted[root] = true;
            const int oid = atomicAdd(d_out_count, 1);
            const Temporal_Refined_Match_GPU& rep = sorted[seg_begin + root];
            fill_refined_edge_hypothesis_from_temporal(
                rep, d_cf_stereo_matches[rep.cf_mate_idx], d_out[oid]);
        }
    }
}

// ---------------------------------------------------------------------------
// temporal_photometric_refine_and_cluster_pipeline
// ---------------------------------------------------------------------------
float temporal_photometric_refine_and_cluster_pipeline(
    int                                       device_id,
    const Match_by_Edge_Index*                d_bnb_sift_matches,
    int                                       num_bnb_sift,
    const Precomputed_Edge_Patches_Photometry* d_kf_left_patches,
    const Precomputed_Edge_Patches_Photometry* d_kf_right_patches,
    const Merged_Refined_Stereo_Match_GPU*      d_kf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU*      d_cf_stereo_matches,
    cudaTextureObject_t                       cf_left_tex,
    cudaTextureObject_t                       cf_right_tex,
    int                                       cf_left_img_w,
    int                                       cf_left_img_h,
    int                                       cf_right_img_w,
    int                                       cf_right_img_h,
    Temporal_Refined_Match_GPU*&              d_refined_matches,
    int*&                                     d_refined_count,
    int&                                      h_refined_count_out,
    Refined_Edge_Hypothesis_Match_GPU*&       d_clustered_matches,
    int*&                                     d_clustered_count,
    int&                                      h_clustered_count_out,
    cudaEvent_t                               start,
    cudaEvent_t                               stop)
{
    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaEventRecord(start));

    h_refined_count_out   = 0;
    h_clustered_count_out = 0;

    if (num_bnb_sift <= 0) {
        cudacheck(cudaEventRecord(stop));
        cudacheck(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        cudacheck(cudaEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }

    // ------------------------------------------------------------------ //
    // Phase 1: 2D Gauss-Newton photometric refinement
    // ------------------------------------------------------------------ //
    const size_t refined_bytes  = static_cast<size_t>(num_bnb_sift) * sizeof(Temporal_Refined_Match_GPU);
    if (d_refined_matches == nullptr)
        cudacheck(cudaMalloc(&d_refined_matches, refined_bytes));
    if (d_refined_count == nullptr)
        cudacheck(cudaMalloc(&d_refined_count, sizeof(int)));
    cudacheck(cudaMemset(d_refined_count, 0, sizeof(int)));

    {
        const int threads = 128;  // lower occupancy to give GN iterations more registers
        const int blocks  = (num_bnb_sift + threads - 1) / threads;
        temporal_photometric_refine_kernel<<<blocks, threads>>>(
            d_bnb_sift_matches,
            d_kf_left_patches, d_kf_right_patches,
            d_kf_stereo_matches, d_cf_stereo_matches,
            cf_left_tex, cf_right_tex,
            d_refined_matches, d_refined_count,
            num_bnb_sift,
            cf_left_img_w, cf_left_img_h,
            cf_right_img_w, cf_right_img_h);
        cudacheck(cudaDeviceSynchronize());
    }

    cudacheck(cudaMemcpy(&h_refined_count_out, d_refined_count,
                         sizeof(int), cudaMemcpyDeviceToHost));

    if (h_refined_count_out <= 0) {
        cudacheck(cudaEventRecord(stop));
        cudacheck(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        cudacheck(cudaEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }

    // ------------------------------------------------------------------ //
    // Phase 2: sort by kf_mate_idx (ties broken by RMS)
    // ------------------------------------------------------------------ //
    thrust::device_ptr<Temporal_Refined_Match_GPU> d_ref_ptr(d_refined_matches);
    thrust::sort(thrust::device, d_ref_ptr, d_ref_ptr + h_refined_count_out,
                 sort_temporal_refined());

    // ------------------------------------------------------------------ //
    // Phase 3: mark segment heads  →  scan  →  get segment boundaries
    // ------------------------------------------------------------------ //
    const int n_ref = h_refined_count_out;

    thrust::device_vector<int> d_flags(static_cast<size_t>(n_ref));
    {
        const int threads = 256;
        const int blocks  = (n_ref + threads - 1) / threads;
        mark_temporal_segment_head_kernel<<<blocks, threads>>>(
            d_refined_matches, n_ref, thrust::raw_pointer_cast(d_flags.data()));
        cudacheck(cudaDeviceSynchronize());
    }

    thrust::device_vector<int> d_seg_starts(static_cast<size_t>(n_ref));
    thrust::exclusive_scan(thrust::device, d_flags.begin(), d_flags.end(),
                           d_seg_starts.begin());

    // Count segments = sum of flags = prefix-sum[-1] + flags[-1]
    int n_segs = 0;
    {
        int last_flag = 0, last_seg = 0;
        cudacheck(cudaMemcpy(&last_flag, thrust::raw_pointer_cast(d_flags.data()) + (n_ref - 1),
                             sizeof(int), cudaMemcpyDeviceToHost));
        cudacheck(cudaMemcpy(&last_seg,  thrust::raw_pointer_cast(d_seg_starts.data()) + (n_ref - 1),
                             sizeof(int), cudaMemcpyDeviceToHost));
        n_segs = last_seg + last_flag;
    }

    if (n_segs <= 0) {
        cudacheck(cudaEventRecord(stop));
        cudacheck(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        cudacheck(cudaEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }

    // Extended segment-start array with a sentinel at the end (= n_ref)
    thrust::device_vector<int> d_seg_starts_ext(static_cast<size_t>(n_segs + 1));
    {
        // Compact: only keep positions where flag == 1
        auto zip_in  = thrust::make_zip_iterator(
                           thrust::make_tuple(d_flags.begin(), d_seg_starts.begin()));
        auto zip_out = d_seg_starts_ext.begin();

        // Gather segment start indices (positions where flag==1)
        // equivalent to: d_seg_starts_ext[seg_id] = position where flag==1
        // We build it by: for each i with flag[i]==1, seg_starts_ext[prefix[i]] = i
        thrust::device_vector<int> d_iota(static_cast<size_t>(n_ref));
        thrust::sequence(thrust::device, d_iota.begin(), d_iota.end(), 0);

        // Copy iota values where flag==1 into seg_starts_ext[0..n_segs-1]
        thrust::copy_if(thrust::device, d_iota.begin(), d_iota.end(),
                        d_flags.begin(), d_seg_starts_ext.begin(),
                        cuda::std::identity{});

        // Set sentinel: seg_starts_ext[n_segs] = n_ref
        cudacheck(cudaMemcpy(thrust::raw_pointer_cast(d_seg_starts_ext.data()) + n_segs,
                             &n_ref, sizeof(int), cudaMemcpyHostToDevice));
    }

    // ------------------------------------------------------------------ //
    // Phase 4: cluster + merge within each segment
    // ------------------------------------------------------------------ //
    const size_t clustered_bytes = static_cast<size_t>(n_ref) * sizeof(Refined_Edge_Hypothesis_Match_GPU);
    if (d_clustered_matches == nullptr)
        cudacheck(cudaMalloc(&d_clustered_matches, clustered_bytes));
    if (d_clustered_count == nullptr)
        cudacheck(cudaMalloc(&d_clustered_count, sizeof(int)));
    cudacheck(cudaMemset(d_clustered_count, 0, sizeof(int)));

    {
        const int threads = 128;
        const int blocks  = (n_segs + threads - 1) / threads;
        temporal_cluster_merge_kernel<<<blocks, threads>>>(
            d_refined_matches,
            d_cf_stereo_matches,
            thrust::raw_pointer_cast(d_seg_starts_ext.data()),
            n_segs,
            d_clustered_matches,
            d_clustered_count);
        cudacheck(cudaDeviceSynchronize());
    }

    cudacheck(cudaMemcpy(&h_clustered_count_out, d_clustered_count,
                         sizeof(int), cudaMemcpyDeviceToHost));

    float elapsed = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&elapsed, start, stop));
    return elapsed;
}

#endif // TEMPORAL_PHOTOMETRIC_REFINE_AND_CLUSTER_CU
