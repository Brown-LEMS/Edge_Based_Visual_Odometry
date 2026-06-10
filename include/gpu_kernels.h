#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <cuda_runtime.h>
#include <cstddef>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include "definitions.h"
#include "gpu_settings.h"

inline void cuda_assert_device_pointer(const void* ptr, const char* name)
{
    if (ptr == nullptr) {
        throw std::runtime_error(std::string("CUDA null device pointer: ") + name);
    }
    cudaPointerAttributes attr{};
    const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        throw std::runtime_error(
            std::string("CUDA expected device pointer for ") + name +
            (err != cudaSuccess ? (": " + std::string(cudaGetErrorString(err)))
                                : ": pointer is not device memory"));
    }
}

//> Helper Edge structure
struct Edge_GPU { 
    float location_x;
    float location_y; 
    float orientation;
};

struct Match_by_Edge_Index {
    int left_edge_idx;
    int right_edge_idx;
};

struct Point2D_GPU {
    float x;
    float y;
};

//> CUDA texture wrapper holds the GPU texture object and its underlying array
struct CUDA_Texture_Wrapper {
    cudaArray_t array;
    cudaTextureObject_t texObj;
};

//> Per edge-pair hypothesis after photometric refinement (stereo right-side refine;
//> temporal quads store refined CF-left position in refined_right_* for post-refine NCC).
struct Refined_Edge_Hypothesis_Match_GPU {
    int left_edge_idx;              //> kf_mate_idx in temporal pipeline
    int right_edge_idx;             //> cf_mate_idx in temporal pipeline
    float refined_right_x;
    float refined_right_y;
    float photometric_rms;
    float source_right_orientation;
};

//> One valid BNB-SIFT candidate after 2D Gauss-Newton refinement on both temporal sides.
struct Temporal_Refined_Match_GPU {
    int   kf_mate_idx;
    int   cf_mate_idx;
    float cf_left_x;
    float cf_left_y;
    float rms;
};

//> The left edge patches are pre-computed and stored in global memory
struct Precomputed_Edge_Patches_Photometry {
    float pL_plus[TOTAL_NUM_OF_PATCH_PIXELS];
    float mL_plus; //> mean
    float vL_plus; //> standard deviation
    
    float pL_minus[TOTAL_NUM_OF_PATCH_PIXELS];
    float mL_minus; //> mean
    float vL_minus; //> standard deviation
};

struct Precomputed_Edge_SIFT_Descriptor_GPU {
    float pL_plus[SIFT_DESCRIPTOR_DIM];
    float pL_minus[SIFT_DESCRIPTOR_DIM];
};

//> One consolidated final stereo match, including the precomputed left-side data
//> gathered from the original left_edge_idx.
struct Merged_Refined_Stereo_Match_GPU {
    int left_edge_idx;
    float left_location_x;
    float left_location_y;
    float left_orientation;
    float merged_right_x;
    float merged_right_y;
    float merged_right_orientation;
    Precomputed_Edge_Patches_Photometry left_patch;
    Precomputed_Edge_Patches_Photometry right_patch;
    Precomputed_Edge_SIFT_Descriptor_GPU left_sift_descriptor;
};

inline CUDA_Texture_Wrapper create_CUDA_texture_object_for_img(float* h_img, int img_width, int img_height) 
{
    CUDA_Texture_Wrapper wrapper;

    //> Define the channel format (1 channel, 32-bit float)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    //> CUDA array allocated in the memory where the swizzled 2D memory lives
    cudacheck( cudaMallocArray(&wrapper.array, &channelDesc, img_width, img_height) );

    //> Copy linear host data to the swizzled CUDA array
    size_t spitch = img_width * sizeof(float);
    cudacheck( cudaMemcpy2DToArray(wrapper.array, 0, 0, h_img, spitch, img_width * sizeof(float), img_height, cudaMemcpyHostToDevice) );
    
    //> Specify the resource descriptor
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = wrapper.array;

    //> Specify the texture descriptor, namely, how the hardware reads it
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp X to edge
    texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp Y to edge
    texDesc.filterMode     = cudaFilterModeLinear; // Enable Hardware Bilinear Interpolation
    texDesc.readMode       = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;               //> Absolute pixel coordinates are used rather than normalized coordinates

    //> Create the texture object
    cudacheck( cudaCreateTextureObject(&wrapper.texObj, &resDesc, &texDesc, nullptr) );

    return wrapper;
}

//> Thrust + GPU kernel + sort pipeline for spatial grid mapping (defined in spatial_grid_mapping.cu)
float edge_spatial_map_thrust_pipeline(
    int             device_id,
    /* Data live in the host memory */
    const Edge_GPU* h_right_edges,
    int*            &h_cell_ids_out,
    /* Data live in the device memory */
    Edge_GPU*       d_right_edges,
    int*            d_cell_ids,
    int*            d_edge_indices,
    int*            d_cell_start_idx,
    int*            d_num_of_edges_in_cell,
    /* Others */
    std::size_t     num_right_edges,
    int             total_num_of_grid_cells,
    int             num_of_grid_cells_in_width,
    int             num_of_grid_cells_in_height,
    cudaEvent_t     start,
    cudaEvent_t     stop
);

//> Build a CF/KF-left spatial grid from final stereo mates (mate index == array index).
//> Uses left_location_* stored in each Merged_Refined_Stereo_Match_GPU.
float temporal_stereo_mate_spatial_map_device_thrust_pipeline(
    int                                  device_id,
    const Merged_Refined_Stereo_Match_GPU* d_stereo_matches,
    int*                                 d_cell_ids,
    int*                                 d_edge_indices,
    int*                                 d_cell_start_idx,
    int*                                 d_num_of_edges_in_cell,
    std::size_t                          num_stereo_mates,
    int                                  total_num_of_grid_cells,
    int                                  num_of_grid_cells_in_width,
    int                                  num_of_grid_cells_in_height,
    cudaEvent_t                          start,
    cudaEvent_t                          stop
);

//> GPU kernel pipeline for epipolar rasterization (defined in epipolar_rasterization.cu)
float epipolar_rasterization_thick_band_pipeline(
    int                     device_id,
    /* Data live in the host memory, used for results retrieval */
    const float*            h_F,
    const Edge_GPU*         h_left_edges,
    int                     &h_match_count_out,
    Match_by_Edge_Index*    &h_matches_out,
    /* Data live in the device memory */
    Edge_GPU*               d_left_edges,
    const Edge_GPU*         d_right_edges, 
    const int*              d_edge_indices,
    const int*              d_cell_start_idx, 
    const int*              d_num_of_edges_in_cell, 
    float*                  d_F,
    Match_by_Edge_Index*    d_matches, 
    int*                    d_match_count, 
    /* Others */
    int                     num_of_left_edges, 
    int                     img_right_W, 
    int                     img_right_H, 
    int                     num_of_grid_cells_in_width, 
    int                     num_of_grid_cells_in_height, 
    int                     max_matches,
    cudaEvent_t             start,
    cudaEvent_t             stop
);

//> GPU kernel pipeline for precompute left edge patches photometry (defined in precompute_left_edge_patches_photometry.cu)
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
    cudaEvent_t     stop
);

//> GPU kernel pipeline for precompute left edge SIFT descriptors (defined in precompute_left_edge_sift_descriptors.cu)
float precompute_left_edge_sift_descriptors_pipeline(
    int             device_id,
    cudaTextureObject_t left_tex,
    Precomputed_Edge_SIFT_Descriptor_GPU*& d_left_sift_descriptors,
    Edge_GPU*       d_left_edges,
    int             left_img_width,
    int             left_img_height,
    std::size_t     num_of_left_edges,
    cudaEvent_t     start,
    cudaEvent_t     stop
);

//> GPU-only SIFT-style descriptor filter (defined in apply_SIFT_descriptor_filter.cu).
float apply_SIFT_descriptor_filter_pipeline(
    int                         device_id,
    /* Data live in the host memory */
    int                         &h_SIFT_match_count_out,
    /* Data live in the device memory */
    cudaTextureObject_t         right_tex,
    const Match_by_Edge_Index*  d_ncc_matches,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_left_sift_descriptors,
    const Edge_GPU*             d_right_edges,
    Match_by_Edge_Index*        &d_final_matches_passing_SIFT_filter,
    int*                        &d_final_count_passing_SIFT_filter,
    float*                      &d_sift_scores_out,
    const float*                d_ncc_scores_in,
    float*&                     d_ncc_scores_out,
    /* Others */
    int                         right_img_width,
    int                         right_img_height,
    int                         num_ncc_matches,
    cudaEvent_t                 start,
    cudaEvent_t                 stop,
    float                       sift_threshold
);

//> GPU kernel pipeline for apply NCC filter (defined in apply_NCC_on_the_fly.cu)
float apply_NCC_filter_pipeline(
    int                         device_id,
    int                         &h_NCC_match_count_out,
    /* Data live in the CUDA texture memory */
    cudaTextureObject_t         right_tex,
    /* Data live in the device memory */
    const Match_by_Edge_Index*  d_candidate_matches,
    const Edge_GPU*             d_left_edges,
    const Edge_GPU*             d_right_edges,
    Match_by_Edge_Index*        &d_final_matches_passing_NCC_filter,
    int*                        &d_match_count_passing_NCC_filter,
    float*                      &d_ncc_scores_out,
    Precomputed_Edge_Patches_Photometry* d_left_patches,
    /* Others */
    int                         num_candidates,
    cudaEvent_t                 start,
    cudaEvent_t                 stop,
    float                       ncc_threshold
);

//> GPU kernel pipeline for Best-Nearly-Best filtering (defined in apply_best_nearly_best_filter.cu)
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
    bool                        higher_is_better,
    cudaEvent_t                 start,
    cudaEvent_t                 stop
);

//> Shift right candidate edges onto the epipolar line, GN refine along the epipolar line based on Huber photometric residual,
//> then merge hypotheses per left edge using distance-only clustering (defined in combined_edge_epipolar_shift_and_refine_and_merge_pipeline.cu)
float combined_edge_epipolar_shift_and_refine_and_merge_pipeline(
    int                                         device_id,
    /* Data live in the CUDA texture memory */
    cudaTextureObject_t                         right_tex,
    /* Data live in the device memory */
    const float*                                d_F,
    const Edge_GPU*                             d_left_edges,
    const Edge_GPU*                             d_right_edges,
    const Precomputed_Edge_Patches_Photometry*  d_left_patches,
    const Match_by_Edge_Index*                  d_matches_in,
    Merged_Refined_Stereo_Match_GPU*&           d_merged_matches_out,
    int*&                                       d_merged_count_out,
    /* Others */
    int                                         num_matches_in,
    int                                         img_right_width,
    int                                         img_right_height,    
    cudaEvent_t                                 start,
    cudaEvent_t                                 stop
);

//> Apply NCC directly on merged refined right hypotheses (defined in apply_second_NCC_filter.cu)
float apply_second_NCC_filter_on_merged_edges_pipeline(
    int                         device_id,
    int                         &h_NCC_match_count_out,
    cudaTextureObject_t         right_tex,
    const Precomputed_Edge_Patches_Photometry* d_left_patches,
    const Merged_Refined_Stereo_Match_GPU* d_merged_matches_in,
    int                         num_candidates,
    Merged_Refined_Stereo_Match_GPU* &d_merged_matches_out,
    int*                        &d_merged_count_out,
    float*                      &d_ncc_scores_out,
    cudaEvent_t                 start,
    cudaEvent_t                 stop
);

//> Apply left-left + right-right NCC on temporal quad candidates (defined in apply_temporal_quad_ncc_filter.cu).
//> Uses mate-indexed precomputed patches; both sides must pass
float apply_temporal_quad_ncc_filter_pipeline(
    int                         device_id,
    int                         &h_match_count_out,
    const Match_by_Edge_Index*  d_candidates,
    const Precomputed_Edge_Patches_Photometry* d_kf_left_patches,
    const Precomputed_Edge_Patches_Photometry* d_kf_right_patches,
    const Precomputed_Edge_Patches_Photometry* d_cf_left_patches,
    const Precomputed_Edge_Patches_Photometry* d_cf_right_patches,
    Match_by_Edge_Index*&       d_final_matches_out,
    int*&                       d_final_count_out,
    float*&                     d_ncc_left_scores_out,
    float*&                     d_ncc_right_scores_out,
    int                         num_candidates,
    int                         num_kf_mates,
    int                         num_cf_mates,
    cudaEvent_t                 start,
    cudaEvent_t                 stop
);

//> Combined left-left + right-right SIFT for temporal quads (defined in apply_temporal_quad_sift_filter.cu).
//> Left-left uses mate-indexed precomputed descriptors; right-right extracts SIFT on-the-fly at merged_right_*.
float apply_temporal_quad_sift_filter_pipeline(
    int                         device_id,
    int                         &h_match_count_out,
    const Match_by_Edge_Index*  d_candidates,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_kf_left_sift,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_cf_left_sift,
    const Merged_Refined_Stereo_Match_GPU* d_kf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU* d_cf_stereo_matches,
    cudaTextureObject_t         kf_right_tex,
    cudaTextureObject_t         cf_right_tex,
    int                         right_img_width,
    int                         right_img_height,
    Match_by_Edge_Index*&       d_final_matches_out,
    int*&                       d_final_count_out,
    float*&                     d_sift_left_scores_out,
    float*&                     d_sift_right_scores_out,
    const float*                d_ncc_left_scores_in,
    float*&                     d_ncc_left_scores_out,
    int                         num_candidates,
    int                         num_kf_mates,
    int                         num_cf_mates,
    cudaEvent_t                 start,
    cudaEvent_t                 stop
);

//> BNB per kf_mate_idx for temporal quads (defined in apply_temporal_quad_bnb_filter.cu).
//> First computes the best score per KF mate, then filters candidates in parallel.
float apply_temporal_quad_bnb_filter_pipeline(
    int                         device_id,
    int                         &h_match_count_out,
    const Match_by_Edge_Index*  d_candidates,
    const float*                d_rank_scores,
    const float*                d_carry_scores,
    Match_by_Edge_Index*&       d_out_matches,
    float*&                     d_out_carry_scores,
    int*&                       d_out_count,
    int                         num_candidates,
    int                         num_kf_mates,
    float                       ratio_threshold,
    bool                        higher_is_better,
    cudaEvent_t                 start,
    cudaEvent_t                 stop
);

//> Temporal quad candidate generation: for each KF stereo mate, queries the CF-left spatial grid
//> within TEMPORAL_SEARCH_RADIUS_PIXELS and checks whether the CF-right mate is also within radius.
//> Orientation filter is applied on both sides.
//> Emits Match_by_Edge_Index { left_edge_idx = kf_mate_idx, right_edge_idx = cf_mate_idx }.
//> (defined in temporal_candidate_generation.cu)
float temporal_candidate_generation_pipeline(
    int                         device_id,
    const Merged_Refined_Stereo_Match_GPU* d_kf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU* d_cf_stereo_matches,
    const int*                  d_cf_left_edge_indices,
    const int*                  d_cf_left_cell_start_idx,
    const int*                  d_cf_left_num_in_cell,
    int                         grid_W,
    int                         grid_H,
    int                         num_kf_mates,
    Match_by_Edge_Index*        d_candidates,
    int*                        d_candidate_count,
    int&                        h_candidate_count_out,
    int                         max_candidates,
    cudaEvent_t                 start,
    cudaEvent_t                 stop
);

//> Best filter: one merged hypothesis per left edge with highest second-NCC score
float apply_best_second_ncc_per_left_edge_pipeline(
    int                         device_id,
    int                         &h_match_count_out,
    Merged_Refined_Stereo_Match_GPU*& d_merged_matches,
    int*&                       d_merged_count,
    float*&                     d_second_ncc_scores,
    const Edge_GPU*             d_left_edges,
    const Precomputed_Edge_Patches_Photometry* d_left_patches,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_left_sift_descriptors,
    Precomputed_Edge_Patches_Photometry*& d_final_left_patches,
    Precomputed_Edge_Patches_Photometry*& d_final_right_patches,
    Precomputed_Edge_SIFT_Descriptor_GPU*& d_final_left_sift_descriptors,
    int                         num_candidates,
    cudaEvent_t                 start,
    cudaEvent_t                 stop
);

//> GPU 2D photometric refinement (Gauss-Newton) + distance-based clustering for temporal quads.
//> (defined in temporal_photometric_refine_and_cluster.cu)
//> Returns elapsed GPU milliseconds.
float temporal_photometric_refine_and_cluster_pipeline(
    int                                                 device_id,
    const Match_by_Edge_Index*                          d_bnb_sift_matches,
    int                                                 num_bnb_sift,
    const Precomputed_Edge_Patches_Photometry*          d_kf_left_patches,
    const Precomputed_Edge_Patches_Photometry*          d_kf_right_patches,
    const Merged_Refined_Stereo_Match_GPU*              d_kf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU*              d_cf_stereo_matches,
    cudaTextureObject_t                                 cf_left_tex,
    cudaTextureObject_t                                 cf_right_tex,
    int                                                 cf_left_img_w,
    int                                                 cf_left_img_h,
    int                                                 cf_right_img_w,
    int                                                 cf_right_img_h,
    //> outputs (lazily allocated by the callee)
    Temporal_Refined_Match_GPU*&                        d_refined_matches,
    int*&                                               d_refined_count,
    int&                                                h_refined_count_out,
    Refined_Edge_Hypothesis_Match_GPU*&                 d_clustered_matches,
    int*&                                               d_clustered_count,
    int&                                                h_clustered_count_out,
    cudaEvent_t                                         start,
    cudaEvent_t                                         stop
);

//> Mate-ordered left SIFT for temporal quads when compact stereo output is missing
//> (defined in extract_mate_ordered_left_sift.cu)
void extract_mate_ordered_left_sift_pipeline(
    int device_id,
    const Merged_Refined_Stereo_Match_GPU* d_merged,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_full_left_sift,
    int num_mates,
    Precomputed_Edge_SIFT_Descriptor_GPU*& d_out_sift);

//> Mate-ordered left SIFT sampled from a left image texture at each mate's left geometry.
void precompute_mate_left_sift_from_texture_pipeline(
    int device_id,
    const Merged_Refined_Stereo_Match_GPU* d_mates,
    cudaTextureObject_t left_tex,
    int left_width,
    int left_height,
    int num_mates,
    Precomputed_Edge_SIFT_Descriptor_GPU*& d_out_sift);

#endif // GPU_KERNELS_H
