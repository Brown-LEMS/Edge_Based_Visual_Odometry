#ifndef STEREO_MATCHES_GPU_HPP
#define STEREO_MATCHES_GPU_HPP

#include <cmath>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <Eigen/Core>

#include "Stereo_Iterator.h"
#include "gpu_settings.h"
#include "definitions.h"
#include "gpu_kernels.h"
#include "prepare_for_GPU.h"
#include "Dataset.h"

struct Stereo_Matches_GPU_Timing_Statistics
{
    float time_spatial_grid_mapping;
    float time_epipolar_rasterization;
    float time_precompute_left_edge_patches_photometry;
    float time_apply_NCC_filter;
    float time_apply_SIFT_descriptor_filter;
    float time_apply_BNB_NCC_filter;
    float time_apply_BNB_SIFT_filter;
    float time_apply_epipolar_shift_photometric_refine_and_merge;
    float time_apply_second_NCC_filter;
    float time_apply_best_second_ncc_per_left;
    float total_time;
};

class Stereo_Matches_GPU_Pipeline {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Stereo_Matches_GPU_Pipeline> Ptr;

    Stereo_Matches_GPU_Pipeline(Dataset::Ptr dataset, const Prepare_For_GPU_Pipeline::Ptr prepare_for_GPU_engin, const StereoFrame& current_frame, int device);
    ~Stereo_Matches_GPU_Pipeline();

    //> main function for constructing stereo edge correspondences
    Stereo_Matches_GPU_Timing_Statistics get_stereo_edge_matching_GPU(size_t frame_idx);

    float build_edge_spatial_map();
    float rasterize_epipolar_band_for_matches();
    float precompute_left_edge_patches_photometry();
    float apply_NCC_filter();
    float apply_SIFT_descriptor_filter();
    float apply_BNB_NCC_filter();
    float apply_BNB_SIFT_filter();
    float apply_epipolar_shift_photometric_refine_and_merge();
    float apply_second_NCC_filter();
    float apply_best_second_ncc_per_left_after_second_NCC();

    //> Write the final stereo matching from GPU to a file; used for offline plotting and visual inspection
    void write_finalized_matches_to_file(size_t frame_idx) const;

    //> Convert the GPU final stereo output (Merged_Refined_Stereo_Match_GPU[]) to a host-side
    //> std::vector<final_stereo_edge_pair> so that the temporal GPU pipeline can use them as inputs.
    //> CH Notes: Only the left/right edge positions (location, orientation) are populated; patches, descriptors,
    //> and Gamma are left as defaults because the temporal GPU pipeline samples them on-the-fly.
    //> However, it might be wise to cache the left edge patches and descriptors since they are unchanged 
    //  in the process of temporal edge matching.
    //> Must be called before the engine is reset/destroyed.
    void retrieve_stereo_mates( const StereoFrame& frame, std::vector<final_stereo_edge_pair>& out_mates) const;

    int device_id;
    int img_left_height, img_right_height;
    int img_left_width, img_right_width;

    //> The pointers/data below need to be public since Pipeline requires accessing them
    Precomputed_Edge_Patches_Photometry* d_final_left_patches = nullptr;
    Precomputed_Edge_Patches_Photometry* d_final_right_patches = nullptr;
    Precomputed_Edge_SIFT_Descriptor_GPU* d_final_left_sift_descriptors = nullptr;
    //> Per-left-edge SIFT table (valid until stereo engine is reset); used to build mate-ordered SIFT for temporal.
    Precomputed_Edge_SIFT_Descriptor_GPU* d_left_sift_descriptors = nullptr;
    Merged_Refined_Stereo_Match_GPU* d_merged_refined_matches = nullptr;
    int h_match_count_passing_NCC_filter = 0;

    // timing
    float time_EP;
    cudaEvent_t start, stop;

private:
    //> Pointers to the host memory
    float* h_right_img = nullptr;
    Edge_GPU* h_left_edges = nullptr;
    Edge_GPU* h_right_edges = nullptr;
    int* h_cell_ids = nullptr;
    int h_match_count_passing_EP_LP_OR_filters = 0;
    // int h_match_count_passing_NCC_filter = 0;
    int h_match_count_passing_SIFT_filter = 0;
    int h_match_count_passing_BNB_NCC_filter = 0;
    int h_match_count_passing_BNB_SIFT_filter = 0;
    int h_match_count_merged_after_refine = 0;
    float h_F_row_major_[9] = {};
    float* h_F = h_F_row_major_;
    Match_by_Edge_Index* h_matches_out = nullptr;

    //> Pointers to the device memory
    Edge_GPU* d_left_edges = nullptr;
    Edge_GPU* d_right_edges = nullptr;
    int* d_cell_ids = nullptr;
    int* d_edge_indices = nullptr;
    int* d_cell_start_idx = nullptr;
    int* d_num_of_edges_in_cell = nullptr;
    int* d_match_count_passing_EP_LP_OR_filters = nullptr;
    Match_by_Edge_Index* d_final_matches_passing_EP_LP_OR_filters = nullptr;
    float* d_F = nullptr;
    Precomputed_Edge_Patches_Photometry* d_left_patches = nullptr;
    int* d_final_count_passing_NCC_filter = nullptr;
    Match_by_Edge_Index* d_final_matches_passing_NCC_filter = nullptr;
    float* d_ncc_scores_passing_NCC_filter = nullptr;
    int* d_final_count_passing_SIFT_filter = nullptr;
    Match_by_Edge_Index* d_final_matches_passing_SIFT_filter = nullptr;
    float* d_sift_scores_passing_SIFT_filter = nullptr;
    float* d_ncc_scores_passing_SIFT_filter = nullptr;
    int* d_final_count_passing_BNB_NCC_filter = nullptr;
    Match_by_Edge_Index* d_final_matches_passing_BNB_NCC_filter = nullptr;
    float* d_ncc_scores_passing_BNB_NCC_filter = nullptr;
    int* d_final_count_passing_BNB_SIFT_filter = nullptr;
    Match_by_Edge_Index* d_final_matches_passing_BNB_SIFT_filter = nullptr;
    float* d_sift_scores_passing_BNB_SIFT_filter = nullptr;
    int* d_merged_refined_count = nullptr;
    Merged_Refined_Stereo_Match_GPU* d_prev_merged_matches = nullptr;
    int* d_prev_merged_count = nullptr;
    float* d_ncc_scores_passing_second_NCC_filter = nullptr;
    CUDA_Texture_Wrapper left_image_texture_ = {nullptr, 0};
    CUDA_Texture_Wrapper right_image_texture_ = {nullptr, 0};

    int num_of_grid_cells_in_width;
    int num_of_grid_cells_in_height;
    int total_num_of_grid_cells;
    int max_stereo_edge_matches_after_epipolar_rasterization;
    size_t num_left_edges_ = 0;
    size_t num_right_edges_ = 0;

    Dataset::Ptr dataset_;
    Eigen::Matrix3f h_F_;

    /** Valid for comparisons only if the StereoFrame outlives this pipeline instance (same as Pipeline::current_frame). */
    const StereoFrame* stereo_frame_ptr_ = nullptr;

    // void allocate_image_textures();
    // void release_image_textures();
    void release_merged_refined_outputs();

    // //> GPU results validation
    // Stereo_Matches_GPU_Test_Context make_test_context() const;
    // void test_spatial_grid_mapping() const;
    // void test_epipolar_rasterization() const;
    // void test_EP_LP_OR_vs_stereo_matches_cpu() const;
    // void test_ncc_filter() const;
    // void test_SIFT_descriptor_filter() const;
};

#endif
