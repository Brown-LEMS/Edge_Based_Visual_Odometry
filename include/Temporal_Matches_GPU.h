#ifndef TEMPORAL_MATCHES_GPU_HPP
#define TEMPORAL_MATCHES_GPU_HPP

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
#include "Dataset.h"

// ---------------------------------------------------------------------------
// Timing breakdown for the temporal GPU pipeline
// ---------------------------------------------------------------------------
struct Temporal_Matches_GPU_Timing_Statistics
{
    float time_build_cf_left_spatial_grid;      //> CF-left edge spatial grid (device-resident edges)
    float time_temporal_candidate_generation;   //> spatial-grid intersection + orientation filter
    float time_apply_ncc_filter;                //> Combined left-left + right-right NCC on temporal quads
    float time_apply_sift_filter;               //> Combined left-left + right-right SIFT on temporal quads
    float time_apply_bnb_ncc_filter;            //> Best-nearly-best on NCC scores
    float time_apply_bnb_sift_filter;           //> Best-nearly-best on SIFT scores
    float time_apply_photometric_refine;        //> 2D Gauss-Newton refinement + GPU clustering
    float total_time;
};

// --------------------------------------------------------------------------------------------------
// GPU temporal quad matching pipeline.
//
// Matching unit: a "quad" is (kf_mate_idx, cf_mate_idx), i.e. one KF stereo mate
// paired with one CF stereo mate. Each mate supplies left+right edges (4 edges total).
// Device geometry/photometry comes from Merged_Refined_Stereo_Match_GPU arrays.
//
// Prerequisites (from stereo GPU, not run here):
//   - KF/CF mate-indexed left+right photometry patches
//   - KF/CF final stereo mates on device; CF-left spatial grid inputs
//
// Pipeline (filter order mirrors run_temporal_quad_pipeline_filters; spatial+OR fused on GPU):
//   1. Build CF-left spatial grid from CF stereo mates
//   2. Candidate generation: left-grid search + right box check + left/right OR filters
//   3. NCC: left-left and right-right must both pass (mate-indexed patches)
//   4. SIFT: left-left (precomputed) + right-right (on-the-fly from right textures)
//   5. BNB-NCC on SIFT survivors, then BNB-SIFT on BNB-NCC survivors (best per KF mate)
//   6. 2D photometric refinement + GPU clustering
//
// Match_by_Edge_Index: { left_edge_idx = kf_mate_idx, right_edge_idx = cf_mate_idx }
// --------------------------------------------------------------------------------------------------
class Temporal_Matches_GPU_Pipeline {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Temporal_Matches_GPU_Pipeline> Ptr;

    //> Constructor
    Temporal_Matches_GPU_Pipeline(
        Dataset::Ptr dataset, const StereoFrame& current_frame, 
        const int num_of_kf_stereo_matches, const int num_of_cf_stereo_matches,
        const Merged_Refined_Stereo_Match_GPU* d_kf_stereo_matches, const Merged_Refined_Stereo_Match_GPU* d_cf_stereo_matches,
        Precomputed_Edge_Patches_Photometry* d_kf_left_patches_in,
        Precomputed_Edge_Patches_Photometry* d_cf_left_patches_in,
        Precomputed_Edge_Patches_Photometry* d_kf_right_patches_in,
        Precomputed_Edge_Patches_Photometry* d_cf_right_patches_in,
        Precomputed_Edge_SIFT_Descriptor_GPU* d_kf_left_sift_descriptors_in,
        Precomputed_Edge_SIFT_Descriptor_GPU* d_cf_left_sift_descriptors_in,
        cudaTextureObject_t kf_left_image_tex,
        cudaTextureObject_t kf_right_image_tex,
        cudaTextureObject_t cf_right_image_tex,
        cudaTextureObject_t cf_left_image_tex,
        int device);

    //> Destructor
    ~Temporal_Matches_GPU_Pipeline();

    //> Run the full GPU temporal quad filter pipeline for KF->CF
    Temporal_Matches_GPU_Timing_Statistics get_temporal_edge_matching_GPU( );

    //> Copy final temporal quads to host, sorted by ascending quads-per-KF-mate (1 = highest rank).
    void retrieve_final_matches(std::vector<Match_by_Edge_Index>& out_matches) const;
    void retrieve_final_hypothesis_matches(std::vector<Refined_Edge_Hypothesis_Match_GPU>& out_matches) const;

    //> Write final temporal matches to a text file for offline inspection.
    void write_finalized_matches_to_file(size_t kf_frame_idx, size_t cf_frame_idx) const;

    int device_id;
    int img_left_height;
    int img_left_width;

    cudaEvent_t start, stop;

private:
    size_t num_kf_mates_          = 0;
    size_t num_cf_mates_          = 0;
    int    num_of_grid_cells_in_width  = 0;
    int    num_of_grid_cells_in_height = 0;
    int    total_num_of_grid_cells     = 0;
    int    max_temporal_candidates     = 0;  //> = num_kf_mates_ * FACTOR_OF_NUM_OF_KF_MATES_FOR_TEMPORAL_CANDIDATES

    int* h_cf_left_cell_ids = nullptr;   //> output of spatial grid mapping (host-side cell id per CF-left edge)

    //> Temporal quad counts (host-side bookkeeping)
    int h_temporal_candidate_count    = 0;
    int h_match_count_after_ncc = 0;
    int h_match_count_after_sift = 0;
    int h_match_count_after_bnb_ncc   = 0;
    int h_match_count_after_bnb_sift  = 0;

    // ----- device: final stereo mates (borrowed from Pipeline; not owned) -----
    const Merged_Refined_Stereo_Match_GPU* d_kf_stereo_matches = nullptr;
    const Merged_Refined_Stereo_Match_GPU* d_cf_stereo_matches = nullptr;

    // ----- device: CF-left spatial grid (grid built on CF-left edge positions) ---
    int* d_cf_left_cell_ids         = nullptr;
    int* d_cf_left_edge_indices     = nullptr;   //> sorted by cell
    int* d_cf_left_cell_start_idx   = nullptr;
    int* d_cf_left_num_in_cell      = nullptr;

    //> Temporal candidate buffer (output temporal edge matches)
    Match_by_Edge_Index* d_temporal_candidates      = nullptr;
    int*                 d_temporal_candidate_count = nullptr;

    // ----- device: precomputed photometry patches ------------------------
    Precomputed_Edge_Patches_Photometry* d_kf_left_patches  = nullptr;
    Precomputed_Edge_Patches_Photometry* d_cf_left_patches  = nullptr;
    Precomputed_Edge_Patches_Photometry* d_kf_right_patches = nullptr;
    Precomputed_Edge_Patches_Photometry* d_cf_right_patches = nullptr;
    bool owns_kf_left_patches_ = false;
    bool owns_kf_right_patches_ = false;
    bool owns_kf_left_sift_descriptors_ = false;
    bool owns_cf_left_sift_descriptors_ = false;
    Precomputed_Edge_SIFT_Descriptor_GPU* d_kf_left_sift_descriptors  = nullptr;
    Precomputed_Edge_SIFT_Descriptor_GPU* d_cf_left_sift_descriptors  = nullptr;
    cudaTextureObject_t kf_left_image_tex_ = 0;
    cudaTextureObject_t kf_right_image_tex_ = 0;
    cudaTextureObject_t cf_right_image_tex_ = 0;
    cudaTextureObject_t cf_left_image_tex_  = 0;
    int right_img_width_  = 0;
    int right_img_height_ = 0;

    // ----- device: NCC filter outputs (lazily allocated per stage) -------
    Match_by_Edge_Index* d_matches_after_ncc  = nullptr;
    int*                 d_count_after_ncc    = nullptr;
    float*               d_ncc_left_scores  = nullptr;
    float*               d_ncc_right_scores = nullptr;

    // ----- device: SIFT filter outputs (lazily allocated per stage) ------
    Match_by_Edge_Index* d_matches_after_sift  = nullptr;
    int*                 d_count_after_sift    = nullptr;
    float*               d_sift_left_scores  = nullptr;
    float*               d_sift_right_scores   = nullptr;
    float*               d_ncc_left_scores_after_sift = nullptr;

    // ----- device: BNB filter outputs (lazily allocated per stage) -------
    Match_by_Edge_Index* d_matches_after_bnb_ncc  = nullptr;
    float*               d_sift_left_scores_after_bnb_ncc = nullptr;
    int*                 d_count_after_bnb_ncc     = nullptr;
    Match_by_Edge_Index* d_matches_after_bnb_sift  = nullptr;
    int*                 d_count_after_bnb_sift    = nullptr;

    // ----- device: photometric refinement + clustering (lazily allocated) ------
    Temporal_Refined_Match_GPU* d_refined_matches         = nullptr;
    int*                        d_refined_count           = nullptr;
    int                         h_match_count_after_refine = 0;
    Refined_Edge_Hypothesis_Match_GPU* d_clustered_matches = nullptr;
    int*                        d_clustered_count         = nullptr;
    int                         h_match_count_after_cluster = 0;

    Dataset::Ptr dataset_;

    float build_cf_left_spatial_grid();
    float generate_temporal_candidates();
    float apply_ncc_filter();
    float apply_sift_filter();
    float apply_bnb_ncc_filter();
    float apply_bnb_sift_filter();
    float apply_photometric_refine_and_cluster();
    void ensure_mate_left_sift_descriptors();

    void free_lazy_allocations();
};

#endif // TEMPORAL_MATCHES_GPU_HPP
