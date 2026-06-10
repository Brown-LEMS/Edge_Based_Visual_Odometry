#include <algorithm>
#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include "definitions.h"
#include "Temporal_Matches_GPU.h"
#include "gpu_kernels.h"
#include "prepare_for_GPU.h"
#include "Dataset.h"

Temporal_Matches_GPU_Pipeline::Temporal_Matches_GPU_Pipeline(
    Dataset::Ptr dataset, const StereoFrame& current_frame,
    const int num_of_kf_stereo_matches, const int num_of_cf_stereo_matches,
    const Merged_Refined_Stereo_Match_GPU* d_kf_stereo_matches, 
    const Merged_Refined_Stereo_Match_GPU* d_cf_stereo_matches,
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
    int device)
    : device_id(device)
    , num_kf_mates_(num_of_kf_stereo_matches)
    , num_cf_mates_(num_of_cf_stereo_matches)
    , img_left_height(mat_rows_prefer_undistorted(current_frame.left_image_undistorted, current_frame.left_image))
    , img_left_width(mat_cols_prefer_undistorted(current_frame.left_image_undistorted, current_frame.left_image))
    , d_kf_left_patches(d_kf_left_patches_in)
    , d_cf_left_patches(d_cf_left_patches_in)
    , d_kf_right_patches(d_kf_right_patches_in)
    , d_cf_right_patches(d_cf_right_patches_in)
    , d_kf_left_sift_descriptors(d_kf_left_sift_descriptors_in)
    , d_cf_left_sift_descriptors(d_cf_left_sift_descriptors_in)
    , kf_left_image_tex_(kf_left_image_tex)
    , kf_right_image_tex_(kf_right_image_tex)
    , cf_right_image_tex_(cf_right_image_tex)
    , cf_left_image_tex_(cf_left_image_tex)
    , right_img_height_(mat_rows_prefer_undistorted(current_frame.right_image_undistorted, current_frame.right_image))
    , right_img_width_(mat_cols_prefer_undistorted(current_frame.right_image_undistorted, current_frame.right_image))
    , d_kf_stereo_matches(d_kf_stereo_matches)
    , d_cf_stereo_matches(d_cf_stereo_matches)
    , dataset_(std::move(dataset))
{
    //> Grid dimensions
    num_of_grid_cells_in_height = static_cast<int>(std::ceil(static_cast<double>(img_left_height) / GRID_CELL_SIZE));
    num_of_grid_cells_in_width  = static_cast<int>(std::ceil(static_cast<double>(img_left_width)  / GRID_CELL_SIZE));
    total_num_of_grid_cells     = num_of_grid_cells_in_width * num_of_grid_cells_in_height;

    max_temporal_candidates = static_cast<int>(num_kf_mates_) * FACTOR_OF_NUM_OF_KF_MATES_FOR_TEMPORAL_CANDIDATES;

    //> Allocate CF-left spatial grid buffers
    //> CH Notes: in temporal edge matching, we don't need CF-right spatial grid buffers as we already own CF stereo mates,
    //  so once a CF-left edge is anchored, its corresponding CF-right edge is identified
    cudacheck(cudaMalloc((void**)&d_cf_left_cell_ids,       num_cf_mates_ * sizeof(int)));
    cudacheck(cudaMalloc((void**)&d_cf_left_edge_indices,   num_cf_mates_ * sizeof(int)));
    cudacheck(cudaMalloc((void**)&d_cf_left_cell_start_idx, total_num_of_grid_cells * sizeof(int)));
    cudacheck(cudaMalloc((void**)&d_cf_left_num_in_cell,    total_num_of_grid_cells * sizeof(int)));

    //> Allocate temporal candidate buffer; this is a pre-sized upper bound
    cudacheck(cudaMalloc((void**)&d_temporal_candidates, static_cast<size_t>(max_temporal_candidates) * sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc((void**)&d_temporal_candidate_count, sizeof(int)));

    //> Set the temporal candidate buffer to zero
    cudacheck(cudaMemset(d_temporal_candidates, 0, static_cast<size_t>(max_temporal_candidates) * sizeof(Match_by_Edge_Index)));

    //> Create CUDA timing events
    cudacheck(cudaEventCreate(&start));
    cudacheck(cudaEventCreate(&stop));
}

// ===========================================================================
// Main Temporal Edge Matching Pipeline Orchestrator
// ===========================================================================
Temporal_Matches_GPU_Timing_Statistics Temporal_Matches_GPU_Pipeline::get_temporal_edge_matching_GPU( )
{
    Temporal_Matches_GPU_Timing_Statistics GPU_time_collection;
    GPU_time_collection.total_time = 0.0f;

    //> CF-left spatial grid
    GPU_time_collection.time_build_cf_left_spatial_grid = build_cf_left_spatial_grid();
    GPU_time_collection.total_time += GPU_time_collection.time_build_cf_left_spatial_grid;

    //> Temporal candidate generation (spatial intersection + OR filter)
    GPU_time_collection.time_temporal_candidate_generation = generate_temporal_candidates();
    GPU_time_collection.total_time += GPU_time_collection.time_temporal_candidate_generation;
    std::cout << "[Temporal GPU] Candidates after spatial+OR filter: " << h_temporal_candidate_count << std::endl;

    //> Combined NCC left-left + right-right (both must pass per candidate)
    GPU_time_collection.time_apply_ncc_filter = apply_ncc_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_ncc_filter;
    std::cout << "[Temporal GPU] Candidates after NCC (left+right): " << h_match_count_after_ncc << std::endl;

    //> Combined SIFT left-left + right-right (both must pass per candidate)
    GPU_time_collection.time_apply_sift_filter = apply_sift_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_sift_filter;
    std::cout << "[Temporal GPU] Candidates after SIFT (left+right): " << h_match_count_after_sift << std::endl;

    //> BNB-NCC (on SIFT survivors)
    GPU_time_collection.time_apply_bnb_ncc_filter = apply_bnb_ncc_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_bnb_ncc_filter;
    std::cout << "[Temporal GPU] Candidates after BNB-NCC: " << h_match_count_after_bnb_ncc << std::endl;

    //> BNB-SIFT (on BNB-NCC survivors)
    GPU_time_collection.time_apply_bnb_sift_filter = apply_bnb_sift_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_bnb_sift_filter;
    std::cout << "[Temporal GPU] Candidates after BNB-SIFT: " << h_match_count_after_bnb_sift << std::endl;

    //> 2D photometric refinement + GPU clustering (final stage)
    GPU_time_collection.time_apply_photometric_refine = apply_photometric_refine_and_cluster();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_photometric_refine;
    std::cout << "[Temporal GPU] Candidates after photometric refinement: " << h_match_count_after_refine  << std::endl;
    std::cout << "[Temporal GPU] Final matches after clustering:          " << h_match_count_after_cluster << std::endl;

#if SHOW_TIMING_BREAKDOWN
    std::cout.precision(3);
    std::cout << "Temporal GPU timing breakdown" << std::endl;
    std::cout << "  build_cf_left_spatial_grid:      " << GPU_time_collection.time_build_cf_left_spatial_grid      << " ms" << std::endl;
    std::cout << "  temporal_candidate_generation:   " << GPU_time_collection.time_temporal_candidate_generation   << " ms" << std::endl;
    std::cout << "  NCC filter (left+right):         " << GPU_time_collection.time_apply_ncc_filter                << " ms" << std::endl;
    std::cout << "  SIFT filter (left+right):        " << GPU_time_collection.time_apply_sift_filter               << " ms" << std::endl;
    std::cout << "  BNB-NCC filter:                  " << GPU_time_collection.time_apply_bnb_ncc_filter            << " ms" << std::endl;
    std::cout << "  BNB-SIFT filter:                 " << GPU_time_collection.time_apply_bnb_sift_filter           << " ms" << std::endl;
    std::cout << "  photometric refine + cluster:    " << GPU_time_collection.time_apply_photometric_refine        << " ms" << std::endl;
    std::cout << "  TOTAL:                           " << GPU_time_collection.total_time                           << " ms" << std::endl;
#endif

    LOG_INFO("Temporal GPU total time: " + std::to_string(GPU_time_collection.total_time) + " (ms)");
    std::cout << "Final quad matches: " << h_match_count_after_cluster << std::endl;

    return GPU_time_collection;
}

// ---------------------------------------------------------------------------
// Stage 1: Build the CF-left spatial grid from final CF stereo mates.
//          After this call d_cf_left_edge_indices / d_cf_left_cell_start_idx / d_cf_left_num_in_cell are ready.
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::build_cf_left_spatial_grid()
{
    if (d_cf_stereo_matches == nullptr || num_cf_mates_ == 0) {
        LOG_WARNING("build_cf_left_spatial_grid: no CF stereo matches on device");
        return 0.0f;
    }

    return temporal_stereo_mate_spatial_map_device_thrust_pipeline(
        device_id,
        d_cf_stereo_matches,
        d_cf_left_cell_ids,
        d_cf_left_edge_indices,
        d_cf_left_cell_start_idx,
        d_cf_left_num_in_cell,
        num_cf_mates_,
        total_num_of_grid_cells,
        num_of_grid_cells_in_width,
        num_of_grid_cells_in_height,
        start,
        stop);
}

// ---------------------------------------------------------------------------
// Stage 2: Temporal candidate generation (spatial grid intersection + OR filter).
//          Reads KF/CF geometry from final stereo mates; cf_mate_idx indexes d_cf_stereo_matches.
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::generate_temporal_candidates()
{
    if (d_kf_stereo_matches == nullptr || d_cf_stereo_matches == nullptr) {
        LOG_WARNING("generate_temporal_candidates: missing KF/CF stereo matches on device");
        h_temporal_candidate_count = 0;
        return 0.0f;
    }

    return temporal_candidate_generation_pipeline(
        device_id,
        d_kf_stereo_matches,
        d_cf_stereo_matches,
        d_cf_left_edge_indices,
        d_cf_left_cell_start_idx,
        d_cf_left_num_in_cell,
        num_of_grid_cells_in_width,
        num_of_grid_cells_in_height,
        static_cast<int>(num_kf_mates_),
        d_temporal_candidates,
        d_temporal_candidate_count,
        h_temporal_candidate_count,
        max_temporal_candidates,
        start, stop);
}

// ---------------------------------------------------------------------------
// Stage 3: Apply NCC filter on temporal quad candidates.
//          Uses mate-indexed precomputed KF/CF patches (from final stereo GPU output).
//          Both sides must pass before a candidate is kept.
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::apply_ncc_filter()
{
    if (h_temporal_candidate_count <= 0) {
        LOG_WARNING("Temporal GPU NCC: no candidates, skipping");
        h_match_count_after_ncc = 0;
        return 0.0f;
    }

    //> Sanity check: make sure the precomputed patches are available
    if (d_kf_left_patches == nullptr || d_kf_right_patches == nullptr || d_cf_left_patches == nullptr || d_cf_right_patches == nullptr) {
        LOG_WARNING("Temporal GPU NCC: missing KF/CF precomputed patches");
        h_match_count_after_ncc = 0;
        return 0.0f;
    }

    return apply_temporal_quad_ncc_filter_pipeline(
        device_id,
        h_match_count_after_ncc,
        d_temporal_candidates,
        d_kf_left_patches,
        d_kf_right_patches,
        d_cf_left_patches,
        d_cf_right_patches,
        d_matches_after_ncc,
        d_count_after_ncc,
        d_ncc_left_scores,
        d_ncc_right_scores,
        h_temporal_candidate_count,
        static_cast<int>(num_kf_mates_),
        static_cast<int>(num_cf_mates_),
        start, stop);
}

// ---------------------------------------------------------------------------
// Ensure mate-indexed left SIFT buffers exist (borrowed from stereo or built here).
// ---------------------------------------------------------------------------
void Temporal_Matches_GPU_Pipeline::ensure_mate_left_sift_descriptors()
{
    if (!owns_kf_left_sift_descriptors_
        && d_kf_stereo_matches != nullptr
        && num_kf_mates_ > 0
        && kf_left_image_tex_ != 0) {
        Precomputed_Edge_SIFT_Descriptor_GPU* rebuilt_kf_sift = nullptr;
        precompute_mate_left_sift_from_texture_pipeline(
            device_id,
            d_kf_stereo_matches,
            kf_left_image_tex_,
            img_left_width,
            img_left_height,
            static_cast<int>(num_kf_mates_),
            rebuilt_kf_sift);
        if (rebuilt_kf_sift != nullptr) {
            d_kf_left_sift_descriptors = rebuilt_kf_sift;
            owns_kf_left_sift_descriptors_ = true;
        }
        if (owns_kf_left_sift_descriptors_) {
            std::cout << "[Temporal GPU] Built KF mate-ordered left SIFT from keyframe texture ("
                      << num_kf_mates_ << " mates)" << std::endl;
        }
    }

    if (!owns_cf_left_sift_descriptors_
        && d_cf_stereo_matches != nullptr
        && num_cf_mates_ > 0
        && cf_left_image_tex_ != 0) {
        Precomputed_Edge_SIFT_Descriptor_GPU* rebuilt_cf_sift = nullptr;
        precompute_mate_left_sift_from_texture_pipeline(
            device_id,
            d_cf_stereo_matches,
            cf_left_image_tex_,
            img_left_width,
            img_left_height,
            static_cast<int>(num_cf_mates_),
            rebuilt_cf_sift);
        if (rebuilt_cf_sift != nullptr) {
            d_cf_left_sift_descriptors = rebuilt_cf_sift;
            owns_cf_left_sift_descriptors_ = true;
        }
        if (owns_cf_left_sift_descriptors_) {
            std::cout << "[Temporal GPU] Built CF mate-ordered left SIFT from current-frame texture ("
                      << num_cf_mates_ << " mates)" << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 4: Apply SIFT filter on temporal quad candidates.
//          Left-left: mate-indexed precomputed KF/CF left descriptors (from stereo GPU).
//          Right-right: on-the-fly extraction at merged_right_* per candidate.
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::apply_sift_filter()
{
    //> A series of sanity checks before applying the SIFT filter
    if (h_match_count_after_ncc <= 0) {
        LOG_WARNING("Temporal GPU SIFT: no NCC survivors, skipping");
        h_match_count_after_sift = 0;
        return 0.0f;
    }

    ensure_mate_left_sift_descriptors();

    if (d_kf_left_sift_descriptors == nullptr || d_cf_left_sift_descriptors == nullptr) {
        LOG_WARNING("Temporal GPU SIFT: missing KF/CF left precomputed descriptors after ensure step");
        h_match_count_after_sift = 0;
        return 0.0f;
    }

    if (d_kf_stereo_matches == nullptr || d_cf_stereo_matches == nullptr
        || kf_right_image_tex_ == 0 || cf_right_image_tex_ == 0) {
        LOG_WARNING("Temporal GPU SIFT: missing stereo mates or right image textures");
        h_match_count_after_sift = 0;
        return 0.0f;
    }

    return apply_temporal_quad_sift_filter_pipeline(
        device_id,
        h_match_count_after_sift,
        d_matches_after_ncc,
        d_kf_left_sift_descriptors,
        d_cf_left_sift_descriptors,
        d_kf_stereo_matches,
        d_cf_stereo_matches,
        kf_right_image_tex_,
        cf_right_image_tex_,
        right_img_width_,
        right_img_height_,
        d_matches_after_sift,
        d_count_after_sift,
        d_sift_left_scores,
        d_sift_right_scores,
        d_ncc_left_scores,
        d_ncc_left_scores_after_sift,
        h_match_count_after_ncc,
        static_cast<int>(num_kf_mates_),
        static_cast<int>(num_cf_mates_),
        start, stop);
}

// ---------------------------------------------------------------------------
// Stage 5: BNB-NCC on SIFT survivors (best per kf_mate_idx; rank by left-left NCC).
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::apply_bnb_ncc_filter()
{
    if (h_match_count_after_sift <= 0 || d_matches_after_sift == nullptr || d_ncc_left_scores_after_sift == nullptr || d_sift_left_scores == nullptr) {
        LOG_WARNING("Temporal GPU BNB-NCC: no SIFT survivors or scores, skipping");
        h_match_count_after_bnb_ncc = 0;
        return 0.0f;
    }

    return apply_temporal_quad_bnb_filter_pipeline(
        device_id,
        h_match_count_after_bnb_ncc,
        d_matches_after_sift,
        d_ncc_left_scores_after_sift,
        d_sift_left_scores,
        d_matches_after_bnb_ncc,
        d_sift_left_scores_after_bnb_ncc,
        d_count_after_bnb_ncc,
        h_match_count_after_sift,
        static_cast<int>(num_kf_mates_),
        static_cast<float>(TEMPORAL_BNB_NCC_THRESHOLD),
        true,
        start, stop);
}

// ---------------------------------------------------------------------------
// Stage 6: BNB-SIFT on BNB-NCC survivors (best per kf_mate_idx; rank by left-left SIFT).
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::apply_bnb_sift_filter()
{
    if (h_match_count_after_bnb_ncc <= 0
        || d_matches_after_bnb_ncc == nullptr
        || d_sift_left_scores_after_bnb_ncc == nullptr) {
        LOG_WARNING("Temporal GPU BNB-SIFT: no BNB-NCC survivors or scores, skipping");
        h_match_count_after_bnb_sift = 0;
        return 0.0f;
    }

    float* sift_carry_out = nullptr;
    return apply_temporal_quad_bnb_filter_pipeline(
        device_id,
        h_match_count_after_bnb_sift,
        d_matches_after_bnb_ncc,
        d_sift_left_scores_after_bnb_ncc,
        nullptr,
        d_matches_after_bnb_sift,
        sift_carry_out,
        d_count_after_bnb_sift,
        h_match_count_after_bnb_ncc,
        static_cast<int>(num_kf_mates_),
        static_cast<float>(TEMPORAL_BNB_SIFT_THRESHOLD),
        false,
        start, stop);
}

// ---------------------------------------------------------------------------
// Stage 7: 2D photometric refinement (Gauss-Newton) + GPU clustering.
//   Input  : BNB-SIFT survivors; KF/CF geometry from stereo mates; CF image textures.
//   Output : Temporal_Refined_Match_GPU (valid only) -> clustered Refined_Edge_Hypothesis_Match_GPU.
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::apply_photometric_refine_and_cluster()
{
    if (h_match_count_after_bnb_sift <= 0 || d_matches_after_bnb_sift == nullptr ||
        d_kf_left_patches == nullptr      || d_kf_right_patches == nullptr ||
        d_kf_stereo_matches == nullptr    || d_cf_stereo_matches == nullptr) {
        LOG_WARNING("Temporal GPU refine+cluster: no BNB-SIFT survivors or missing inputs, skipping");
        h_match_count_after_refine  = 0;
        h_match_count_after_cluster = 0;
        return 0.0f;
    }

    return temporal_photometric_refine_and_cluster_pipeline(
        device_id,
        d_matches_after_bnb_sift, h_match_count_after_bnb_sift,
        d_kf_left_patches, d_kf_right_patches,
        d_kf_stereo_matches, d_cf_stereo_matches,
        cf_left_image_tex_, cf_right_image_tex_,
        img_left_width, img_left_height,
        right_img_width_, right_img_height_,
        d_refined_matches, d_refined_count, h_match_count_after_refine,
        d_clustered_matches, d_clustered_count, h_match_count_after_cluster,
        start, stop);
}

// ===========================================================================
// retrieve_final_matches / retrieve_final_hypothesis_matches
// ===========================================================================
namespace {

// Rank final quads by ambiguity: KF mates with fewer surviving quads come first
// (1 quad per KF = highest rank / most unambiguous).
void sort_final_quads_by_kf_mate_count(std::vector<Refined_Edge_Hypothesis_Match_GPU>& matches)
{
    if (matches.size() <= 1) {
        return;
    }

    std::unordered_map<int, int> quads_per_kf;
    quads_per_kf.reserve(matches.size());
    for (const auto& m : matches) {
        if (m.left_edge_idx >= 0) {
            ++quads_per_kf[m.left_edge_idx];
        }
    }

    std::stable_sort(matches.begin(), matches.end(),
        [&quads_per_kf](const Refined_Edge_Hypothesis_Match_GPU& a, const Refined_Edge_Hypothesis_Match_GPU& b) {
            const int count_a = quads_per_kf[a.left_edge_idx];
            const int count_b = quads_per_kf[b.left_edge_idx];
            if (count_a != count_b) {
                return count_a < count_b;
            }
            if (a.left_edge_idx != b.left_edge_idx) {
                return a.left_edge_idx < b.left_edge_idx;
            }
            return a.right_edge_idx < b.right_edge_idx;
        });
}

}  // namespace

void Temporal_Matches_GPU_Pipeline::retrieve_final_hypothesis_matches(
    std::vector<Refined_Edge_Hypothesis_Match_GPU>& out_matches) const
{
    out_matches.clear();
    if (d_clustered_matches == nullptr || h_match_count_after_cluster <= 0) {
        return;
    }

    out_matches.resize(static_cast<size_t>(h_match_count_after_cluster));
    cudacheck(cudaMemcpy(out_matches.data(), d_clustered_matches,
                         static_cast<size_t>(h_match_count_after_cluster) * sizeof(Refined_Edge_Hypothesis_Match_GPU),
                         cudaMemcpyDeviceToHost));

    sort_final_quads_by_kf_mate_count(out_matches);
}

void Temporal_Matches_GPU_Pipeline::retrieve_final_matches(
    std::vector<Match_by_Edge_Index>& out_matches) const
{
    out_matches.clear();

    if (d_clustered_matches != nullptr && h_match_count_after_cluster > 0) {
        std::vector<Refined_Edge_Hypothesis_Match_GPU> hypotheses;
        retrieve_final_hypothesis_matches(hypotheses);
        out_matches.reserve(hypotheses.size());
        for (const auto& h : hypotheses) {
            out_matches.push_back({h.left_edge_idx, h.right_edge_idx});
        }
        return;
    }

    if (d_matches_after_bnb_sift != nullptr && h_match_count_after_bnb_sift > 0) {
        out_matches.resize(static_cast<size_t>(h_match_count_after_bnb_sift));
        cudacheck(cudaMemcpy(out_matches.data(), d_matches_after_bnb_sift,
                             static_cast<size_t>(h_match_count_after_bnb_sift) * sizeof(Match_by_Edge_Index),
                             cudaMemcpyDeviceToHost));
    }
}

// ===========================================================================
// write_finalized_matches_to_file
// ===========================================================================
void Temporal_Matches_GPU_Pipeline::write_finalized_matches_to_file(
    size_t kf_frame_idx, size_t cf_frame_idx) const
{
    std::vector<Refined_Edge_Hypothesis_Match_GPU> hypotheses;
    retrieve_final_hypothesis_matches(hypotheses);
    if (hypotheses.empty()) {
        LOG_WARNING("write_finalized_matches_to_file: no final temporal matches to write");
        return;
    }

    const std::string out_dir = (dataset_ != nullptr) ? dataset_->get_output_path() : std::string("output_files");
    const std::string fname   = out_dir + "/gpu_temporal_quad_matches_kf"
                                + std::to_string(kf_frame_idx) + "_cf"
                                + std::to_string(cf_frame_idx) + ".txt";
    std::ofstream ofs(fname);
    if (!ofs.is_open()) {
        LOG_ERROR("write_finalized_matches_to_file: cannot open " + fname);
        return;
    }

    ofs << "kf_mate_idx cf_mate_idx "
           "kf_left_x kf_left_y kf_left_orientation "
           "kf_right_x kf_right_y kf_right_orientation "
           "cf_left_x cf_left_y cf_left_orientation "
           "cf_right_x cf_right_y cf_right_orientation\n";
    ofs.precision(8);

    for (size_t i = 0; i < hypotheses.size(); ++i) {
        const int kf_idx = hypotheses[i].left_edge_idx;
        const int cf_idx = hypotheses[i].right_edge_idx;
        if (kf_idx < 0 || cf_idx < 0
            || kf_idx >= static_cast<int>(num_kf_mates_)
            || cf_idx >= static_cast<int>(num_cf_mates_)) {
            continue;
        }
        Merged_Refined_Stereo_Match_GPU kf_mate{};
        Merged_Refined_Stereo_Match_GPU cf_mate{};
        if (cudaMemcpy(&kf_mate, d_kf_stereo_matches + kf_idx, sizeof(Merged_Refined_Stereo_Match_GPU), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(&cf_mate, d_cf_stereo_matches + cf_idx, sizeof(Merged_Refined_Stereo_Match_GPU), cudaMemcpyDeviceToHost) != cudaSuccess) {
            continue;
        }
        ofs << kf_idx << " " << cf_idx << " "
            << kf_mate.left_location_x << " " << kf_mate.left_location_y << " " << kf_mate.left_orientation << " "
            << kf_mate.merged_right_x << " " << kf_mate.merged_right_y << " " << kf_mate.merged_right_orientation << " "
            << cf_mate.left_location_x << " " << cf_mate.left_location_y << " " << cf_mate.left_orientation << " "
            << cf_mate.merged_right_x << " " << cf_mate.merged_right_y << " " << cf_mate.merged_right_orientation << "\n";
    }
    ofs.close();
    std::cout << "[Temporal GPU] Wrote " << hypotheses.size()
              << " temporal quad matches to " << fname
              << " (sorted by quads-per-KF-mate ascending; highest rank = most unambiguous)" << std::endl;
}

void Temporal_Matches_GPU_Pipeline::free_lazy_allocations()
{
    if (owns_kf_left_patches_)          safe_free(d_kf_left_patches);
    if (owns_kf_right_patches_)         safe_free(d_kf_right_patches);
    if (owns_kf_left_sift_descriptors_) safe_free(d_kf_left_sift_descriptors);
    if (owns_cf_left_sift_descriptors_) safe_free(d_cf_left_sift_descriptors);
    owns_kf_left_patches_           = false;
    owns_kf_right_patches_          = false;
    owns_kf_left_sift_descriptors_  = false;
    owns_cf_left_sift_descriptors_  = false;
    safe_free(d_matches_after_ncc);
    safe_free(d_count_after_ncc);
    safe_free(d_ncc_left_scores);
    safe_free(d_ncc_right_scores);

    safe_free(d_matches_after_sift);
    safe_free(d_count_after_sift);
    safe_free(d_sift_left_scores);
    safe_free(d_sift_right_scores);
    safe_free(d_ncc_left_scores_after_sift);

    safe_free(d_matches_after_bnb_ncc);
    safe_free(d_sift_left_scores_after_bnb_ncc);
    safe_free(d_count_after_bnb_ncc);
    safe_free(d_matches_after_bnb_sift);
    safe_free(d_count_after_bnb_sift);

    safe_free(d_refined_matches);
    safe_free(d_refined_count);
    safe_free(d_clustered_matches);
    safe_free(d_clustered_count);
    h_match_count_after_refine  = 0;
    h_match_count_after_cluster = 0;
}

Temporal_Matches_GPU_Pipeline::~Temporal_Matches_GPU_Pipeline()
{
    //> Frees buffers that stages allocate lazily inside the temporal GPU pipeline, 
    //  and only when the pipeline owns them
    free_lazy_allocations();

    if (d_cf_left_cell_ids)       cudacheck_noexcept(cudaFree(d_cf_left_cell_ids));
    if (d_cf_left_edge_indices)   cudacheck_noexcept(cudaFree(d_cf_left_edge_indices));
    if (d_cf_left_cell_start_idx) cudacheck_noexcept(cudaFree(d_cf_left_cell_start_idx));
    if (d_cf_left_num_in_cell)    cudacheck_noexcept(cudaFree(d_cf_left_num_in_cell));

    if (d_temporal_candidates)      cudacheck_noexcept(cudaFree(d_temporal_candidates));
    if (d_temporal_candidate_count) cudacheck_noexcept(cudaFree(d_temporal_candidate_count));

    cudacheck_noexcept(cudaEventDestroy(start));
    cudacheck_noexcept(cudaEventDestroy(stop));
}
